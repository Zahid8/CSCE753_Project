import argparse
import copy
import csv
import json
import math
import os
import time
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from data import PairedImageDataset, UnpairedImageDataset
from losses import CompositeLoss
from metrics import psnr, ssim_metric, uciqe_proxy_metric, uiqm_proxy_metric
from models import PhysGuidedUFormer
from utils import load_checkpoint, save_checkpoint, set_seed


os.environ.setdefault("KMP_DISABLE_SHM", "1")


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not torch.is_floating_point(v):
                v.copy_(msd[k])
            else:
                v.mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)


def save_image(tensor, path):
    tensor = torch.clamp(tensor, 0.0, 1.0)
    img = (tensor * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(path)


def image_to_tensor(img):
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.ascontiguousarray(arr)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def make_named_run_dir(base_dir, prefix):
    ts = time.strftime("%H%M%S_%d%m%Y")
    run_dir = os.path.join(base_dir, f"{prefix}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_grid(images, path):
    if not images:
        return
    h, w = images[0].shape[1], images[0].shape[2]
    grid = torch.zeros(3, h, w * len(images))
    for i, img in enumerate(images):
        grid[:, :, i * w : (i + 1) * w] = img
    save_image(grid, path)


def build_loader(dataset, batch_size, workers, shuffle, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
        drop_last=drop_last,
    )


def _split_csv_paths(text):
    if not text:
        return []
    return [p.strip() for p in text.split(",") if p.strip()]


def build_paired_dataset(degraded_paths, clean_paths, crop, augment):
    if len(degraded_paths) != len(clean_paths):
        raise ValueError(
            f"Paired path count mismatch: {len(degraded_paths)} degraded vs {len(clean_paths)} clean"
        )
    datasets = [
        PairedImageDataset(d, c, crop_size=crop, augment=augment)
        for d, c in zip(degraded_paths, clean_paths)
    ]
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def cosine_lr(base_lr, min_lr, epoch, max_epoch):
    if max_epoch <= 1:
        return base_lr
    ratio = float(epoch) / float(max_epoch - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


def validate(model, loader, device):
    model.eval()
    psnr_vals, ssim_vals, uciqe_vals, uiqm_vals = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred, _ = model(x)
            psnr_vals.append(psnr(pred, y))
            ssim_vals.append(ssim_metric(pred, y))
            uciqe_vals.append(uciqe_proxy_metric(pred))
            uiqm_vals.append(uiqm_proxy_metric(pred))

    metrics = {
        "psnr_mean": float(sum(psnr_vals) / max(1, len(psnr_vals))),
        "ssim_mean": float(sum(ssim_vals) / max(1, len(ssim_vals))),
        "uciqe_proxy_mean": float(sum(uciqe_vals) / max(1, len(uciqe_vals))),
        "uiqm_proxy_mean": float(sum(uiqm_vals) / max(1, len(uiqm_vals))),
    }
    print(
        "Val PSNR {:.3f} SSIM {:.4f} UCIQE_proxy {:.4f} UIQM_proxy {:.4f}".format(
            metrics["psnr_mean"],
            metrics["ssim_mean"],
            metrics["uciqe_proxy_mean"],
            metrics["uiqm_proxy_mean"],
        ),
        flush=True,
    )
    return metrics


def train(args):
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    train_root = os.path.join(args.out_dir, "train")
    os.makedirs(train_root, exist_ok=True)
    run_dir = make_named_run_dir(train_root, "train")

    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    writer = None
    if args.tb and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))

    model = PhysGuidedUFormer(
        in_ch=3,
        out_ch=3,
        base=args.base,
        depth=args.depth,
        blocks_per_stage=args.blocks_per_stage,
    ).to(device)

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    amp_enabled = args.amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = CompositeLoss(
        w_l1=args.w_l1,
        w_ssim=args.w_ssim,
        w_edge=args.w_edge,
        w_nr=args.w_nr,
        use_charbonnier=not args.use_l1_only,
    )

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler, map_location=device)
        if ema is not None:
            ema.ema.load_state_dict(model.state_dict())

    paired_loader = None
    unpaired_loader = None
    paired_degraded_paths = _split_csv_paths(args.paired_degraded)
    paired_clean_paths = _split_csv_paths(args.paired_clean)
    if paired_degraded_paths and paired_clean_paths:
        paired_ds = build_paired_dataset(
            paired_degraded_paths,
            paired_clean_paths,
            crop=args.crop,
            augment=True,
        )
        paired_loader = build_loader(
            paired_ds, args.batch, args.workers, shuffle=True, drop_last=True
        )

    if args.unpaired_degraded:
        unpaired_ds = UnpairedImageDataset(args.unpaired_degraded, args.crop, augment=True)
        unpaired_loader = build_loader(
            unpaired_ds, args.batch, args.workers, shuffle=True, drop_last=True
        )

    if paired_loader is None and unpaired_loader is None:
        raise ValueError("Provide paired and/or unpaired training data paths")
    if paired_loader is None and args.w_nr <= 0:
        raise ValueError("Unpaired training requires --w-nr > 0, or provide paired data.")

    val_loader = None
    val_degraded_paths = _split_csv_paths(args.val_degraded)
    val_clean_paths = _split_csv_paths(args.val_clean)
    if val_degraded_paths and val_clean_paths:
        if len(val_degraded_paths) == 1 and len(val_clean_paths) == 1:
            val_ds = PairedImageDataset(
                val_degraded_paths[0],
                val_clean_paths[0],
                args.val_crop,
                augment=False,
                resize_eval=True,
            )
        else:
            val_ds = build_paired_dataset(
                val_degraded_paths,
                val_clean_paths,
                crop=args.val_crop,
                augment=False,
            )
        val_loader = build_loader(
            val_ds, args.val_batch, args.workers, shuffle=False, drop_last=False
        )

    best_psnr = -1e9
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        lr_now = cosine_lr(args.lr, args.min_lr, epoch, args.epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_ssim = 0.0
        total_edge = 0.0
        steps = 0

        paired_iter = iter(paired_loader) if paired_loader else None
        unpaired_iter = iter(unpaired_loader) if unpaired_loader else None
        max_steps = max(
            len(paired_loader) if paired_loader else 0,
            len(unpaired_loader) if unpaired_loader else 0,
        )
        if args.max_steps > 0:
            max_steps = min(max_steps, args.max_steps)

        step_iter = range(max_steps)
        if tqdm is not None:
            step_iter = tqdm(step_iter, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=120)

        for step in step_iter:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            running_comp = {
                "recon": 0.0,
                "ssim": 0.0,
                "edge": 0.0,
            }

            if paired_iter is not None:
                try:
                    x, y = next(paired_iter)
                except StopIteration:
                    paired_iter = iter(paired_loader)
                    x, y = next(paired_iter)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    pred, _ = model(x)
                    paired_loss, comp = criterion(pred, y, return_components=True)
                loss = loss + paired_loss
                running_comp["recon"] += float(comp["recon"].item())
                running_comp["ssim"] += float(comp["ssim"].item())
                running_comp["edge"] += float(comp["edge"].item())

            if unpaired_iter is not None and args.w_nr > 0:
                try:
                    x_u = next(unpaired_iter)
                except StopIteration:
                    unpaired_iter = iter(unpaired_loader)
                    x_u = next(unpaired_iter)
                x_u = x_u.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    pred_u, _ = model(x_u)
                    unpaired_loss, _ = criterion(pred_u, None, return_components=True)
                loss = loss + unpaired_loss

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            total_loss += float(loss.item())
            total_recon += running_comp["recon"]
            total_ssim += running_comp["ssim"]
            total_edge += running_comp["edge"]
            steps += 1
            global_step += 1

            if writer is not None and global_step % args.log_interval == 0:
                writer.add_scalar("train/loss_total", total_loss / max(1, steps), global_step)
                writer.add_scalar("train/loss_recon", total_recon / max(1, steps), global_step)
                writer.add_scalar("train/loss_ssim", total_ssim / max(1, steps), global_step)
                writer.add_scalar("train/loss_edge", total_edge / max(1, steps), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            if (step + 1) % args.log_interval == 0:
                avg_loss = total_loss / max(1, steps)
                if tqdm is not None:
                    step_iter.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                else:
                    print(
                        f"Epoch {epoch+1}/{args.epochs} "
                        f"Step {step+1}/{max_steps} "
                        f"Loss {avg_loss:.4f}",
                        flush=True,
                    )

        epoch_loss = total_loss / max(1, steps)
        print(
            f"Epoch {epoch+1} done. Loss={epoch_loss:.4f} "
            f"Recon={total_recon/max(1,steps):.4f} "
            f"SSIM={total_ssim/max(1,steps):.4f} "
            f"Edge={total_edge/max(1,steps):.4f}",
            flush=True,
        )

        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                os.path.join(run_dir, f"ckpt_epoch_{epoch+1}.pt"),
                model,
                optimizer,
                epoch + 1,
                scaler,
            )

        if val_loader is not None and args.val_interval > 0 and (epoch + 1) % args.val_interval == 0:
            eval_model = ema.ema if ema is not None else model
            metrics = validate(eval_model, val_loader, device)
            if writer is not None:
                writer.add_scalar("val/psnr", metrics["psnr_mean"], epoch + 1)
                writer.add_scalar("val/ssim", metrics["ssim_mean"], epoch + 1)

            if metrics["psnr_mean"] > best_psnr:
                best_psnr = metrics["psnr_mean"]
                epochs_no_improve = 0
                save_checkpoint(
                    os.path.join(run_dir, "ckpt_best.pt"),
                    eval_model,
                    optimizer,
                    epoch + 1,
                    scaler,
                )
            else:
                epochs_no_improve += 1
                if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                    print(
                        f"Early stopping at epoch {epoch+1} "
                        f"(no PSNR improvement for {epochs_no_improve} evals)",
                        flush=True,
                    )
                    break

    if writer is not None:
        writer.close()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysGuidedUFormer(
        in_ch=3,
        out_ch=3,
        base=args.base,
        depth=args.depth,
        blocks_per_stage=args.blocks_per_stage,
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    eval_root = os.path.join(args.out_dir, "eval")
    os.makedirs(eval_root, exist_ok=True)
    run_dir = make_named_run_dir(eval_root, "eval")
    img_dir = os.path.join(run_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    ds = PairedImageDataset(
        args.val_degraded,
        args.val_clean,
        args.crop,
        augment=False,
        resize_eval=True,
    )
    loader = build_loader(ds, batch_size=1, workers=args.workers, shuffle=False, drop_last=False)

    rows = []
    psnr_vals, ssim_vals, uciqe_vals, uiqm_vals = [], [], [], []

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred, t = model(x)
            p = psnr(pred, y)
            s = ssim_metric(pred, y)
            uq = uciqe_proxy_metric(pred)
            uiq = uiqm_proxy_metric(pred)
            psnr_vals.append(p)
            ssim_vals.append(s)
            uciqe_vals.append(uq)
            uiqm_vals.append(uiq)

            name = f"{idx:05d}"
            if idx < args.save_images:
                save_grid(
                    [x.squeeze(0).cpu(), pred.squeeze(0).cpu(), y.squeeze(0).cpu()],
                    os.path.join(img_dir, f"{name}_inp_pred_gt.png"),
                )
                if args.save_transmission:
                    t_img = t.squeeze(0).repeat(3, 1, 1).cpu()
                    save_image(t_img, os.path.join(img_dir, f"{name}_T.png"))

            rows.append([name, p, s, uq, uiq])

    metrics = {
        "psnr_mean": float(sum(psnr_vals) / max(1, len(psnr_vals))),
        "ssim_mean": float(sum(ssim_vals) / max(1, len(ssim_vals))),
        "uciqe_proxy_mean": float(sum(uciqe_vals) / max(1, len(uciqe_vals))),
        "uiqm_proxy_mean": float(sum(uiqm_vals) / max(1, len(uiqm_vals))),
        "num_images": len(psnr_vals),
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "psnr", "ssim", "uciqe_proxy", "uiqm_proxy"])
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2), flush=True)


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysGuidedUFormer(
        in_ch=3,
        out_ch=3,
        base=args.base,
        depth=args.depth,
        blocks_per_stage=args.blocks_per_stage,
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob(os.path.join(args.in_dir, ext)))

    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        for path in paths:
            img = Image.open(path).convert("RGB")
            if args.resize > 0:
                img = img.resize((args.resize, args.resize), Image.BICUBIC)
            x = image_to_tensor(img).unsqueeze(0).to(device)
            pred, t = model(x)
            save_image(pred.squeeze(0), os.path.join(args.out_dir, os.path.basename(path)))
            if args.save_transmission:
                t_img = t.squeeze(0).repeat(3, 1, 1)
                t_name = os.path.splitext(os.path.basename(path))[0] + "_T.png"
                save_image(t_img, os.path.join(args.out_dir, t_name))


def build_parser():
    p = argparse.ArgumentParser(description="Physics-guided underwater enhancement")
    sub = p.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument(
        "--paired-degraded",
        type=str,
        default="/home/zahid/Projects/CVPR/EUVP/Paired/underwater_scenes/trainA",
        help="Comma-separated degraded dirs for paired training.",
    )
    train_p.add_argument(
        "--paired-clean",
        type=str,
        default="/home/zahid/Projects/CVPR/EUVP/Paired/underwater_scenes/trainB",
        help="Comma-separated clean dirs matching --paired-degraded order.",
    )
    train_p.add_argument("--unpaired-degraded", type=str, default="")
    train_p.add_argument("--val-degraded", type=str, default="", help="Comma-separated validation degraded dirs.")
    train_p.add_argument("--val-clean", type=str, default="", help="Comma-separated validation clean dirs.")
    train_p.add_argument("--out-dir", type=str, default="outputs")
    train_p.add_argument("--resume", type=str, default="")
    train_p.add_argument("--epochs", type=int, default=120)
    train_p.add_argument("--batch", type=int, default=8)
    train_p.add_argument("--val-batch", type=int, default=4)
    train_p.add_argument("--crop", type=int, default=256)
    train_p.add_argument("--val-crop", type=int, default=256)
    train_p.add_argument("--lr", type=float, default=2e-4)
    train_p.add_argument("--min-lr", type=float, default=1e-6)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--base", type=int, default=48)
    train_p.add_argument("--depth", type=int, default=4)
    train_p.add_argument("--blocks-per-stage", type=int, default=1)
    train_p.add_argument("--amp", action="store_true", default=True)
    train_p.add_argument("--no-amp", dest="amp", action="store_false")
    train_p.add_argument("--workers", type=int, default=4)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--w-l1", type=float, default=1.0)
    train_p.add_argument("--w-ssim", type=float, default=0.3)
    train_p.add_argument("--w-edge", type=float, default=0.1)
    train_p.add_argument("--w-nr", type=float, default=0.0)
    train_p.add_argument("--use-l1-only", action="store_true")
    train_p.add_argument("--grad-clip", type=float, default=1.0)
    train_p.add_argument("--ema-decay", type=float, default=0.0)
    train_p.add_argument("--tb", action="store_true", default=True)
    train_p.add_argument("--no-tb", dest="tb", action="store_false")
    train_p.add_argument("--log-interval", type=int, default=20)
    train_p.add_argument("--save-interval", type=int, default=5)
    train_p.add_argument("--val-interval", type=int, default=2)
    train_p.add_argument("--max-steps", type=int, default=0)
    train_p.add_argument("--early-stop-patience", type=int, default=15)

    infer_p = sub.add_parser("infer")
    infer_p.add_argument("--in-dir", type=str, required=True)
    infer_p.add_argument("--out-dir", type=str, required=True)
    infer_p.add_argument("--checkpoint", type=str, required=True)
    infer_p.add_argument("--base", type=int, default=48)
    infer_p.add_argument("--depth", type=int, default=4)
    infer_p.add_argument("--blocks-per-stage", type=int, default=1)
    infer_p.add_argument("--resize", type=int, default=0)
    infer_p.add_argument("--save-transmission", action="store_true")

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--val-degraded", type=str, required=True)
    eval_p.add_argument("--val-clean", type=str, required=True)
    eval_p.add_argument("--checkpoint", type=str, required=True)
    eval_p.add_argument("--out-dir", type=str, default="outputs")
    eval_p.add_argument("--crop", type=int, default=256)
    eval_p.add_argument("--save-images", type=int, default=16)
    eval_p.add_argument("--save-transmission", action="store_true")
    eval_p.add_argument("--base", type=int, default=48)
    eval_p.add_argument("--depth", type=int, default=4)
    eval_p.add_argument("--blocks-per-stage", type=int, default=1)
    eval_p.add_argument("--workers", type=int, default=4)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "infer":
        infer(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
