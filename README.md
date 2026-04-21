# Physics-Guided Underwater Image Enhancement

Hybrid learning-based and physics-guided underwater image enhancement pipeline for research workflows (CVPR-style experimentation), implemented in PyTorch.

## Overview

This repository implements a residual U-shaped enhancement network with a physics-prior branch:

- A **main restoration branch** predicts a residual correction for underwater images.
- A **physics-guided prior branch** estimates a pseudo-transmission map.
- Multi-scale feature modulation applies:
  - `F_mod^(s) = F^(s) * phi_s(resize(T))`

Final output uses residual reconstruction:

- `DeltaI = 0.2 * tanh(head(features))`
- `I_hat = clip(I + DeltaI, 0, 1)`

## Key Features

- Physics-guided multi-scale feature modulation.
- Lightweight residual U-Net backbone (single-GPU friendly).
- Mixed precision training (`torch.cuda.amp`).
- Composite loss (Charbonnier/L1 + SSIM + edge consistency).
- Full-reference metrics: PSNR, SSIM.
- Underwater proxy metrics: UCIQE proxy, UIQM proxy.
- Train / eval / infer CLI in one script.

## Repository Layout

```text
CVPR/
├── main.py          # Train / eval / infer entrypoint
├── models.py        # Network architecture (restoration + prior branch)
├── data.py          # Paired/unpaired datasets + augmentation
├── losses.py        # Composite training losses
├── metrics.py       # Evaluation metrics
├── utils.py         # Seed + checkpoint helpers
├── detail.md        # Architecture change log + hyperparameters
├── info.md          # Methodology notes
├── diag.tex         # Paper-ready architecture diagram (TikZ)
└── outputs*/        # Experiment artifacts
```

## Environment

The recommended environment used in this project is `gappy`.

Example:

```bash
conda activate gappy
python -V
```

Install core dependencies if needed:

```bash
pip install torch torchvision numpy pillow tqdm matplotlib tensorboard
```

## Data Format

Paired training expects one-to-one filename matching between degraded and clean directories.

Example EUVP paired layout:

```text
EUVP/Paired/underwater_scenes/trainA  # degraded
EUVP/Paired/underwater_scenes/trainB  # clean
```

Multi-domain paired training is supported using comma-separated paths (quoted as a single argument).

## Training

### Recommended Multi-Domain Training Command

```bash
python -u /home/zahid/Projects/CVPR/main.py train \
  --paired-degraded "/home/zahid/Projects/CVPR/EUVP/Paired/underwater_scenes/trainA,/home/zahid/Projects/CVPR/EUVP/Paired/underwater_dark/trainA,/home/zahid/Projects/CVPR/EUVP/Paired/underwater_imagenet/trainA" \
  --paired-clean "/home/zahid/Projects/CVPR/EUVP/Paired/underwater_scenes/trainB,/home/zahid/Projects/CVPR/EUVP/Paired/underwater_dark/trainB,/home/zahid/Projects/CVPR/EUVP/Paired/underwater_imagenet/trainB" \
  --val-degraded "/home/zahid/Projects/CVPR/EUVP/test_samples/Inp" \
  --val-clean "/home/zahid/Projects/CVPR/EUVP/test_samples/GTr" \
  --out-dir "/home/zahid/Projects/CVPR/outputs_v3" \
  --epochs 200 \
  --batch 8 \
  --val-batch 4 \
  --crop 256 \
  --val-crop 256 \
  --base 48 \
  --lr 2e-4 \
  --min-lr 1e-6 \
  --weight-decay 1e-4 \
  --w-l1 1.0 \
  --w-ssim 0.3 \
  --w-edge 0.1 \
  --grad-clip 1.0 \
  --workers 4 \
  --log-interval 1 \
  --val-interval 2 \
  --save-interval 5 \
  --amp \
  --no-tb
```

Notes:

- Keep comma-separated path lists inside quotes.
- If memory is limited, reduce `--batch` first (for example, `8 -> 4`) and optionally `--base` (`48 -> 32`).
- Validation metrics are printed every `--val-interval` epochs.

## Evaluation

```bash
python -u /home/zahid/Projects/CVPR/main.py eval \
  --val-degraded /home/zahid/Projects/CVPR/EUVP/test_samples/Inp \
  --val-clean /home/zahid/Projects/CVPR/EUVP/test_samples/GTr \
  --checkpoint /home/zahid/Projects/CVPR/outputs_v3/train/train_203448_19042026/ckpt_best.pt \
  --out-dir /home/zahid/Projects/CVPR/outputs_v3 \
  --crop 256 \
  --base 48 \
  --workers 4 \
  --save-images 32 \
  --save-transmission
```

Eval writes:

- `.../eval/eval_<timestamp>/images/*_inp_pred_gt.png`
- `.../eval/eval_<timestamp>/images/*_T.png` (if enabled)
- `.../eval/eval_<timestamp>/metrics.json`
- `.../eval/eval_<timestamp>/metrics.csv`

Panel order in `*_inp_pred_gt.png`:

1. Input degraded image
2. Predicted enhanced image
3. Ground-truth clean image

## Inference

```bash
python -u /home/zahid/Projects/CVPR/main.py infer \
  --in-dir /home/zahid/Projects/CVPR/EUVP/test_samples/Inp \
  --out-dir /home/zahid/Projects/CVPR/outputs_v3/infer_samples \
  --checkpoint /home/zahid/Projects/CVPR/outputs_v3/train/train_203448_19042026/ckpt_best.pt \
  --base 48 \
  --save-transmission
```

Infer writes enhanced outputs to `--out-dir` with original filenames, plus `*_T.png` maps if requested.

## Output Artifacts

### Training (`outputs_v*/train/train_<timestamp>`)

- `args.json`
- `ckpt_epoch_*.pt`
- `ckpt_best.pt`

### Eval (`outputs_v*/eval/eval_<timestamp>`)

- `metrics.json`
- `metrics.csv`
- `images/` comparisons and transmission maps

### Inference (`outputs_v*/infer_samples`)

- enhanced images
- optional transmission maps (`*_T.png`)

## Architecture Diagram

Source:

- `diag.tex`

Compile:

```bash
pdflatex -interaction=nonstopmode -halt-on-error diag.tex
```

Output:

- `diag.pdf`

## Important Training Notes

- `I_gt` is the clean target in paired supervision.
- `T` is the pseudo-transmission map from the physics prior branch.
- `phi_s(T)` maps transmission to per-scale modulation weights.
- Backpropagation updates all connected modules:
  - residual head
  - decoder, bottleneck, encoder
  - modulation modules
  - physics prior branch via `T`

## Troubleshooting

### Command parsing fails (ambiguous options or missing paths)

Cause is usually line breaks or accidental spaces inside arguments.

Fix:

- Use one clean command line, or proper `\` continuation.
- Always quote comma-separated directory lists.

### No training progress appears

Try:

- `python -u ...` for unbuffered output.
- `--workers 0 --log-interval 1` for debugging visibility.

### Pairing error: "No paired samples found"

Check:

- degraded and clean filenames must match by basename.
- paired degraded/clean list lengths must be identical for multi-domain mode.

## Citation

If you use this codebase in academic work, please cite your project paper and mention this implementation repository in the supplementary material.
