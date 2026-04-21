import torch
import torch.nn.functional as F
from losses import ssim


def psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
    psnr_val = 10 * torch.log10((data_range ** 2) / (mse + 1e-8))
    return psnr_val.mean().item()


def ssim_metric(pred, target):
    return ssim(pred, target).mean().item()


def uciqe_proxy_metric(pred):
    from losses import uciqe_proxy
    return uciqe_proxy(pred).mean().item()


def uiqm_proxy_metric(pred):
    from losses import uiqm_proxy
    return uiqm_proxy(pred).mean().item()
