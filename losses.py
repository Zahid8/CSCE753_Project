import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size, sigma, device, channels):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.view(1, 1, window_size)
    window_2d = window_1d.transpose(2, 1) @ window_1d
    window_2d = window_2d / window_2d.sum()
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(x, y, window_size=11, sigma=1.5, data_range=1.0):
    channels = x.size(1)
    window = _gaussian_window(window_size, sigma, x.device, channels)
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8)
    return ssim_map.mean(dim=(1, 2, 3))


class SSIMLoss(nn.Module):
    def forward(self, x, y):
        return 1.0 - ssim(x, y).mean()


def uciqe_proxy(x):
    # Approximate UCIQE using differentiable operations in RGB
    # x in [0,1]
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    chroma = torch.sqrt((r - g) ** 2 + (g - b) ** 2 + (b - r) ** 2 + 1e-8)
    c_std = chroma.flatten(1).std(dim=1)

    lum = 0.299 * r + 0.587 * g + 0.114 * b
    l_contrast = (lum.flatten(1).max(dim=1).values - lum.flatten(1).min(dim=1).values)

    sat = chroma / (lum + 1e-6)
    s_mean = sat.flatten(1).mean(dim=1)

    # weights based on original UCIQE formulation
    return 0.4680 * c_std + 0.2745 * l_contrast + 0.2576 * s_mean


def uiqm_proxy(x):
    # Approximate UIQM with colorfulness and sharpness terms
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    rg = r - g
    yb = 0.5 * (r + g) - b
    rg_mean = rg.flatten(1).mean(dim=1)
    yb_mean = yb.flatten(1).mean(dim=1)
    rg_std = rg.flatten(1).std(dim=1)
    yb_std = yb.flatten(1).std(dim=1)
    colorfulness = torch.sqrt(rg_std ** 2 + yb_std ** 2) + 0.3 * torch.sqrt(rg_mean ** 2 + yb_mean ** 2)

    # sharpness via Laplacian
    lap = torch.abs(F.conv2d(x, weight=_laplacian_kernel(x.device), padding=1, groups=3))
    sharpness = lap.flatten(1).mean(dim=1)

    return 0.5 * colorfulness + 0.5 * sharpness


def _laplacian_kernel(device):
    k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=device, dtype=torch.float32)
    k = k.view(1, 1, 3, 3)
    return k.repeat(3, 1, 1, 1)


class NRQualityLoss(nn.Module):
    def __init__(self, weight_uciqe=0.2, weight_uiqm=0.2):
        super().__init__()
        self.weight_uciqe = weight_uciqe
        self.weight_uiqm = weight_uiqm

    def forward(self, x):
        uciqe = uciqe_proxy(x)
        uiqm = uiqm_proxy(x)
        # Maximize quality => minimize negative proxy
        loss = -self.weight_uciqe * uciqe.mean() - self.weight_uiqm * uiqm.mean()
        return loss


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


def _sobel_kernels(device, channels):
    kx = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    return kx.repeat(channels, 1, 1, 1), ky.repeat(channels, 1, 1, 1)


def _edge_magnitude(x):
    c = x.size(1)
    kx, ky = _sobel_kernels(x.device, c)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.l1(_edge_magnitude(pred), _edge_magnitude(target))


class CompositeLoss(nn.Module):
    def __init__(
        self,
        w_l1=1.0,
        w_ssim=0.3,
        w_edge=0.1,
        w_nr=0.0,
        use_charbonnier=True,
    ):
        super().__init__()
        self.recon = CharbonnierLoss() if use_charbonnier else nn.L1Loss()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.nr = NRQualityLoss()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_edge = w_edge
        self.w_nr = w_nr

    def forward(self, pred, target=None, return_components=False):
        loss = pred.new_tensor(0.0)
        components = {
            "recon": pred.new_tensor(0.0),
            "ssim": pred.new_tensor(0.0),
            "edge": pred.new_tensor(0.0),
            "nr": pred.new_tensor(0.0),
        }

        if target is not None:
            recon = self.recon(pred, target)
            ssim_term = self.ssim(pred, target)
            edge_term = self.edge(pred, target)
            components["recon"] = recon.detach()
            components["ssim"] = ssim_term.detach()
            components["edge"] = edge_term.detach()
            loss = loss + self.w_l1 * recon
            loss = loss + self.w_ssim * ssim_term
            loss = loss + self.w_edge * edge_term

        if self.w_nr > 0:
            nr_term = self.nr(pred)
            components["nr"] = nr_term.detach()
            loss = loss + self.w_nr * nr_term

        if return_components:
            components["total"] = loss.detach()
            return loss, components
        return loss
