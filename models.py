import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(ch):
    max_groups = min(8, ch)
    for g in range(max_groups, 0, -1):
        if ch % g == 0:
            return g
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualDSBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.dw_norm = nn.GroupNorm(_group_count(ch), ch)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.pw_norm = nn.GroupNorm(_group_count(ch), ch)
        self.act = nn.SiLU(inplace=True)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(ch // 8, 4), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(ch // 8, 4), ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.dw(x)
        out = self.act(self.dw_norm(out))
        out = self.pw(out)
        out = self.pw_norm(out)
        out = out * self.se(out)
        return self.act(x + out)


class DoubleResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(ResidualDSBlock(ch), ResidualDSBlock(ch))

    def forward(self, x):
        return self.block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = ConvNormAct(in_ch, out_ch, kernel_size=3, stride=2)
        self.body = DoubleResidualBlock(out_ch)

    def forward(self, x):
        x = self.down(x)
        x = self.body(x)
        return x


class UpsampleFuseBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.fuse = ConvNormAct(in_ch + skip_ch, out_ch, kernel_size=1)
        self.body = DoubleResidualBlock(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.body(x)
        return x


class PhysicsBranch(nn.Module):
    def __init__(self, in_ch=3, base=16):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.stem = nn.Sequential(ConvNormAct(in_ch, c1), ResidualDSBlock(c1))
        self.down1 = DownsampleBlock(c1, c2)
        self.down2 = DownsampleBlock(c2, c3)
        self.up1 = UpsampleFuseBlock(c3, c2, c2)
        self.up2 = UpsampleFuseBlock(c2, c1, c1)
        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.down1(s1)
        b = self.down2(s2)
        d2 = self.up1(b, s2)
        d1 = self.up2(d2, s1)
        t = torch.sigmoid(self.out_conv(d1))
        return t


class PhiModule(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        hidden = max(8, out_ch // 4)
        self.map = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, t):
        return self.map(t)


class PhysGuidedUFormer(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        base=32,
        depth=4,
        blocks_per_stage=1,
    ):
        super().__init__()
        del depth, blocks_per_stage  # kept for backward-compatible constructor signature

        c1 = base
        c2 = base * 2
        c3 = base * 4
        c4 = base * 8
        c5 = base * 16

        self.physics = PhysicsBranch(in_ch=in_ch, base=max(12, base // 2))

        self.stem = nn.Sequential(ConvNormAct(in_ch, c1), DoubleResidualBlock(c1))
        self.enc2 = DownsampleBlock(c1, c2)
        self.enc3 = DownsampleBlock(c2, c3)
        self.enc4 = DownsampleBlock(c3, c4)
        self.bottleneck = DownsampleBlock(c4, c5)

        self.dec4 = UpsampleFuseBlock(c5, c4, c4)
        self.dec3 = UpsampleFuseBlock(c4, c3, c3)
        self.dec2 = UpsampleFuseBlock(c3, c2, c2)
        self.dec1 = UpsampleFuseBlock(c2, c1, c1)

        self.phi1 = PhiModule(c1)
        self.phi2 = PhiModule(c2)
        self.phi3 = PhiModule(c3)
        self.phi4 = PhiModule(c4)
        self.phi5 = PhiModule(c5)
        self.phid4 = PhiModule(c4)
        self.phid3 = PhiModule(c3)
        self.phid2 = PhiModule(c2)
        self.phid1 = PhiModule(c1)

        self.out_conv = nn.Conv2d(c1, out_ch, kernel_size=1)
        self.residual_scale = 0.2
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _modulate(feat, t, phi):
        t_resized = F.interpolate(t, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        return feat * phi(t_resized)

    def forward(self, x):
        t = self.physics(x)

        e1 = self._modulate(self.stem(x), t, self.phi1)
        e2 = self._modulate(self.enc2(e1), t, self.phi2)
        e3 = self._modulate(self.enc3(e2), t, self.phi3)
        e4 = self._modulate(self.enc4(e3), t, self.phi4)
        b = self._modulate(self.bottleneck(e4), t, self.phi5)

        d4 = self._modulate(self.dec4(b, e4), t, self.phid4)
        d3 = self._modulate(self.dec3(d4, e3), t, self.phid3)
        d2 = self._modulate(self.dec2(d3, e2), t, self.phid2)
        d1 = self._modulate(self.dec1(d2, e1), t, self.phid1)

        residual = self.residual_scale * torch.tanh(self.out_conv(d1))
        pred = torch.clamp(x + residual, 0.0, 1.0)
        return pred, t
