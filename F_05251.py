import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math
try:
    _has_builtin_dct = hasattr(torch.fft, "dct")
except AttributeError:
    _has_builtin_dct = False

if _has_builtin_dct:
    _dct  = torch.fft.dct
    _idct = torch.fft.idct
else:
    from torch_dct import dct as _dct, idct as _idct

def dct_2d(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    if _has_builtin_dct:
        y = _dct(x, type=2, norm=norm, dim=-1)
        y = _dct(y.transpose(-1, -2), type=2, norm=norm, dim=-1)
        return y.transpose(-1, -2)
    else:
        y = _dct(x, norm=norm)
        y = _dct(y.transpose(-1, -2), norm=norm).transpose(-1, -2)
        return y

def idct_2d(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    if _has_builtin_dct:
        y = _idct(x, type=3, norm=norm, dim=-1)
        y = _idct(y.transpose(-1, -2), type=3, norm=norm, dim=-1)
        return y.transpose(-1, -2)
    else:
        y = _idct(x, norm=norm)
        y = _idct(y.transpose(-1, -2), norm=norm).transpose(-1, -2)
        return y

class DCTDropUnit(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((channels, 1, 1), 0.5))

    def forward(self, x):
        B, C, H, W = x.shape
        X = dct_2d(x)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij")
        r = torch.sqrt(xx ** 2 + yy ** 2)
        mask = r[None, None]

        X_mod = X * (1 - self.alpha) + X * self.alpha * mask
        x_rec = idct_2d(X_mod)
        return x_rec.type_as(x)

class WaveletShrinkUnit(nn.Module):
    def __init__(self, channels, lam_init=0.02):
        super().__init__()
        self.lam = nn.Parameter(torch.full((channels, 1, 1), lam_init))

        k = torch.tensor([[.5, .5], [.5, .5]], dtype=torch.float32)
        ll = k
        lh = torch.tensor([[.5, .5], [-.5, -.5]])
        hl = torch.tensor([[.5, -.5], [.5, -.5]])
        hh = torch.tensor([[.5, -.5], [-.5, .5]])
        ker = torch.stack([ll, lh, hl, hh]).unsqueeze(1)
        self.register_buffer("ker", ker)

    def forward(self, x):
        B, C, H, W = x.shape
        weight = self.ker.repeat(C, 1, 1, 1)
        Y = F.conv2d(x, weight, stride=2, groups=C)
        LL, LH, HL, HH = torch.split(Y, C, dim=1)

        def shrink(t, lam): return torch.sign(t) * F.relu(torch.abs(t) - lam)
        LH = shrink(LH, self.lam)
        HL = shrink(HL, self.lam)
        HH = shrink(HH, self.lam)

        Y_shrink = torch.cat([LL, LH, HL, HH], 1)
        x_rec = F.conv_transpose2d(
            Y_shrink, weight, stride=2, groups=C)
        return x_rec[:, :, :H, :W]

class LowRankFFTUnit(nn.Module):
    def __init__(self, channels, rank_ratio=0.25):
        super().__init__()
        self.rank_ratio = rank_ratio
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, C, H, W = x.shape

        x_flat = x.view(B, C, -1)
        U, S, Vh = torch.linalg.svd(x_flat, full_matrices=False)
        r = max(1, int(C * self.rank_ratio))
        x_low = (U[:, :, :r] @ torch.diag_embed(S[:, :r]) @ Vh[:, :r]).view(B, C, H, W)

        fy = torch.fft.fftshift(torch.fft.fftfreq(H, d=1.0)).to(x.device)
        fx = torch.fft.fftshift(torch.fft.fftfreq(W, d=1.0)).to(x.device)
        yy, xx = torch.meshgrid(fy, fx, indexing='ij')
        freq_radius = torch.sqrt(xx ** 2 + yy ** 2)

        cutoff = 0.2
        highpass_mask = (freq_radius >= cutoff).float()
        highpass_mask = highpass_mask.unsqueeze(0).unsqueeze(0)

        X_fft = torch.fft.fft2(x)
        X_fft_shift = torch.fft.fftshift(X_fft, dim=(-2, -1))
        X_fft_filtered = X_fft_shift * highpass_mask

        X_fft_unshift = torch.fft.ifftshift(X_fft_filtered, dim=(-2, -1))
        res = torch.fft.ifft2(X_fft_unshift).real

        return x_low + self.scale * res.type_as(x)

class ButterworthUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.log_fc = nn.Parameter(torch.tensor(0.0))     # ln fc
        self.log_n  = nn.Parameter(torch.tensor(math.log(2.0)))

    def forward(self, x):
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij")
        r = torch.sqrt(xx ** 2 + yy ** 2)
        fc = torch.exp(self.log_fc) + 1e-4
        n  = torch.exp(self.log_n)

        G = 1 / torch.sqrt(1 + (r / fc) ** (2 * n))
        X_fft = torch.fft.rfft2(x)
        G_r = G[..., :W // 2 + 1]
        x_bp = torch.fft.irfft2(X_fft * G_r, s=(H, W))
        return x_bp.type_as(x)

class ReconBlock(nn.Module):
    """
    kind: 'dct' | 'wavelet' | 'lowrank' | 'butter'
    """
    def __init__(self, channels: int, kind: str = "dct"):
        super().__init__()
        if kind == "dct":
            self.core = DCTDropUnit(channels)
        elif kind == "wavelet":
            self.core = WaveletShrinkUnit(channels)
        elif kind == "lowrank":
            self.core = LowRankFFTUnit(channels)
        elif kind == "butter":
            self.core = ButterworthUnit(channels)
        else:
            raise ValueError(f"Unknown kind {kind}")

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * self.core(x)

class HaarDWT(nn.Module):
    def __init__(self, levels: int = 1):
        super().__init__()
        self.levels = levels
        k = torch.tensor([[0.5, 0.5],
                          [0.5, 0.5]], dtype=torch.float32)
        ll = k
        lh = torch.tensor([[0.5, 0.5],
                           [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5],
                           [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]])
        kernel = torch.stack([ll, lh, hl, hh], 0)          # (4,2,2)
        self.register_buffer("kernel", kernel.unsqueeze(1))

    def _analysis(self, x: torch.Tensor):
        B, C, H, W = x.shape
        weight = self.kernel.repeat(C, 1, 1, 1)
        out = F.conv2d(x, weight, stride=2, groups=C)
        ll, lh, hl, hh = torch.split(out, C, dim=1)
        high = torch.cat([lh, hl, hh], dim=1)
        return ll, high

    def forward(self, x: torch.Tensor):
        highs: List[torch.Tensor] = []
        cur = x
        for _ in range(self.levels):
            cur, high = self._analysis(cur)
            highs.append(high)
        return highs, cur


class SFU(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        fft = torch.fft.rfft2(x, norm='ortho')
        amp = torch.abs(fft)
        v = amp.mean((-2, -1))
        w = self.fc(v).view(B, C, 1, 1)
        amp_mod = amp * w
        fft_mod = torch.polar(amp_mod, torch.angle(fft))
        x_hi = torch.fft.irfft2(fft_mod, s=(H, W), norm='ortho')
        return x_hi


class HFInjector(nn.Module):
    def __init__(self, c_main: int, c_hf: int):
        super().__init__()
        self.conv = nn.Conv2d(c_main + c_hf, c_main, 1, bias=False)
        self.norm = nn.BatchNorm2d(c_main)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, f_main, f_hf):
        x = torch.cat([f_main, f_hf], 1)
        return self.act(self.norm(self.conv(x)))


class UpConv(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.SiLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class SceneProbability(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, vis_y: torch.Tensor, delta: torch.Tensor):
        vis_y  = vis_y.squeeze(1)
        delta  = delta.squeeze(1)
        v      = torch.stack([vis_y, delta], dim=1)
        alpha  = self.mlp(v).unsqueeze(-1).unsqueeze(-1)
        return alpha

class ECA(nn.Module):
    def __init__(self, channels, k_size=None):
        super().__init__()
        if k_size is None:
            k_size = int(abs(torch.log2(torch.tensor(channels, dtype=torch.float32)) + 1))
            k_size = k_size + 1 if k_size % 2 == 0 else k_size
        self.conv = nn.Conv1d(1, 1, k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, 1, c)
        y = self.sigmoid(self.conv(y)).view(b, c, 1, 1)
        return x * y


class CoordAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y_h = F.adaptive_avg_pool2d(x, (h, 1))
        y_w = F.adaptive_avg_pool2d(x, (1, w)).transpose(2, 3)
        y = torch.cat([y_h, y_w], 2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], 2)
        y_w = y_w.transpose(2, 3)
        a_h = self.sigmoid(self.conv_h(y_h))
        a_w = self.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


class SceneAwareFusion(nn.Module):
    def __init__(self, c_half: int):
        super().__init__()
        self.est = SceneProbability()
        self.c_half = c_half

    def forward(self, x, vis, ir):
        f_ir, f_vis = torch.split(x, self.c_half, dim=1)
        vis_y  = vis.mean((2, 3))
        delta  = (ir - vis).abs().mean((2, 3))
        alpha  = self.est(vis_y, delta)
        f_ir  = f_ir  * alpha
        f_vis = f_vis * (1 - alpha)
        return torch.cat([f_ir, f_vis], dim=1)


class SpatialBranchIRVIS(nn.Module):
    def __init__(self, in_c, mid_c):
        super().__init__()
        assert mid_c % 2 == 0
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(inplace=True)
        )
        self.fuse = SceneAwareFusion(mid_c // 2)
        self.eca  = ECA(mid_c)
        self.ca   = CoordAttention(mid_c)
        self.out_proj = (nn.Identity() if in_c == mid_c
                         else nn.Conv2d(mid_c, in_c, 1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, vis, ir):
        y = self.bottleneck(x)
        y = self.fuse(y, vis, ir)
        y = self.eca(y)
        y = self.ca(y)
        y = self.out_proj(y)
        return x + self.gamma * y
class FusionNet(nn.Module):
    def __init__(self, base_c=16):
        super().__init__()
        self.init_conv = nn.Conv2d(2, base_c, 3, padding=1, bias=False)
        self.enc1 = SpatialBranchIRVIS(base_c, base_c)
        self.down1 = nn.Conv2d(base_c, base_c, 3, 2, 1, bias=False)

        self.enc2 = SpatialBranchIRVIS(base_c, base_c)
        self.down2 = nn.Conv2d(base_c, base_c, 3, 2, 1, bias=False)

        self.enc3 = SpatialBranchIRVIS(base_c, base_c)
        self.down3 = nn.Conv2d(base_c, base_c, 3, 2, 1, bias=False)

        self.enc4 = SpatialBranchIRVIS(base_c, base_c)

        self.dwt256 = HaarDWT(levels=3)
        self.dwt128 = HaarDWT(levels=2)
        self.sfu32  = SFU(base_c)
        self.sfu16  = SFU(base_c)

        self.inj_e2 = HFInjector(base_c, 54)
        self.inj_e3 = HFInjector(base_c, 102)
        self.inj_e4 = HFInjector(base_c, 134)

        self.up3 = UpConv(base_c)
        self.up2 = UpConv(base_c)
        self.up1 = UpConv(base_c)
        self.recon3 = ReconBlock(channels= 16,kind = "butter")
        self.recon2 = ReconBlock(channels= 16,kind = "butter")
        self.recon1 = ReconBlock(channels= 16,kind = "butter")
        self.out_conv = nn.Conv2d(base_c, 1, 3, padding=1)

    @staticmethod
    def _dwt_ir_vis(dwt_module, ir, vis, level_idx) -> torch.Tensor:
        hi_ir, _ = dwt_module(ir)
        hi_vis, _ = dwt_module(vis)
        return torch.cat([hi_ir[level_idx], hi_vis[level_idx]], 1)

    @staticmethod
    def _dwt_single(dwt_module, x: torch.Tensor, level_idx: int) -> torch.Tensor:
        hi, _ = dwt_module(x)
        return hi[level_idx]

    def forward(self, ir, vis):
        x0 = self.init_conv(torch.cat([ir, vis], 1))

        e1 = self. enc1(x0, vis, ir)
        e1d = self.down1(e1)

        e2 = self.enc2(e1d, vis, ir)
        e2d = self.down2(e2)

        e3 = self.enc3(e2d, vis, ir)
        e3d = self.down3(e3)

        e4 = self.enc4(e3d, vis, ir)

        hf128_basic = self._dwt_ir_vis(self.dwt256, ir, vis, 0)
        hf64_basic = self._dwt_ir_vis(self.dwt256, ir, vis, 1)
        hf32_basic = self._dwt_ir_vis(self.dwt256, ir, vis, 2)

        hf128_e1 = self._dwt_single(self.dwt256, e1, 0)
        hf64_e1 = self._dwt_single(self.dwt256, e1, 1)
        hf32_e1 = self._dwt_single(self.dwt256, e1, 2)

        hf64_e2 = self._dwt_single(self.dwt128, e2, 0)
        hf32_e2 = self._dwt_single(self.dwt128, e2, 1)

        hf32 = torch.cat([hf32_basic, hf32_e1, hf32_e2, self.sfu32(e3d), self.sfu16(e4)], 1)
        hf64 = torch.cat([hf64_basic,hf64_e1,hf64_e2], 1)
        hf128 = torch.cat([hf128_basic, hf128_e1], 1)

        f4 = self.inj_e4(e4, hf32)
        r_f4 = self.recon3(f4)
        d3 = self.up3(r_f4)

        f3 = self.inj_e3(e3 + d3, hf64)
        r_f3 = self.recon2(f3)
        d2 = self.up2(r_f3)

        f2 = self.inj_e2(e2 + d2, hf128)
        r_f2 = self.recon1(f2)
        d1 = self.up1(r_f2)
        out = torch.tanh(self.out_conv(d1)) / 2 + 0.5
        return out
