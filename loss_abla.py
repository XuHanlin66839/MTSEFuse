import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pytorch_wavelets import DWTForward, DWTInverse
def haar_dwt_1level(x):
    """
    x : [B,C,H,W]，H、W 均需为 2 的倍数
    返回:
        LL, (LH, HL, HH) : 4 个 [B,C,H/2,W/2] 张量
    """
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    h = 1 / 2**0.5
    ll = torch.tensor([[h, h], [h, h]],  dtype=dtype, device=device)
    lh = torch.tensor([[h,-h], [h,-h]], dtype=dtype, device=device)
    hl = torch.tensor([[h, h],[-h,-h]],  dtype=dtype, device=device)
    hh = torch.tensor([[h,-h],[-h, h]], dtype=dtype, device=device)
    kernels = torch.stack([ll, lh, hl, hh])            # [4,2,2]
    kernels = kernels.unsqueeze(1)                     # [4,1,2,2]

    # ---------- 关键修正：扩展到 4*B*C ----------
    kernels = kernels.repeat(B * C, 1, 1, 1)           # [4BC,1,2,2]

    # depthwise conv (每个通道 4 个滤波器)
    x_ = x.reshape(1, B * C, H, W)                     # 组合 batch & ch
    out = F.conv2d(x_, kernels, stride=2, groups=B * C)
    out = out.reshape(B, C, 4, H // 2, W // 2)

    LL, LH, HL, HH = out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]
    return LL, (LH, HL, HH)
def _gaussian_kernel1d(sigma: float, truncate: float = 4.0):
    """
    创建 1-D 高斯核；返回 torch.Tensor, 形状 [k]
    """
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel

def gaussian_blur_tensor(x: torch.Tensor,
                          sigma: float,
                          truncate: float = 4.0) -> torch.Tensor:
    """
    纯 PyTorch 高斯模糊，支持 autograd & CUDA。
    x      : [B,C,H,W]
    sigma  : 标准差 (像素)
    """
    if sigma <= 0:
        return x
    B, C, H, W = x.shape
    kernel1d = _gaussian_kernel1d(sigma, truncate).to(x.device, x.dtype)
    k = kernel1d.numel()
    # 1-D depthwise conv：先水平方向，再垂直方向
    kernel_x = kernel1d.view(1, 1, 1, k).expand(C, 1, 1, k)
    kernel_y = kernel1d.view(1, 1, k, 1).expand(C, 1, k, 1)
    padding = k // 2
    x = F.conv2d(x, kernel_x, padding=(0, padding), groups=C)
    x = F.conv2d(x, kernel_y, padding=(padding, 0), groups=C)
    return x
def _grad_mag(x, kx, ky):
    """
    x  : [B,C,H,W]
    kx/ky : torch.tensor 形状 [3,3]
    返回 grad_mag : [B,C,H,W]
    """
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    # [C,1,3,3] 深度分组卷积，每个通道用同一核
    kx = kx.to(device=device, dtype=dtype).expand(C, 1, 3, 3)
    ky = ky.to(device=device, dtype=dtype).expand(C, 1, 3, 3)

    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


# ------------------------- Sobel -------------------------
_SOBEL_X = torch.tensor([[[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]]])
_SOBEL_Y = _SOBEL_X.transpose(1, 2)  # y 核

def Sobelxy(x):
    return _grad_mag(x, _SOBEL_X, _SOBEL_Y)


# ------------------------- Scharr ------------------------
# 3×3 Scharr (更多权重集中在中心行/列)
_SCHARR_X = torch.tensor([[[ 3,  0, -3],
                           [10,  0,-10],
                           [ 3,  0, -3]]])
_SCHARR_Y = _SCHARR_X.transpose(1, 2)

def Scharrxy(x):
    return _grad_mag(x, _SCHARR_X, _SCHARR_Y)


# ------------------------ Prewitt ------------------------
_PREWITT_X = torch.tensor([[[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]]])
_PREWITT_Y = _PREWITT_X.transpose(1, 2)

def Prewittxy(x):
    return _grad_mag(x, _PREWITT_X, _PREWITT_Y)
# def Sobelxy(x):
#     kernelx = [[-1, 0, 1],
#               [-2,0 , 2],
#               [-1, 0, 1]]
#     kernely = [[1, 2, 1],
#               [0,0 , 0],
#               [-1, -2, -1]]
#     kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#     kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
#     # weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
#     # weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
#     weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)
#     sobelx=F.conv2d(x, weightx, padding=1)
#     sobely=F.conv2d(x, weighty, padding=1)
#     return torch.abs(sobelx)+torch.abs(sobely)

# ------------------ 卷积核辅助 ------------------
def _conv2d(weight, padding=1, groups=1):
    k = weight.shape[-1]
    conv = nn.Conv2d(weight.size(0) // groups,
                     weight.size(0) // groups,
                     k, padding=padding, bias=False, groups=groups)
    conv.weight.data.copy_(weight)
    conv.weight.requires_grad_(False)
    return conv


class GradEnhancer(nn.Module):
    """
    kind ∈ {'usm', 'dog', 'lap', 'wavelet'}
    - usm    : 非锐化掩模（你的原方法，λ 和 σ 可调）
    - dog    : Difference-of-Gaussian，近似 LoG，锐化纹理
    - lap    : 高提升 + Laplacian（二阶导数增强）
    - wavelet: DWT 高频层增强（与 ReconBlock 的 wavelet 同思路）
    kind	λ 推荐范围	σ（如适用）	说明
    usm	      0.5-3	      1-5 px	经典非锐化掩模，λ 控强度
    dog	       1-6	      1-4 px	σ₁≈σ/1.6；纹理更突出
    lap   	 0.2-1.5	    —	    防溢出建议 λ≤1
    wavelet	 0.1-0.8	    —	    高频通道能量大，λ 通常更小
    """

    def gauss_blur(self, x, sigma):
        if hasattr(F, "gaussian_blur"):
            # torchvision / torch >=0.14.0 才有
            return F.gaussian_blur(x, kernel_size=0, sigma=sigma)
        else:
            return gaussian_blur_tensor(x, sigma)

    def __init__(self, kind='usm', lamb=1.0, sigma=3.0):
        super().__init__()
        self.kind, self.lamb, self.sigma = kind, lamb, sigma

        # Laplacian 核 (边长 3)
        lap_kernel = torch.tensor([[[[0,  1, 0],
                                     [1, -4, 1],
                                     [0,  1, 0]]]], dtype=torch.float32)
        self.lap_conv = _conv2d(lap_kernel, padding=1)

    # -------------------------------------------------------------
    # def gauss_blur(self, x, sigma):
    #     # torchvision 的 GaussianBlur 只能在 PIL; 这里用 F.gaussian_blur
    #     return F.gaussian_blur(x, kernel_size=0, sigma=sigma)
    # -------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == 'usm':
            blurred = self.gauss_blur(x, self.sigma)
            return (1 + self.lamb) * x - self.lamb * blurred

        elif self.kind == 'dog':
            g1 = self.gauss_blur(x, 0.3)
            g2 = self.gauss_blur(x, 3)
            return x + self.lamb * (g1 - g2)          # λ 控制差分幅度

        elif self.kind == 'lap':
            high = self.lap_conv(x)
            return x + self.lamb * high               # 高提升

        # elif self.kind == 'wavelet':
        #     # 仅示例：直接取 1-level Haar 高频
        #     LL, (LH, HL, HH) = TF.haar_wavelet(x, mode='reflect')
        #     high = torch.cat([LH, HL, HH], 1)
        #     return x + self.lamb * F.interpolate(high, size=x.shape[-2:],
        #                                          mode='bilinear',
        #                                          align_corners=False)
        elif self.kind == 'wavelet':
            self.dwt = DWTForward(J=1, wave='haar')
            self.iwt = DWTInverse(wave='haar')
            # 保证权重搬到同 device / dtype
            self.dwt = self.dwt.to(x.device, x.dtype)
            self.iwt = self.iwt.to(x.device, x.dtype)

            LL, Yh = self.dwt(x)  # ← 只两项

            # 单层增强：整体乘 λ
            Yh_boost = [Yh[0] * self.lamb]

            x_sharp = self.iwt((LL, Yh_boost))
            return x_sharp
        elif self.kind == 'gabor':
            # five 3×3 Gabor kernels at 0°/45°/90°/135°/180°
            thetas = [0, 45, 90, 135, 180]
            kernels = []
            for t in thetas:
                theta = torch.deg2rad(torch.tensor(t, dtype=torch.float32))
                # σ≈0.8, λ≈2, γ=0.5, φ=0
                sigma, lam, gamma, psi = 0.8, 2.0, 0.5, 0.0
                x, y = torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2))
                x_theta = x * torch.cos(theta) + y * torch.sin(theta)
                y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
                gb = torch.exp(-(x_theta ** 2 + gamma ** 2 * y_theta ** 2) /
                               (2 * sigma ** 2)) * torch.cos(2 * torch.pi * x_theta / lam + psi)
                kernels.append(gb)
            kernels = torch.stack(kernels).unsqueeze(1)  # [5,1,3,3]
            k = kernels.to(x.device, x.dtype).expand(-1, x.shape[1], -1, -1)  # 按通道广播
            gabor_resp = F.conv2d(x, k, padding=1, groups=x.shape[1])
            high = gabor_resp.abs().sum(dim=0, keepdim=True)  # 方向幅值求和
            return x + self.lamb * high
        else:
            raise ValueError(f"Unknown enhancer kind '{self.kind}'")

def L_GradX(img_A, img_B, img_f, kind='usm', lamb=1.0, sigma=3.0,
            grad='sobel'):
    """
    - kind : 梯度增强方式，同 GradEnhancer.kind
    - grad : {'sobel', 'scharr', 'prewitt'} 任选
    """
    enhancer = GradEnhancer(kind, lamb, sigma).to(img_A.device)

    # ---- 只取 Y 通道 ----
    Y = lambda t: t[:, :1]
    A, B, Fused = map(Y, (img_A, img_B, img_f))

    # ---- 先增强再取梯度 ----
    A_e = enhancer(A)
    B_e = enhancer(B)

    if grad == 'sobel':
        G = Sobelxy
    elif grad == 'scharr':
        G = Scharrxy
    else:
        G = Prewittxy

    gA, gB = G(A_e), G(B_e)
    gF = G(Fused)

    target = torch.max(gA, gB)
    return F.l1_loss(gF, target)

def L_GGrad(image_A, image_B, image_fused):
    image_A_Y = image_A[:, :1, :, :]
    image_B_Y = image_B[:, :1, :, :]
    image_fused_Y = image_fused[:, :1, :, :]
    gradient_A = Sobelxy(image_A_Y)
    gradient_B = Sobelxy(image_B_Y)
    gradient_fused = Sobelxy(image_fused_Y)
    gradient_joint = torch.max(gradient_A, gradient_B)
    Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
    return Loss_gradient