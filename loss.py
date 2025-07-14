import torch
import torch.nn as nn
import torch.nn.functional as F
def Sobelxy(x):
    kernelx = [[-1, 0, 1],
              [-2,0 , 2],
              [-1, 0, 1]]
    kernely = [[1, 2, 1],
              [0,0 , 0],
              [-1, -2, -1]]
    kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
    kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
    weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)
    sobelx=F.conv2d(x, weightx, padding=1)
    sobely=F.conv2d(x, weighty, padding=1)
    return torch.abs(sobelx)+torch.abs(sobely)

def gaussian_blur(input_tensor, kernel_size=5, sigma=3.0):
    channels = input_tensor.shape[1]
    kernel = get_gaussian_kernel(kernel_size, sigma).to(input_tensor.device)
    kernel = kernel[:, None] * kernel[None, :]
    kernel = kernel[None, None, :, :].repeat(channels, 1, 1, 1)

    return F.conv2d(input_tensor, kernel, padding=kernel_size // 2, groups=channels)


def get_gaussian_kernel(size, sigma):
    coords = torch.arange(size).float() - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g

def L_Grad(image_A, image_B, image_fused,
           lamb: float = 2.0,
           sigma: float = 3.0):

    Y = lambda t: t[:, :1, :, :]
    image_A_Y, image_B_Y, image_fused_Y = map(Y, (image_A, image_B, image_fused))

    blurred_A = gaussian_blur(image_A_Y, sigma=sigma)
    blurred_B = gaussian_blur(image_B_Y, sigma=sigma)

    unsharp_A = (1.0 + lamb) * image_A_Y - lamb * blurred_A
    unsharp_B = (1.0 + lamb) * image_B_Y - lamb * blurred_B

    # ---- Gradient (Sobel) ----
    grad_A = Sobelxy(unsharp_A)
    grad_B = Sobelxy(unsharp_B)
    grad_F = Sobelxy(image_fused_Y)

    target_grad = torch.max(grad_A, grad_B)
    return F.l1_loss(grad_F, target_grad)
def L_Int(image_A, image_B, image_fused):
    image_A_Y = image_A[:, :1, :, :]
    image_B_Y = image_B[:, :1, :, :]
    image_fused_Y = image_fused[:, :1, :, :]
    x_in_max = torch.max(image_A_Y, image_B_Y)
    loss_in = F.l1_loss(x_in_max, image_fused_Y)
    return loss_in

def L_SD(image_fused):
    mean = torch.mean(image_fused, dim=[2,3], keepdim=True)
    sd = torch.sqrt(torch.mean((image_fused - mean)**2))
    return -sd