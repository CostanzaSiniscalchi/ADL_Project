# Install required packages (only for demonstration purposes, not executable here)
# !pip install torch torchvision numpy scipy lpips piq

import torch
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3
from torchvision import transforms
from piq import ssim, multi_scale_ssim
import lpips 
from torchmetrics.image.kid import KernelInceptionDistance
from typing import Tuple
import numpy as np
from scipy import linalg




def calculate_fid(real_features: torch.Tensor, gen_features: torch.Tensor) -> float:
    mu1, sigma1 = real_features.mean(0), torch.cov(real_features.T)
    mu2, sigma2 = gen_features.mean(0), torch.cov(gen_features.T)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def calculate_kid(real_images: torch.Tensor, gen_images: torch.Tensor) -> Tuple[float, float]:
    real_images = (real_images * 255).clamp(0, 255).to(torch.uint8)
    gen_images = (gen_images * 255).clamp(0, 255).to(torch.uint8)
    kid = KernelInceptionDistance(subset_size=1)
    kid.update(real_images, real=True)
    kid.update(gen_images, real=False)
    return kid.compute()


def calculate_ms_ssim(real_images: torch.Tensor, gen_images: torch.Tensor) -> float:
    return multi_scale_ssim(gen_images, real_images, data_range=1.0).mean().item()


def calculate_4gr_ssim(real_images: torch.Tensor, gen_images: torch.Tensor) -> float:
    # Use piq SSIM or extend it with gradient and multi-orientation filters
    return ssim(gen_images, real_images, data_range=1.0).mean().item()

def calculate_mmd(x: torch.Tensor, y: torch.Tensor, kernel: str = "rbf", sigma: float = 1.0) -> float:
    def rbf_kernel(a, b, sigma):
        a_norm = (a ** 2).sum(1).view(-1, 1)
        b_norm = (b ** 2).sum(1).view(1, -1)
        dist = a_norm + b_norm - 2 * torch.mm(a, b.T)
        return torch.exp(-dist / (2 * sigma ** 2))

    k_xx = rbf_kernel(x, x, sigma).mean()
    k_yy = rbf_kernel(y, y, sigma).mean()
    k_xy = rbf_kernel(x, y, sigma).mean()
    return float(k_xx + k_yy - 2 * k_xy)


def calculate_coverage(real_features: torch.Tensor, gen_features: torch.Tensor) -> float:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(real_features)
    distances, _ = nn.kneighbors(gen_features)
    threshold = np.percentile(distances, 95)
    covered = np.sum(distances <= threshold)
    return covered / len(real_features)


def calculate_lpips(real_images: torch.Tensor, gen_images: torch.Tensor) -> float:
    loss_fn = lpips.LPIPS(net='alex').cuda()
    return loss_fn(gen_images, real_images).mean().item()

