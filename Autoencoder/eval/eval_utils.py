"""
Filename: eval_utils.py
Author: Sanmati Choudhary
Description: Defines evaluation metrics for SSL and prediction task.
"""

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



def calculate_ms_ssim(real_images: torch.Tensor, gen_images: torch.Tensor) -> float:
    return multi_scale_ssim(gen_images, real_images, data_range=1.0).mean().item()


def calculate_ssim(real_images: torch.Tensor, gen_images: torch.Tensor) -> float:
    # Use piq SSIM or extend it with gradient and multi-orientation filters
    return ssim(gen_images, real_images, data_range=1.0).mean().item()


def calculate_coverage(real_features: torch.Tensor, gen_features: torch.Tensor) -> float:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(real_features)
    distances, _ = nn.kneighbors(gen_features)
    threshold = np.percentile(distances, 95)
    covered = np.sum(distances <= threshold)
    return covered, len(gen_features)


def calculate_lpips(real_images: torch.Tensor, gen_images: torch.Tensor) -> float:
    loss_fn = lpips.LPIPS(net='alex').cuda()
    return loss_fn(gen_images, real_images).mean().item()

