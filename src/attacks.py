"""
Label-agnostic inference-time attacks on ImageClient embeddings.
Maximize MSE between clean and adversarial embeddings; output images clamped to [0, 1].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet normalization (matches data_utils / torchvision eval transforms)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denormalize_to_01(x_norm: torch.Tensor) -> torch.Tensor:
    """Map ImageNet-normalized tensor to approximately [0, 1]."""
    mean = IMAGENET_MEAN.to(device=x_norm.device, dtype=x_norm.dtype)
    std = IMAGENET_STD.to(device=x_norm.device, dtype=x_norm.dtype)
    return (x_norm * std + mean).clamp(0.0, 1.0)


def normalize_from_01(x_01: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] tensor to ImageNet normalization."""
    mean = IMAGENET_MEAN.to(device=x_01.device, dtype=x_01.dtype)
    std = IMAGENET_STD.to(device=x_01.device, dtype=x_01.dtype)
    return (x_01 - mean) / std


def adaptive_spectral_filter_gradient(grad: torch.Tensor, sparsity_k: float) -> torch.Tensor:
    """
    Keep only the top ``sparsity_k`` fraction of 2D-FFT magnitude bins **per image**
    (shared mask across RGB); zero the rest; IFFT per channel.

    Magnitude per frequency bin is summed over channels so one (H, W) mask applies
    to each image in the batch.

    Args:
        grad: (B, C, H, W) spatial gradient.
        sparsity_k: fraction in (0, 1] of frequency bins (H×W) to retain per image.

    Returns:
        Filtered gradient (B, C, H, W), real.
    """
    if not 0.0 < sparsity_k <= 1.0:
        raise ValueError(f"sparsity_k must be in (0, 1], got {sparsity_k}")

    b, c, h, w = grad.shape
    n_bins = h * w
    n_keep = max(1, min(int(round(sparsity_k * n_bins)), n_bins))

    G = torch.fft.fft2(grad, dim=(-2, -1))
    mag = torch.abs(G)
    # Per-image combined magnitude over channels → one mask (H, W) per batch index
    mag_img = mag.sum(dim=1)  # (B, H, W)
    mag_flat = mag_img.reshape(b, n_bins)
    _, idx = torch.topk(mag_flat, k=n_keep, dim=1, largest=True)
    mask_flat = torch.zeros(b, n_bins, device=grad.device, dtype=grad.dtype)
    mask_flat.scatter_(1, idx, 1.0)
    mask = mask_flat.view(b, 1, h, w).expand_as(mag)
    G_f = G * mask.to(dtype=G.dtype)
    out = torch.fft.ifft2(G_f, dim=(-2, -1)).real
    return out


class BaseEmbeddingAttack(ABC):
    """Maximize MSE(image_emb(x_adv), image_emb(x_clean)); x_adv in [0,1]."""

    def __init__(self, image_client: nn.Module):
        self.image_client = image_client

    @abstractmethod
    def __call__(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_norm: (B,3,H,W) ImageNet-normalized (as from dataloader).
        Returns:
            x_adv_01: (B,3,H,W) in [0,1].
        """

    def _embedding_loss_grad(
        self,
        x_01: torch.Tensor,
        clean_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return loss (scalar per batch mean) and gradient w.r.t. x_01."""
        x_01 = x_01.clone().detach().requires_grad_(True)
        x_n = normalize_from_01(x_01)
        adv_emb = self.image_client(x_n)
        loss = F.mse_loss(adv_emb, clean_emb.detach())
        grad = torch.autograd.grad(loss, x_01)[0]
        return loss, grad


class SpatialFGSM(BaseEmbeddingAttack):
    """Single-step FGSM in [0,1] space."""

    def __init__(self, image_client: nn.Module, epsilon: float):
        super().__init__(image_client)
        self.epsilon = epsilon

    def __call__(self, x_norm: torch.Tensor) -> torch.Tensor:
        device = x_norm.device
        x_01 = denormalize_to_01(x_norm)
        with torch.no_grad():
            clean_emb = self.image_client(x_norm)

        x_01 = x_01.clone().detach().requires_grad_(True)
        x_n = normalize_from_01(x_01)
        adv_emb = self.image_client(x_n)
        loss = F.mse_loss(adv_emb, clean_emb)
        grad = torch.autograd.grad(loss, x_01)[0]

        x_adv = x_01 + self.epsilon * grad.sign()
        return x_adv.clamp(0.0, 1.0)


class SpatialPGD(BaseEmbeddingAttack):
    """PGD maximizing embedding MSE under L_inf."""

    def __init__(
        self,
        image_client: nn.Module,
        epsilon: float,
        alpha: float,
        num_steps: int = 10,
        random_start: bool = True,
    ):
        super().__init__(image_client)
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start

    def __call__(self, x_norm: torch.Tensor) -> torch.Tensor:
        x_clean_01 = denormalize_to_01(x_norm)
        with torch.no_grad():
            clean_emb = self.image_client(x_norm)

        if self.random_start:
            delta = torch.empty_like(x_clean_01).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(x_clean_01)
        delta = delta.clamp(
            -self.epsilon,
            self.epsilon,
        )
        x_adv = (x_clean_01 + delta).clamp(0.0, 1.0)

        for _ in range(self.num_steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            x_n = normalize_from_01(x_adv)
            adv_emb = self.image_client(x_n)
            loss = F.mse_loss(adv_emb, clean_emb)
            grad = torch.autograd.grad(loss, x_adv)[0]

            x_adv = x_adv + self.alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_clean_01 + self.epsilon), x_clean_01 - self.epsilon)
            x_adv = x_adv.clamp(0.0, 1.0)

        return x_adv.detach()


class AdaptiveSpectralPGD(BaseEmbeddingAttack):
    """
    PGD with per-image adaptive spectral sparsity: each step keeps only the top
    ``sparsity_k`` fraction of frequency bins (by gradient FFT magnitude), then
    applies a linear step using the filtered gradient normalized per image by its
    own L_inf norm (avoids ``sign()`` broadband harmonics).
    """

    def __init__(
        self,
        image_client: nn.Module,
        epsilon: float,
        alpha: float,
        sparsity_k: float,
        num_steps: int = 10,
        random_start: bool = True,
    ):
        super().__init__(image_client)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.sparsity_k = float(sparsity_k)
        self.num_steps = int(num_steps)
        self.random_start = random_start
        if not 0.0 < self.sparsity_k <= 1.0:
            raise ValueError(f"sparsity_k must be in (0, 1], got {sparsity_k}")

    def __call__(self, x_norm: torch.Tensor) -> torch.Tensor:
        x_clean_01 = denormalize_to_01(x_norm)
        with torch.no_grad():
            clean_emb = self.image_client(x_norm)

        if self.random_start:
            delta = torch.empty_like(x_clean_01).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(x_clean_01)
        delta = delta.clamp(-self.epsilon, self.epsilon)

        for _ in range(self.num_steps):
            x_adv = (x_clean_01 + delta).clamp(0.0, 1.0)
            x_adv = x_adv.clone().detach().requires_grad_(True)
            x_n = normalize_from_01(x_adv)
            adv_emb = self.image_client(x_n)
            loss = F.mse_loss(adv_emb, clean_emb)
            grad = torch.autograd.grad(loss, x_adv)[0]
            grad_f = adaptive_spectral_filter_gradient(grad, self.sparsity_k)

            b = x_adv.size(0)
            max_abs = grad_f.abs().view(b, -1).max(dim=1)[0].view(b, 1, 1, 1) + 1e-8
            normalized_grad = grad_f / max_abs
            delta = delta + self.alpha * normalized_grad
            delta = delta.clamp(-self.epsilon, self.epsilon)

        x_out = (x_clean_01 + delta).clamp(0.0, 1.0)
        return x_out.detach()
