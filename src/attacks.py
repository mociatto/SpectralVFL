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


def create_frequency_mask(
    shape: Tuple[int, int],
    band: str,
    device: torch.device,
    r_low: float = 40.0,
    r_mid: float = 80.0,
) -> torch.Tensor:
    """
    Radial mask in fftshifted frequency coordinates (H, W).

    Bands (center distance r):
      - 'low':   r < r_low
      - 'mid':   r_low <= r < r_mid
      - 'high':  r >= r_mid
      - 'all':   full spectrum
    """
    h, w = shape
    yy = torch.arange(h, device=device, dtype=torch.float32).view(-1, 1)
    xx = torch.arange(w, device=device, dtype=torch.float32).view(1, -1)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    band = band.lower()
    if band == "low":
        m = (r < r_low).float()
    elif band == "mid":
        m = ((r >= r_low) & (r < r_mid)).float()
    elif band == "high":
        m = (r >= r_mid).float()
    elif band == "all":
        m = torch.ones(h, w, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown band '{band}'. Use 'low', 'mid', 'high', or 'all'.")

    return m


def _spectral_filter_gradient(
    grad: torch.Tensor,
    mask_hw: torch.Tensor,
) -> torch.Tensor:
    """
    Apply band mask to 2D FFT of gradient (per spatial plane, summed over channels for direction).
    grad: (B, C, H, W)
    mask_hw: (H, W)
    """
    b, c, h, w = grad.shape
    m = mask_hw.view(1, 1, h, w).to(grad.device, grad.dtype)
    out = torch.zeros_like(grad)
    for ch in range(c):
        g = grad[:, ch : ch + 1, :, :]  # (B,1,H,W)
        G = torch.fft.fft2(g, dim=(-2, -1))
        Gs = torch.fft.fftshift(G, dim=(-2, -1))
        Gs = Gs * m
        Gu = torch.fft.ifftshift(Gs, dim=(-2, -1))
        g_f = torch.fft.ifft2(Gu, dim=(-2, -1)).real
        out[:, ch, :, :] = g_f.squeeze(1)
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


class SpectralFGSM(BaseEmbeddingAttack):
    """FGSM with gradient filtered in frequency domain."""

    def __init__(self, image_client: nn.Module, epsilon: float, band: str = "all"):
        super().__init__(image_client)
        self.epsilon = epsilon
        self.band = band

    def __call__(self, x_norm: torch.Tensor) -> torch.Tensor:
        x_01 = denormalize_to_01(x_norm)
        _, _, h, w = x_01.shape
        device = x_01.device
        mask = create_frequency_mask((h, w), self.band, device)

        with torch.no_grad():
            clean_emb = self.image_client(x_norm)

        x_01 = x_01.clone().detach().requires_grad_(True)
        x_n = normalize_from_01(x_01)
        adv_emb = self.image_client(x_n)
        loss = F.mse_loss(adv_emb, clean_emb)
        grad = torch.autograd.grad(loss, x_01)[0]
        grad_f = _spectral_filter_gradient(grad, mask)

        x_adv = x_01 + self.epsilon * grad_f.sign()
        return x_adv.clamp(0.0, 1.0)


class SpectralPGD(BaseEmbeddingAttack):
    """PGD with spectral filtering of the gradient each step."""

    def __init__(
        self,
        image_client: nn.Module,
        epsilon: float,
        alpha: float,
        band: str = "all",
        num_steps: int = 10,
        random_start: bool = True,
    ):
        super().__init__(image_client)
        self.epsilon = epsilon
        self.alpha = alpha
        self.band = band
        self.num_steps = num_steps
        self.random_start = random_start

    def __call__(self, x_norm: torch.Tensor) -> torch.Tensor:
        x_clean_01 = denormalize_to_01(x_norm)
        _, _, h, w = x_clean_01.shape
        device = x_clean_01.device
        mask = create_frequency_mask((h, w), self.band, device)

        with torch.no_grad():
            clean_emb = self.image_client(x_norm)

        if self.random_start:
            delta = torch.empty_like(x_clean_01).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(x_clean_01)
        delta = delta.clamp(-self.epsilon, self.epsilon)
        x_adv = (x_clean_01 + delta).clamp(0.0, 1.0)

        for _ in range(self.num_steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            x_n = normalize_from_01(x_adv)
            adv_emb = self.image_client(x_n)
            loss = F.mse_loss(adv_emb, clean_emb)
            grad = torch.autograd.grad(loss, x_adv)[0]
            grad_f = _spectral_filter_gradient(grad, mask)

            x_adv = x_adv + self.alpha * grad_f.sign()
            x_adv = torch.max(torch.min(x_adv, x_clean_01 + self.epsilon), x_clean_01 - self.epsilon)
            x_adv = x_adv.clamp(0.0, 1.0)

        return x_adv.detach()
