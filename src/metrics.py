"""
Perceptual / stealth metrics between clean and adversarial images in [0, 1].
Uses torchmetrics (PSNR, SSIM, LPIPS with AlexNet).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from torchmetrics.functional.image import (
        learned_perceptual_image_patch_similarity,
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )
except ImportError:  # pragma: no cover
    learned_perceptual_image_patch_similarity = None
    peak_signal_noise_ratio = None
    structural_similarity_index_measure = None


def _tensor_to_float_scalar(t: torch.Tensor) -> float:
    """Detach from graph and move to CPU before scalar conversion (avoids autograd warnings)."""
    x = t.detach().cpu()
    if x.dim() > 0:
        return float(x.mean().item())
    return float(x.item())


def compute_stealth_metrics(
    clean_images_01: torch.Tensor,
    adv_images_01: torch.Tensor,
    reduction: str = "mean",
    lpips_chunk_size: Optional[int] = 32,
) -> Dict[str, float]:
    """
    Compare clean vs adversarial images in [0, 1] range, shape (N, 3, H, W).

    Returns dict with keys: psnr, ssim, lpips (lower LPIPS = more similar).

    All metrics run on ``adv_images_01.device`` so LPIPS (AlexNet) matches the input device.
    ``lpips_chunk_size`` caps LPIPS micro-batches to reduce VRAM spikes when N is large.
    Set ``lpips_chunk_size=None`` to compute LPIPS in one shot (legacy behavior).
    """
    if clean_images_01.shape != adv_images_01.shape:
        raise ValueError("clean and adv tensors must have the same shape.")

    # Detach inputs so torchmetrics never sees tensors that may require grad
    preds = adv_images_01.detach().clamp(0.0, 1.0)
    target = clean_images_01.detach().clamp(0.0, 1.0)
    dev = preds.device
    preds = preds.to(dev)
    target = target.to(dev)

    if peak_signal_noise_ratio is None or structural_similarity_index_measure is None:
        raise ImportError("torchmetrics is required for compute_stealth_metrics. pip install torchmetrics")

    # PSNR / SSIM: data in [0, 1] — detach metric tensors before float conversion
    with torch.no_grad():
        psnr = peak_signal_noise_ratio(preds, target, data_range=1.0)
        ssim = structural_similarity_index_measure(preds, target, data_range=1.0)
        psnr = psnr.detach()
        ssim = ssim.detach()

    out: Dict[str, float] = {}
    if reduction == "mean":
        out["psnr"] = _tensor_to_float_scalar(psnr.mean() if psnr.dim() > 0 else psnr)
        out["ssim"] = _tensor_to_float_scalar(ssim.mean() if ssim.dim() > 0 else ssim)
    else:
        out["psnr"] = _tensor_to_float_scalar(psnr)
        out["ssim"] = _tensor_to_float_scalar(ssim)

    # LPIPS (AlexNet); normalize=True for inputs in [0, 1]; chunked to avoid VRAM spikes
    if learned_perceptual_image_patch_similarity is None:
        out["lpips"] = float("nan")
        return out

    n = preds.size(0)
    with torch.no_grad():
        if lpips_chunk_size is None or n <= lpips_chunk_size:
            lpips_val = learned_perceptual_image_patch_similarity(
                preds,
                target,
                net_type="alex",
                normalize=True,
                reduction="mean",
            )
            lpips_val = lpips_val.detach()
            out["lpips"] = _tensor_to_float_scalar(
                lpips_val.mean() if lpips_val.dim() > 0 else lpips_val
            )
        else:
            acc = 0.0
            for i in range(0, n, lpips_chunk_size):
                end = min(i + lpips_chunk_size, n)
                chunk_p = preds[i:end]
                chunk_t = target[i:end]
                lp = learned_perceptual_image_patch_similarity(
                    chunk_p,
                    chunk_t,
                    net_type="alex",
                    normalize=True,
                    reduction="mean",
                )
                lp = lp.detach()
                acc += float(lp.mean().item() if lp.dim() > 0 else lp.item()) * (end - i)
            out["lpips"] = acc / float(n)

    return out


def aggregate_metrics_list(rows: list) -> Dict[str, float]:
    """Average over list of dicts with same keys."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(sum(d[k] for d in rows) / len(rows)) for k in keys}


@torch.no_grad()
def compute_attack_success_rate(
    image_client: nn.Module,
    tabular_client: nn.Module,
    vfl_server: nn.Module,
    images_norm: torch.Tensor,
    tabular: torch.Tensor,
    adv_images_01: torch.Tensor,
    device: torch.device,
) -> float:
    """
    ASR: fraction of samples where argmax server logits change after replacing image branch
    with adversarial embedding (tabular unchanged).

    images_norm: (N,3,H,W) normalized; adv_images_01: (N,3,H,W) in [0,1].
    """
    from .attacks import normalize_from_01

    images_norm = images_norm.to(device)
    tabular = tabular.to(device)
    adv_images_01 = adv_images_01.to(device)

    clean_emb = image_client(images_norm)
    tab_emb = tabular_client(tabular)
    clean_logits = vfl_server(clean_emb, tab_emb)
    clean_pred = clean_logits.argmax(dim=1)

    x_adv_n = normalize_from_01(adv_images_01)
    adv_emb = image_client(x_adv_n)
    adv_logits = vfl_server(adv_emb, tab_emb)
    adv_pred = adv_logits.argmax(dim=1)

    changed = (clean_pred != adv_pred).float().mean().item()
    return 100.0 * changed
