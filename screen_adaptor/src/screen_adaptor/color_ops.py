from __future__ import annotations

import torch

from .colorspace import DKL2RGB as _DKL2RGB_NP
from .colorspace import RGB2DKL as _RGB2DKL_NP


def clamp_rgb(rgb: torch.Tensor) -> torch.Tensor:
    return torch.clamp(rgb, 0.0, 1.0)

def weighted_power(rgb: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (rgb * weights).sum(dim=-1)


def relative_power(original_rgb: torch.Tensor, optimized_rgb: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    original_power = weighted_power(original_rgb, weights).clamp_min(1e-6)
    optimized_power = weighted_power(optimized_rgb, weights)
    return optimized_power / original_power


def _matrix_from_numpy(matrix: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(matrix, dtype=reference.dtype, device=reference.device)


def srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    srgb = torch.clamp(srgb, 0.0, 1.0)
    mask = srgb <= 0.04045
    return torch.where(mask, srgb / 12.92, torch.pow((srgb + 0.055) / 1.055, 2.4))


def linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    rgb = torch.clamp(rgb, 0.0, 1.0)
    mask = rgb <= 0.0031308
    return torch.where(mask, 12.92 * rgb, 1.055 * torch.pow(rgb, 1.0 / 2.4) - 0.055)


def rgb_to_dkl(rgb: torch.Tensor) -> torch.Tensor:
    matrix = _matrix_from_numpy(_RGB2DKL_NP, rgb)
    return torch.matmul(rgb, matrix.t())


def dkl_to_rgb(dkl: torch.Tensor) -> torch.Tensor:
    matrix = _matrix_from_numpy(_DKL2RGB_NP, dkl)
    return torch.matmul(dkl, matrix.t())


def srgb_to_dkl(srgb: torch.Tensor) -> torch.Tensor:
    return rgb_to_dkl(srgb_to_linear(srgb))


def dkl_to_srgb(dkl: torch.Tensor) -> torch.Tensor:
    return linear_to_srgb(dkl_to_rgb(dkl))


def dkl_to_delta(dkl: torch.Tensor, pedestal: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (dkl - pedestal) / scale.clamp_min(1e-6)


def delta_to_dkl(delta: torch.Tensor, pedestal: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return pedestal + delta * scale


