"""
OLED display power model for the Gradual Chromatic Adaptation (GCA) module.

Implements gradual.md "阶段一.1 测量显示功耗模型":

    p(c) = p_disp^T @ c + p_static

The blue subpixel is ~2x more expensive than red/green, so shifting the
scene illuminant toward yellow-green (decreasing blue) lowers total power.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import P_DISP_RAW, P_STATIC, POWER_WEIGHTS_RGB


def power_model(
    rgb: np.ndarray,
    weights: Tuple[float, float, float] = POWER_WEIGHTS_RGB,
    static: float = P_STATIC,
) -> np.ndarray:
    """
    Per-pixel OLED display power: p(c) = w^T @ c + static.

    Args:
        rgb: linear RGB image in [0, 1], shape (..., 3)
        weights: (R, G, B) display power coefficients
        static: static (offset) power term

    Returns:
        per-pixel power, shape (...)
    """
    w = np.asarray(weights, dtype=np.float64)
    return (rgb * w).sum(axis=-1) + static


def compute_power_reduction(
    original: np.ndarray,
    adapted: np.ndarray,
    weights: Tuple[float, float, float] = POWER_WEIGHTS_RGB,
    static: float = P_STATIC,
) -> float:
    """
    Compute the total display power saving fraction:

        saving = 1 - power(adapted) / power(original)

    Args:
        original: original sRGB image in [0, 1], shape (H, W, 3)
        adapted: adapted sRGB image in [0, 1], shape (H, W, 3)

    Returns:
        saving in [0, 1] (1.0 => zero display power)
    """
    orig_power = power_model(original, weights, static).sum()
    opt_power = power_model(adapted, weights, static).sum()
    if orig_power <= 0:
        return 0.0
    return float(1.0 - opt_power / orig_power)