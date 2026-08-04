"""
Color space utilities for the Gradual Chromatic Adaptation (GCA) module.

Implements (all formulas from gradual.md "色彩空间" section):
  - sRGB <-> linear RGB (gamma encode/decode)
  - linear RGB <-> XYZ (Judd-Vos corrected sRGB->XYZ matrix)
  - XYZ <-> CIE u'v' (for illuminant trajectories)
  - Bradford chromatic adaptation (CAT) between two illuminants
"""

from __future__ import annotations

import numpy as np

from .constants import BRADFORD_MATRIX, BRADFORD_MATRIX_INV, D65_XYZ


# ── RGB -> XYZ (Judd-Vos corrected sRGB primaries) ─────────────────────
def _compute_rgb2xyz_jv():
    """Judd-Vos corrected sRGB->XYZ matrix (matches color_optimizer/colorspace.py)."""
    old_rgb2xyz = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ])

    def xy_to_xyz(x, y, yy):
        return np.array([yy / y * x, yy, yy / y * (1 - x - y)])

    def jv_correct(x, y, yy):
        denom = 0.03845 * x + 0.01496 * y + 1
        xp = (1.0271 * x - 0.00008 * y - 0.00009) / denom
        yp = (0.00376 * x + 1.0072 * y + 0.00764) / denom
        return xp, yp, yy

    def xyz_to_xy(xyz):
        s = xyz.sum()
        return xyz[0] / s, xyz[1] / s

    cols = []
    for prim in np.eye(3):
        # xyz = old_rgb2xyz @ prim  ->  Y luminance comes from this column
        prim_xyz = old_rgb2xyz @ prim
        x, y = xyz_to_xy(prim_xyz)
        yy = prim_xyz[1]
        cols.append(xy_to_xyz(*jv_correct(x, y, yy)))
    return np.stack(cols, axis=1)


RGB2XYZ = _compute_rgb2xyz_jv()
XYZ2RGB = np.linalg.inv(RGB2XYZ)


# ── sRGB <-> linear RGB ────────────────────────────────────────────────
def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Decode sRGB (0..1) to linear RGB (0..1). Handles (H,W,3) arrays."""
    srgb = np.clip(srgb, 0.0, 1.0)
    lo = srgb <= 0.04045
    out = np.empty_like(srgb)
    out[lo] = srgb[lo] / 12.92
    out[~lo] = ((srgb[~lo] + 0.055) / 1.055) ** 2.4
    return out


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Encode linear RGB (0..1) to sRGB (0..1). Handles (H,W,3) arrays."""
    rgb = np.clip(rgb, 0.0, 1.0)
    lo = rgb <= 0.0031308
    out = np.empty_like(rgb)
    out[lo] = 12.92 * rgb[lo]
    out[~lo] = 1.055 * rgb[~lo] ** (1.0 / 2.4) - 0.055
    return out


# ── XYZ <-> CIE u'v' ───────────────────────────────────────────────────
def xyz_to_upvp(xyz: np.ndarray) -> np.ndarray:
    """
    Convert XYZ (last dim = 3) to CIE 1976 u'v' chromaticity.
    Returns the same shape with last dim = 2 (u', v').
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    denom = x + 15.0 * y + 3.0 * z
    u = 4.0 * x / np.maximum(denom, 1e-12)
    v = 9.0 * y / np.maximum(denom, 1e-12)
    return np.stack([u, v], axis=-1)


def upvp_to_xyz(upvp: np.ndarray, y: float | np.ndarray = 1.0) -> np.ndarray:
    """
    Convert CIE 1976 u'v' + Y to XYZ.
    Assumes u'v' shape (..., 2), Y broadcastable to (..., ).

    Steps: u'v' -> xy chromaticity, then scale by Y:
        denom = 6u - 16v + 12
        x = 9u/denom, y_c = 4v/denom
        X = Y * x/y_c, Z = Y * (1 - x - y_c)/y_c
    """
    u, v = upvp[..., 0], upvp[..., 1]
    denom = np.maximum(6.0 * u - 16.0 * v + 12.0, 1e-12)
    x_chrom = 9.0 * u / denom
    y_chrom = 4.0 * v / denom

    xyz = np.empty(upvp.shape[:-1] + (3,), dtype=np.float64)
    xyz[..., 0] = x_chrom / y_chrom * y
    xyz[..., 1] = y
    xyz[..., 2] = (1.0 - x_chrom - y_chrom) / y_chrom * y
    return xyz


def d65_upvp() -> np.ndarray:
    """D65 white point in CIE u'v' space."""
    return xyz_to_upvp(np.asarray(D65_XYZ, dtype=np.float64))


# ── Bradford chromatic adaptation ──────────────────────────────────────
def compute_cat_matrix(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
) -> np.ndarray:
    """
    Compute the 3x3 Bradford chromatic adaptation matrix that maps colors
    viewed under illuminant ``src_xyz`` to the same appearance under
    illuminant ``dst_xyz``.

    Standard CAT:  M = inv(B) @ diag(dst_LMS / src_LMS) @ B
    """
    src_xyz = np.asarray(src_xyz, dtype=np.float64).reshape(3)
    dst_xyz = np.asarray(dst_xyz, dtype=np.float64).reshape(3)

    src_lms = BRADFORD_MATRIX @ src_xyz
    dst_lms = BRADFORD_MATRIX @ dst_xyz
    scale = dst_lms / np.maximum(src_lms, 1e-12)
    cat = BRADFORD_MATRIX_INV @ np.diag(scale) @ BRADFORD_MATRIX
    return cat


def bradford_cat(
    linear_rgb: np.ndarray,
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    rgb2xyz: np.ndarray = RGB2XYZ,
    xyz2rgb: np.ndarray = XYZ2RGB,
) -> np.ndarray:
    """
    Apply Bradford chromatic adaptation to linear RGB image (last dim 3).

    Path: linearRGB -> XYZ -> (CAT) -> XYZ -> linearRGB
    """
    src_xyz = np.asarray(src_xyz, dtype=np.float64).reshape(3)
    dst_xyz = np.asarray(dst_xyz, dtype=np.float64).reshape(3)
    cat = compute_cat_matrix(src_xyz, dst_xyz)

    # linear RGB -> XYZ
    xyz = linear_rgb @ rgb2xyz.T
    # CAT
    xyz_adapted = xyz @ cat.T
    # XYZ -> linear RGB
    out = xyz_adapted @ xyz2rgb.T
    return out