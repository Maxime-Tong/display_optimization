from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from .scene_matcher import IMAGE_EXTENSIONS


def generate_phi_map(
    image_shape,
    steepness: float = 4.0,
    center_fraction: float = 0.3,
    foveal_fraction: Optional[float] = None,
    transition_width: Optional[float] = None,
    phi_min: float = 0.8,
    phi_max: float = 1.2,
):
    """Generate a foveated weight map using a smooth sigmoid ramp.

    Legacy approach restored for a simple, smooth center→periphery transition:

    * A logistic (sigmoid) curve centered at ``center_fraction`` climbed from
      a low value at the center to a high value at the periphery.
    * The output is normalized to [0, 1] and then mapped into ``[phi_min,
      phi_max]`` (default 0.8–1.2).  Keeping the range narrow around 1.0 means
      the whole frame is always optimized (no hard foveal plateau that leaves
      the center untouched), while the small 0.4-wide modulation avoids a
      visible circular boundary and keeps the transition extremely gentle.

    Parameters
    ----------
    image_shape : tuple
        (h, w) or (h, w, c) view of the target image.
    steepness : float
        Slope of the sigmoid curve (larger = sharper transition).
    center_fraction : float
        Normalized distance (fraction of max corner distance) at which the
        sigmoid reaches 0.5.
    foveal_fraction : float | None
        Backward-compat alias.  When given, overrides ``center_fraction`` as
        ``foveal_fraction * 2`` (keeps the old plateau-onset relationship).
    transition_width : float | None
        Backward-compat alias.  When given, overrides ``steepness`` as
        ``1 / (2 * transition_width)``.
    phi_min, phi_max : float
        Output range for the normalized sigmoid (default 0.8–1.2).
    """
    h, w = image_shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2

    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2) + 1e-9

    normalized_dist = dist / max_dist

    if foveal_fraction is not None:
        center_fraction = foveal_fraction * 2.0
    if transition_width is not None:
        steepness = 1.0 / max(2.0 * transition_width, 1e-6)

    # Legacy sigmoid curve (smooth S-shape from center to periphery)
    sig = 1.0 / (1.0 + np.exp(-steepness * (normalized_dist - center_fraction)))

    # Normalize to [0, 1] then map into [phi_min, phi_max]
    sig_norm = (sig - sig.min()) / max(sig.max() - sig.min(), 1e-9)
    phi = phi_min + (phi_max - phi_min) * sig_norm

    return phi


def collect_image_paths(data_dir: Union[str, Path]) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError("data_dir does not exist: %s" % root)
    image_paths = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    image_paths.sort()
    if not image_paths:
        raise ValueError("no image files found under: %s" % root)
    return image_paths


def load_image_tensor(path: Union[str, Path], image_size: int, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """Load image as tensor on specified device."""
    image = Image.open(path).convert("RGB")
    if image_size > 0:
        image = image.resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).to(device)


def sample_image(image_paths: Sequence[Path], image_size: int, device: Union[str, torch.device]) -> torch.Tensor:
    """Randomly pick one image and load it as a tensor."""
    image_path = random.choice(list(image_paths))
    return load_image_tensor(image_path, image_size, device)