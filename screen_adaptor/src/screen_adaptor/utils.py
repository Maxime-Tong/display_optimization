from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import torch
from PIL import Image

from .scene_matcher import IMAGE_EXTENSIONS


def generate_phi_map(image_shape, steepness=4.0, center_fraction=0.3):
    """Generate a foveated weight map: center=0 (full LUT), edge=1 (keep original)."""
    h, w = image_shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2

    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)

    normalized_dist = dist / max_dist
    scaled = (normalized_dist - center_fraction) * steepness
    phi = 1 / (1 + np.exp(-scaled))
    phi = (phi - phi.min()) / (phi.max() - phi.min())

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