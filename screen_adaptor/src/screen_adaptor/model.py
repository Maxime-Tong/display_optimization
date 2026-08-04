from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator
from torch import nn

from .color_ops import srgb_to_dkl
from .utils import generate_phi_map


class EllipsoidRadiusNet(nn.Module):
    def __init__(self, hidden_dim: int = 32, depth: int = 2) -> None:
        super().__init__()
        layers = []
        input_dim = 3
        
        for _ in range(depth):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, 3)
        
        self.hidden_dim = hidden_dim
        self.depth = depth

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        dkl = srgb_to_dkl(rgb)
        features = self.backbone(dkl)
        residual = self.head(features)
        out_rgb = rgb - F.tanh(residual) * 0.5
        return out_rgb.clamp(0.0, 1.0)


def save_checkpoint(model: EllipsoidRadiusNet, path: Union[str, Path]) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": model.hidden_dim,
            "depth": model.depth,
        },
        Path(path),
    )


def load_model(checkpoint_path: Union[str, Path]) -> EllipsoidRadiusNet:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = EllipsoidRadiusNet(
        hidden_dim=int(checkpoint.get("hidden_dim", 32)),
        depth=int(checkpoint.get("depth", 2)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


class LUTColorTransformer:
    def __init__(self, lut: torch.Tensor, foveated: float = 0, temporal: float = 0) -> None:
        if lut.ndim != 4 or lut.shape[-1] != 3:
            raise ValueError("lut must have shape [R, G, B, 3]")
        self.lut = lut.float().contiguous()
        self.volume = self.lut.permute(3, 0, 1, 2).unsqueeze(0)

        n_dims = lut.shape[:3]
        grid_coords = [np.linspace(0, 1, n) for n in n_dims]
        self.interpolator = RegularGridInterpolator(grid_coords, self.lut.cpu().numpy())

        self.foveated = foveated
        self.temporal = temporal
        self.phi_map = None

    def transform(self, rgb: torch.Tensor) -> torch.Tensor:
        out_torch = self.transform_blockwise(rgb)
        out_torch = self.apply_forveated(rgb, out_torch)
        return out_torch
    
    def apply_forveated(self, rgb: torch.Tensor, rgb_opt: torch.Tensor) -> torch.Tensor:
        if self.foveated <= 0:
            return rgb_opt
        
        downsample_gray = rgb[::4, ::4, :].mean(axis=-1)
        std = downsample_gray.std()
        if std < 0.2:
            return rgb_opt
        
        if self.phi_map is None or self.phi_map.shape[0] != rgb.shape[0] or self.phi_map.shape[1] != rgb.shape[1]:
            phi_map = torch.from_numpy(generate_phi_map(rgb.shape)).to(rgb.device)
            phi_map = phi_map.unsqueeze(-1).expand(-1, -1, 3)
            self.phi_map = phi_map * self.foveated

        return rgb + self.phi_map * (rgb_opt - rgb)
        
        
    def transform_blockwise(self, rgb: torch.Tensor, block_size: int = 4) -> torch.Tensor:
        h, w = rgb.shape[:2]

        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        rgb_pad = F.pad(rgb.permute(2, 0, 1).unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect')

        block_avg = F.avg_pool2d(rgb_pad, kernel_size=block_size, stride=block_size).squeeze(0).permute(1, 2, 0)
        blocks_h, blocks_w = block_avg.shape[:2]

        block_avg_flat = block_avg.reshape(-1, 3)
        block_out_flat = self.interpolator(block_avg_flat.cpu().numpy())
        block_out = torch.from_numpy(block_out_flat).to(rgb.device).reshape(blocks_h, blocks_w, 3)

        block_out_upsampled = block_out.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)[:h, :w, :]
        block_avg_upsampled = block_avg.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)[:h, :w, :]

        eps = 1e-6
        result = block_out_upsampled * (rgb / (block_avg_upsampled + eps))

        return result.clamp(0, 1)


def generate_lut(model: EllipsoidRadiusNet, resolution: int, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """Generate LUT from trained model."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    axis = torch.linspace(0.0, 1.0, resolution, device=device)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    with torch.no_grad():
        optimized = model(grid)
    return optimized.reshape(resolution, resolution, resolution, 3).cpu()


def save_lut(lut: torch.Tensor, path: Union[str, Path]) -> None:
    torch.save({"lut": lut}, Path(path))


def load_lut_transformer(path: Union[str, Path], foveated: float = 0, temporal: float = 0) -> LUTColorTransformer:
    payload = torch.load(path, map_location="cpu")
    lut = payload.get("lut") if isinstance(payload, dict) else payload
    return LUTColorTransformer(lut, foveated=foveated, temporal=temporal)