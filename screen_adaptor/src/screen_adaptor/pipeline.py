"""Unified pipeline: training loop + DKL clustering + per‑cluster LUT training + manifest export."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .model import EllipsoidRadiusNet, generate_lut, load_model, save_checkpoint
from .scene_matcher import (
    IMAGE_EXTENSIONS,
    MANIFEST_VERSION,
    ScenePrototype,
    _extract_feature,
    _extract_feature_fast,
    _kmeans,
    _mean_and_std,
    _normalize,
    load_image_array,
    load_scene_manifest,
)
from .utils import generate_phi_map, sample_image

from odak.learn.perception import MetamericLossUniform  # noqa: E402

# ── defaults ───────────────────────────────────────────────────────────────

DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "image_size": 128,
    "batch_size": 32,
    "steps": 1000,
    "lr": 1e-3,
    "device": "cuda",
    "hidden_dim": 32,
    "depth": 2,
    "power_weights": [0.22970384, 0.24373232, 0.5265638],
    "alpha_power": 0.8,
    "lambda_perceptual": 0.5,
    "lambda_ssim": 0.5,
    "lut_resolution": 16,
    "log_interval": 50,
}


def _to_triple(value: Union[float, List[float], Tuple[float, ...]]) -> Tuple[float, float, float]:
    if isinstance(value, (int, float)):
        return (float(value), float(value), float(value))
    return (float(value[0]), float(value[1]), float(value[2]))


def load_train_config(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Merge a JSON file over ``DEFAULT_TRAIN_CONFIG``.  Only keys present
    in the file override defaults."""
    with open(json_path, "r", encoding="utf-8") as f:
        overrides = json.load(f)
    return {**DEFAULT_TRAIN_CONFIG, **overrides}


@dataclass
class TrainConfig:
    data_dir: str = ""
    image_size: int = -1
    batch_size: int = 32
    steps: int = 1000
    lr: float = 1e-3
    device: str = "cuda"
    hidden_dim: int = 32
    depth: int = 2
    power_weights: Tuple[float, float, float] = (0.22970384, 0.24373232, 0.5265638)
    alpha_power: float = 0.75
    lambda_perceptual: float = 0.5
    lambda_ssim: float = 0.5
    lut_resolution: int = 16
    log_interval: int = 50

    def print(self) -> "TrainConfig":
        print("============ Train Config ============")
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            print(f"  {field}: {value}")
        print("======================================")
        return self


# ── SSIM helper ────────────────────────────────────────────────────────────

def _compute_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Return the mean SSIM between two (1, 3, H, W) image tensors."""
    device = x.device
    gauss = torch.tensor(
        [math.exp(-(i - window_size // 2) ** 2 / (2 * sigma ** 2)) for i in range(window_size)],
        device=device,
    )
    _1d = gauss / gauss.sum()
    _2d = torch.mm(_1d.unsqueeze(1), _1d.unsqueeze(0))
    window = _2d.unsqueeze(0).unsqueeze(0).expand(3, 1, window_size, window_size)

    if x.shape[1] != 3:
        x = x.expand(-1, 3, -1, -1)
    if y.shape[1] != 3:
        y = y.expand(-1, 3, -1, -1)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2

    mu1 = F.conv2d(x, window, padding=pad, groups=3)
    mu2 = F.conv2d(y, window, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(x * x, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(x * y, window, padding=pad, groups=3) - mu1_mu2

    sigma1_sq = sigma1_sq.clamp(min=0.0)
    sigma2_sq = sigma2_sq.clamp(min=0.0)

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1).clamp(min=1e-10) * (sigma1_sq + sigma2_sq + C2).clamp(min=1e-10)
    ssim_map = (num / den).clamp(min=-1.0, max=1.0)
    return ssim_map.mean()


# ── loss ───────────────────────────────────────────────────────────────────

class ColorOptimizationLoss(nn.Module):
    """
    Combined loss for screen colour optimisation.
    """

    def __init__(
        self,
        power_weights: Tuple[float, float, float] = (0.22970384, 0.24373232, 0.5265638),
        alpha_power: float = 0.7,
        lambda_perceptual: float = 0.5,
        lambda_ssim: float = 0.5,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.register_buffer(
            "power_weights", torch.tensor(power_weights, dtype=torch.float32, device=self.device)
        )
        self.alpha_power = alpha_power
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim

        self.metameric_loss = MetamericLossUniform(n_pyramid_levels=5, n_orientations=4, pooling_size=64, device=device, loss_type="L1")
        # self.lpips_model = lpips.LPIPS(net="vgg", spatial=False, model_path="models/vgg16-397923af.pth")
        # self.lpips_model.to(self.device)
        # for p in self.lpips_model.parameters():
        #     p.requires_grad = False

    def forward(self, rgb: torch.Tensor, optimized: torch.Tensor) -> Dict[str, torch.Tensor]:
        H, W, _ = rgb.shape
        rgb_flat = rgb.reshape(-1, 3)

        # --- power loss ---
        weights = self.power_weights.unsqueeze(0).expand_as(rgb_flat)
        power_loss = (optimized * weights).mean() * self.alpha_power

        # --- perceptual losses ---
        x = rgb_flat.unsqueeze(0).permute(0, 2, 1).reshape(1, 3, H, W)
        y = optimized.unsqueeze(0).permute(0, 2, 1).reshape(1, 3, H, W)

        ssim_loss = 1.0 - _compute_ssim(x, y)
        metameric_loss = self.metameric_loss(y, x)
        # perceptual_loss = self.lpips_model.forward(x, y).mean()

        total = (
            power_loss
            + self.lambda_perceptual * metameric_loss
            + self.lambda_ssim * ssim_loss
        )
        return {
            "total": total,
            "power": power_loss,
            "perceptual": metameric_loss,
            "ssim": ssim_loss,
        }


# ── training loop ──────────────────────────────────────────────────────────

def _train_loop(
    model: EllipsoidRadiusNet,
    loss_fn: ColorOptimizationLoss,
    optimizer: torch.optim.Optimizer,
    image_paths: Sequence[Path],
    config: TrainConfig,
) -> List[Dict[str, float]]:
    device = torch.device(config.device)
    model.to(device)
    model.train()
    history: List[Dict[str, float]] = []
    running: Dict[str, float] = {}

    for _step in tqdm(range(config.steps), desc="Training", unit="step"):
        rgb = sample_image(image_paths, config.image_size, device)
        optimized = model(rgb.view(-1, 3))
        losses = loss_fn(rgb, optimized)
        losses["total"].backward()

        if _step != 0 and _step % config.batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        step_losses = {k: float(v.detach().cpu()) for k, v in losses.items() if k != "optimized_rgb"}
        history.append(step_losses)

        for k, v in step_losses.items():
            running[k] = running.get(k, 0.0) + v

        if (_step + 1) % config.log_interval == 0 or _step == config.steps - 1:
            n = min(_step + 1, config.log_interval)
            avg = {k: running[k] / n for k in running}
            parts = " | ".join(f"{k}: {avg[k]:.4f}" for k in avg)
            print(f"\nStep {_step + 1}/{config.steps} | {parts}")
            running.clear()

    return history


# ── scene sampling ─────────────────────────────────────────────────────────

def sample_images_per_scene(data_dir: str | Path, samples_per_scene: int = 100, seed: int = 0) -> List[Path]:
    """Sample up to ``samples_per_scene`` images from each sub‑directory (scene) under ``data_dir``."""
    root = Path(data_dir)
    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not subdirs:
        raise FileNotFoundError(f"no scene sub‑directories under {root}")

    rng = random.Random(seed)
    all_sampled: List[Path] = []
    for subdir in subdirs:
        images = sorted(p for p in subdir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            continue
        sampled = rng.sample(images, min(samples_per_scene, len(images)))
        all_sampled.extend(sampled)
        print(f"  {subdir.name}: {len(sampled)} images sampled (out of {len(images)})")

    if not all_sampled:
        raise ValueError(f"no images found in any sub‑directory under {root}")
    print(f"Total pretrain images: {len(all_sampled)}")
    return all_sampled


# ── config builders ────────────────────────────────────────────────────────

def _make_train_config_from_json(config_json: Optional[str | Path], device: str) -> Dict[str, Any]:
    """Load JSON config, falling back to defaults."""
    if config_json is not None:
        config = load_train_config(Path(config_json))
        print(f"Loaded training config: {config_json}")
    else:
        config = {**DEFAULT_TRAIN_CONFIG}
    config["device"] = device
    return config


def _make_train_config(params: Dict[str, Any]) -> TrainConfig:
    """Build a TrainConfig from a loaded dict of params."""
    return TrainConfig(
        data_dir=params.get("data_dir", ""),
        image_size=params.get("image_size", 128),
        batch_size=params.get("batch_size", 32),
        steps=params.get("steps", 1000),
        lr=params.get("lr", 1e-3),
        device=params.get("device", "cuda"),
        hidden_dim=params.get("hidden_dim", 32),
        depth=params.get("depth", 2),
        power_weights=_to_triple(params.get("power_weights", (0.22970384, 0.24373232, 0.5265638))),
        alpha_power=params.get("alpha_power", 0.7),
        lambda_perceptual=params.get("lambda_perceptual", 0.5),
        lambda_ssim=params.get("lambda_ssim", 0.5),
        lut_resolution=params.get("lut_resolution", 16),
        log_interval=params.get("log_interval", 50),
    ).print()


# ── derive phase ───────────────────────────────────────────────────────────

def _collect_all_images(data_dir: Path) -> Tuple[List[Path], Dict[str, List[Path]]]:
    all_images: List[Path] = []
    dataset_map: Dict[str, List[Path]] = {}
    subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not subdirs:
        raise FileNotFoundError(f"no dataset sub‑directories under {data_dir}")
    print(f"Found {len(subdirs)} datasets: {[d.name for d in subdirs]}")
    for subdir in subdirs:
        images = sorted(p for p in subdir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if images:
            dataset_map[subdir.name] = images
            all_images.extend(images)
    if not all_images:
        raise ValueError("no images found across all datasets")
    print(f"Total images: {len(all_images)}")
    return all_images, dataset_map


def _extract_all_features(image_paths: List[Path], max_size: int = 128) -> np.ndarray:
    features_list: List[np.ndarray] = []
    for path in tqdm(image_paths, desc="Extracting DKL features", unit="img"):
        arr = load_image_array(path, max_size=max_size)
        features_list.append(_extract_feature_fast(arr))
    return np.stack(features_list, axis=0)


def derive_manifest(
    data_dir: str | Path,
    manifest_path: str | Path,
    n_clusters: int,
    max_size: int = 128,
    seed: int = 0,
) -> Path:
    """Phase 1: extract per‑image DKL features → k‑means → save manifest + assignments."""
    data_dir = Path(data_dir)
    manifest_path = Path(manifest_path)

    all_images, _dataset_map = _collect_all_images(data_dir)
    features = _extract_all_features(all_images, max_size=max_size)
    print(f"Feature matrix: {features.shape}")

    feature_mean, feature_std = _mean_and_std(features)
    normalized = _normalize(features, feature_mean, feature_std)
    labels, centroids = _kmeans(normalized, n_clusters=n_clusters, seed=seed)

    prototypes: List[ScenePrototype] = []
    for cluster_idx in range(n_clusters):
        members = np.where(labels == cluster_idx)[0]
        if len(members) == 0:
            continue
        proto = ScenePrototype(
            name=f"cluster_{cluster_idx}",
            feature=centroids[cluster_idx].astype(np.float32),
            strategy_name=f"cluster_{cluster_idx}",
        )
        prototypes.append(proto)
        print(f"  Cluster {cluster_idx}: {len(members)} images → {proto.name}")

    payload = {
        "version": MANIFEST_VERSION,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "prototypes": [
            {
                "name": p.name,
                "feature": np.asarray(p.feature, dtype=np.float32).tolist(),
                "lut_path": p.lut_path,
                "strategy_name": p.strategy_name,
            }
            for p in prototypes
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Manifest saved → {manifest_path}")

    assignments_path = manifest_path.with_suffix(".assignments.json")
    assignments: Dict[str, int] = {str(img.absolute()): int(labels[i]) for i, img in enumerate(all_images)}
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_path.write_text(json.dumps(assignments))
    print(f"Assignments saved → {assignments_path}")
    return manifest_path


# ── pretrain phase ─────────────────────────────────────────────────────────

def train_pretrain(
    data_dir: str | Path,
    output_dir: str | Path,
    config_json: Optional[str | Path] = None,
    device: str = "cuda",
    seed: int = 0,
    samples_per_scene: int = 100,
) -> Tuple[Path, Path]:
    """Pretrain a base model on sampled images from every scene under ``data_dir``.

    Returns
    -------
    (checkpoint_path, lut_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = _make_train_config_from_json(config_json, device)
    samples_per_scene = params.get("pretrain_samples_per_scene", samples_per_scene)
    config = _make_train_config(params)

    print(f"\nSampling up to {samples_per_scene} images per scene for pretrain ...")
    image_paths = sample_images_per_scene(data_dir, samples_per_scene=samples_per_scene, seed=seed)

    model = EllipsoidRadiusNet(hidden_dim=config.hidden_dim, depth=config.depth)
    loss_fn = ColorOptimizationLoss(
        power_weights=config.power_weights,
        alpha_power=config.alpha_power,
        lambda_perceptual=config.lambda_perceptual,
        lambda_ssim=config.lambda_ssim,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history = _train_loop(model, loss_fn, optimizer, image_paths, config)

    checkpoint_path = output_dir / "base_checkpoint.pt"
    save_checkpoint(model, str(checkpoint_path))
    print(f"Base checkpoint saved → {checkpoint_path}")

    lut_path = output_dir / "base_lut.pt"
    lut = generate_lut(model, config.lut_resolution, device=device)
    torch.save({"lut": lut}, str(lut_path))
    print(f"Base LUT saved → {lut_path}  shape={lut.shape}")

    if history:
        print(f"pretrain_final_total={history[-1]['total']:.6f}")

    return checkpoint_path, lut_path


# ── finetune (per-cluster) phase ───────────────────────────────────────────

def train_per_cluster(
    manifest_path: str | Path,
    lut_dir: str | Path,
    config_json: Optional[str | Path] = None,
    base_checkpoint: Optional[str | Path] = None,
    device: str = "cuda",
    seed: int = 0,
) -> Path:
    """Phase 2 (or finetune): for every cluster in the manifest, train a LUT on its member images.

    If ``base_checkpoint`` is provided, each cluster model is initialised from that checkpoint
    (finetune) instead of from scratch.
    """
    manifest_path = Path(manifest_path)
    lut_dir = Path(lut_dir)
    lut_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = manifest_path.with_suffix(".assignments.json")
    if not assignments_path.exists():
        raise FileNotFoundError(f"assignments file not found: {assignments_path} (run derive first)")

    assignments: Dict[str, int] = json.loads(assignments_path.read_text(encoding="utf-8"))
    cluster_to_images: Dict[int, List[str]] = {}
    for img_str, label in assignments.items():
        cluster_to_images.setdefault(label, []).append(img_str)

    prototypes, _mean, _std = load_scene_manifest(manifest_path)
    n_clusters = len(prototypes)
    print(f"Training on {n_clusters} clusters, total images: {len(assignments)}")

    if base_checkpoint is not None:
        print(f"Finetune mode: loading base checkpoint from {base_checkpoint}")

    params = _make_train_config_from_json(config_json, device)
    config = _make_train_config(params)

    luts: Dict[int, Path] = {}

    for cluster_idx in range(n_clusters):
        member_paths = [Path(p) for p in cluster_to_images.get(cluster_idx, []) if Path(p).exists()]
        if not member_paths:
            print(f"  Skipping cluster_{cluster_idx}: no valid images")
            continue

        print(f"\n{'=' * 50}")
        suffix = "finetune" if base_checkpoint else "train"
        print(f"{suffix.upper()} cluster_{cluster_idx} on {len(member_paths)} images")
        print(f"{'=' * 50}")

        if base_checkpoint is not None:
            model = load_model(str(base_checkpoint)).to(device)
        else:
            model = EllipsoidRadiusNet(hidden_dim=config.hidden_dim, depth=config.depth).to(device)

        loss_fn = ColorOptimizationLoss(
            power_weights=config.power_weights,
            alpha_power=config.alpha_power,
            lambda_perceptual=config.lambda_perceptual,
            lambda_ssim=config.lambda_ssim,
            device=device,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        history = _train_loop(model, loss_fn, optimizer, member_paths, config)

        checkpoint_path = str(lut_dir / f"cluster_{cluster_idx}.pt")
        export_lut = str(lut_dir / f"cluster_{cluster_idx}_lut.pt")

        save_checkpoint(model, checkpoint_path)
        lut = generate_lut(model, config.lut_resolution, device=device)
        torch.save({"lut": lut}, export_lut)
        luts[cluster_idx] = export_lut
        print(f"  LUT saved → {export_lut}  shape={lut.shape}")

        if history:
            print(f"  final_total={history[-1]['total']:.6f}")

    if luts:
        _assign_luts_to_manifest(manifest_path, luts)
        print(f"\nUpdated manifest with LUT paths → {manifest_path}")

    return manifest_path


def _assign_luts_to_manifest(manifest_path: str | Path, lut_paths: Dict[int, str | Path]) -> None:
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in payload.get("prototypes", []):
        name = item.get("strategy_name", "")
        if name.startswith("cluster_"):
            idx = int(name.split("_")[1])
            if idx in lut_paths:
                item["lut_path"] = str(Path(lut_paths[idx]))
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── full pipelines ─────────────────────────────────────────────────────────

def run_full_pipeline(
    data_dir: str | Path,
    output_dir: str | Path,
    n_clusters: int,
    max_size: int = 128,
    config_json: Optional[str | Path] = None,
    device: str = "cuda",
    seed: int = 0,
) -> Path:
    """End‑to‑end: derive manifest → train per‑cluster (from scratch) → final manifest with LUTs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "scene_manifest.json"

    print("=" * 60)
    print("PHASE 1/2 – DERIVE")
    print("=" * 60)
    derive_manifest(data_dir, manifest_path, n_clusters=n_clusters, max_size=max_size, seed=seed)

    print("\n" + "=" * 60)
    print("PHASE 2/2 – TRAIN (from scratch)")
    print("=" * 60)
    train_per_cluster(
        manifest_path,
        lut_dir=output_dir / "luts",
        config_json=config_json,
        device=device,
        seed=seed,
    )

    print(f"\nPipeline complete. Manifest: {manifest_path}")
    return manifest_path


def run_pretrain_finetune_pipeline(
    data_dir: str | Path,
    output_dir: str | Path,
    n_clusters: int,
    max_size: int = 128,
    pretrain_config: Optional[str | Path] = None,
    finetune_config: Optional[str | Path] = None,
    device: str = "cuda",
    seed: int = 0,
    samples_per_scene: int = 100,
) -> Path:
    """End‑to‑end: pretrain base model → derive manifest → finetune per‑cluster → final manifest with LUTs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "scene_manifest.json"

    print("=" * 60)
    print("PHASE 1/3 – PRETRAIN BASE MODEL")
    print("=" * 60)
    base_checkpoint, base_lut = train_pretrain(
        data_dir,
        output_dir=output_dir,
        config_json=pretrain_config,
        device=device,
        seed=seed,
        samples_per_scene=samples_per_scene,
    )

    print("\n" + "=" * 60)
    print("PHASE 2/3 – DERIVE")
    print("=" * 60)
    derive_manifest(data_dir, manifest_path, n_clusters=n_clusters, max_size=max_size, seed=seed)

    print("\n" + "=" * 60)
    print("PHASE 3/3 – FINETUNE PER CLUSTER")
    print("=" * 60)
    train_per_cluster(
        manifest_path,
        lut_dir=output_dir / "luts",
        config_json=finetune_config,
        base_checkpoint=base_checkpoint,
        device=device,
        seed=seed,
    )

    print(f"\nPipeline complete. Manifest: {manifest_path}")
    print(f"Base checkpoint: {base_checkpoint}")
    print(f"Base LUT: {base_lut}")
    return manifest_path


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Scene clustering & per‑cluster training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_derive = sub.add_parser("derive", help="Extract DKL features + cluster → manifest")
    p_derive.add_argument("--data-dir", type=str, required=True)
    p_derive.add_argument("--manifest", type=str, required=True)
    p_derive.add_argument("--clusters", type=int, default=8)
    p_derive.add_argument("--max-size", type=int, default=128)
    p_derive.add_argument("--seed", type=int, default=0)

    p_pretrain = sub.add_parser("pretrain", help="Pretrain a base model on sampled images from all scenes")
    p_pretrain.add_argument("--data-dir", type=str, required=True)
    p_pretrain.add_argument("--output-dir", type=str, required=True)
    p_pretrain.add_argument("--config", type=str, default='configs/pretrain_config.json', help="JSON config for pretrain")
    p_pretrain.add_argument("--samples-per-scene", type=int, default=100)
    p_pretrain.add_argument("--device", type=str, default="cuda")
    p_pretrain.add_argument("--seed", type=int, default=0)

    p_train = sub.add_parser("train", help="Train per‑cluster LUTs from existing manifest")
    p_train.add_argument("--manifest", type=str, required=True)
    p_train.add_argument("--lut-dir", type=str, required=True)
    p_train.add_argument("--base-checkpoint", type=str, default=None)
    p_train.add_argument("--config", type=str, default='configs/train_config.json', help="JSON file with hyper‑parameter overrides")
    p_train.add_argument("--device", type=str, default="cuda")
    p_train.add_argument("--seed", type=int, default=0)

    p_full = sub.add_parser("full", help="End‑to‑end: derive + train (from scratch)")
    p_full.add_argument("--data-dir", type=str, required=True)
    p_full.add_argument("--output-dir", type=str, required=True)
    p_full.add_argument("--config", type=str, default='configs/train_config.json', help="JSON file with hyper‑parameter overrides")
    p_full.add_argument("--clusters", type=int, default=8)
    p_full.add_argument("--max-size", type=int, default=128)
    p_full.add_argument("--device", type=str, default="cuda")
    p_full.add_argument("--seed", type=int, default=0)

    p_full_pt = sub.add_parser("full-pt", help="End‑to‑end: pretrain + derive + finetune")
    p_full_pt.add_argument("--data-dir", type=str, required=True)
    p_full_pt.add_argument("--output-dir", type=str, required=True)
    p_full_pt.add_argument("--pretrain-config", type=str, default='configs/pretrain_config.json')
    p_full_pt.add_argument("--finetune-config", type=str, default='configs/finetune_config.json')
    p_full_pt.add_argument("--clusters", type=int, default=8)
    p_full_pt.add_argument("--max-size", type=int, default=128)
    p_full_pt.add_argument("--samples-per-scene", type=int, default=100)
    p_full_pt.add_argument("--device", type=str, default="cuda")
    p_full_pt.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.command == "pretrain":
        checkpoint, lut = train_pretrain(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            config_json=args.config,
            device=args.device,
            seed=args.seed,
            samples_per_scene=args.samples_per_scene,
        )
        print(f"pretrain_complete checkpoint={checkpoint} lut={lut}")
    elif args.command == "derive":
        manifest_path = derive_manifest(
            data_dir=Path(args.data_dir),
            manifest_path=Path(args.manifest),
            n_clusters=args.clusters,
            max_size=args.max_size,
            seed=args.seed,
        )
        print(f"derive_complete={manifest_path}")
    elif args.command == "train":
        train_per_cluster(
            manifest_path=Path(args.manifest),
            lut_dir=Path(args.lut_dir),
            config_json=args.config,
            seed=args.seed,
            device=args.device,
            base_checkpoint=args.base_checkpoint,
        )
        print(f"train_complete={args.manifest}")
    elif args.command == "full":
        manifest_path = run_full_pipeline(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            n_clusters=args.clusters,
            max_size=args.max_size,
            config_json=args.config,
            device=args.device,
            seed=args.seed,
        )
        print(f"full_complete={manifest_path}")
    elif args.command == "full-pt":
        manifest_path = run_pretrain_finetune_pipeline(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            n_clusters=args.clusters,
            max_size=args.max_size,
            pretrain_config=args.pretrain_config,
            finetune_config=args.finetune_config,
            device=args.device,
            seed=args.seed,
            samples_per_scene=args.samples_per_scene,
        )
        print(f"full_pt_complete={manifest_path}")


if __name__ == "__main__":
    main()