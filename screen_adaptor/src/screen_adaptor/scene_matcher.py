from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from .colorspace import RGB2DKL, sRGB2RGB

MANIFEST_VERSION = 1
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ScenePrototype:
    name: str
    feature: np.ndarray
    lut_path: Optional[str] = None
    strategy_name: Optional[str] = None


def load_image_array(image_path: str | Path, max_size: int = 128) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    if max_size > 0:
        image = image.resize((max_size, max_size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _extract_feature(image: np.ndarray) -> np.ndarray:
    """Extract a 27‑dim DKL feature vector: mean, std, percentiles, skewness, kurtosis."""
    linear_rgb = sRGB2RGB(image)
    dkl = linear_rgb.reshape(-1, 3) @ RGB2DKL.T
    flat = dkl.reshape(-1, 3)

    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    p10, p25, p50, p75, p90 = np.percentile(flat, [10, 25, 50, 75, 90], axis=0)
    centered = flat - mean
    m3 = (centered ** 3).mean(axis=0)
    m4 = (centered ** 4).mean(axis=0)
    denom = std ** 3
    denom = np.where(denom < 1e-8, 1.0, denom)
    skew = m3 / denom
    denom4 = std ** 4
    denom4 = np.where(denom4 < 1e-8, 1.0, denom4)
    kurt = m4 / denom4 - 3.0

    return np.concatenate([
        mean, std, p10, p25, p50, p75, p90, skew, kurt,
    ]).astype(np.float32)
    
def _extract_feature_fast(image: np.ndarray) -> np.ndarray:
    linear_rgb = sRGB2RGB(image)
    dkl = linear_rgb.reshape(-1, 3) @ RGB2DKL.T
    
    mean = dkl.mean(axis=0)
    std = dkl.std(axis=0)
    median = np.median(dkl, axis=0)
    
    return np.concatenate([mean, std, median]).astype(np.float32)

def _sample_window_feature(image_paths: Sequence[str | Path], max_size: int = 128) -> np.ndarray:
    features = [_extract_feature_fast(load_image_array(path, max_size=max_size)) for path in image_paths]
    return np.mean(np.stack(features, axis=0), axis=0)


def _window_paths(image_paths: Sequence[str | Path], window_size: int, step: int) -> list[list[Path]]:
    ordered = [Path(path) for path in image_paths]
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive")
    if len(ordered) < window_size:
        return []
    return [ordered[start : start + window_size] for start in range(0, len(ordered) - window_size + 1, step)]


def _mean_and_std(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def _kmeans(features: np.ndarray, n_clusters: int, seed: int = 0, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(features, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("features must be a 2D array")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if data.shape[0] < n_clusters:
        raise ValueError("n_clusters cannot exceed number of samples")

    rng = np.random.default_rng(seed)
    centroids = data[rng.choice(data.shape[0], size=n_clusters, replace=False)].copy()

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        updated = centroids.copy()
        for index in range(n_clusters):
            members = data[labels == index]
            if members.size:
                updated[index] = members.mean(axis=0)
        if np.allclose(updated, centroids):
            break
        centroids = updated

    distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    return labels, centroids


class SceneMatcher:
    def __init__(self, prototypes: Sequence[ScenePrototype], feature_mean: np.ndarray, feature_std: np.ndarray) -> None:
        if not prototypes:
            raise ValueError("prototypes must not be empty")
        self.prototypes = list(prototypes)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)

    def match(self, feature: np.ndarray) -> tuple[int, ScenePrototype, float]:
        query = _normalize(np.asarray(feature, dtype=np.float32), self.feature_mean, self.feature_std)
        prototype_features = np.stack([
            np.asarray(proto.feature, dtype=np.float32)
            for proto in self.prototypes
        ], axis=0)
        distances = np.linalg.norm(prototype_features - query[None, :], axis=1)
        best_index = int(np.argmin(distances))
        return best_index, self.prototypes[best_index], float(distances[best_index])

    def match_paths(self, image_paths: Sequence[str | Path], max_size: int = 128) -> tuple[int, ScenePrototype, float]:
        return self.match(_sample_window_feature(image_paths, max_size=max_size))


def build_scene_manifest(
    data_dir: str | Path,
    manifest_path: str | Path,
    n_clusters: int,
    window_size: int = 8,
    step: int = 4,
    max_size: int = 128,
    seed: int = 0,
) -> Path:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data_dir does not exist: {root}")

    image_paths = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    windows = _window_paths(image_paths, window_size=window_size, step=step)
    if not windows:
        raise ValueError("not enough images to build windows")

    window_features = np.stack([_sample_window_feature(window, max_size=max_size) for window in windows], axis=0)
    feature_mean, feature_std = _mean_and_std(window_features)
    normalized = _normalize(window_features, feature_mean, feature_std)
    labels, centroids = _kmeans(normalized, n_clusters=n_clusters, seed=seed)

    prototypes = []
    for cluster_index in range(n_clusters):
        members = np.where(labels == cluster_index)[0]
        if members.size == 0:
            continue
        representative = windows[int(members[0])]
        prototypes.append(
            ScenePrototype(
                name=representative[0].parent.name,
                feature=centroids[cluster_index].astype(np.float32),
                strategy_name=f"cluster_{cluster_index}",
            )
        )

    payload = {
        "version": MANIFEST_VERSION,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "prototypes": [
            {
                "name": prototype.name,
                "feature": np.asarray(prototype.feature, dtype=np.float32).tolist(),
                "lut_path": prototype.lut_path,
                "strategy_name": prototype.strategy_name,
            }
            for prototype in prototypes
        ],
    }

    output_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_scene_manifest(manifest_path: str | Path) -> tuple[list[ScenePrototype], np.ndarray, np.ndarray]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version: {payload.get('version')}")

    prototypes = [
        ScenePrototype(
            name=str(item["name"]),
            feature=np.asarray(item["feature"], dtype=np.float32),
            lut_path=item.get("lut_path"),
            strategy_name=item.get("strategy_name"),
        )
        for item in payload.get("prototypes", [])
    ]
    if not prototypes:
        raise ValueError("scene manifest has no prototypes")

    return (
        prototypes,
        np.asarray(payload["feature_mean"], dtype=np.float32),
        np.asarray(payload["feature_std"], dtype=np.float32),
    )