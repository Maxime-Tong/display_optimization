"""Shared helpers for the LUT specialization visualization suite.

Kept independent from ``visualize_lut.py`` so the heavy lifting (loading,
metrics, colour difference, image statistics, correlations) can be reused or
unit-tested separately.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Optional reuse of the project's own DKL colour matrices.  The cluster
# assignment logic in ``screen_adaptor`` was built on these, so reusing them
# keeps scene-feature statistics consistent with how clusters were derived.
try:  # pragma: no cover - environment dependent
    _SA_SRC = Path(__file__).resolve().parents[1] / "screen_adaptor" / "src"
    sys.path.insert(0, str(_SA_SRC))
    from screen_adaptor.colorspace import RGB2DKL, sRGB2RGB

    HAS_DKL = True
except Exception:  # pragma: no cover
    HAS_DKL = False


_CLUSTER_LUT_RE = re.compile(r"cluster_(\d+)_lut\.(?:pt|npy|npz)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# LUT loading
# ---------------------------------------------------------------------------
def load_lut(path: str | Path) -> np.ndarray:
    """Load a LUT from ``.pt`` / ``.npy`` / ``.npz`` into float32 [N,N,N,3].

    The real ``screen_adaptor`` LUTs are PyTorch checkpoints containing
    ``{"lut": tensor}``; raw tensors, plain numpy arrays and npz archives are
    also accepted so the suite works with 17^3 ``.npy`` files too.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LUT file not found: {path}")

    if path.suffix == ".pt":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            lut = payload.get("lut")
            if lut is None or not hasattr(lut, "shape"):
                raise ValueError(
                    f"Unexpected .pt payload in {path}: dict without a 'lut' tensor"
                )
        else:
            lut = payload
        lut = lut.numpy() if hasattr(lut, "numpy") else np.asarray(lut)
    elif path.suffix == ".npy":
        lut = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path) as archive:
            key = "lut" if "lut" in archive else next(iter(archive.keys()))
            lut = archive[key]
    else:
        raise ValueError(
            f"Unsupported LUT extension '{path.suffix}' (use .pt / .npy / .npz)"
        )

    lut = np.asarray(lut, dtype=np.float32)
    if lut.ndim != 4 or lut.shape[-1] != 3:
        raise ValueError(f"LUT must have shape [N,N,N,3]; got {lut.shape} in {path}")
    if not (lut.shape[0] == lut.shape[1] == lut.shape[2]):
        raise ValueError(f"LUT grid must be cubic; got {lut.shape} in {path}")
    return lut


def discover_lut_paths(lut_dir: str | Path, pattern: str = "") -> list[Path]:
    """Return the sorted cluster LUT paths found under ``lut_dir``.

    ``cluster_0_lut.pt`` .. ``cluster_7_lut.pt`` are discovered automatically.
    An explicit ``pattern`` containing ``{i}`` is used only when no files match
    the default naming convention.
    """
    lut_dir = Path(lut_dir)
    if not lut_dir.is_dir():
        raise FileNotFoundError(f"LUT directory not found: {lut_dir}")

    matches: dict[int, Path] = {}
    for path in lut_dir.iterdir():
        match = _CLUSTER_LUT_RE.match(path.name)
        if match:
            matches[int(match.group(1))] = path

    if not matches and pattern:
        # Explicit pattern fallback, e.g. "lut_{i}.pt".
        for index in range(64):
            path = lut_dir / pattern.format(i=index)
            if path.exists():
                matches[index] = path

    if not matches:
        raise FileNotFoundError(
            f"No cluster LUTs matching 'cluster_<i>_lut.*' in {lut_dir}"
        )
    return [matches[index] for index in sorted(matches)]


def lut_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, values) for an N x N x N input grid.

    ``indices`` is int [N^3, 3] with one row per grid point and columns
    (r_idx, g_idx, b_idx); ``values`` is the corresponding float [N^3, 3] in
    [0, 1].  The flattened order matches ``numpy.meshgrid(..., indexing='ij')``
    so slices along any axis reshape cleanly.
    """
    axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    indices = np.stack(
        np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"),
        axis=-1,
    )
    return indices.reshape(-1, 3), grid.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------
def pairwise_distances(
    luts: Sequence[np.ndarray],
    metrics: Iterable[str] = ("euclidean", "manhattan", "cosine"),
) -> dict[str, np.ndarray]:
    """Pairwise LUT distance matrices (n_luts x n_luts) for each metric.

    Each LUT is flattened to a single vector of length N^3 * 3; distances are
    computed with scikit-learn so cosine is handled correctly.
    """
    from sklearn.metrics import pairwise_distances as sk_pairwise_distances

    vectors = np.stack([lut.reshape(-1) for lut in luts], axis=0)
    return {
        metric: sk_pairwise_distances(vectors, metric=metric)
        for metric in metrics
    }


# ---------------------------------------------------------------------------
# Colour difference (Delta E)
# ---------------------------------------------------------------------------
def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> CIE Lab (D65) via colour-science."""
    import colour

    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float64)
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb))


def compute_delta_e(
    base: np.ndarray,
    finetuned: np.ndarray,
    metric: str = "cie2000",
) -> np.ndarray:
    """Per-grid-point colour difference between two LUTs (N^3,).

    The two LUTs are sampled at the same grid coordinates, so this measures how
    perceptually different the finetuned output color is from the base output
    at every input color.  ``metric`` supports ``cie2000`` (default) and
    ``cie76`` (simple Lab Euclidean).
    """
    base_flat = base.reshape(-1, 3)
    ft_flat = finetuned.reshape(-1, 3)
    lab_base = _srgb_to_lab(base_flat)
    lab_ft = _srgb_to_lab(ft_flat)

    if metric == "cie76":
        delta = np.linalg.norm(lab_base - lab_ft, axis=1)
    elif metric == "cie2000":
        import colour

        delta = colour.difference.delta_E_CIE2000(lab_base, lab_ft)
        delta = np.asarray(delta, dtype=np.float64)
    else:
        raise ValueError(f"Unknown delta_e_metric '{metric}' (cie2000 | cie76)")
    return delta.reshape(-1)


def adjustment_stats(base: np.ndarray, finetuned: np.ndarray) -> dict[str, float]:
    """Summary statistics of the per-point adjustment field F - B."""
    delta = finetuned.reshape(-1, 3).astype(np.float64) - base.reshape(-1, 3)
    magnitude = np.linalg.norm(delta, axis=1)
    return {
        "mean_abs_magnitude": float(np.abs(delta).mean()),
        "adjustment_std": float(magnitude.std()),
        "max_magnitude": float(magnitude.max()),
        "p95_magnitude": float(np.percentile(magnitude, 95)),
        "mean_abs_delta_r": float(np.abs(delta[:, 0]).mean()),
        "mean_abs_delta_g": float(np.abs(delta[:, 1]).mean()),
        "mean_abs_delta_b": float(np.abs(delta[:, 2]).mean()),
    }


# ---------------------------------------------------------------------------
# Scene metadata & image statistics
# ---------------------------------------------------------------------------
def load_manifest(
    manifest_path: str | Path,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Load scene_manifest.json -> (prototypes, feature_mean, feature_std)."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    prototypes = [
        {
            "name": item.get("name", f"cluster_{i}"),
            "feature": np.asarray(item["feature"], dtype=np.float32),
            "strategy_name": item.get("strategy_name"),
        }
        for i, item in enumerate(payload.get("prototypes", []))
    ]
    return (
        prototypes,
        np.asarray(payload.get("feature_mean"), dtype=np.float32),
        np.asarray(payload.get("feature_std"), dtype=np.float32),
    )


def load_assignments(assignments_path: str | Path) -> dict[str, int]:
    """Load scene_manifest.assignments.json -> {resolved image path: cluster}.

    Only entries whose file still exists on disk are kept.
    """
    raw = json.loads(Path(assignments_path).read_text(encoding="utf-8"))
    assignments: dict[str, int] = {}
    for path_str, cluster in raw.items():
        path = Path(path_str)
        if path.exists():
            assignments[str(path.resolve())] = int(cluster)
    return assignments


def cluster_image_paths(
    assignments: dict[str, int], n_clusters: int
) -> dict[int, list[str]]:
    """Group assigned image paths by cluster index."""
    grouped: dict[int, list[str]] = {i: [] for i in range(n_clusters)}
    for path, cluster in assignments.items():
        if 0 <= cluster < n_clusters:
            grouped[cluster].append(path)
    return grouped


def load_image_rgb(image_path: str | Path, max_size: int = 128) -> np.ndarray:
    """Load an image as float32 RGB [max_size, max_size, 3] in [0, 1]."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = image.resize((max_size, max_size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def dkl_fast_feature(rgb: np.ndarray) -> np.ndarray | None:
    """9-dim DKL feature (mean, std, median) matching ``_extract_feature_fast``."""
    if not HAS_DKL:
        return None
    linear_rgb = sRGB2RGB(rgb)
    dkl = linear_rgb.reshape(-1, 3) @ RGB2DKL.T
    return np.concatenate(
        [dkl.mean(axis=0), dkl.std(axis=0), np.median(dkl, axis=0)]
    ).astype(np.float32)


def fallback_feature(rgb: np.ndarray) -> np.ndarray:
    """6-dim luminance/RGB fallback when the DKL matrices are unavailable."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    saturation = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)
    return np.array(
        [
            luma.mean(),
            luma.std(),
            r.mean(),
            g.mean(),
            b.mean(),
            saturation.mean(),
        ],
        dtype=np.float32,
    )


def _histogram_entropy(histogram: np.ndarray) -> float:
    """Shannon entropy (bits) of a normalized histogram."""
    p = histogram / max(histogram.sum(), 1e-12)
    eps = 1e-12
    return float(-(p * np.log2(p + eps)).sum())


def compute_scene_stats(
    cluster_paths: dict[int, list[str]],
    max_images: int = 60,
    max_size: int = 128,
    bins: int = 32,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-cluster image statistics: luminance, RGB, saturation, histograms.

    Returns one row per cluster with mean/STD of the sampled images, per-channel
    histogram entropies, and (when available) the 9-dim DKL fast features that
    mirror the clustering feature space.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for cluster in sorted(cluster_paths):
        paths = sorted(cluster_paths[cluster])
        if not paths:
            rows.append({"cluster": cluster, "n_images": 0})
            continue
        if len(paths) > max_images:
            paths = list(rng.choice(paths, size=max_images, replace=False))

        lum_means, lum_stds = [], []
        channel_means = {"r": [], "g": [], "b": []}
        channel_stds = {"r": [], "g": [], "b": []}
        saturations = []
        hist_sum = np.zeros((3, bins))
        dkl_feats: list[np.ndarray] = []

        for path in paths:
            rgb = load_image_rgb(path, max_size=max_size)
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            lum_means.append(luma.mean())
            lum_stds.append(luma.std())
            channel_means["r"].append(r.mean())
            channel_means["g"].append(g.mean())
            channel_means["b"].append(b.mean())
            channel_stds["r"].append(r.std())
            channel_stds["g"].append(g.std())
            channel_stds["b"].append(b.std())
            saturations.append((np.max(rgb, axis=-1) - np.min(rgb, axis=-1)).mean())
            for channel in range(3):
                hist, _ = np.histogram(
                    rgb[..., channel], bins=bins, range=(0.0, 1.0), density=True
                )
                hist_sum[channel] += hist
            dkl = dkl_fast_feature(rgb)
            if dkl is not None:
                dkl_feats.append(dkl)

        n = len(paths)
        hist_mean = hist_sum / n
        row: dict = {
            "cluster": cluster,
            "n_images": n,
            "luminance_mean": float(np.mean(lum_means)),
            "luminance_std": float(np.mean(lum_stds)),
            "r_mean": float(np.mean(channel_means["r"])),
            "g_mean": float(np.mean(channel_means["g"])),
            "b_mean": float(np.mean(channel_means["b"])),
            "r_std": float(np.mean(channel_stds["r"])),
            "g_std": float(np.mean(channel_stds["g"])),
            "b_std": float(np.mean(channel_stds["b"])),
            "saturation_mean": float(np.mean(saturations)),
            "hist_entropy_r": _histogram_entropy(hist_mean[0]),
            "hist_entropy_g": _histogram_entropy(hist_mean[1]),
            "hist_entropy_b": _histogram_entropy(hist_mean[2]),
        }
        if dkl_feats:
            mean_dkl = np.mean(np.stack(dkl_feats), axis=0)
            for i, value in enumerate(mean_dkl):
                row[f"dkl_{i}"] = float(value)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
def pearson_spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Pearson r and Spearman rho with p-values for two equal-length arrays."""
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def build_correlation_table(
    adjustment_df: pd.DataFrame,
    scene_df: pd.DataFrame,
    adjustment_cols: Sequence[str],
    scene_cols: Sequence[str],
) -> pd.DataFrame:
    """Long-format correlation table (adjustment stat x scene stat)."""
    rows = []
    for adj_col in adjustment_cols:
        for scene_col in scene_cols:
            result = pearson_spearman(
                adjustment_df[adj_col].to_numpy(dtype=float),
                scene_df[scene_col].to_numpy(dtype=float),
            )
            rows.append({"adjustment_stat": adj_col, "scene_stat": scene_col, **result})
    table = pd.DataFrame(rows)
    return table.reindex(table["pearson_r"].abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_fig(fig, path: str | Path, dpi: int = 150) -> None:
    """Save a matplotlib figure and close it to free memory."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):{floatfmt}}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
