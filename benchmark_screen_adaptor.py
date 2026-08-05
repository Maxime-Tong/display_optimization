#!/usr/bin/env python
"""
Benchmark: screen_adaptor LUT color transform, applied to screen_adaptor's
datasets and measured with screen display power + quality metrics.

This script:
  1. Loads the screen_adaptor LUT transformers:
       - single  : base_lut.pt
       - cluster : scene_manifest.json + per-cluster LUTs (dynamic switching)
  2. For each dataset, applies the LUT (optionally foveated/temporal) to every image
  3. Computes: Saving%, PSNR, SSIM, MetaM (metameric loss)
  4. Outputs per-dataset summary CSV + JSON + Markdown
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ── Add screen_adaptor src to path ────────────────────────────────────────
_SA_ROOT = Path(__file__).resolve().parent / "screen_adaptor"
_SA_SRC = _SA_ROOT / "src"
sys.path.insert(0, str(_SA_SRC))
sys.path.insert(0, str(_SA_ROOT))  # for bundled odak (MetamericLoss)

from screen_adaptor.model import LUTColorTransformer, load_lut_transformer  # noqa: E402
from screen_adaptor.scene_matcher import SceneMatcher, load_scene_manifest  # noqa: E402

from odak.learn.perception import MetamericLoss  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────

# Datasets from screen_adaptor eval.ps1
DATASETS = [
    ("cf",            _SA_ROOT / "datasets" / "cf"),
    ("delta_force",   _SA_ROOT / "datasets" / "delta_force"),
    ("dfm300",        _SA_ROOT / "datasets" / "dfm300"),
    ("genshin_impact",_SA_ROOT / "datasets" / "genshin_impact"),
    ("jkchess",       _SA_ROOT / "datasets" / "jkchess"),
    ("miHoYo",        _SA_ROOT / "datasets" / "miHoYo"),
    ("nrc",           _SA_ROOT / "datasets" / "nrc"),
    ("sgame0",        _SA_ROOT / "datasets" / "sgame0"),
]

# screen_adaptor outputs
BASE_LUT = _SA_ROOT / "outputs" / "base_lut.pt"
SCENE_MANIFEST = _SA_ROOT / "outputs" / "scene_manifest.json"

# OLED power weights used in screen_adaptor eval.ps1: 0.229 0.243 0.526 (R,G,B)
POWER_WEIGHTS_RGB = (0.229, 0.243, 0.526)

# MetaM loss instance (matching screen_adaptor eval.py)
_fov_hvs_loss = MetamericLoss(
    real_image_width=1.4,
    real_viewing_distance=0.7,
    equi=False,
    alpha=5.0,
    mode="quadratic",
    loss_type="L1",
    use_l2_foveal_loss=False,
    n_pyramid_levels=5,
    n_orientations=4,
    use_radial_weight=True,
)


# ── Image loading helpers ─────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(directory: Path) -> List[Path]:
    paths = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths


def resolve_lut_path(path: Optional[str]) -> Path:
    """Resolve a manifest lut_path to an existing file (fallback to base LUT).

    Mirrors generate_video.py's _resolve_lut_path: the manifest may reference
    absolute paths (e.g. .../screen_adaptor/clusters/...) that no longer exist,
    so we search the known output directories for the file by name.
    """
    if not path:
        return BASE_LUT
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        _SA_ROOT / "outputs" / "luts" / p.name,
        _SA_ROOT / "outputs" / p,
        _SA_ROOT / "outputs" / "luts" / p,
        _SA_ROOT / "clusters" / p.name,
        p,
    ]
    for c in candidates:
        if c.exists():
            return c
    print(f"  [warn] LUT not found for {path}, fallback to base_lut.pt")
    return BASE_LUT


# ── LUT pipeline construction ─────────────────────────────────────────────

def build_transformers(
    lut_mode: str,
    foveated: float,
    temporal: float,
) -> Tuple[List[LUTColorTransformer], Optional[SceneMatcher]]:
    """Build transformer(s) + optional SceneMatcher.

    Returns:
        (transformers, matcher_or_None)
        - lut_mode=="single": one transformer, matcher=None
        - lut_mode=="cluster": one transformer per prototype, matcher=SceneMatcher
    """
    if lut_mode == "single":
        transformer = load_lut_transformer(
            BASE_LUT, foveated=foveated, temporal=temporal,
        )
        return [transformer], None

    # cluster 模式：从 scene_manifest 加载 prototypes + 每簇 LUT
    if not SCENE_MANIFEST.exists():
        raise FileNotFoundError(f"scene manifest not found: {SCENE_MANIFEST}")
    prototypes, feat_mean, feat_std = load_scene_manifest(SCENE_MANIFEST)
    transformers = []
    for proto in prototypes:
        lut = resolve_lut_path(proto.lut_path)
        transformers.append(
            load_lut_transformer(lut, foveated=foveated, temporal=temporal)
        )
    matcher = SceneMatcher(prototypes, feature_mean=feat_mean, feature_std=feat_std)
    return transformers, matcher


def apply_lut_image(
    img_path: Path,
    img: np.ndarray,
    transformers: List[LUTColorTransformer],
    matcher: Optional[SceneMatcher],
) -> Tuple[np.ndarray, int, Optional[str], float]:
    """Apply the selected LUT to a single image (full native resolution).

    Args:
        img_path: image path (used for scene matching)
        img: float32 sRGB in [0, 1], shape (H, W, 3)
        transformers: list of LUT transformers
        matcher: SceneMatcher or None (single-LUT mode)

    Returns:
        (optimized, lut_index, prototype_name, match_distance)
    """
    rgb = torch.from_numpy(np.asarray(img, dtype=np.float32))

    if matcher is None:
        transformer = transformers[0]
        lut_idx, proto_name, dist = 0, None, 0.0
    else:
        best_idx, proto, dist = matcher.match_paths([img_path])
        transformer = transformers[best_idx]
        lut_idx = best_idx
        proto_name = proto.name

    with torch.no_grad():
        optimized = transformer.transform(rgb)
    optimized_np = optimized.cpu().numpy()

    return optimized_np, lut_idx, proto_name, float(dist)


# ── Metrics (matching screen_adaptor eval.py / other benchmarks) ─────────

def weighted_power_np(rgb: np.ndarray, weights: Tuple[float, float, float]) -> np.ndarray:
    """Compute per-pixel OLED weighted power: sum(rgb * weights)."""
    w = np.array(weights, dtype=np.float64)
    return (rgb * w).sum(axis=-1)


def resize_for_metametric(
    original: np.ndarray,
    optimized: np.ndarray,
    max_side: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downscale both images (float32 [0,1]) so the longer side <= max_side."""
    h, w = original.shape[0], original.shape[1]
    if max(h, w) <= max_side:
        return original, optimized

    scale = max_side / float(max(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    def _resize(img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray((np.clip(img, 0, 1) * 255.0).astype(np.uint8))
        pil = pil.resize((new_w, new_h), Image.LANCZOS)
        return np.asarray(pil, dtype=np.float32) / 255.0

    return _resize(original), _resize(optimized)


def compute_metametric(
    original: np.ndarray,
    optimized: np.ndarray,
    max_side: int = 768,
) -> float:
    """
    Compute MetamericLoss using odak (matching screen_adaptor eval.py),
    optionally downscaled to bound runtime for very large images.

    Args:
        original, optimized: float32 in [0,1], shape (H, W, 3)
    Returns:
        metametric: scalar float
    """
    orig_r, opt_r = resize_for_metametric(original, optimized, max_side)

    # NCHW layout
    orig_nchw = torch.from_numpy(orig_r).unsqueeze(0).permute(0, 3, 1, 2)
    opt_nchw = torch.from_numpy(opt_r).unsqueeze(0).permute(0, 3, 1, 2)
    with torch.no_grad():
        val = float(_fov_hvs_loss(opt_nchw, orig_nchw, gaze=[0.5, 0.5]))
    return val


def evaluate_image(
    original: np.ndarray,
    optimized: np.ndarray,
    power_weights: Tuple[float, float, float],
    max_metam_side: int = 768,
) -> Dict[str, float]:
    """
    Compute SCREEN DISPLAY POWER + quality metrics for a single image pair
    (float32 [0, 1]).

    Returns dict with keys: saving, psnr, ssim, metametric
    """
    # OLED display power saving
    orig_power = weighted_power_np(original, power_weights).sum()
    opt_power = weighted_power_np(optimized, power_weights).sum()
    saving = 0.0 if orig_power <= 0 else float(1.0 - (opt_power / orig_power))

    # PSNR
    psnr = float(peak_signal_noise_ratio(original, optimized, data_range=1.0))

    # SSIM
    ssim = float(structural_similarity(
        original, optimized, channel_axis=2, data_range=1.0,
    ))

    # MetaM (perceptual losslessness check)
    metametric = compute_metametric(original, optimized, max_side=max_metam_side)

    return {
        "saving": saving,
        "psnr": psnr,
        "ssim": ssim,
        "metametric": metametric,
    }


# ── Main benchmark loop ───────────────────────────────────────────────────

def run_benchmark(
    max_images: int = 10,
    output_dir: Path = Path("screen_adaptor_benchmark_results"),
    max_metam_side: int = 768,
    lut_mode: str = "cluster",
    foveated: float = 1.0,
    temporal: float = 0.0,
) -> None:
    """Run screen_adaptor LUT on all datasets; measure screen power + quality."""
    output_dir.mkdir(parents=True, exist_ok=True)

    transformers, matcher = build_transformers(lut_mode, foveated, temporal)
    n_luts = len(transformers)

    print(f"\n{'='*70}")
    print(f"  screen_adaptor benchmark")
    print(f"  lut-mode : {lut_mode} ({n_luts} LUT{'s' if n_luts > 1 else ''})")
    print(f"  foveated : {foveated}")
    print(f"  temporal : {temporal}")
    if matcher is not None:
        print(f"  scene matching enabled (dynamic LUT switching)")
    print(f"{'='*70}")

    all_summaries: List[Dict[str, Any]] = []
    per_dataset_results: Dict[str, Dict[str, Any]] = {}

    for dataset_name, dataset_dir in DATASETS:
        if not dataset_dir.exists():
            print(f"\n[SKIP] Dataset not found: {dataset_dir}")
            continue

        print(f"\n{'='*64}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Path: {dataset_dir}")
        print(f"{'='*64}")

        images = collect_images(dataset_dir)
        if not images:
            print(f"  No images found, skipping.")
            continue

        selected = images[:max_images]
        print(f"  Found {len(images)} images, processing {len(selected)}...")

        results: List[Dict[str, Any]] = []
        savings: List[float] = []
        psnrs: List[float] = []
        ssims: List[float] = []
        metametrics: List[float] = []

        for idx, img_path in enumerate(selected):
            try:
                # Load image (float sRGB in [0, 1]); full native resolution
                img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0

                # Apply LUT (single or scene-matched cluster)
                t0 = time.perf_counter()
                optimized, lut_idx, proto_name, match_dist = apply_lut_image(
                    img_path, img, transformers, matcher,
                )
                elapsed = time.perf_counter() - t0

                # Metrics
                metrics = evaluate_image(
                    img, optimized, POWER_WEIGHTS_RGB,
                    max_metam_side=max_metam_side,
                )

                results.append({
                    "filename": img_path.name,
                    **metrics,
                    "lut_index": lut_idx,
                    "time_sec": round(elapsed, 3),
                })
                if matcher is not None:
                    results[-1]["matched_prototype"] = proto_name
                    results[-1]["match_distance"] = round(match_dist, 4)

                savings.append(metrics["saving"])
                psnrs.append(metrics["psnr"])
                ssims.append(metrics["ssim"])
                metametrics.append(metrics["metametric"])

                extra = f" [LUT:{lut_idx}]"
                if matcher is not None and proto_name is not None:
                    extra += f" ({proto_name}, d={match_dist:.3f})"
                print(
                    f"  [{idx+1:4d}/{len(selected)}] {img_path.name}{extra}: "
                    f"saving={metrics['saving']*100:.2f}%, "
                    f"PSNR={metrics['psnr']:.2f}dB, "
                    f"SSIM={metrics['ssim']:.4f}, "
                    f"MetaM={metrics['metametric']:.6f} "
                    f"({elapsed:.2f}s)"
                )

            except Exception as e:
                print(f"  [{idx+1:4d}/{len(selected)}] {img_path.name}: ERROR - {e}")

        # Summary for this dataset
        if savings:
            avg_saving = float(np.mean(savings))
            avg_psnr = float(np.mean(psnrs))
            avg_ssim = float(np.mean(ssims))
            avg_metametric = float(np.mean(metametrics))
        else:
            avg_saving = avg_psnr = avg_ssim = avg_metametric = 0.0

        summary = {
            "dataset": dataset_name,
            "total_images": len(results),
            "avg_saving_percent": avg_saving * 100,
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "avg_metametric": avg_metametric,
        }
        all_summaries.append(summary)
        per_dataset_results[dataset_name] = {
            "summary": summary,
            "per_image": results,
        }

        print(f"\n  --- {dataset_name} Summary ---")
        print(f"  Images:          {len(results)}")
        print(f"  Avg screen power saving: {avg_saving*100:.2f}%")
        print(f"  Avg PSNR:        {avg_psnr:.2f} dB")
        print(f"  Avg SSIM:        {avg_ssim:.4f}")
        print(f"  Avg MetaM:       {avg_metametric:.6f}")

    # ── Save Results ──────────────────────────────────────────────────────

    # JSON (full details)
    json_path = output_dir / "screen_adaptor_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "method": ("screen_adaptor: EllipsoidRadiusNet-derived 3D LUT "
                           "color transform; cluster mode dynamically switches "
                           "LUTs via DKL scene matching"),
                "lut_mode": lut_mode,
                "n_luts": n_luts,
                "foveated": foveated,
                "temporal": temporal,
                "scene_matching": matcher is not None,
                "base_lut": str(BASE_LUT),
                "scene_manifest": str(SCENE_MANIFEST),
                "benchmark_metric": "screen (OLED) display power: "
                                    "saving = 1 - power(opt)/power(orig), "
                                    "power = sum(R*0.229 + G*0.243 + B*0.526)",
                "power_weights_rgb": list(POWER_WEIGHTS_RGB),
                "max_metam_side": max_metam_side,
                "max_images_per_dataset": max_images,
            },
            "per_dataset": per_dataset_results,
            "summary_table": all_summaries,
        }, f, indent=2)
    print(f"\nFull results saved to: {json_path}")

    # CSV summary
    csv_path = output_dir / "screen_adaptor_benchmark_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "total_images", "avg_saving_percent",
            "avg_psnr", "avg_ssim", "avg_metametric",
        ])
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"CSV summary saved to: {csv_path}")

    # Markdown summary
    md_path = output_dir / "screen_adaptor_benchmark_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# screen_adaptor Benchmark Results\n\n")
        f.write("- Method: EllipsoidRadiusNet-derived 3D LUT color transform\n")
        f.write("- LUT mode: **%s** (%d LUT%s)\n" % (
            lut_mode, n_luts, "s" if n_luts > 1 else ""))
        f.write("- Foveated: %s, Temporal: %s\n" % (foveated, temporal))
        if matcher is not None:
            f.write("- Scene matching enabled (dynamic LUT switching via DKL "
                    "features)\n")
        f.write("- Benchmark metric: **screen (OLED) display power** saving with "
                "weights (R,G,B) = %s\n" % (POWER_WEIGHTS_RGB,))
        f.write("- MetaM computed with max side %d px (odak MetamericLoss)\n"
                % max_metam_side)
        f.write("- Images per dataset: up to %d\n\n" % max_images)
        f.write("| Dataset | Images | Power Saving (%) | PSNR (dB) | SSIM "
                "| MetaM |\n")
        f.write("|---------|--------|:----------------:|:---------:|:-----:"
                "|:------:|\n")
        for s in all_summaries:
            f.write(
                f"| {s['dataset']} | {s['total_images']} | "
                f"{s['avg_saving_percent']:.2f} | "
                f"{s['avg_psnr']:.2f} | "
                f"{s['avg_ssim']:.4f} | "
                f"{s['avg_metametric']:.6f} |\n"
            )
    print(f"Markdown summary saved to: {md_path}")

    # ── Final console summary ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  FINAL BENCHMARK: screen_adaptor ({lut_mode} mode) on All Datasets")
    print(f"  Metric: screen (OLED) display power saving")
    print(f"{'='*80}")
    header = (
        f"  {'Dataset':<16s} {'Imgs':>4s} {'Saving%':>9s} {'PSNR':>7s} "
        f"{'SSIM':>7s} {'MetaM':>11s}"
    )
    print(header)
    print(f"  {'-'*16} {'-'*4} {'-'*9} {'-'*7} {'-'*7} {'-'*11}")
    for s in all_summaries:
        print(
            f"  {s['dataset']:<16s} {s['total_images']:4d} "
            f"{s['avg_saving_percent']:8.2f}% "
            f"{s['avg_psnr']:6.2f} "
            f"{s['avg_ssim']:7.4f} "
            f"{s['avg_metametric']:11.6f}"
        )
    print(f"{'='*80}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark screen_adaptor LUT color transform on "
                    "screen_adaptor datasets, measuring screen display power"
    )
    parser.add_argument("--max-images", type=int, default=10,
                        help="Max images per dataset (default: 10)")
    parser.add_argument("--output-dir", type=str,
                        default="screen_adaptor_benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--max-metam-side", type=int, default=768,
                        help="Max image side (px) used for MetaM computation "
                             "(default: 768)")
    parser.add_argument("--lut-mode", type=str, default="cluster",
                        choices=["single", "cluster"],
                        help="single=base LUT; cluster=scene-matched per-cluster LUTs")
    parser.add_argument("--foveated", type=float, default=1.0,
                        help="Foveated modulation strength [0, 1] (default: 1.0)")
    parser.add_argument("--temporal", type=float, default=0.0,
                        help="Temporal smoothing strength [0, 1] (default: 0.0)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_benchmark(
        max_images=args.max_images,
        output_dir=output_dir,
        max_metam_side=args.max_metam_side,
        lut_mode=args.lut_mode,
        foveated=args.foveated,
        temporal=args.temporal,
    )


if __name__ == "__main__":
    main()