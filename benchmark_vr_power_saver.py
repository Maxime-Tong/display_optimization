#!/usr/bin/env python
"""
Benchmark: Apply vr-power-saver's color-perception-guided power reduction method
to screen_adaptor's datasets, using screen_adaptor's evaluation metrics.

This script:
  1. Loads the vr-power-saver BaseColorModel (RBF network trained on HVS data)
  2. For each dataset, applies the vr-power-saver pipeline to every image
  3. Computes: Saving%, PSNR, SSIM, MetaM (metameric loss)
  4. Outputs per-dataset summary CSV + JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ── Add vr-power-saver to path ───────────────────────────────────────────
_VR_ROOT = Path(__file__).resolve().parent / "vr-power-saver"
sys.path.insert(0, str(_VR_ROOT))

from color_model.base_color_model import BaseColorModel
from util.vr_tools import build_ecc_map, build_transition_mask

# ── Add screen_adaptor for odak (MetamericLoss) ──────────────────────────
_SA_ROOT = Path(__file__).resolve().parent / "screen_adaptor"
sys.path.insert(0, str(_SA_ROOT))

from odak.learn.perception import MetamericLoss

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

# VR pipeline defaults (from power_saver_demo.py)
FOV = 60                      # degrees
TRANSITION_WIDTH = 3          # degrees
MODEL_PATH = _VR_ROOT / "io" / "color_model" / "model.pth"

# Power weights used in screen_adaptor eval.ps1: 0.229 0.243 0.526 (R, G, B)
# vr-power-saver uses 4 weights (R, G, B, W). We match the RGB triplet.
# The power_vec for apply_filter = -[w_r, w_g, w_b] (negative gradient direction)
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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def collect_images(directory: Path) -> List[Path]:
    paths = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths


# ── VR Power Saver pipeline ───────────────────────────────────────────────

def apply_vr_pipeline(
    img_srgb: np.ndarray,
    model: BaseColorModel,
    power_weights: Tuple[float, float, float],
) -> np.ndarray:
    """
    Apply the full VR power-saver pipeline to a single image.

    Args:
        img_srgb: float32 sRGB image in [0, 1], shape (H, W, 3)
        model: loaded BaseColorModel
        power_weights: (R, G, B) OLED power coefficients

    Returns:
        optimized_srgb: float32 sRGB image in [0, 1], shape (H, W, 3)
    """
    # Use the full original image (no square crop — benchmark runs at native size)
    inp = img_srgb

    # Power gradient vector (negative of weights, excluding W channel)
    # vr-power-saver model was trained with 4 weights [231.5, 245.7, 530.8, 977.3]
    # but we use the normalized direction only; apply_filter normalizes internally
    power_vec = -np.array(power_weights, dtype=np.float64)

    # Build eccentricity map (simulates VR headset viewing)
    ecc_map = build_ecc_map(
        FOV, 0.0, 0.0,
        model.opt.max_eccentricity,
        inp.shape[0], inp.shape[1],
    )

    # Apply the color shift filter
    out = model.apply_filter(inp, ecc_map, power_vec)

    # Blend with transition mask (smooth falloff at fovea edge)
    mask = build_transition_mask(
        ecc_map,
        model.opt.min_eccentricity,
        TRANSITION_WIDTH,
    )
    out = inp * (1 - mask) + out * mask

    return out.astype(np.float32)


# ── Metrics (matching screen_adaptor eval.py) ────────────────────────────

def weighted_power_np(rgb: np.ndarray, weights: Tuple[float, float, float]) -> np.ndarray:
    """Compute per-pixel weighted power: sum(rgb * weights)."""
    w = np.array(weights, dtype=np.float64)
    return (rgb * w).sum(axis=-1)


def compute_metametric(
    original: np.ndarray,
    optimized: np.ndarray,
) -> float:
    """
    Compute MetamericLoss using odak (matching screen_adaptor eval.py).

    Args:
        original, optimized: float32 in [0,1], shape (H, W, 3)
    Returns:
        metametric: scalar float
    """
    import torch
    # NCHW layout
    orig_nchw = torch.from_numpy(original).unsqueeze(0).permute(0, 3, 1, 2)
    opt_nchw = torch.from_numpy(optimized).unsqueeze(0).permute(0, 3, 1, 2)
    with torch.no_grad():
        val = float(_fov_hvs_loss(opt_nchw, orig_nchw, gaze=[0.5, 0.5]))
    return val


def evaluate_image(
    original: np.ndarray,
    optimized: np.ndarray,
    power_weights: Tuple[float, float, float],
) -> Dict[str, float]:
    """
    Compute all metrics for a single image pair.

    Returns dict with keys: saving, psnr, ssim, metametric
    """
    # Power saving
    orig_power = weighted_power_np(original, power_weights).sum()
    opt_power = weighted_power_np(optimized, power_weights).sum()
    saving = 0.0 if orig_power <= 0 else float(1.0 - (opt_power / orig_power))

    # PSNR
    psnr = float(peak_signal_noise_ratio(original, optimized, data_range=1.0))

    # SSIM
    ssim = float(structural_similarity(
        original, optimized, channel_axis=2, data_range=1.0,
    ))

    # MetaM
    metametric = compute_metametric(original, optimized)

    return {
        "saving": saving,
        "psnr": psnr,
        "ssim": ssim,
        "metametric": metametric,
    }


# ── Main benchmark loop ───────────────────────────────────────────────────

def run_benchmark(
    max_images: int = 100,
    output_dir: Path = Path("vr_power_saver_benchmark_results"),
) -> None:
    """Run vr-power-saver on all datasets and collect metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load VR model
    print(f"Loading VR model from: {MODEL_PATH}")
    model = BaseColorModel()
    model.load(str(MODEL_PATH))
    print(f"  Model loaded. min_ecc={model.opt.min_eccentricity}, max_ecc={model.opt.max_eccentricity}")

    all_summaries: List[Dict[str, Any]] = []
    per_dataset_results: Dict[str, List[Dict[str, Any]]] = {}

    for dataset_name, dataset_dir in DATASETS:
        if not dataset_dir.exists():
            print(f"\n[SKIP] Dataset not found: {dataset_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Path: {dataset_dir}")
        print(f"{'='*60}")

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
                # Load image
                img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0

                # Apply VR pipeline (uses the full native-resolution image)
                t0 = time.perf_counter()
                optimized = apply_vr_pipeline(img, model, POWER_WEIGHTS_RGB)
                elapsed = time.perf_counter() - t0

                # Metrics (compare full-size versions)
                metrics = evaluate_image(img, optimized, POWER_WEIGHTS_RGB)

                results.append({
                    "filename": img_path.name,
                    **metrics,
                    "time_sec": round(elapsed, 3),
                })
                savings.append(metrics["saving"])
                psnrs.append(metrics["psnr"])
                ssims.append(metrics["ssim"])
                metametrics.append(metrics["metametric"])

                print(
                    f"  [{idx+1:4d}/{len(selected)}] {img_path.name}: "
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
        print(f"  Images:         {len(results)}")
        print(f"  Avg Saving:     {avg_saving*100:.2f}%")
        print(f"  Avg PSNR:       {avg_psnr:.2f} dB")
        print(f"  Avg SSIM:       {avg_ssim:.4f}")
        print(f"  Avg MetaM:      {avg_metametric:.6f}")

    # ── Save Results ──────────────────────────────────────────────────────

    # JSON (full details)
    json_path = output_dir / "vr_power_saver_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": str(MODEL_PATH),
                "fov": FOV,
                "transition_width": TRANSITION_WIDTH,
                "power_weights_rgb": list(POWER_WEIGHTS_RGB),
                "max_images_per_dataset": max_images,
            },
            "per_dataset": per_dataset_results,
            "summary_table": all_summaries,
        }, f, indent=2)
    print(f"\nFull results saved to: {json_path}")

    # CSV summary
    csv_path = output_dir / "vr_power_saver_benchmark_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "total_images", "avg_saving_percent",
            "avg_psnr", "avg_ssim", "avg_metametric",
        ])
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"CSV summary saved to: {csv_path}")

    # Markdown summary
    md_path = output_dir / "vr_power_saver_benchmark_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# VR Power Saver Benchmark Results\n\n")
        f.write(f"- Model: `{MODEL_PATH}`\n")
        f.write(f"- FOV: {FOV}°, Transition Width: {TRANSITION_WIDTH}°\n")
        f.write(f"- Power Weights (R,G,B): {POWER_WEIGHTS_RGB}\n")
        f.write(f"- Images per dataset: up to {max_images}\n\n")
        f.write("| Dataset | Images | Avg Saving (%) | Avg PSNR (dB) | Avg SSIM | Avg MetaM |\n")
        f.write("|---------|--------|:-------------:|:-----------:|:------:|:----------:|\n")
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
    print(f"\n{'='*70}")
    print(f"  FINAL BENCHMARK: VR Power Saver on All Datasets")
    print(f"{'='*70}")
    print(f"  {'Dataset':<20s} {'Imgs':>5s} {'Saving%':>10s} {'PSNR':>8s} {'SSIM':>8s} {'MetaM':>12s}")
    print(f"  {'-'*20} {'-'*5} {'-'*10} {'-'*8} {'-'*8} {'-'*12}")
    for s in all_summaries:
        print(
            f"  {s['dataset']:<20s} {s['total_images']:5d} "
            f"{s['avg_saving_percent']:9.2f}% "
            f"{s['avg_psnr']:7.2f} "
            f"{s['avg_ssim']:8.4f} "
            f"{s['avg_metametric']:12.6f}"
        )
    print(f"{'='*70}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vr-power-saver on screen_adaptor datasets"
    )
    parser.add_argument("--max-images", type=int, default=10,
                        help="Max images per dataset (default: 10)")
    parser.add_argument("--output-dir", type=str, default="vr_power_saver_benchmark_results",
                        help="Output directory for results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_benchmark(max_images=args.max_images, output_dir=output_dir)


if __name__ == "__main__":
    main()