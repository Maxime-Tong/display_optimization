#!/usr/bin/env python
"""
Benchmark: Gradual Chromatic Adaptation (GCA) + GUD power-saving method applied
to screen_adaptor's datasets, measured with screen display power + quality metrics.

This script:
  1. Uses the gradual_adaptation package (implementation of gradual.md),
     with BOTH improvements for higher power savings:
       - adaptation-bounded trajectory  (perceived cast |A-a| <= 5 JND while
         the illuminant itself drifts ~2.5x further than DELTA_T)
       - combined GUD (gradual uniform dimming)
  2. For each dataset, simulates the GRADUAL ramp by FRAME SKIPPING:
     - Dataset image k of N is rendered at t_k = (k+0.5)/N * t_max
       (a different wall-clock time across the 2-minute ramp)
     - Per-image saving starts near 0% and RAMPS UP to its maximum across
       the N images, directly validating the progressive power decline
     - The dataset mean estimates the TIME-AVERAGED ramp saving
  3. Computes Saving% = 1 - power(opt) / power(orig) plus PSNR, SSIM and
     MetaM (odak MetamericLoss) for EVERY image on ITS OWN time-stamp state,
     matching the baseline benchmark methodology per-image.
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
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ── Add gradual_adaptation to path ─────────────────────────────────────
_GA_ROOT = Path(__file__).resolve().parent / "gradual_adaptation"
sys.path.insert(0, str(_GA_ROOT.parent))

from gradual_adaptation import (  # noqa: E402
    GradualAdaptationImageOptimizer,
)
from gradual_adaptation.constants import (  # noqa: E402
    DEFAULT_TRAJECTORY,
    DEFAULT_VELOCITY,
    GUD_ENABLED,
    GUD_TARGET,
    POWER_WEIGHTS_RGB,
    T_MAX,
)

# ── Add screen_adaptor for odak (MetamericLoss) ──────────────────────────
_SA_ROOT = Path(__file__).resolve().parent / "screen_adaptor"
sys.path.insert(0, str(_SA_ROOT))

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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def collect_images(directory: Path) -> List[Path]:
    paths = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths


def square_crop(img: np.ndarray) -> np.ndarray:
    """Center square crop. Mimics power_saver_demo.py / benchmark_vr_power_saver.py."""
    h, w = img.shape[0], img.shape[1]
    if h == w:
        return img.copy()
    if h > w:
        vpad = (h - w) // 2
        return img[vpad:vpad + w, ...].copy()
    else:
        hpad = (w - h) // 2
        return img[:, hpad:hpad + h, ...].copy()


# ── Optimizer construction (GCA + optional GUD) ──────────────────────────

def make_optimizer(
    trajectory: str,
    velocity: float,
    gud_enabled: bool = GUD_ENABLED,
    gud_target: float = GUD_TARGET,
) -> GradualAdaptationImageOptimizer:
    """Build a per-dataset GradualAdaptationImageOptimizer (GCA + GUD)."""
    return GradualAdaptationImageOptimizer(
        trajectory=trajectory,
        velocity=velocity,
        t_max=T_MAX,
        gud_target=gud_target,
        gud_enabled=gud_enabled,
    )


# ── Metrics (matching screen_adaptor eval.py) ────────────────────────────

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
    import torch

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
    (MetaM computed per image, matching baseline methodology.)
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
    output_dir: Path = Path("gradual_adaptation_benchmark_results"),
    max_metam_side: int = 768,
    time_s: float = T_MAX,
    trajectory: str = DEFAULT_TRAJECTORY,
    velocity: float = DEFAULT_VELOCITY,
    gud_enabled: bool = GUD_ENABLED,
    gud_target: float = GUD_TARGET,
) -> None:
    """
    Run GCA (+ GUD) on all datasets; measure screen power + quality metrics.

    Frame skipping: each of the N dataset images is rendered at a distinct
    time t_k = (k+0.5)/N * time_s across the gradual ramp, so the per-image
    power saving shows the progressive decline and the dataset mean estimates
    the time-averaged ramp saving.

    Saving%, PSNR, SSIM and MetaM are computed PER IMAGE on its own
    time-stamp state (t_img); no shared worst-case state is used, so the
    comparison with the other baselines is fair.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict[str, Any]] = []
    per_dataset_results: Dict[str, Dict[str, Any]] = {}

    print(f"Gradual Chromatic Adaptation configuration:")
    print(f"  trajectory   : {trajectory}")
    print(f"  velocity     : {velocity} u'v'/s (adaptation-bounded)")
    print(f"  time (t)     : {time_s} s   (t_max = {T_MAX} s)")
    print(f"  GUD          : enabled={gud_enabled}, target={gud_target}")
    print(f"  weights RGB  : {POWER_WEIGHTS_RGB}")
    print(f"  frame skip   : image k rendered at t=(k+0.5)/N*{time_s}s; "
          f"per-image saving ramps 0 -> max")
    print()

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
        n_imgs = len(selected)
        print(f"  Found {len(images)} images, processing {len(selected)}...")

        # One optimizer per dataset; its GCA clock stays monotonic.
        optimizer = make_optimizer(
            trajectory=trajectory,
            velocity=velocity,
            gud_enabled=gud_enabled,
            gud_target=gud_target,
        )

        results: List[Dict[str, Any]] = []
        savings: List[float] = []
        psnrs: List[float] = []
        ssims: List[float] = []
        metametrics: List[float] = []

        for idx, img_path in enumerate(selected):
            try:
                # Load image (float sRGB in [0, 1])
                img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0

                # Apply same square crop to original for fair comparison
                original_cropped = square_crop(img)

                # ── Temporal FRAME-SKIPPED power evaluation ─────────────
                # GCA+GUD is a GRADUAL technique: the power saving starts at
                # ~0% at t=0 and grows toward its maximum at t=time_s.
                # Rendering every real frame (e.g. 90 fps * 120 s) is
                # prohibitive, so we FRAME-SKIP: dataset image k (of N) is
                # rendered at the midpoint of its time window, t_k =
                # (k + 0.5)/N * time_s. This mimics a video whose sampled
                # frames are shown at progressively later times: per-image
                # saving ramps up across the N images, and the dataset mean
                # estimates the time-averaged ramp saving.
                t0 = time.perf_counter()

                t_img = float(((idx + 0.5) / n_imgs) * time_s)
                adapted_t = optimizer.process_frame(original_cropped, t=t_img)

                # All metrics (Saving%, PSNR, SSIM, MetaM) are computed on THIS
                # FRAME'S OWN TIME state (t_img), matching the baseline
                # methodology where every image is measured independently.
                # No shared worst-case final state is used, so the comparison
                # with the other baselines is fair.
                metrics = evaluate_image(
                    original_cropped, adapted_t, POWER_WEIGHTS_RGB,
                    max_metam_side=max_metam_side,
                )
                elapsed = time.perf_counter() - t0

                results.append({
                    "filename": img_path.name,
                    **metrics,
                    "saving_at_time": metrics["saving"],
                    "time_stamp_sec": round(t_img, 3),
                    "time_sec": round(elapsed, 3),
                })
                savings.append(metrics["saving"])
                psnrs.append(metrics["psnr"])
                ssims.append(metrics["ssim"])
                metametrics.append(metrics["metametric"])

                print(
                    f"  [{idx+1:4d}/{n_imgs}] {img_path.name} "
                    f"(t={t_img:.1f}s): "
                    f"power_saving={metrics['saving']*100:.2f}%, "
                    f"PSNR={metrics['psnr']:.2f}dB, "
                    f"SSIM={metrics['ssim']:.4f}, "
                    f"MetaM={metrics['metametric']:.6f} "
                    f"({elapsed:.2f}s)"
                )

            except Exception as e:
                print(f"  [{idx+1:4d}/{n_imgs}] {img_path.name}: ERROR - {e}")

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
    json_path = output_dir / "gradual_adaptation_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "method": ("Gradual Chromatic Adaptation (GCA): slow white-point "
                           "shift along a yellow-green u'v' trajectory from D65 "
                           "(Bradford CAT), exploiting slow human chromatic "
                           "adaptation to be perceptually invisible. Plus "
                           "GUD (gradual uniform dimming). Velocity is "
                           "adaptation-bounded: the perceived cast |A-a| stays "
                           "within 5 JND while the absolute illuminant drift "
                           "exceeds DELTA_T."),
                "benchmark_metric": "screen (OLED) display power: "
                                    "saving = 1 - power(opt)/power(orig), "
                                    "power = sum(R*0.229 + G*0.243 + B*0.526)",
                "trajectory": trajectory,
                "velocity_upvp_per_sec": velocity,
                "time_sec": time_s,
                "t_max_sec": T_MAX,
                "gud_enabled": bool(gud_enabled),
                "gud_target": float(gud_target),
                "power_weights_rgb": list(POWER_WEIGHTS_RGB),
                "max_metam_side": max_metam_side,
                "max_images_per_dataset": max_images,
                "temporal_averaging": {
                    "method": ("frame skipping: dataset image k of N rendered "
                               "at t_k = (k+0.5)/N * time_s across the gradual "
                               "ramp; per-image saving ramps 0 -> max, dataset "
                               "mean estimates the time-averaged ramp saving"),
                    "n_time_samples": int(max_images),
                    "note": ("All metrics (Saving%, PSNR, SSIM, MetaM) computed "
                             "per image on its own time-stamp state (no shared "
                             "worst-case state); dataset avg = ramp average"),
                },
            },
            "per_dataset": per_dataset_results,
            "summary_table": all_summaries,
        }, f, indent=2)
    print(f"\nFull results saved to: {json_path}")

    # CSV summary
    csv_path = output_dir / "gradual_adaptation_benchmark_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "total_images", "avg_saving_percent",
            "avg_psnr", "avg_ssim", "avg_metametric",
        ])
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"CSV summary saved to: {csv_path}")

    # Markdown summary
    md_path = output_dir / "gradual_adaptation_benchmark_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Gradual Chromatic Adaptation (GCA) + GUD Benchmark Results\n\n")
        f.write("- Method: slow white-point shift along a yellow-green u'v' "
                "trajectory from D65 (Bradford CAT), perceptually invisible via "
                "human chromatic adaptation, + GUD uniform dimming\n")
        f.write("- Trajectory: **%s**, velocity: **%s u'v'/s** (adaptation-bounded, "
                "perceived cast <= 5 JND), time: **%s s** (t_max = %s s)\n"
                % (trajectory, velocity, time_s, T_MAX))
        f.write("- GUD: enabled=%s, target=%.2f\n" % (gud_enabled, gud_target))
        f.write("- Benchmark metric: **screen (OLED) display power** saving with "
                "weights (R,G,B) = %s\n" % (POWER_WEIGHTS_RGB,))
        f.write("- MetaM computed with max side %d px (odak MetamericLoss), "
                "per image on its own time-stamp state\n" % max_metam_side)
        f.write("- Images per dataset: up to %d\n" % max_images)
        f.write("- Temporal frame-skipped power: %d images sampled across "
                "[0, %s s]; image k rendered at t=(k+0.5)/N*%s s, per-image "
                "saving ramps 0 -> max, dataset mean = ramp average\n\n"
                % (max_images, time_s, time_s))
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
    print(f"  FINAL BENCHMARK: Gradual Chromatic Adaptation (+GUD) on All Datasets")
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
        description="Benchmark Gradual Chromatic Adaptation (GCA) + GUD on "
                    "screen_adaptor datasets, measuring screen display power"
    )
    parser.add_argument("--max-images", type=int, default=10,
                        help="Max images per dataset (default: 10)")
    parser.add_argument("--output-dir", type=str,
                        default="gradual_adaptation_benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--max-metam-side", type=int, default=768,
                        help="Max image side (px) used for MetaM computation "
                             "(default: 768)")
    parser.add_argument("--time", type=float, default=T_MAX,
                        help=f"Elapsed time (s) of the gradual shift "
                             f"(default: {T_MAX})")
    parser.add_argument("--trajectory", type=str, default=DEFAULT_TRAJECTORY,
                        choices=["daylight", "1.47", "1.863", "2.256"],
                        help=f"Illuminant trajectory (default: {DEFAULT_TRAJECTORY})")
    parser.add_argument("--velocity", type=float, default=DEFAULT_VELOCITY,
                        help=f"u'v' per second advance speed "
                             f"(default: {DEFAULT_VELOCITY}, adaptation-bounded)")
    parser.add_argument("--no-gud", action="store_true",
                        help="Disable gradual uniform dimming")
    parser.add_argument("--gud-target", type=float, default=GUD_TARGET,
                        help=f"Final uniform dim factor (default: {GUD_TARGET})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_benchmark(
        max_images=args.max_images,
        output_dir=output_dir,
        max_metam_side=args.max_metam_side,
        time_s=args.time,
        trajectory=args.trajectory,
        velocity=args.velocity,
        gud_enabled=not args.no_gud,
        gud_target=args.gud_target,
    )


if __name__ == "__main__":
    main()