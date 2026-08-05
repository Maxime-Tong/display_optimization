#!/usr/bin/env python
"""
Benchmark: hvs_vr_encoding with a MODIFIED optimization objective, applied to
screen_adaptor's datasets and measured with screen display power metrics.

--------------------------------------------------------------------------------
Modification to the compression algorithm
--------------------------------------------------------------------------------
Original (hvs_vr_encoding, ASPLOS 2024): for each 4x4 tile, every pixel is
converged toward a SINGLE consistent color plane (`col_plane` — the midpoint of
the tile's just-noticeable-difference intersection range). All deltas collapse
toward that shared target, making the Base-Delta encoder highly effective, but
the target color is forced to be consistent within the tile.

Modified (this benchmark): the optimization objective is changed from "keep the
target consistent" to "maximally reduce each pixel's changeable delta". Each
pixel's updated color is its OWN maximum-reduction point — the boundary of its
JND ellipsoid along the decreasing direction (`min_p`). Every changeable delta
is reduced by the largest amount the HVS tolerates, and the image is darkened
as far as perception allows. This directly minimizes the screen (OLED) display
power while remaining numerically lossy but perceptually lossless.

--------------------------------------------------------------------------------
Benchmark metrics (screen display power)
--------------------------------------------------------------------------------
The benchmark function measures the SCREEN DISPLAY POWER, matching
screen_adaptor eval.ps1 / benchmark_vr_power_saver.py:

  - Saving% = 1 - power(optimized) / power(original)
      power(img) = sum over pixels of (R*0.229 + G*0.243 + B*0.526)
  - PSNR, SSIM            (screen_adaptor quality metrics)
  - MetaM                 (odak metameric loss; must stay ≈ 0 to be
                           perceptually lossless)
  - BD compression rates  (context: the method is also a framebuffer
                           compressor, so its BD rate on original and
                           optimized images is reported as secondary info)
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
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ── Add hvs_vr_encoding color_optimizer to path ───────────────────────────
_HVS_ROOT = Path(__file__).resolve().parent / "hvs_vr_encoding"
_HVS_COLOR_OPT = _HVS_ROOT / "host" / "color_optimizer"
sys.path.insert(0, str(_HVS_COLOR_OPT))

from red_blue_optimization_cpu import (  # noqa: E402
    Image_color_optimizer as _OrigImageColorOptimizer,
)
# Subclass hooks we need to modify the tile-level objective
from red_blue_optimization_cpu import Tile_color_optimizer_hw_part  # noqa: E402
from util.opt_BD_enc import bd_compress_rate  # noqa: E402

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

# hvs_vr_encoding pipeline defaults (from red_blue_optimization_cpu.py)
TILE_SIZE = 4
H_FOV = 110                     # degrees
MAX_ECC = 35                    # degrees (max eccentricity)
ECC_NO_COMPRESS = 10            # below this eccentricity, no compression
ONLY_BLUE = False               # optimize both R and B channels

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


def crop_to_tile_multiple(img: np.ndarray, tile_size: int = TILE_SIZE) -> np.ndarray:
    """
    Center-crop image (MINIMAL, no square crop) so the color optimizer 4x4
    tiling works.  This matches generate_video.py's crop behavior so the
    benchmark and the video generation use the exact same resolution.

    The hvs_vr_encoding method requires both dimensions divisible by 4 AND
    H*W divisible by 48 (its internal ecc-map reshape is (-1, 4, 4, 3)).
    """
    img = np.asarray(img)
    h, w = img.shape[0], img.shape[1]

    h = (h // tile_size) * tile_size
    w = (w // tile_size) * tile_size

    # Ensure H*W divisible by 48 (needed by internal (-1, 4, 4, 3) reshape):
    # make H a multiple of 12 (= 4 * 3) so H*W % 48 == 0
    if (h * w) % 48 != 0:
        h = (h // 12) * 12

    vpad = (img.shape[0] - h) // 2
    hpad = (img.shape[1] - w) // 2
    return img[vpad:vpad + h, hpad:hpad + w, ...].copy()


# ── Modified compression algorithm ────────────────────────────────────────
#
# The ONLY change to the algorithm is in Tile_color_optimizer_hw_part.adjust_tile:
#
#   Original : all pixels of a tile converge to a single consistent col_plane
#              (target kept consistent within the tile)
#   Modified : each pixel's updated color = its own maximum-reduction point,
#              i.e. the boundary of its just-noticeable-difference (JND)
#              ellipsoid along the decreasing direction (min_p). Every
#              changeable delta is reduced by the maximum amount perception
#              allows, and the image is darkened as far as the HVS allows ->
#              directly minimizes screen (OLED) display power.


class MaxReduceTileHW(Tile_color_optimizer_hw_part):
    """
    Tile color optimizer with the modified objective:

    Instead of converging every pixel onto the shared `col_plane`, each pixel is
    updated to the maximum-reduction point of its own delta — the boundary of
    its JND ellipsoid in the decreasing direction (`min_p`). The updated color
    target is per-pixel (not consistent) and maximally reduces the changeable
    delta for that pixel, which darkens the image as much as human color
    discrimination tolerates.
    """

    def adjust_tile(self, dkl_centers):
        # Point where each pixel's color is reduced MOST: the boundary of its
        # JND ellipsoid along the decreasing direction (min_vec).
        min_p = self.line_ell_inter(dkl_centers, self.min_vec_dkl)
        self.fix_bounds(min_p)

        # Clamp to valid RGB [0, 1] while preserving maximum reduction
        np.clip(min_p, 0.0, 1.0, out=min_p)
        return min_p


# ── hvs_vr_encoding pipeline (with modified objective) ───────────────────

_optimizer_cache: Dict[Tuple[int, int], _OrigImageColorOptimizer] = {}


def get_optimizer(h: int, w: int) -> _OrigImageColorOptimizer:
    key = (h, w)
    if key not in _optimizer_cache:
        opt = _OrigImageColorOptimizer(
            img_height=h,
            img_width=w,
            tile_size=TILE_SIZE,
            foveated=True,
            max_ecc=MAX_ECC,
            h_fov=H_FOV,
            ecc_no_compress=ECC_NO_COMPRESS,
            only_blue=ONLY_BLUE,
        )
        # Swap in the MAX-REDUCTION tile optimizer (the modified objective)
        tc = opt.Tile_color_optimizer
        tc.hw_tile_optimizer = MaxReduceTileHW(
            tc.color_channel, opt.r_max_vec, opt.b_max_vec,
        )
        _optimizer_cache[key] = opt
    return _optimizer_cache[key]


def apply_modified_hvs_encoding(img_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply hvs_vr_encoding with the modified (max-delta-reduction) objective.

    Args:
        img_uint8: uint8 RGB image in [0, 255], shape (H, W, 3)

    Returns:
        (original_cropped, optimized): uint8 RGB images in [0, 255]
    """
    inp = crop_to_tile_multiple(img_uint8, TILE_SIZE)
    h, w = inp.shape[0], inp.shape[1]

    if h < 12 or w < 4:
        raise ValueError(f"image too small after crop: {inp.shape}")

    optimizer = get_optimizer(h, w)

    # color_conversion expects float32 in [0, 255]; returns uint8 RGB in [0, 255]
    optimized = optimizer.color_conversion(inp.astype(np.float32))

    return inp, optimized


# ── Metrics ────────────────────────────────────────────────────────────────

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


def evaluate_bd(
    original_uint8: np.ndarray,
    optimized_uint8: np.ndarray,
) -> Dict[str, float]:
    """
    Compute both images' BD compression rates (secondary/context metric: the
    method is a framebuffer compressor).

    BD compression rate = 1 - compressed_size / uncompressed_size.
    """
    orig_rate = float(bd_compress_rate(original_uint8.astype(np.int16)))
    opt_rate = float(bd_compress_rate(optimized_uint8.astype(np.int16)))
    return {
        "bd_compression_orig": orig_rate,
        "bd_compression_opt": opt_rate,
        "bd_improvement": opt_rate - orig_rate,
    }


# ── Main benchmark loop ───────────────────────────────────────────────────

def run_benchmark(
    max_images: int = 10,
    output_dir: Path = Path("hvs_vr_encoding_benchmark_results"),
    max_metam_side: int = 768,
) -> None:
    """Run the modified hvs-vr-encoding on all datasets; measure screen power."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict[str, Any]] = []
    per_dataset_results: Dict[str, List[Dict[str, Any]]] = {}

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
        bd_origs: List[float] = []
        bd_opts: List[float] = []

        for idx, img_path in enumerate(selected):
            try:
                # Load image (uint8 RGB in [0, 255])
                img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)

                # Apply modified hvs_vr_encoding pipeline (internally center-crops)
                t0 = time.perf_counter()
                original_cropped, optimized = apply_modified_hvs_encoding(img)
                elapsed = time.perf_counter() - t0

                # Float [0,1] versions for the power/quality metrics
                original_f = original_cropped.astype(np.float32) / 255.0
                optimized_f = optimized.astype(np.float32) / 255.0

                # SCREEN DISPLAY POWER + quality
                metrics = evaluate_image(
                    original_f, optimized_f, POWER_WEIGHTS_RGB,
                    max_metam_side=max_metam_side,
                )

                # BD compression (context)
                bd_metrics = evaluate_bd(original_cropped, optimized)

                results.append({
                    "filename": img_path.name,
                    **metrics,
                    **bd_metrics,
                    "time_sec": round(elapsed, 3),
                })
                savings.append(metrics["saving"])
                psnrs.append(metrics["psnr"])
                ssims.append(metrics["ssim"])
                metametrics.append(metrics["metametric"])
                bd_origs.append(bd_metrics["bd_compression_orig"])
                bd_opts.append(bd_metrics["bd_compression_opt"])

                print(
                    f"  [{idx+1:4d}/{len(selected)}] {img_path.name}: "
                    f"power_saving={metrics['saving']*100:.2f}%, "
                    f"PSNR={metrics['psnr']:.2f}dB, "
                    f"SSIM={metrics['ssim']:.4f}, "
                    f"MetaM={metrics['metametric']:.6f}, "
                    f"BD={bd_metrics['bd_compression_orig']*100:.2f}%->"
                    f"{bd_metrics['bd_compression_opt']*100:.2f}% "
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
            avg_bd_orig = float(np.mean(bd_origs))
            avg_bd_opt = float(np.mean(bd_opts))
        else:
            avg_saving = avg_psnr = avg_ssim = avg_metametric = 0.0
            avg_bd_orig = avg_bd_opt = 0.0

        summary = {
            "dataset": dataset_name,
            "total_images": len(results),
            "avg_saving_percent": avg_saving * 100,
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "avg_metametric": avg_metametric,
            "avg_bd_orig_percent": avg_bd_orig * 100,
            "avg_bd_opt_percent": avg_bd_opt * 100,
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
        print(f"  Avg BD rate:     {avg_bd_orig*100:.2f}% -> {avg_bd_opt*100:.2f}%")

    # ── Save Results ──────────────────────────────────────────────────────

    # JSON (full details)
    json_path = output_dir / "hvs_vr_encoding_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "method": ("hvs_vr_encoding (ASPLOS 2024) with MODIFIED objective: "
                           "per-pixel max delta reduction (each pixel -> boundary "
                           "of its JND ellipsoid along the decreasing direction "
                           "min_p) instead of tile-consistent col_plane"),
                "benchmark_metric": "screen (OLED) display power: "
                                    "saving = 1 - power(opt)/power(orig), "
                                    "power = sum(R*0.229 + G*0.243 + B*0.526)",
                "tile_size": TILE_SIZE,
                "h_fov": H_FOV,
                "max_ecc": MAX_ECC,
                "ecc_no_compress": ECC_NO_COMPRESS,
                "only_blue": ONLY_BLUE,
                "power_weights_rgb": list(POWER_WEIGHTS_RGB),
                "max_metam_side": max_metam_side,
                "max_images_per_dataset": max_images,
            },
            "per_dataset": per_dataset_results,
            "summary_table": all_summaries,
        }, f, indent=2)
    print(f"\nFull results saved to: {json_path}")

    # CSV summary
    csv_path = output_dir / "hvs_vr_encoding_benchmark_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "total_images", "avg_saving_percent",
            "avg_psnr", "avg_ssim", "avg_metametric",
            "avg_bd_orig_percent", "avg_bd_opt_percent",
        ])
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"CSV summary saved to: {csv_path}")

    # Markdown summary
    md_path = output_dir / "hvs_vr_encoding_benchmark_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# hvs_vr_encoding (Modified Objective) Benchmark Results\n\n")
        f.write("- Method: hvs_vr_encoding color-discrimination-guided "
                "framebuffer compression with MODIFIED optimization objective "
                "(per-pixel max delta reduction)\n")
        f.write("- H-FOV: %d°, Max Ecc: %d°, Ecc No-Compress: %d°, "
                "Tile Size: %d\n" % (H_FOV, MAX_ECC, ECC_NO_COMPRESS, TILE_SIZE))
        f.write("- Benchmark metric: **screen (OLED) display power** saving with "
                "weights (R,G,B) = %s\n" % (POWER_WEIGHTS_RGB,))
        f.write("- MetaM computed with max side %d px (odak MetamericLoss)\n"
                % max_metam_side)
        f.write("- Images per dataset: up to %d\n\n" % max_images)
        f.write("| Dataset | Images | Power Saving (%) | PSNR (dB) | SSIM "
                "| MetaM | BD orig (%) | BD opt (%) |\n")
        f.write("|---------|--------|:----------------:|:---------:|:-----:"
                "|:------:|:-----------:|:----------:|\n")
        for s in all_summaries:
            f.write(
                f"| {s['dataset']} | {s['total_images']} | "
                f"{s['avg_saving_percent']:.2f} | "
                f"{s['avg_psnr']:.2f} | "
                f"{s['avg_ssim']:.4f} | "
                f"{s['avg_metametric']:.6f} | "
                f"{s['avg_bd_orig_percent']:.2f} | "
                f"{s['avg_bd_opt_percent']:.2f} |\n"
            )
    print(f"Markdown summary saved to: {md_path}")

    # ── Final console summary ─────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  FINAL BENCHMARK: hvs_vr_encoding (modified objective) "
          f"on All Datasets")
    print(f"  Metric: screen (OLED) display power saving")
    print(f"{'='*90}")
    header = (
        f"  {'Dataset':<16s} {'Imgs':>4s} {'Saving%':>9s} {'PSNR':>7s} "
        f"{'SSIM':>7s} {'MetaM':>11s} {'BD_orig%':>9s} {'BD_opt%':>8s}"
    )
    print(header)
    print(f"  {'-'*16} {'-'*4} {'-'*9} {'-'*7} {'-'*7} {'-'*11} "
          f"{'-'*9} {'-'*8}")
    for s in all_summaries:
        print(
            f"  {s['dataset']:<16s} {s['total_images']:4d} "
            f"{s['avg_saving_percent']:8.2f}% "
            f"{s['avg_psnr']:6.2f} "
            f"{s['avg_ssim']:7.4f} "
            f"{s['avg_metametric']:11.6f} "
            f"{s['avg_bd_orig_percent']:8.2f}% "
            f"{s['avg_bd_opt_percent']:7.2f}%"
        )
    print(f"{'='*90}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark hvs_vr_encoding (modified max-delta-reduction "
                    "objective) on screen_adaptor datasets, measuring screen "
                    "display power"
    )
    parser.add_argument("--max-images", type=int, default=10,
                        help="Max images per dataset (default: 10)")
    parser.add_argument("--output-dir", type=str,
                        default="hvs_vr_encoding_benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--max-metam-side", type=int, default=768,
                        help="Max image side (px) used for MetaM computation "
                             "(default: 768)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_benchmark(
        max_images=args.max_images,
        output_dir=output_dir,
        max_metam_side=args.max_metam_side,
    )


if __name__ == "__main__":
    main()