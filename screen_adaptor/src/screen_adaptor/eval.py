"""Inference / evaluation: apply LUT(s) to images and report power-saving + quality metrics.

Supports single‑LUT mode and scene‑manifest mode with dynamic LUT switching.
Also supports foveated rendering and temporal smoothing (from the former inference_v2 module).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .color_ops import weighted_power
from .model import LUTColorTransformer, load_lut_transformer
from .scene_matcher import IMAGE_EXTENSIONS, SceneMatcher, load_scene_manifest
from .utils import collect_image_paths, generate_phi_map

from odak.learn.perception import MetamericLoss

fov_hvs_loss = MetamericLoss(real_image_width=1.4, real_viewing_distance=0.7, equi=False, alpha=5.0, mode="quadratic", loss_type="L1", use_l2_foveal_loss=False, n_pyramid_levels=5, n_orientations=4, use_radial_weight=True)

# ── core LUT application ───────────────────────────────────────────────────

def _apply_lut_core(
    rgb: torch.Tensor,
    transformer: LUTColorTransformer,
    power_weights: Tuple[float, float, float],
    foveated: float = 0,
    temporal: float = 0,
) -> Tuple[torch.Tensor, float, float, float, float]:
    """Apply LUT transform, compute saving. Returns (optimized, saving, psnr, ssim, metametric)."""
    device = rgb.device
    h, w = rgb.shape[0], rgb.shape[1]
    arr = np.asarray(rgb.cpu().numpy(), dtype=np.float32)

    with torch.no_grad():
        optimized = transformer.transform(rgb)
        if optimized.shape != rgb.shape:
            optimized = optimized.reshape(rgb.shape)

    weights = torch.tensor(power_weights, dtype=torch.float32)
    orig_power = weighted_power(rgb.reshape(-1, 3), weights).sum().item()
    opt_power = weighted_power(optimized.reshape(-1, 3), weights).sum().item()
    saving = 0.0 if orig_power <= 0 else 1.0 - (opt_power / orig_power)

    optimized_np = optimized.cpu().numpy()
    psnr = float(peak_signal_noise_ratio(arr, optimized_np, data_range=1.0))
    ssim = float(structural_similarity(
        arr, optimized_np, channel_axis=2, data_range=1.0,
    ))

    # Metameric metric: NCHW layout on original device
    orig_nchw = rgb.unsqueeze(0).permute(0, 3, 1, 2)
    opt_nchw = optimized.unsqueeze(0).permute(0, 3, 1, 2)
    print(f"Computing MetaM: orig_nchw={orig_nchw.shape}, opt_nchw={opt_nchw.shape}, device={device}")
    metametric = float(fov_hvs_loss(opt_nchw, orig_nchw, gaze=[0.5, 0.5]))

    return optimized, saving, psnr, ssim, metametric


def apply_lut_single(
    image_path: Path,
    transformer: LUTColorTransformer,
    power_weights: Tuple[float, float, float],
    out_path: Optional[Path] = None,
    eval_mode: bool = False,
    foveated: float = 0,
    temporal: float = 0,
) -> Dict[str, Any]:
    """Apply a single LUT to one image."""
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    rgb = torch.from_numpy(arr)

    optimized, saving, psnr, ssim, metametric = _apply_lut_core(
        rgb, transformer, power_weights, foveated, temporal,
    )

    if not eval_mode and out_path is not None:
        out_arr = (optimized.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(out_arr).save(out_path)

    return {
        "filename": image_path.name,
        "saving": saving,
        "psnr": psnr,
        "ssim": ssim,
        "metametric": metametric,
    }


def apply_lut_with_matcher(
    image_path: Path,
    transformers: Sequence[LUTColorTransformer],
    matcher: SceneMatcher,
    power_weights: Tuple[float, float, float],
    out_path: Optional[Path] = None,
    eval_mode: bool = False,
    foveated: float = 0,
    temporal: float = 0,
) -> Dict[str, Any]:
    """Match scene → select LUT → apply."""
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.shape[0] > arr.shape[1]:
        arr = arr.transpose(1, 0, 2)
    rgb = torch.from_numpy(arr)

    best_index, prototype, distance = matcher.match_paths([image_path])

    optimized, saving, psnr, ssim, metametric = _apply_lut_core(
        rgb, transformers[best_index], power_weights, foveated, temporal,
    )

    if not eval_mode and out_path is not None:
        out_arr = (optimized.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(out_arr).save(out_path)

    return {
        "filename": image_path.name,
        "saving": saving,
        "psnr": psnr,
        "ssim": ssim,
        "metametric": metametric,
        "matched_prototype": prototype.name,
        "match_distance": float(distance),
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Apply LUT to images and report metrics")
    parser.add_argument("--lut", type=str, default="", help="Path to LUT model (ignored if using scene manifest)")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--power-weights", nargs=3, type=float, default=[0.18, 0.58, 1.0])
    parser.add_argument("--foveated", type=float, default=0.5, help="Foveated modulation strength [0, 1]")
    parser.add_argument("--temporal", type=float, default=0., help="Temporal smoothing strength [0, 1]")
    parser.add_argument("--scene-manifest", type=str, default=None,
                        help="Optional JSON manifest with scene prototypes and LUT paths")
    parser.add_argument("--eval-mode", action="store_true",
                        help="Evaluation mode: don't save images, output metrics to JSON")
    parser.add_argument("--json-output", type=str, default="evaluation_results.json",
                        help="Path to JSON output file (only used in eval mode)")
    args = parser.parse_args()

    image_paths = collect_image_paths(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    transformers: List[LUTColorTransformer]
    matcher: Optional[SceneMatcher] = None
    if args.scene_manifest:
        prototypes, feature_mean, feature_std = load_scene_manifest(args.scene_manifest)
        if not prototypes:
            raise ValueError(f"scene manifest has no prototypes: {args.scene_manifest}")
        transformers = [
            load_lut_transformer(
                proto.lut_path or args.lut,
                foveated=args.foveated,
                temporal=args.temporal,
            )
            for proto in prototypes
        ]
        matcher = SceneMatcher(prototypes, feature_mean=feature_mean, feature_std=feature_std)
    else:
        transformer = load_lut_transformer(
            args.lut, foveated=args.foveated, temporal=args.temporal,
        )
        transformers = [transformer]

    selected = image_paths[:args.max_images]
    results: List[Dict[str, Any]] = []
    savings: List[float] = []
    psnrs: List[float] = []
    ssims: List[float] = []
    metametrics: List[float] = []

    for p in selected:
        if args.eval_mode:
            out_p = None
        else:
            out_p = Path(args.output_dir) / p.name
            out_p.parent.mkdir(parents=True, exist_ok=True)

        if matcher is None:
            result = apply_lut_single(
                p, transformers[0], tuple(args.power_weights), out_p,
                eval_mode=args.eval_mode,
                foveated=args.foveated, temporal=args.temporal,
            )
        else:
            result = apply_lut_with_matcher(
                p, transformers, matcher, tuple(args.power_weights), out_p,
                eval_mode=args.eval_mode,
                foveated=args.foveated, temporal=args.temporal,
            )
        results.append(result)
        savings.append(result["saving"])
        psnrs.append(result["psnr"])
        ssims.append(result["ssim"])
        metametrics.append(result["metametric"])

        parts = f"{p.name}: saving={result['saving']*100:.2f}%"
        parts += f", PSNR={result['psnr']:.2f}dB, SSIM={result['ssim']:.4f}, MetaM={result['metametric']:.6f}"
        # if not args.eval_mode:
        print(parts)
        if matcher is not None:
            print(f"  matched={result['matched_prototype']} distance={result['match_distance']:.4f}")

    avg_saving = float(np.mean(savings)) if savings else 0.0
    avg_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    avg_ssim = float(np.mean(ssims)) if ssims else 0.0
    avg_metametric = float(np.mean(metametrics)) if metametrics else 0.0

    final_results = {
        "summary": {
            "total_images": len(results),
            "average_saving": avg_saving,
            "average_saving_percent": avg_saving * 100,
            "average_psnr": avg_psnr,
            "average_ssim": avg_ssim,
            "average_metametric": avg_metametric,
        },
        "per_image": results,
    }

    if args.eval_mode:
        output_file = Path(args.json_output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    print(f"\n{'='*50}")
    print(f"SUMMARY over {len(results)} images:")
    print(f"Average Saving: {avg_saving*100:.2f}%")
    if not args.eval_mode or True:
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average MetaM: {avg_metametric:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()