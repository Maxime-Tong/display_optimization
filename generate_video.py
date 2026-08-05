#!/usr/bin/env python3
"""
generate_video.py — 使用不同屏幕省电优化方法将数据集图像序列生成视频。

支持方法:
  original              原始图像，不做处理
  screen_adaptor        LUT 颜色变换 (screen_adaptor)，支持单LUT / scene-cluster / foveated / temporal
  vr_power_saver        VR 色彩感知功耗优化 (vr-power-saver)
  gradual_adaptation    渐进色适应 GCA + GUD (gradual_adaptation)
  hvs_vr_encoding       HVS VR 帧缓冲压缩 (hvs_vr_encoding)

screen_adaptor 额外选项:
  --lut-mode single|cluster   single=使用基础LUT；cluster=按场景清单动态切换LUT
  --foveated 0~1              foveated 调制强度（0=关闭）
  --temporal 0~1              时序平滑强度（0=关闭）

用法示例:
  # 原始
  python generate_video.py --method original --dataset miHoYo --output-dir videos

  # screen_adaptor 单LUT + foveated
  python generate_video.py --method screen_adaptor --dataset miHoYo --output-dir videos \
      --lut-mode single --foveated 0.5

  # screen_adaptor scene-cluster
  python generate_video.py --method screen_adaptor --dataset miHoYo --output-dir videos \
      --lut-mode cluster

  # screen_adaptor scene-cluster + foveated
  python generate_video.py --method screen_adaptor --dataset miHoYo --output-dir videos \
      --lut-mode cluster --foveated 0.5

  # 其余方法
  python generate_video.py --method vr_power_saver --dataset miHoYo --output-dir videos
  python generate_video.py --method gradual_adaptation --dataset miHoYo --output-dir videos
  python generate_video.py --method hvs_vr_encoding --dataset miHoYo --output-dir videos
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2
from PIL import Image

# ── 路径配置 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

_SA_ROOT = ROOT / "screen_adaptor"
_SA_SRC = _SA_ROOT / "src"
_SA_CLUSTERS = _SA_ROOT / "outputs"
_VR_ROOT = ROOT / "vr-power-saver"
_GA_ROOT = ROOT / "gradual_adaptation"
_HVS_ROOT = ROOT / "hvs_vr_encoding"
_HVS_COLOR_OPT = _HVS_ROOT / "host" / "color_optimizer"

for p in (_SA_SRC, _VR_ROOT, _GA_ROOT.parent, _HVS_COLOR_OPT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ── 常量 ──────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
DATASET_ROOT = _SA_ROOT / "datasets"

# screen_adaptor
SCREEN_ADAPTOR_LUT = _SA_CLUSTERS / "base_lut.pt"
SCREEN_MANIFEST = _SA_CLUSTERS / "scene_manifest.json"

# vr-power-saver
VR_MODEL_PATH = _VR_ROOT / "io" / "color_model" / "model.pth"
VR_FOV = 60
VR_TRANSITION_WIDTH = 3
VR_POWER_WEIGHTS = (0.229, 0.243, 0.526)

# hvs_vr_encoding
HVS_TILE_SIZE = 4
HVS_FOV = 110
HVS_MAX_ECC = 35
HVS_ECC_NO_COMPRESS = 10

POWER_WEIGHTS_RGB = (0.229, 0.243, 0.526)


# ── 图片加载 / 工具 ───────────────────────────────────────────────────────

def collect_images(directory: Path) -> List[Path]:
    """收集目录下全部支持的图片，按文件名排序。"""
    paths = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths


def crop_to_tile_multiple(img: np.ndarray, tile_size: int = HVS_TILE_SIZE) -> np.ndarray:
    """裁剪到 4x4 tile 倍数并对齐 H*W%48==0（hvs_vr_encoding 内部要求）。

    Note: 只做最小中心裁剪以满足 tile 约束，不做方形裁剪，保持与原图一致的宽高比。
    """
    img = np.asarray(img)
    h, w = img.shape[0], img.shape[1]
    h = (h // tile_size) * tile_size
    w = (w // tile_size) * tile_size
    if (h * w) % 48 != 0:
        h = (h // 12) * 12
    vpad = (img.shape[0] - h) // 2
    hpad = (img.shape[1] - w) // 2
    return img[vpad:vpad + h, hpad:hpad + w, ...].copy()


def _resolve_lut_path(path: Optional[str | Path]) -> Path:
    """将 scene_manifest 中的相对 lut_path 解析为绝对路径；找不到则回退到基础 LUT。"""
    if not path:
        return SCREEN_ADAPTOR_LUT
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        _SA_CLUSTERS / "luts" / p.name,
        _SA_CLUSTERS / p,
        _SA_CLUSTERS / "luts" / p,
        _SA_CLUSTERS / "outputs" / "luts" / p.name,
        p,
    ]
    for c in candidates:
        if c.exists():
            return c
    print(f"  [warn] LUT not found for {path}, fallback to base_lut.pt")
    return SCREEN_ADAPTOR_LUT


# ── 各方法处理逻辑 ───────────────────────────────────────────────────────

def _build_screen_adaptor_pipeline(
    lut_mode: str,
    foveated: float,
    temporal: float,
) -> Tuple[List[Any], Any]:
    """构建 screen_adaptor 的 transformer(s) + 可选 SceneMatcher。

    Returns:
        (transformers, matcher_or_None)
        - lut_mode=="single": 一个 transformer，matcher=None
        - lut_mode=="cluster": 每个原型一个 transformer，matcher=SceneMatcher
    """
    from screen_adaptor.model import load_lut_transformer
    from screen_adaptor.scene_matcher import SceneMatcher, load_scene_manifest

    if lut_mode == "single":
        transformer = load_lut_transformer(
            SCREEN_ADAPTOR_LUT, foveated=foveated, temporal=temporal,
        )
        return [transformer], None

    # cluster 模式：从 scene_manifest 加载 prototypes + 每簇 LUT
    if not SCREEN_MANIFEST.exists():
        raise FileNotFoundError(f"scene manifest not found: {SCREEN_MANIFEST}")
    prototypes, feat_mean, feat_std = load_scene_manifest(SCREEN_MANIFEST)
    transformers = []
    for proto in prototypes:
        lut = _resolve_lut_path(proto.lut_path)
        transformers.append(
            load_lut_transformer(lut, foveated=foveated, temporal=temporal)
        )
    matcher = SceneMatcher(prototypes, feature_mean=feat_mean, feature_std=feat_std)
    return transformers, matcher


def _process_screen_adaptor_frame(
    img_path: Path,
    img_uint8: np.ndarray,
    transformers: List[Any],
    matcher: Any,
) -> np.ndarray:
    """按帧应用 screen_adaptor LUT（单 LUT 或 scene-cluster 动态切换）。"""
    import torch

    # Use the full native-resolution image (no transpose, no square crop)
    rgb = torch.from_numpy(img_uint8.astype(np.float32) / 255.0)

    if matcher is None:
        transformer = transformers[0]
    else:
        best_idx, proto, dist = matcher.match_paths([img_path])
        transformer = transformers[best_idx]

    with torch.no_grad():
        out = transformer.transform(rgb)
    out_np = out.cpu().numpy()
    frame = (np.clip(out_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    if matcher is None:
        return frame, 0
    return frame, best_idx


def _draw_lut_label(frame_bgr: np.ndarray, lut_idx: int) -> None:
    """在帧左上角绘制红色 LUT 编号（frame 为 BGR，写入视频前调用）。"""
    cv2.putText(
        frame_bgr, f"LUT:{lut_idx}", (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA,
    )


def _build_vr_model():
    from color_model.base_color_model import BaseColorModel
    model = BaseColorModel()
    model.load(str(VR_MODEL_PATH))
    return model


def _process_vr_frame(img_uint8: np.ndarray, model: Any) -> np.ndarray:
    """VR 感知功耗优化（foveated + 颜色压缩）。保持原始图片尺寸（不做方形裁剪）。"""
    from util.vr_tools import build_ecc_map, build_transition_mask

    inp = img_uint8.astype(np.float32) / 255.0
    power_vec = -np.array(VR_POWER_WEIGHTS, dtype=np.float64)
    ecc_map = build_ecc_map(
        VR_FOV, 0.0, 0.0,
        model.opt.max_eccentricity, inp.shape[0], inp.shape[1],
    )
    out = model.apply_filter(inp, ecc_map, power_vec)
    mask = build_transition_mask(
        ecc_map, model.opt.min_eccentricity, VR_TRANSITION_WIDTH,
    )
    out = inp * (1 - mask) + out * mask
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _build_gradual_optimizer():
    from gradual_adaptation import GradualAdaptationImageOptimizer
    from gradual_adaptation.constants import DEFAULT_TRAJECTORY, DEFAULT_VELOCITY, T_MAX
    return GradualAdaptationImageOptimizer(
        trajectory=DEFAULT_TRAJECTORY,
        velocity=DEFAULT_VELOCITY,
        t_max=T_MAX,
        gud_enabled=True,
    )


def _process_gradual_frame(
    img_uint8: np.ndarray,
    optimizer: Any,
    t: float,
) -> np.ndarray:
    """渐进色适应 GCA+GUD（按时间戳 t 处理）。保持原始图片尺寸（不做方形裁剪）。"""
    inp = img_uint8.astype(np.float32) / 255.0
    out = optimizer.process_frame(inp, t=t)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _build_hvs_optimizer(img_uint8: np.ndarray):
    # hvs_vr_encoding/host/color_optimizer/util has NO __init__.py (namespace
    # package) while vr-power-saver/util is a regular package. When both are on
    # sys.path the regular package always shadows the namespace one, breaking
    # `from util.ecc_map import ...`. Temporarily remove vr-power-saver from
    # sys.path for this method only (each generate_video run uses one method).
    _vroot = str(_VR_ROOT)
    _popped = _vroot in sys.path
    if _popped:
        sys.path.remove(_vroot)
    try:
        from red_blue_optimization_cpu import Image_color_optimizer
        inp = crop_to_tile_multiple(img_uint8, HVS_TILE_SIZE)
        h, w = inp.shape[0], inp.shape[1]
        if h < 12 or w < 4:
            raise ValueError(f"image too small after crop: {inp.shape}")
        return Image_color_optimizer(
            img_height=h, img_width=w, tile_size=HVS_TILE_SIZE,
            foveated=True, max_ecc=HVS_MAX_ECC, h_fov=HVS_FOV,
            ecc_no_compress=HVS_ECC_NO_COMPRESS,
        )
    finally:
        if _popped and _vroot not in sys.path:
            sys.path.insert(0, _vroot)


def _process_hvs_frame(img_uint8: np.ndarray, optimizer: Any) -> np.ndarray:
    """HVS VR 帧缓冲压缩（每帧颜色降维到 JND 边界）。"""
    inp = crop_to_tile_multiple(img_uint8, HVS_TILE_SIZE)
    if (optimizer.img_height, optimizer.img_width) != (inp.shape[0], inp.shape[1]):
        return _build_hvs_optimizer(img_uint8).color_conversion(
            inp.astype(np.float32)
        ).astype(np.uint8)
    out = optimizer.color_conversion(inp.astype(np.float32))
    return out.astype(np.uint8)


# ── 视频生成主流程 ────────────────────────────────────────────────────────

def generate_video(
    method: str,
    dataset_name: str,
    output_dir: Path,
    lut_mode: str = "single",
    foveated: float = 0.0,
    temporal: float = 0.0,
    max_images: int = 0,
    fps: int = 30,
    time_s: float = 120.0,
    resolution: Optional[Tuple[int, int]] = None,
    show_lut: bool = False,
) -> Path:
    """生成视频。"""
    dataset_dir = DATASET_ROOT / dataset_name
    if not dataset_dir.exists():
        dataset_dir = Path(dataset_name)  # 允许传绝对路径
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_dir}")

    images = collect_images(dataset_dir)
    if not images:
        raise ValueError(f"no images found in {dataset_dir}")
    if max_images and max_images > 0:
        images = images[:max_images]
    n = len(images)

    print(f"\n{'='*64}")
    print(f"  Dataset : {dataset_name}  ({n} images)")
    print(f"  Method  : {method}")
    if method == "screen_adaptor":
        print(f"    lut-mode : {lut_mode}")
        print(f"    foveated : {foveated}")
        print(f"    temporal : {temporal}")
    if method == "gradual_adaptation":
        print(f"    ramp time: {time_s}s")
    print(f"{'='*64}")

    # ── 初始化方法组件 ────────────────────────────────────────────────────
    transformers: List[Any] = []
    matcher: Any = None
    vr_model: Any = None
    ga_optimizer: Any = None
    hvs_optimizer: Any = None

    if method == "screen_adaptor":
        transformers, matcher = _build_screen_adaptor_pipeline(lut_mode, foveated, temporal)
    elif method == "vr_power_saver":
        vr_model = _build_vr_model()
    elif method == "gradual_adaptation":
        ga_optimizer = _build_gradual_optimizer()
    elif method == "hvs_vr_encoding":
        first_img = np.asarray(Image.open(images[0]).convert("RGB"), dtype=np.uint8)
        hvs_optimizer = _build_hvs_optimizer(first_img)

    # ── 用第一帧决定输出尺寸 ──────────────────────────────────────────────
    first_img = np.asarray(Image.open(images[0]).convert("RGB"), dtype=np.uint8)
    if method == "original":
        frame0 = first_img  # 与读取图片尺寸一致，不裁剪
    elif method == "screen_adaptor":
        frame0, _ = _process_screen_adaptor_frame(images[0], first_img, transformers, matcher)
    elif method == "vr_power_saver":
        frame0 = _process_vr_frame(first_img, vr_model)
    elif method == "gradual_adaptation":
        frame0 = _process_gradual_frame(first_img, ga_optimizer, t=0.0)
    elif method == "hvs_vr_encoding":
        frame0 = _process_hvs_frame(first_img, hvs_optimizer)
    else:
        raise ValueError(f"unknown method: {method}")

    # 视频画布默认与原始图片尺寸一致（除非 --resolution 显式指定）。
    out_h, out_w = first_img.shape[0], first_img.shape[1]
    if resolution:
        out_w, out_h = resolution
    if out_w % 2:
        out_w -= 1
    if out_h % 2:
        out_h -= 1

    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{dataset_name}_{method}"
    if method == "screen_adaptor":
        if lut_mode == "cluster":
            out_name += "_cluster"
        if foveated and foveated > 0:
            out_name += "_foveated"
    video_path = output_dir / f"{out_name}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, fps, (out_w, out_h))
    if not vw.isOpened():
        raise RuntimeError(f"failed to create video writer: {video_path}")

    print(f"  Video size : {out_w}x{out_h} @ {fps} fps")
    print(f"  Output     : {video_path}\n")

    t_start = time.perf_counter()
    for idx, img_path in enumerate(images):
        t0 = time.perf_counter()
        try:
            img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)

            lut_idx = 0
            if method == "original":
                frame = img  # 与读取图片尺寸一致，不裁剪
            elif method == "screen_adaptor":
                frame, lut_idx = _process_screen_adaptor_frame(img_path, img, transformers, matcher)
            elif method == "vr_power_saver":
                frame = _process_vr_frame(img, vr_model)
            elif method == "gradual_adaptation":
                t = float(((idx + 0.5) / n) * time_s)
                frame = _process_gradual_frame(img, ga_optimizer, t=t)
            elif method == "hvs_vr_encoding":
                frame = _process_hvs_frame(img, hvs_optimizer)
            else:
                raise ValueError(f"unknown method: {method}")

            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if show_lut and method == "screen_adaptor":
                _draw_lut_label(frame_bgr, lut_idx)
            vw.write(frame_bgr)

            elapsed = time.perf_counter() - t0
            print(f"  [{idx+1:4d}/{n}] {img_path.name}  {elapsed:.2f}s")
        except Exception as e:
            print(f"  [{idx+1:4d}/{n}] {img_path.name}: ERROR - {e}")

    vw.release()
    total = time.perf_counter() - t_start
    print(f"\n  Done in {total:.1f}s -> {video_path}")
    return video_path


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用不同屏幕省电优化方法将数据集图像序列生成视频"
    )
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["original", "screen_adaptor", "vr_power_saver",
                 "gradual_adaptation", "hvs_vr_encoding"],
    )
    parser.add_argument("--dataset", type=str, default="miHoYo")
    parser.add_argument("--output-dir", type=str, default="videos")
    parser.add_argument("--max-images", type=int, default=0,
                        help="最多处理的图片数（0=全部）")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--time", type=float, default=120.0,
                        help="gradual_adaptation 的总耗时（秒）")
    parser.add_argument("--resolution", type=str, default=None,
                        help="输出分辨率 WxH，如 1920x1080")
    parser.add_argument("--show-lut", action="store_true",
                        help="在 screen_adaptor 视频左上角绘制当前 LUT 编号（红色）")

    # screen_adaptor 专属
    parser.add_argument("--lut-mode", type=str, default="cluster",
                        choices=["single", "cluster"])
    parser.add_argument("--foveated", type=float, default=1.0,
                        help="foveated 调制强度 [0,1]，仅 screen_adaptor")
    parser.add_argument("--temporal", type=float, default=0.0,
                        help="时序平滑强度 [0,1]，仅 screen_adaptor")

    args = parser.parse_args()

    resolution = None
    if args.resolution:
        parts = args.resolution.lower().split("x")
        resolution = (int(parts[0]), int(parts[1]))

    generate_video(
        method=args.method,
        dataset_name=args.dataset,
        output_dir=Path(args.output_dir),
        lut_mode=args.lut_mode,
        foveated=args.foveated,
        temporal=args.temporal,
        max_images=args.max_images,
        fps=args.fps,
        time_s=args.time,
        resolution=resolution,
        show_lut=args.show_lut,
    )


if __name__ == "__main__":
    main()