#!/usr/bin/env python3
"""
extract_frames.py — 将视频文件逐帧提取为 PNG 图片。

用法示例:
  # 默认: 最多 1000 帧，保存到 screen_adaptor/datasets/miHoYo_2/
  python extract_frames.py --video screen_adaptor/datasets/miHoYo_2.flv

  # 指定输出目录与最大帧数
  python extract_frames.py --video video.flv --output-dir out/frames --max-frames 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# ── 路径配置 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "screen_adaptor" / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将视频文件逐帧提取为 PNG 图片序列。"
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="输入视频文件路径（如 screen_adaptor/datasets/miHoYo_2.flv）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "输出目录（默认为视频同名的子目录，"
            "例如 miHoYo_2.flv -> datasets/miHoYo_2/）。"
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="最多提取的帧数（默认 1000，0 表示不限制）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path: Path = args.video
    if not video_path.is_file():
        print(f"[错误] 视频文件不存在: {video_path}")
        sys.exit(1)

    # 默认输出目录: 视频文件所在目录下的同名子目录
    output_dir: Path = args.output_dir
    if output_dir is None:
        output_dir = video_path.parent / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[信息] 视频信息: {fps:.2f} fps, {width}x{height}, "
        f"{total} 帧"
    )

    max_frames = args.max_frames
    cap_frames = total if max_frames <= 0 else min(total, max_frames)
    print(f"[信息] 将提取最多 {cap_frames} 帧到: {output_dir}")

    frame_idx = 0
    saved_count = 0
    while frame_idx < cap_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"[警告] 第 {frame_idx} 帧读取失败，提前停止。")
            break

        # 6 位零填充命名，与现有数据集命名规范一致 (000001.png, 000002.png, ...)
        out_path = output_dir / f"{frame_idx + 1:06d}.png"
        cv2.imwrite(str(out_path), frame)
        saved_count += 1

        if frame_idx % 100 == 0:
            print(f"[进度] 已保存 {saved_count}/{cap_frames} 帧...")

        frame_idx += 1

    cap.release()
    print(f"[完成] 共保存 {saved_count} 张 PNG 图片到: {output_dir}")


if __name__ == "__main__":
    main()