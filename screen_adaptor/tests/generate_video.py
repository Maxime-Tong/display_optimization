#!/usr/bin/env python3
"""
图片合并视频工具
将指定目录下的图片按文件名顺序合并成视频
支持多种图片格式和视频编码选项
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def get_supported_images(folder_path):
    """
    获取文件夹内所有支持的图片文件，按文件名排序
    
    Args:
        folder_path: 文件夹路径
    
    Returns:
        list: 排序后的图片文件路径列表
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    for file in Path(folder_path).iterdir():
        if file.is_file() and file.suffix.lower() in supported_extensions:
            image_files.append(str(file))
    
    # 按文件名自然排序
    image_files.sort()
    return image_files


def get_image_dimensions(image_path):
    """
    获取图片尺寸
    
    Args:
        image_path: 图片路径
    
    Returns:
        tuple: (width, height)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    height, width = img.shape[:2]
    return width, height


def resize_to_fit(image, target_width, target_height, fit_mode='pad'):
    """
    调整图片大小以适应目标尺寸
    
    Args:
        image: 输入图片
        target_width: 目标宽度
        target_height: 目标高度
        fit_mode: 适应模式 ('stretch', 'pad', 'crop')
    
    Returns:
        numpy.ndarray: 调整后的图片
    """
    h, w = image.shape[:2]
    
    if fit_mode == 'stretch':
        # 直接拉伸
        return cv2.resize(image, (target_width, target_height))
    
    elif fit_mode == 'pad':
        # 保持比例，填充黑边
        ratio = min(target_width / w, target_height / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        resized = cv2.resize(image, (new_w, new_h))
        
        # 创建画布并居中放置
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas
    
    elif fit_mode == 'crop':
        # 保持比例，裁剪多余部分
        ratio = max(target_width / w, target_height / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        resized = cv2.resize(image, (new_w, new_h))
        
        # 居中裁剪
        y_offset = (new_h - target_height) // 2
        x_offset = (new_w - target_width) // 2
        return resized[y_offset:y_offset + target_height, x_offset:x_offset + target_width]
    
    else:
        raise ValueError(f"不支持的适应模式: {fit_mode}")


def images_to_video(
    input_dir,
    output_path,
    fps=30,
    duration=None,
    frame_duration=None,
    fit_mode='pad',
    target_width=None,
    target_height=None,
    codec='mp4v'
):
    """
    将图片合并成视频
    
    Args:
        input_dir: 输入目录
        output_path: 输出视频路径
        fps: 帧率
        duration: 视频总时长(秒)，与frame_duration互斥
        frame_duration: 每张图片显示时长(秒)，与duration互斥
        fit_mode: 图片适应模式 ('stretch', 'pad', 'crop')
        target_width: 目标宽度，不指定则使用第一张图片宽度
        target_height: 目标高度，不指定则使用第一张图片高度
        codec: 视频编码器
    
    Returns:
        bool: 是否成功
    """
    # 获取所有图片
    image_files = get_supported_images(input_dir)
    
    if not image_files:
        print(f"错误: 在 {input_dir} 中没有找到支持的图片文件")
        print("支持的格式: jpg, jpeg, png, bmp, tiff, webp")
        return False
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 确定尺寸
    first_width, first_height = get_image_dimensions(image_files[0])
    
    if target_width is None:
        target_width = first_width
    if target_height is None:
        target_height = first_height
    
    print(f"输出视频尺寸: {target_width}x{target_height}")
    
    # 计算每张图片的帧数
    if duration is not None and frame_duration is not None:
        print("警告: 同时指定了 duration 和 frame_duration，将使用 frame_duration")
    
    if frame_duration is not None:
        frames_per_image = max(1, int(fps * frame_duration))
    elif duration is not None:
        total_frames = int(fps * duration)
        frames_per_image = max(1, total_frames // len(image_files))
    else:
        # 默认每张图片1秒
        frames_per_image = fps
    
    total_frames = frames_per_image * len(image_files)
    video_duration = total_frames / fps
    
    print(f"帧率: {fps} fps")
    print(f"每张图片帧数: {frames_per_image}")
    print(f"总帧数: {total_frames}")
    print(f"视频时长: {video_duration:.2f} 秒")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 根据文件扩展名确定编码器
    ext = output_path.suffix.lower()
    if ext == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif ext == '.mov':
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (target_width, target_height)
    )
    
    if not video_writer.isOpened():
        print(f"错误: 无法创建视频文件 {output_path}")
        return False
    
    # 处理每张图片
    print("正在生成视频...")
    with tqdm(total=len(image_files), desc="处理图片") as pbar:
        for img_path in image_files:
            try:
                # 读取图片
                img = cv2.imread(img_path)
                if img is None:
                    print(f"警告: 跳过无法读取的图片 {img_path}")
                    pbar.update(1)
                    continue
                
                # 调整尺寸
                resized_img = resize_to_fit(img, target_width, target_height, fit_mode)
                
                # 写入多帧
                for _ in range(frames_per_image):
                    video_writer.write(resized_img)
                    
            except Exception as e:
                print(f"警告: 处理图片 {img_path} 时出错: {e}")
            
            pbar.update(1)
    
    # 释放资源
    video_writer.release()
    print(f"视频已保存: {output_path}")
    
    return True


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="将指定目录下的图片合并成视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认设置，每张图片显示1秒
  python images_to_video.py -i ./images -o output.mp4
  
  # 指定帧率和每张图片显示时长
  python images_to_video.py -i ./images -o output.mp4 --fps 24 --frame-duration 0.5
  
  # 指定视频总时长（图片时长自动分配）
  python images_to_video.py -i ./images -o output.mp4 --duration 10 --fps 30
  
  # 调整图片适应方式和输出尺寸
  python images_to_video.py -i ./images -o output.mp4 --fit crop --width 1920 --height 1080
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入图片目录路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='输出视频文件路径（如: output.mp4）'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='视频帧率（默认: 30）'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='视频总时长（秒），将自动分配每张图片的显示时间'
    )
    
    parser.add_argument(
        '--frame-duration',
        type=float,
        default=None,
        help='每张图片的显示时长（秒）'
    )
    
    parser.add_argument(
        '--fit',
        choices=['stretch', 'pad', 'crop'],
        default='pad',
        help='图片适应模式: stretch(拉伸), pad(填充黑边), crop(裁剪)（默认: pad）'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=None,
        help='输出视频宽度，默认使用第一张图片的宽度'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=None,
        help='输出视频高度，默认使用第一张图片的高度'
    )
    
    parser.add_argument(
        '--codec',
        choices=['mp4v', 'XVID', 'MJPG', 'H264'],
        default='mp4v',
        help='视频编码器（默认: mp4v）'
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.isdir(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        sys.exit(1)
    
    # 执行转换
    success = images_to_video(
        input_dir=args.input,
        output_path=args.output,
        fps=args.fps,
        duration=args.duration,
        frame_duration=args.frame_duration,
        fit_mode=args.fit,
        target_width=args.width,
        target_height=args.height,
        codec=args.codec
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()