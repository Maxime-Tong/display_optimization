from __future__ import annotations
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Tuple

from src.screen_adaptor.model import LUTColorTransformer, load_lut_transformer


def generate_ellipsoid_surface(
    center: np.ndarray, 
    rx: float, 
    ry: float, 
    rz: float,
    n_theta: int = 30,
    n_phi: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成椭球网格点
    
    Args:
        center: [R,G,B] 中心点坐标
        rx, ry, rz: 三个轴向的半轴长度
        n_theta: 经线分辨率
        n_phi: 纬线分辨率
    
    Returns:
        x, y, z: 网格坐标数组
    """
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = center[0] + rx * np.sin(phi_grid) * np.cos(theta_grid)
    y = center[1] + ry * np.sin(phi_grid) * np.sin(theta_grid)
    z = center[2] + rz * np.cos(phi_grid)
    return x, y, z


def sample_rgb_from_lut(
    lut_transformer: LUTColorTransformer, 
    sample_num: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从LUT空间随机采样归一化RGB中心点，并查表获取三轴椭球偏移rx,ry,rz
    
    Args:
        lut_transformer: LUT变换器
        sample_num: 采样数量
        seed: 随机种子，保证可复现
    
    Returns:
        centers: (N,3) 采样RGB中心点 [0,1]
        radii: (N,3) 每个点对应椭球半轴(rx,ry,rz)
    """
    np.random.seed(seed)
    
    lut_tensor = lut_transformer.lut
    grid_res = lut_tensor.shape[:3]  # (R_res, G_res, B_res)
    N = sample_num

    # 随机采样网格索引
    r_idx = np.random.randint(0, grid_res[0], size=N)
    g_idx = np.random.randint(0, grid_res[1], size=N)
    b_idx = np.random.randint(0, grid_res[2], size=N)

    # 归一化RGB中心坐标 [0,1]
    r_center = r_idx / (grid_res[0] - 1)
    g_center = g_idx / (grid_res[1] - 1)
    b_center = b_idx / (grid_res[2] - 1)
    centers = np.stack([r_center, g_center, b_center], axis=-1)

    # 查表获取椭球三轴偏移量 rx ry rz
    radii = lut_tensor[r_idx, g_idx, b_idx].cpu().numpy()
    
    # 确保半径非负，避免显示问题
    radii = np.maximum(radii, 1e-6)
    
    return centers, radii


def draw_lut_ellipsoids(lut_path: str, sample_count: int = 3, save_path: str = None):
    """
    绘制LUT感知椭球可视化
    
    Args:
        lut_path: LUT模型路径
        sample_count: 绘制椭球数量
        save_path: 保存图片路径，None则显示
    """
    # 1. 加载LUT
    transformer = load_lut_transformer(lut_path)
    rgb_centers, ellipsoid_radii = sample_rgb_from_lut(transformer, sample_count)

    # 2. matplotlib 3D画布配置
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    fig = plt.figure(figsize=(12, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # 3. 隐藏网格线
    ax.grid(False)  # 移除网格线
    
    # 4. 设置坐标轴
    # 隐藏刻度标签
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # 设置坐标轴标签 - 使用RGB颜色
    ax.set_xlabel('R', fontsize=16, fontweight='bold', color='#CC0000', labelpad=15)
    ax.set_ylabel('G', fontsize=16, fontweight='bold', color='#00AA00', labelpad=15)
    ax.set_zlabel('B', fontsize=16, fontweight='bold', color='#0000CC', labelpad=15)
    
    
    # 移除面板填充
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # 设置面板边框透明度（使背景更干净）
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    
    # 设置视角
    ax.view_init(elev=25, azim=-45)

    # 5. 绘制椭球
    for i, (rgb_c, (rx, ry, rz)) in enumerate(zip(rgb_centers, ellipsoid_radii)):
        # 生成椭球表面
        surf_x, surf_y, surf_z = generate_ellipsoid_surface(
            rgb_c, rx, ry, rz,
            n_theta=30,  # 适当提高分辨率
            n_phi=20
        )
        
        # 绘制半透明椭球，添加边缘线条增强立体感
        ax.plot_surface(
            surf_x, surf_y, surf_z,
            color=rgb_c,
            alpha=0.35,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=True
        )
        
        # 添加椭球边缘线框（细线，增强视觉效果）
        edge_alpha = 0.15
        ax.plot_wireframe(
            surf_x, surf_y, surf_z,
            color=rgb_c,
            alpha=edge_alpha,
            rstride=4,
            cstride=4,
            linewidth=0.5
        )
        
        # 标记中心点
        ax.scatter(
            rgb_c[0], rgb_c[1], rgb_c[2],
            color=rgb_c,
            s=30,
            edgecolors='black',
            linewidth=0.5,
            alpha=0.8,
            zorder=10
        )

    # # 6. 添加RGB立方体参考边框（半透明）
    # cube_range = [0, 1]
    # for i, (start, end) in enumerate([(0, 1), (0, 1), (0, 1)]):
    #     # 绘制立方体边框
    #     if i == 0:  # R方向
    #         x = [start, end, end, start, start]
    #         y = [0, 0, 0, 0, 0]
    #         z = [0, 0, 1, 1, 0]
    #     elif i == 1:  # G方向
    #         x = [0, 0, 0, 0, 0]
    #         y = [start, end, end, start, start]
    #         z = [0, 0, 1, 1, 0]
    #     else:  # B方向
    #         x = [0, 0, 1, 1, 0]
    #         y = [0, 0, 0, 0, 0]
    #         z = [start, end, end, start, start]
        
    #     ax.plot(x, y, z, color='gray', alpha=0.2, linewidth=0.5)

    # 7. 调整布局
    plt.tight_layout()
    
    # 8. 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图片已保存至: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize RGB LUT Perception Ellipsoids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visualize_lut.py --lut model.pth --samples 3
  python visualize_lut.py --lut model.pth --samples 5 --output ellipsoids.png
        """
    )
    parser.add_argument(
        "--lut", 
        type=str, 
        required=True, 
        help="Trained LUT model path (.pth)"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=3, 
        help="Number of ellipsoids to draw (2~5 recommended)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (e.g., ellipsoids.png). If not specified, display interactively."
    )
    
    args = parser.parse_args()
    
    # 参数验证
    if args.samples < 1:
        print("警告: samples参数至少为1，已自动调整为1")
        args.samples = 1
    elif args.samples > 10:
        print("警告: samples参数建议不超过5，当前值可能使图像拥挤")
    
    draw_lut_ellipsoids(args.lut, sample_count=args.samples, save_path=args.output)


if __name__ == "__main__":
    main()