# Gradual Chromatic Adaptation (GCA)

> 渐进色适应（Gradual Chromatic Adaptation）技术实现 —— 依据 `gradual.md` 技术方案。
> 基于人类视觉色适应的时间特性，在 VR 显示中**无眼动追踪**地逐步偏移照明色温（白点），
> 在感知不可见的前提下降低 OLED 显示功耗（蓝色子像素功耗最高）。

## 核心原理

- **色适应是渐进过程**（需数分钟完全适应）：瞬时改变照明会被察觉，缓慢偏移则不可见。
- 功耗模型 `p(c) = p_dispᵀ·c + p_static`：蓝色子像素功耗约为红/绿 2 倍，
  将照明向**黄绿方向**偏移可降低蓝色分量从而省电（推荐轨迹 ~31%）。
- 数学模型：
  ```
  a'(t) = k₁(k₂·A(t) - a(t))     # 用户适应状态演化
  A(t)  = D65 + min(v·t, ΔT)·dir # 场景照明（u'v' 空间沿轨迹移动）
  ```

## 模块结构

| 文件 | 职责 |
|------|------|
| `constants.py` | OLED 功耗权重、Bradford 矩阵、D65、轨迹角度、k₁/k₂、速度/时长/ΔT |
| `colorspace.py` | sRGB↔线性↔XYZ、XYZ↔u'v'、Bradford CAT（`compute_cat_matrix`/`bradford_cat`） |
| `trajectory.py` | 日光轨迹 + 3 条线性轨迹（1.47/1.863/2.256 rad）；`get_illuminant_xyz(t,...)` |
| `color_adaptation.py` | `GradualChromaticOptimizer`：逐帧 `apply_to_frame(img, t)` 应用 GCA |
| `power_model.py` | `power_model`（逐像素功耗）与 `compute_power_reduction`（省电比例） |
| `image_optimizer.py` | `GradualAdaptationImageOptimizer`：完整逐帧管线（可组合 GUD 均匀调光） |

## 快速使用

```python
import numpy as np
from gradual_adaptation import GradualChromaticOptimizer

img = np.random.rand(480, 640, 3).astype(np.float32)  # sRGB [0,1]

gca = GradualChromaticOptimizer(trajectory="1.47", velocity=0.00015, t_max=120)
out = gca.apply_to_frame(img, t=120.0)                # 2 分钟末的帧
```

推荐配置（gradual.md）：轨迹 **1.47 rad**（黄绿方向），`velocity=0.00015 u'v'/s`，
`t_max=120 s`，ΔT = 5 JND（≈0.02 u'v'）。

## 关键注意事项

1. 在线性 sRGB 空间计算；sRGB 必须先反 Gamma（`srgb_to_linear`）。
2. GCA 时间 `t` 必须**单调递增**（`GradualChromaticOptimizer.advance` 已内建单调，
   且剪辑到 `[0, t_max]`）；用户中断使用需 `reset()`。
3. 总时长 2 分钟，变化平滑，禁止跳变。
4. 可与 GUD（均匀调光）组合：先 GCA 1 分钟，再 GUD 1 分钟，组合省电达 ~31%。