"""
Global constants for the Gradual Chromatic Adaptation (GCA) module.

Documented in: gradual.md — "渐进色适应（Gradual Chromatic Adaptation）技术实现方案"
"""

from __future__ import annotations

import numpy as np

# ── OLED display power model ────────────────────────────────────────────
# p(c) = p_disp^T @ c + p_static   (per pixel)
# Physical per-channel OLED power coefficients (blue subpixel is the least
# efficient, roughly 2x red/green -> shifting toward yellow-green saves power).
P_DISP_RAW = np.array([231.53, 245.67, 530.75], dtype=np.float64)
P_DISP_NORM = P_DISP_RAW / P_DISP_RAW.sum()      # ≈ [0.230, 0.244, 0.527]
P_STATIC = 0.0                                   # static (offset) power term

# Normalized OLED weights used by screen_adaptor eval.ps1 (R, G, B)
POWER_WEIGHTS_RGB = (0.229, 0.243, 0.526)

# ── D65 reference white (sRGB) ──────────────────────────────────────────
D65_XYZ = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)   # XYZ (Y=1)
D65_CCT = 6504.0                                               # Kelvin

# ── Bradford chromatic adaptation matrix ────────────────────────────────
BRADFORD_MATRIX = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
], dtype=np.float64)
BRADFORD_MATRIX_INV = np.linalg.inv(BRADFORD_MATRIX)

# ── Illuminant trajectories in CIE u'v' space ───────────────────────────
# Three linear trajectories (angles in radians) plus the daylight locus.
TRAJECTORY_ANGLES_RAD = {
    "1.47": 1.47,
    "1.863": 1.863,
    "2.256": 2.256,
}
# Daylight locus: CCT ramp from D65 (6504 K) down to a warmer white
# (~5300 K) over T_MAX, whose u'v' displacement ≈ ΔT (0.02).
DAYLIGHT_CCT_START = D65_CCT
DAYLIGHT_CCT_END = 5300.0

# ── Adaptation model parameters (psychophysics, per gradual.md) ─────────
# a'(t) = k1 * (k2 * A(t) - a(t))
ADAPTATION_PARAMS = {
    "daylight": {"k1": 0.127, "k2": 0.712},
    "1.47":     {"k1": 0.101, "k2": 0.685},
    "1.863":    {"k1": 0.107, "k2": 0.638},
    "2.256":    {"k1": 0.069, "k2": 0.707},
}

# ── Recommended operating point (gradual.md "推荐配置") ─────────────────
DEFAULT_TRAJECTORY = "1.47"     # yellow-green direction
# Max adaptation-safe constant velocity for the 1.47 trajectory:
#   v_max = DELTA_T / (1/k1 + (1-k2)*t_max)
#         = 0.02 / (1/0.101 + 0.315*120)  ≈ 4.19e-4 u'v'/s
# This exploits chromatic adaptation (the perceived quantity is the mismatch
# |A-a|, not |A - D65|) so the illuminant drifts ~2.5x further than DELTA_T
# while the perceived cast remains within 5 JND.
DEFAULT_VELOCITY = 0.000419      # u'v' / second (adaptation-derived max)
T_MAX = 120.0                   # seconds (2 minutes)
DELTA_T = 0.02                  # 5 JND ≈ 0.02 u'v'

# ── Combined GUD (Gradual Uniform Dimming) configuration ────────────────
# gradual.md "组合优化": GCA first (染色 1 min), then GUD (dimming 1 min).
# GUD dims display brightness toward GUD_TARGET by its own ramp. Uniform
# dimming reduces ALL channels proportionally -> directly lowers power with
# no chromatic cast. Combined saving up to ~31% with no significant
# perceived difference.
GUD_ENABLED = True
GUD_TARGET = 0.85               # final uniform dim factor (1.0 = no dim)
GUD_T_MAX = T_MAX               # dim ramp duration (s)
