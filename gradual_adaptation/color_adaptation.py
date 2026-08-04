"""
Gradual Chromatic Adaptation (GCA) optimizer.

Implements gradual.md technique: instead of instantly changing the illumination
(color temperature), the white point is shifted GRADUALLY along a power-saving
trajectory in CIE u'v' space while the user's visual system adapts at the same
pace. Because chromatic adaptation is a slow process (~minutes), a sufficiently
slowly drifting illuminant is perceptually invisible, but the blue subpixel
emission (highest OLED power) is reduced over time.
"""

from __future__ import annotations

import numpy as np

from .colorspace import (
    bradford_cat,
    linear_to_srgb,
    srgb_to_linear,
)
from .constants import D65_XYZ, DEFAULT_TRAJECTORY, DEFAULT_VELOCITY, T_MAX
from .trajectory import get_illuminant_xyz


class GradualChromaticOptimizer:
    """
    Per-frame chromatic adaptation that slowly moves the display white point
    from D65 along a power-saving trajectory.

    a'(t) = k1 * (k2 * A(t) - a(t))       (model of user adaptation)
    A(t)  = D65 + min(velocity * t, DELTA_T) * dir   (scene illuminant)

    The frame is rendered under the *current* illuminant A(t) via Bradford
    chromatic adaptation, so the displayed content corresponds to what the
    user's slowly-adapting visual system expects to see.
    """

    def __init__(
        self,
        trajectory: str = DEFAULT_TRAJECTORY,
        velocity: float = DEFAULT_VELOCITY,
        t_max: float = T_MAX,
        k1: float | None = None,
        k2: float | None = None,
    ):
        self.trajectory = trajectory
        self.velocity = velocity
        self.t_max = t_max

        # Adaptation parameters from psychophysics (gradual.md §阶段二)
        # a'(t) = k1 * (k2 * A(t) - a(t))
        from .constants import ADAPTATION_PARAMS

        params = ADAPTATION_PARAMS.get(trajectory, ADAPTATION_PARAMS[DEFAULT_TRAJECTORY])
        self.k1 = params["k1"] if k1 is None else k1
        self.k2 = params["k2"] if k2 is None else k2

        # Absolute accumulated time; must remain monotonic across frames.
        self.t = 0.0

    def reset(self, t: float = 0.0) -> None:
        """Reset the accumulated time (e.g. when the user removes the headset)."""
        self.t = float(t)

    def advance(self, dt: float) -> float:
        """
        Advance the internal clock by a frame dt and return the new time.

        The clock is clipped to [0, t_max]; once t_max is reached the
        illuminant stays at its final (maximum-power-saving) value.
        """
        self.t = float(np.clip(self.t + dt, 0.0, self.t_max))
        return self.t

    def current_illuminant_xyz(self) -> np.ndarray:
        """XYZ of the display white point at the current time."""
        return get_illuminant_xyz(self.t, self.trajectory, self.velocity)

    def apply_to_frame(self, img_srgb: np.ndarray, t: float | None = None) -> np.ndarray:
        """
        Apply gradual chromatic adaptation to one sRGB frame.

        Args:
            img_srgb: float sRGB image in [0, 1], shape (H, W, 3)
            t: explicit time in seconds. If None, uses the internal clock.

        Returns:
            adapted sRGB image, same shape/dtype as input
        """
        if t is not None:
            time_s = float(np.clip(t, 0.0, self.t_max))
        else:
            time_s = self.t

        # Current display illuminant (the scene light the frame is rendered
        # under). adaptation_bounded clamps the velocity to the largest value
        # that keeps the perceived cast |A - a| within DELTA_T (5 JND) even
        # though the absolute illuminant drift exceeds DELTA_T.
        dst_xyz = get_illuminant_xyz(time_s, self.trajectory, self.velocity)

        # Strictly gradual: never jump the illuminant; the displacement from D65
        # is capped at DELTA_T (5 JND) by get_illuminant_xyz.
        linear = srgb_to_linear(img_srgb)
        adapted_linear = bradford_cat(linear, D65_XYZ, dst_xyz)
        out = linear_to_srgb(adapted_linear)

        if isinstance(img_srgb, np.ndarray):
            out = out.astype(img_srgb.dtype, copy=False)
        return out