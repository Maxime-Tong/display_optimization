"""
Per-frame image optimizer for the Gradual Chromatic Adaptation (GCA) module.

Wraps GradualChromaticOptimizer in a complete per-frame pipeline:

    sRGB frame -> GCA (gradual illuminant shift) -> [optional GUD dimming] -> sRGB

Supports the combined optimization mentioned in gradual.md "组合优化":
GCA first (color temperature gradually shifted), then GUD (gradual uniform
dimming) — combined saving up to ~31% with no statistically significant
perceptual difference.
"""

from __future__ import annotations

import numpy as np

from .color_adaptation import GradualChromaticOptimizer
from .constants import (
    DEFAULT_TRAJECTORY,
    DEFAULT_VELOCITY,
    GUD_ENABLED,
    GUD_TARGET,
    GUD_T_MAX,
    T_MAX,
)


class GradualAdaptationImageOptimizer:
    """
    High-level per-frame optimizer that applies gradual chromatic adaptation
    (optionally combined with gradual uniform dimming).
    """

    def __init__(
        self,
        trajectory: str = DEFAULT_TRAJECTORY,
        velocity: float = DEFAULT_VELOCITY,
        t_max: float = T_MAX,
        gud_target: float | None = None,
        gud_t_max: float = GUD_T_MAX,
        gud_enabled: bool = GUD_ENABLED,
    ):
        """
        Args:
            trajectory: illuminant trajectory name
                ("daylight", "1.47", "1.863", "2.256")
            velocity: u'v' per second advance speed
            t_max: duration (s) of the chromatic adaptation ramp
            gud_target: optional final uniform dimming factor in [0, 1]
                (1.0 = no dimming; 0.5 = halve display brightness); defaults
                to GUD_TARGET from constants
            gud_t_max: duration (s) of the GUD ramp (defaults to GUD_T_MAX)
            gud_enabled: whether to apply gradual uniform dimming
        """
        self.gca = GradualChromaticOptimizer(
            trajectory=trajectory,
            velocity=velocity,
            t_max=t_max,
        )
        self.gud_target = GUD_TARGET if gud_target is None else gud_target
        self.gud_t_max = gud_t_max
        self.gud_enabled = gud_enabled

    def process_frame(
        self,
        img_srgb: np.ndarray,
        dt: float = 1.0 / 90.0,
        t: float | None = None,
    ) -> np.ndarray:
        """
        Process one frame at the current time.

        Args:
            img_srgb: float sRGB image in [0, 1], shape (H, W, 3)
            dt: frame delta time (s) when using the internal clock
            t: explicit time (s). If provided, the internal clock is NOT
                advanced (the caller controls the timeline).

        Returns:
            optimized sRGB image, same shape as input
        """
        if t is None:
            t = self.gca.advance(dt)

        out = self.gca.apply_to_frame(img_srgb, t=t)

        # Optional gradual uniform dimming (GUD): combined GUD + GCA mode.
        # GCA first (chromatic shift), then GUD (uniform dimming) per
        # gradual.md 组合优化 — combined saving up to ~31%.
        if self.gud_enabled and self.gud_target is not None:
            t_norm = float(np.clip(t / max(self.gud_t_max, 1e-9), 0.0, 1.0))
            # GUD ramps brightness toward the target (e.g. 0.85 => 15% dimmer).
            dim = 1.0 + (self.gud_target - 1.0) * t_norm
            out = out * dim
            out = np.clip(out, 0.0, 1.0)

        return out

    def reset(self, t: float = 0.0) -> None:
        """Reset the internal GCA clock."""
        self.gca.reset(t)