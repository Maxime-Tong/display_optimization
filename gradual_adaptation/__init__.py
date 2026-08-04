"""
Gradual Chromatic Adaptation (GCA) package.

Implements the power-saving technique from gradual.md:
slowly shifting the display white point (CIE u'v') from D65 along a
yellow-green trajectory while the user's visual system chromatically
adapts, reducing OLED blue-subpixel power with no perceptible artifact.

Public API:
    - GradualChromaticOptimizer      (per-frame CAT)
    - GradualAdaptationImageOptimizer (full pipeline, optional GUD)
    - compute_power_reduction         (energy metric)
"""

from __future__ import annotations

from .color_adaptation import GradualChromaticOptimizer
from .image_optimizer import GradualAdaptationImageOptimizer
from .power_model import compute_power_reduction, power_model

__all__ = [
    "GradualChromaticOptimizer",
    "GradualAdaptationImageOptimizer",
    "compute_power_reduction",
    "power_model",
]