"""
Illuminant trajectories for the Gradual Chromatic Adaptation (GCA) module.

Implements gradual.md "阶段一：确定省电照明轨迹":
  - Daylight locus (CCT ramp from D65 toward warmer white)
  - Three linear trajectories in CIE u'v' space (angles 1.47 / 1.863 / 2.256 rad)

All trajectories share the recommended constraint |A(t) - a(t)| <= DELTA_T
(5 JND ~ 0.02 u'v'), and advance at a constant velocity.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .colorspace import d65_upvp, upvp_to_xyz
from .constants import (
    ADAPTATION_PARAMS,
    DAYLIGHT_CCT_END,
    DAYLIGHT_CCT_START,
    DEFAULT_TRAJECTORY,
    DEFAULT_VELOCITY,
    DELTA_T,
    T_MAX,
    TRAJECTORY_ANGLES_RAD,
)

# Daylight locus endpoints as CCTs (deg K).  The locus is computed by
# interpolating CCT linearly from D65 down to a warmer white over T_MAX,
# which produces a predominantly yellow-green chromatic shift.
_DAYLIGHT_CCTS = [DAYLIGHT_CCT_START, DAYLIGHT_CCT_END]


def _cct_to_upvp(cct: float) -> np.ndarray:
    """
    Approximate CIE 1976 u'v' chromaticity for a correlated color temperature
    (Kelvin) using the Robertson-style polynomial approximation.

    Reference: Krystek (1985) — "An algorithm to calculate correlated colour
    temperature", Phys. Med. Biol. 30 (1985) 89-92.
    """
    if cct <= 4000.0:
        u = 0.179187 * (cct ** -1) + 0.0671979
        v = 0.260743 * (cct ** -1) + 0.134662
    else:
        u = 0.121453 / cct + 0.197382
        v = 0.027980 / cct + 0.133581
    return np.array([u, v], dtype=np.float64)


def _daylight_endpoint_upvp() -> np.ndarray:
    """u'v' chromaticity at the warm end of the daylight ramp."""
    return _cct_to_upvp(DAYLIGHT_CCT_END)


def _normalized_dir(upvp: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(upvp)
    return upvp / (norm + 1e-12)


def trajectory_direction(name: str) -> np.ndarray:
    """
    Return the unit direction vector (in u'v' space) for a trajectory.

    Args:
        name: one of "daylight", "1.47", "1.863", "2.256"

    Returns:
        normalized 2-vector (du', dv')
    """
    key = str(name)
    if key == "daylight":
        # Direction toward the warm (yellow-green) end of the daylight locus.
        start = d65_upvp()
        end = _daylight_endpoint_upvp()
        return _normalized_dir(end - start)
    if key in TRAJECTORY_ANGLES_RAD:
        angle = TRAJECTORY_ANGLES_RAD[key]
        # u'v' displacement along the trajectory angle (yellow-green sector).
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    raise ValueError(
        f"Unknown trajectory '{name}'. "
        f"Options: daylight, {sorted(TRAJECTORY_ANGLES_RAD.keys())}"
    )


def trajectory_end_upvp(name: str) -> np.ndarray:
    """
    Return the u'v' chromaticity reached at t = T_MAX for a trajectory,
    clamped to the DELTA_T constraint.
    """
    dir_vec = trajectory_direction(name)
    max_shift = DELTA_T
    return d65_upvp() + dir_vec * max_shift


def compute_max_adaptation_velocity(
    trajectory: str = DEFAULT_TRAJECTORY,
    t_max: float = T_MAX,
    delta_t: float = DELTA_T,
    k1: float | None = None,
    k2: float | None = None,
) -> float:
    """
    Largest constant illuminant velocity (u'v'/s) such that the PERCEIVED
    mismatch is never perceptible:

        max_t ||A(t) - a(t)|| <= delta_t

    where the user's adaptation state evolves as  a'(t) = k1 (k2 A(t) - a(t)).

    For a constant velocity v, the mismatch grows linearly in time and the
    worst case (at t = t_max) is:
        ||A - a||(t_max) = v/k1 + (1 - k2) * v * t_max + (k2 - 1)*v/k1*(1-e^{-k1 t_max})
    A generous, exact-for-k2=1 upper bound used here is
        v/k1 + (1 - k2) * v * t_max
    so the constraint gives
        v_max = delta_t / (1/k1 + (1 - k2) * t_max)

    This is the "快速求速方法" from gradual.md 阶段三: because the visual
    system ADAPTS (a(t) tracks A(t) with lag), the illuminant can drift much
    further than delta_t while the perceived cast stays within ~delta_t.
    """
    params = ADAPTATION_PARAMS.get(
        str(trajectory), ADAPTATION_PARAMS[DEFAULT_TRAJECTORY]
    )
    k1 = params["k1"] if k1 is None else float(k1)
    k2 = params["k2"] if k2 is None else float(k2)

    t_max = float(t_max)
    delta_t = float(delta_t)
    denom = 1.0 / max(k1, 1e-9) + (1.0 - k2) * t_max
    return float(delta_t / denom)


def get_illuminant_upvp(
    t: float,
    trajectory: str = DEFAULT_TRAJECTORY,
    velocity: float = DEFAULT_VELOCITY,
    adaptation_bounded: bool = True,
) -> np.ndarray:
    """
    Compute the current illuminant white point in CIE u'v' space at time ``t``.

    A(t) = D65 + shift(t) * trajectory_dir

    Args:
        t: elapsed time in seconds (monotonically increasing; clipped to [0, T_MAX])
        trajectory: trajectory name ("daylight", "1.47", "1.863", "2.256")
        velocity: u'v' per second advance speed
        adaptation_bounded: if True (default), velocity is clamped to the
            adaptation-derived maximum so the perceived mismatch
            ||A(t) - a(t)|| stays within DELTA_T (5 JND) even though the
            absolute illuminant displacement exceeds DELTA_T. If False, the
            conservative old behavior is used (absolute shift capped at
            DELTA_T).

    Returns:
        (u', v') array
    """
    t = float(np.clip(t, 0.0, T_MAX))
    dir_vec = trajectory_direction(trajectory)

    if adaptation_bounded:
        # Exploit chromatic adaptation: the perceived quantity is the mismatch
        # |A - a|, not |A - D65|. Clamp to the largest safe velocity.
        v = float(min(float(velocity), compute_max_adaptation_velocity(trajectory)))
        shift = v * t
    else:
        shift = min(DELTA_T, float(velocity) * t)

    return d65_upvp() + dir_vec * shift


def get_illuminant_xyz(
    t: float,
    trajectory: str = DEFAULT_TRAJECTORY,
    velocity: float = DEFAULT_VELOCITY,
    adaptation_bounded: bool = True,
) -> np.ndarray:
    """
    Compute the current illuminant white point as XYZ (Y=1) at time ``t``.
    """
    upvp = get_illuminant_upvp(t, trajectory, velocity, adaptation_bounded)
    return upvp_to_xyz(upvp, y=1.0)
