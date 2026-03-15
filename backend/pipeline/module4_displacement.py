"""
BeanHealth CLR Tool — Module 4: Displacement Measurement
=========================================================

Responsibility:
    Given the pupil centre and CLR position for each eye (from Modules 2 & 3),
    calculate how far and in which direction the CLR is displaced from the pupil.

    Displacement is normalised by iris radius so the measurement is
    scale-invariant — independent of how far the phone is from the face.

Key formula:
    dx            = clr_x - pupil_x
    dy            = clr_y - pupil_y
    magnitude     = sqrt(dx² + dy²)          [pixels, raw]
    normalised    = magnitude / iris_radius  [scale-invariant ratio]
    direction     = atan2(dy, dx)            [radians]
    direction_label = nasal / temporal / superior / inferior

Clinical note:
    In a normal eye the CLR sits at or near the pupil centre (normalised ≈ 0).
    In a squinting eye the CLR is displaced from the pupil centre.
    The normalised value feeds directly into Module 5's Hirschberg conversion.

Pipeline position:  FOURTH — depends on Modules 2 (pupil) and 3 (CLR).
Failure behaviour:  PipelineError if iris_radius is 0 (prevents division by zero).

Author: BeanHealth
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from utils.constants import (
    DIRECTION_INFERIOR,
    DIRECTION_NASAL,
    DIRECTION_SUPERIOR,
    DIRECTION_TEMPORAL,
)
from utils.exceptions import PipelineError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class DisplacementResult:
    """
    Displacement of the CLR from the pupil centre for both eyes.

    All pixel values are in crop-local coordinates.
    Normalised values are dimensionless ratios (0.0 = centred, 1.0 = displaced by full iris radius).
    """

    # Raw pixel displacement vectors
    left_dx:  float   # positive = CLR is to the RIGHT of pupil in crop
    left_dy:  float   # positive = CLR is BELOW pupil in crop
    right_dx: float
    right_dy: float

    # Magnitudes
    left_displacement_px:  float   # Euclidean distance in pixels
    right_displacement_px: float

    # Normalised (scale-invariant) — key clinical value
    left_displacement_norm:  float  # magnitude / iris_radius
    right_displacement_norm: float

    # Directions in anatomical terms (accounts for left/right eye mirroring)
    left_direction:  str   # nasal / temporal / superior / inferior
    right_direction: str

    # Raw angle in radians (for debugging / visualisation)
    left_angle_rad:  float
    right_angle_rad: float

    # Propagated flags from upstream modules
    flags: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Direction mapping helpers
# ─────────────────────────────────────────────────────────────

def _angle_to_cardinal(angle_rad: float) -> str:
    """
    Convert a raw atan2 angle to one of four cardinal directions.

    In crop-local coordinates (x right, y down):
        right  (+x) → angle ≈ 0
        down   (+y) → angle ≈ π/2
        left   (-x) → angle ≈ ±π
        up     (-y) → angle ≈ -π/2

    Returns one of: "right", "left", "up", "down"
    (anatomical labelling happens in _cardinal_to_anatomical)
    """
    # Normalise angle to [-π, π]
    while angle_rad > math.pi:
        angle_rad -= 2 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2 * math.pi

    # 45° sectors
    if -math.pi / 4 <= angle_rad < math.pi / 4:
        return "right"
    elif math.pi / 4 <= angle_rad < 3 * math.pi / 4:
        return "down"
    elif angle_rad >= 3 * math.pi / 4 or angle_rad < -3 * math.pi / 4:
        return "left"
    else:  # -3π/4 <= angle < -π/4
        return "up"


def _cardinal_to_anatomical(cardinal: str, eye: str) -> str:
    """
    Convert a crop-space direction to an anatomical direction label.

    The mapping depends on which eye we're looking at because the nasal
    side of the left eye is to the right in the image, and vice versa.

    Image convention (Photo Booth / front camera — mirrored):
        Left eye  in image → person's RIGHT eye (nasal = image-right)
        Right eye in image → person's LEFT eye  (nasal = image-left)

    For back camera (production app — not mirrored):
        Left eye  in image → person's LEFT eye  (nasal = image-right)
        Right eye in image → person's RIGHT eye (nasal = image-left)

    We assume FRONT camera (mirrored) as per current test setup.
    This can be toggled with the `mirror` parameter in measure_displacement().

    Vertical directions are consistent:
        up   → superior (CLR above pupil)
        down → inferior (CLR below pupil)
    """
    if cardinal == "up":
        return DIRECTION_SUPERIOR
    if cardinal == "down":
        return DIRECTION_INFERIOR

    # Horizontal — depends on which eye and camera mirroring
    if eye == "left":
        # In a mirrored (front-camera) image, left eye's nasal side is image-RIGHT
        return DIRECTION_NASAL if cardinal == "right" else DIRECTION_TEMPORAL
    else:
        # Right eye's nasal side is image-LEFT
        return DIRECTION_NASAL if cardinal == "left" else DIRECTION_TEMPORAL


# ─────────────────────────────────────────────────────────────
# Core displacement calculation — unit-testable
# ─────────────────────────────────────────────────────────────

def measure_displacement(
    pupil: Tuple[float, float],
    clr: Tuple[float, float],
    iris_radius: float,
    eye: str = "left",
    mirror: bool = True,
) -> dict:
    """
    Compute displacement of CLR from pupil centre for a single eye.

    Args:
        pupil:       (x, y) pupil centre in crop-local coordinates
        clr:         (x, y) CLR position in crop-local coordinates
        iris_radius: iris radius in pixels (for normalisation)
        eye:         "left" or "right" — used for anatomical direction labelling
        mirror:      True if image is front-camera (mirrored); False for back camera

    Returns:
        dict with keys:
            dx, dy, magnitude, normalised, direction, angle_rad

    Raises:
        PipelineError: if iris_radius <= 0
    """
    if iris_radius <= 0:
        raise PipelineError(
            f"iris_radius must be > 0, got {iris_radius}. "
            "Cannot normalise displacement without a valid iris size."
        )

    dx = clr[0] - pupil[0]
    dy = clr[1] - pupil[1]
    magnitude = math.sqrt(dx * dx + dy * dy)
    normalised = magnitude / iris_radius
    angle_rad = math.atan2(dy, dx)

    cardinal = _angle_to_cardinal(angle_rad)

    # For back camera (not mirrored), flip horizontal direction
    if not mirror:
        if cardinal == "right":
            cardinal = "left"
        elif cardinal == "left":
            cardinal = "right"

    direction = _cardinal_to_anatomical(cardinal, eye)

    logger.debug(
        f"[M4] {eye} eye: pupil={pupil}, clr={clr}, "
        f"dx={dx:.2f}, dy={dy:.2f}, mag={magnitude:.2f}px, "
        f"norm={normalised:.4f}, dir={direction}"
    )

    return {
        "dx":         dx,
        "dy":         dy,
        "magnitude":  magnitude,
        "normalised": normalised,
        "direction":  direction,
        "angle_rad":  angle_rad,
    }


# ─────────────────────────────────────────────────────────────
# Public API — process both eyes
# ─────────────────────────────────────────────────────────────

def compute_displacement(
    left_pupil:   Tuple[float, float],
    right_pupil:  Tuple[float, float],
    left_clr:     Tuple[float, float],
    right_clr:    Tuple[float, float],
    left_iris_radius:  float,
    right_iris_radius: float,
    upstream_flags: Optional[List[str]] = None,
    mirror: bool = True,
) -> DisplacementResult:
    """
    Compute CLR displacement from pupil centre for both eyes.

    Args:
        left_pupil:         (x, y) left pupil centre in crop coords
        right_pupil:        (x, y) right pupil centre in crop coords
        left_clr:           (x, y) left CLR position in crop coords
        right_clr:          (x, y) right CLR position in crop coords
        left_iris_radius:   left iris radius in pixels
        right_iris_radius:  right iris radius in pixels
        upstream_flags:     flags carried forward from Modules 1–3
        mirror:             True for front camera (mirrored image)

    Returns:
        DisplacementResult

    Raises:
        PipelineError: if either iris_radius is zero or negative
    """
    flags: List[str] = list(upstream_flags or [])

    # Compute per eye
    left  = measure_displacement(left_pupil,  left_clr,  left_iris_radius,  eye="left",  mirror=mirror)
    right = measure_displacement(right_pupil, right_clr, right_iris_radius, eye="right", mirror=mirror)

    # Warn if displacement is extremely large (> 1 iris radius) — may indicate bad detection
    if left["normalised"] > 1.0:
        flags.append("large_displacement_left")
        logger.warning(f"[M4] Left eye displacement extremely large: {left['normalised']:.3f}")
    if right["normalised"] > 1.0:
        flags.append("large_displacement_right")
        logger.warning(f"[M4] Right eye displacement extremely large: {right['normalised']:.3f}")

    result = DisplacementResult(
        left_dx=left["dx"],
        left_dy=left["dy"],
        right_dx=right["dx"],
        right_dy=right["dy"],
        left_displacement_px=left["magnitude"],
        right_displacement_px=right["magnitude"],
        left_displacement_norm=left["normalised"],
        right_displacement_norm=right["normalised"],
        left_direction=left["direction"],
        right_direction=right["direction"],
        left_angle_rad=left["angle_rad"],
        right_angle_rad=right["angle_rad"],
        flags=flags,
    )

    logger.info(
        f"[M4] Left:  norm={result.left_displacement_norm:.4f}, dir={result.left_direction} | "
        f"Right: norm={result.right_displacement_norm:.4f}, dir={result.right_direction}"
    )

    return result
