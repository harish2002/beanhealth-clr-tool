"""
BeanHealth CLR Tool — Module 5: Asymmetry Score & Hirschberg Angle
===================================================================

Responsibility:
    Given the normalised displacement of both eyes (from Module 4), compute:

      1. Asymmetry score — the KEY clinical red flag.
         A large displacement in ONE eye relative to the other indicates squint.
         Symmetric displacement (even if large) is less clinically significant.

      2. Hirschberg angle — converts normalised displacement to clinical degrees
         using the established ophthalmology formula:
             1mm of CLR displacement ≈ 7° of ocular deviation

      3. Severity tier — maps the deviation angle to:
             NORMAL / MILD / MODERATE / SEVERE

Clinical background:
    The Hirschberg test is the standard method for estimating ocular deviation.
    The formula: angle = displacement_mm × 7°/mm
    where displacement_mm = normalised_displacement × IRIS_RADIUS_MM (5.75mm average)

    We use the MORE DISPLACED eye's normalised value for the angle calculation,
    because that eye is the one showing the squint.

    The asymmetry score compares both eyes:
        asymmetry = |left_norm - right_norm|
    Near 0 → symmetric (both eyes equal, regardless of absolute displacement)
    Large  → one eye significantly more displaced than the other → squint

Severity thresholds (from CLAUDE.md spec):
    < 5°    → NORMAL
    5–15°   → MILD
    15–30°  → MODERATE
    ≥ 30°   → SEVERE

Pipeline position:  FIFTH — depends on Module 4 output.
Failure behaviour:  No failure modes — pure arithmetic, always returns a result.

Author: BeanHealth
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

from utils.constants import (
    HIRSCHBERG_CONSTANT,
    IRIS_RADIUS_MM,
    SEVERITY_MILD,
    SEVERITY_MODERATE,
    SEVERITY_NORMAL,
    SEVERITY_SEVERE,
    SEVERITY_MILD_DEG,
    SEVERITY_MODERATE_DEG,
    SEVERITY_SEVERE_DEG,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class AsymmetryResult:
    """
    Asymmetry score and Hirschberg angle for the current image.

    asymmetry_score:      The primary clinical red flag (0 = perfect symmetry).
    dominant_eye:         Which eye has the larger displacement (the squinting eye).
    deviation_degrees:    Hirschberg angle of the dominant eye in clinical degrees.
    deviation_mm:         Displacement in mm using average iris radius.
    severity:             NORMAL / MILD / MODERATE / SEVERE
    flags:                Propagated flags from upstream modules.
    """
    asymmetry_score:    float   # |left_norm - right_norm|
    dominant_eye:       str     # "left" | "right" | "equal"
    deviation_degrees:  float   # Hirschberg angle in degrees
    deviation_mm:       float   # physical displacement in mm
    severity:           str     # NORMAL / MILD / MODERATE / SEVERE
    flags:              List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Core computations — individually unit-testable
# ─────────────────────────────────────────────────────────────

def compute_asymmetry(
    left_norm: float,
    right_norm: float,
) -> dict:
    """
    Compute asymmetry score and determine dominant eye.

    Args:
        left_norm:  normalised CLR displacement for the left eye (0.0–1.0+)
        right_norm: normalised CLR displacement for the right eye

    Returns:
        dict with keys: asymmetry_score, dominant_eye, dominant_norm
    """
    asymmetry_score = abs(left_norm - right_norm)

    if left_norm > right_norm:
        dominant_eye  = "left"
        dominant_norm = left_norm
    elif right_norm > left_norm:
        dominant_eye  = "right"
        dominant_norm = right_norm
    else:
        dominant_eye  = "equal"
        dominant_norm = left_norm   # both equal, use either

    logger.debug(
        f"[M5] left_norm={left_norm:.4f}, right_norm={right_norm:.4f}, "
        f"asymmetry={asymmetry_score:.4f}, dominant={dominant_eye}"
    )

    return {
        "asymmetry_score": asymmetry_score,
        "dominant_eye":    dominant_eye,
        "dominant_norm":   dominant_norm,
    }


def compute_angle(displacement_norm: float) -> dict:
    """
    Convert normalised displacement to clinical angle using Hirschberg formula.

    Formula:
        displacement_mm = displacement_norm × IRIS_RADIUS_MM
        angle_degrees   = displacement_mm   × HIRSCHBERG_CONSTANT

    Args:
        displacement_norm: scale-invariant displacement ratio from Module 4

    Returns:
        dict with keys: deviation_degrees, deviation_mm
    """
    displacement_mm = displacement_norm * IRIS_RADIUS_MM
    angle_degrees   = displacement_mm   * HIRSCHBERG_CONSTANT

    logger.debug(
        f"[M5] Hirschberg: norm={displacement_norm:.4f} → "
        f"{displacement_mm:.3f}mm → {angle_degrees:.2f}°"
    )

    return {
        "deviation_degrees": angle_degrees,
        "deviation_mm":      displacement_mm,
    }


def compute_angle_severity(angle_degrees: float) -> dict:
    """
    Map a deviation angle to a severity tier.

    Thresholds (from clinical spec):
        < 5°    → NORMAL
        5–15°   → MILD
        15–30°  → MODERATE
        ≥ 30°   → SEVERE

    Args:
        angle_degrees: Hirschberg angle in degrees

    Returns:
        dict with key: severity (str)
    """
    if angle_degrees < SEVERITY_MILD_DEG:
        severity = SEVERITY_NORMAL
    elif angle_degrees < SEVERITY_MODERATE_DEG:
        severity = SEVERITY_MILD
    elif angle_degrees < SEVERITY_SEVERE_DEG:
        severity = SEVERITY_MODERATE
    else:
        severity = SEVERITY_SEVERE

    logger.debug(f"[M5] angle={angle_degrees:.2f}° → severity={severity}")

    return {"severity": severity}


# ─────────────────────────────────────────────────────────────
# Public API — full Module 5 computation
# ─────────────────────────────────────────────────────────────

def compute_asymmetry_and_angle(
    left_displacement_norm:  float,
    right_displacement_norm: float,
    upstream_flags: Optional[List[str]] = None,
) -> AsymmetryResult:
    """
    Full Module 5 pipeline: asymmetry score + Hirschberg angle + severity.

    Args:
        left_displacement_norm:  normalised CLR displacement for left eye (Module 4)
        right_displacement_norm: normalised CLR displacement for right eye (Module 4)
        upstream_flags:          flags carried forward from Modules 1–4

    Returns:
        AsymmetryResult
    """
    flags: List[str] = list(upstream_flags or [])

    # Step 1: Asymmetry and dominant eye
    asym = compute_asymmetry(left_displacement_norm, right_displacement_norm)

    # Step 2: Hirschberg angle from dominant eye's displacement
    angle = compute_angle(asym["dominant_norm"])

    # Step 3: Severity tier
    sev = compute_angle_severity(angle["deviation_degrees"])

    # Additional flag: both eyes have very low displacement → likely NORMAL but low confidence
    if left_displacement_norm < 0.05 and right_displacement_norm < 0.05:
        flags.append("very_low_displacement_both")

    result = AsymmetryResult(
        asymmetry_score=asym["asymmetry_score"],
        dominant_eye=asym["dominant_eye"],
        deviation_degrees=angle["deviation_degrees"],
        deviation_mm=angle["deviation_mm"],
        severity=sev["severity"],
        flags=flags,
    )

    logger.info(
        f"[M5] asymmetry={result.asymmetry_score:.4f}, "
        f"dominant={result.dominant_eye}, "
        f"angle={result.deviation_degrees:.2f}°, "
        f"severity={result.severity}"
    )

    return result
