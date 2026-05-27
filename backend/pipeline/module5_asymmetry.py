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

    asymmetry_score:      Scalar legacy field — |left_norm - right_norm|.
                          Retained for backwards compatibility; NOT used for severity.
    asymmetry_degrees:    Bilateral asymmetry vector magnitude converted to clinical
                          degrees.  THIS drives severity.  It cancels kappa because
                          kappa is anatomically mirrored: in true alignment the vector
                          difference between both eyes ≈ 0 regardless of absolute
                          kappa offset.  In true strabismus the vector difference is
                          non-zero in proportion to the ocular deviation.
    bav_nasal:            Nasal component of the bilateral asymmetry vector (normalised).
                          Positive = left eye more nasal than right (esotropia signal).
    bav_vertical:         Vertical component of the bilateral asymmetry vector.
                          Positive = left eye more superior than right (hypertropia signal).
    dominant_eye:         Which eye has the larger absolute displacement.
    deviation_degrees:    Hirschberg angle of the dominant eye.  Reference only — NOT
                          used for severity (includes kappa offset).
    deviation_mm:         Physical displacement of dominant eye in mm.
    severity:             NORMAL / MILD / MODERATE / SEVERE  (from asymmetry_degrees)
    flags:                Propagated flags from upstream modules.
    """
    asymmetry_score:    float   # |left_norm - right_norm| (legacy scalar)
    asymmetry_degrees:  float   # |BAV| × IRIS_RADIUS_MM × HIRSCHBERG_CONSTANT
    bav_nasal:          float   # bilateral asymmetry vector — nasal axis
    bav_vertical:       float   # bilateral asymmetry vector — vertical axis
    dominant_eye:       str     # "left" | "right" | "equal"
    deviation_degrees:  float   # Hirschberg angle of dominant eye (reference only)
    deviation_mm:       float   # physical displacement in mm
    severity:           str     # NORMAL / MILD / MODERATE / SEVERE
    flags:              List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Core computations — individually unit-testable
# ─────────────────────────────────────────────────────────────

def compute_bilateral_asymmetry_vector(
    left_dx: float,  left_dy: float,  left_iris_radius: float,
    right_dx: float, right_dy: float, right_iris_radius: float,
    mirror: bool = True,
) -> dict:
    """
    Compute the bilateral asymmetry vector (BAV) in a canonical anatomical frame.

    Canonical frame:
        nasal+    — displacement toward the nose (positive)
        temporal− — displacement away from the nose
        superior+ — displacement upward (positive)
        inferior− — displacement downward

    Coordinate mapping (front camera, mirrored=True):
        Left  eye: nasal = image-right (+dx),  superior = image-up (−dy)
        Right eye: nasal = image-left  (−dx),  superior = image-up (−dy)

    For a fixating patient with kappa angle κ:
        left_canonical  ≈ (κ, 0)
        right_canonical ≈ (κ, 0)
        BAV             ≈ (0, 0)   ← kappa cancels

    For a patient with esotropia (left eye turns inward by angle α):
        left_canonical  ≈ (κ + α, 0)
        right_canonical ≈ (κ, 0)
        BAV             ≈ (α, 0)   ← true deviation isolated

    Args:
        left_dx, left_dy:   raw pixel displacement vector for left eye (crop coords)
        left_iris_radius:   left iris radius in pixels (for normalisation)
        right_dx, right_dy: raw pixel displacement vector for right eye
        right_iris_radius:  right iris radius in pixels
        mirror:             True for front camera (mirrored image)

    Returns:
        dict with keys:
            bav_nasal, bav_vertical,   — canonical vector components (normalised)
            bav_magnitude,             — Euclidean magnitude (the asymmetry score)
            left_norm_vec, right_norm_vec,  — per-eye canonical vectors (for debug)
            asymmetry_score,           — legacy scalar |‖L‖ − ‖R‖|
            dominant_eye, dominant_norm
    """
    # Normalise raw pixel vectors by iris radius → scale-invariant
    ln_x = left_dx  / left_iris_radius   # left  normalised image-x
    ln_y = left_dy  / left_iris_radius   # left  normalised image-y  (positive = down)
    rn_x = right_dx / right_iris_radius
    rn_y = right_dy / right_iris_radius

    # Convert to canonical anatomical frame
    # Front camera (mirrored): left eye nasal = image+x; right eye nasal = image−x
    # Back camera (not mirrored): left eye nasal = image−x (TODO: flip if needed)
    if mirror:
        left_nasal    = +ln_x          # image-right = nasal for left eye
        right_nasal   = -rn_x         # image-left  = nasal for right eye
    else:
        left_nasal    = -ln_x
        right_nasal   = +rn_x

    left_vertical  = -ln_y            # image-up = superior (positive) for both eyes
    right_vertical = -rn_y

    # Bilateral asymmetry vector = left_canonical − right_canonical
    # Kappa angle (symmetric in both eyes) cancels out here
    bav_nasal    = left_nasal    - right_nasal
    bav_vertical = left_vertical - right_vertical
    bav_magnitude = math.sqrt(bav_nasal ** 2 + bav_vertical ** 2)

    # Legacy scalar asymmetry (magnitude difference — kept for backwards compat)
    left_norm_scalar  = math.sqrt(ln_x ** 2 + ln_y ** 2)
    right_norm_scalar = math.sqrt(rn_x ** 2 + rn_y ** 2)
    asymmetry_score   = abs(left_norm_scalar - right_norm_scalar)

    # Dominant eye = the one with larger absolute displacement
    if left_norm_scalar > right_norm_scalar:
        dominant_eye  = "left"
        dominant_norm = left_norm_scalar
    elif right_norm_scalar > left_norm_scalar:
        dominant_eye  = "right"
        dominant_norm = right_norm_scalar
    else:
        dominant_eye  = "equal"
        dominant_norm = left_norm_scalar

    logger.debug(
        f"[M5] BAV: nasal={bav_nasal:.4f}, vertical={bav_vertical:.4f}, "
        f"magnitude={bav_magnitude:.4f} | "
        f"left_canon=({left_nasal:.4f},{left_vertical:.4f}), "
        f"right_canon=({right_nasal:.4f},{right_vertical:.4f})"
    )

    return {
        "bav_nasal":        bav_nasal,
        "bav_vertical":     bav_vertical,
        "bav_magnitude":    bav_magnitude,
        "left_norm_vec":    (left_nasal,  left_vertical),
        "right_norm_vec":   (right_nasal, right_vertical),
        "asymmetry_score":  asymmetry_score,
        "dominant_eye":     dominant_eye,
        "dominant_norm":    dominant_norm,
    }


def compute_asymmetry(
    left_norm: float,
    right_norm: float,
) -> dict:
    """
    Legacy scalar asymmetry — |left_norm - right_norm|.
    Kept for unit-test compatibility.  New code should use
    compute_bilateral_asymmetry_vector() instead.
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
        dominant_norm = left_norm

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
    # Bilateral asymmetry vector inputs (required for vector-based scoring)
    left_dx:  Optional[float] = None,
    left_dy:  Optional[float] = None,
    right_dx: Optional[float] = None,
    right_dy: Optional[float] = None,
    left_iris_radius:  Optional[float] = None,
    right_iris_radius: Optional[float] = None,
    mirror: bool = True,
) -> AsymmetryResult:
    """
    Full Module 5 pipeline: bilateral asymmetry vector + Hirschberg angle + severity.

    When dx/dy/radius arguments are supplied the function uses the bilateral
    asymmetry vector (BAV) — a 2D vector difference in canonical anatomical space
    (nasal+, superior+).  The kappa angle cancels by subtraction, leaving only true
    inter-ocular deviation as the severity signal.

    If vector inputs are omitted (legacy mode) the function falls back to the scalar
    |left_norm − right_norm| approach.

    Args:
        left_displacement_norm:  normalised CLR magnitude for left eye (Module 4)
        right_displacement_norm: normalised CLR magnitude for right eye (Module 4)
        upstream_flags:          flags carried forward from Modules 1–4
        left_dx, left_dy:        raw pixel CLR displacement vector, left eye
        right_dx, right_dy:      raw pixel CLR displacement vector, right eye
        left_iris_radius:        left iris radius in pixels
        right_iris_radius:       right iris radius in pixels
        mirror:                  True for front-facing (mirrored) camera

    Returns:
        AsymmetryResult
    """
    flags: List[str] = list(upstream_flags or [])

    vector_inputs_available = all(
        v is not None for v in
        [left_dx, left_dy, right_dx, right_dy, left_iris_radius, right_iris_radius]
    )

    if vector_inputs_available:
        # ── Bilateral asymmetry vector path (preferred) ──────────────────────
        bav = compute_bilateral_asymmetry_vector(
            left_dx=left_dx,   left_dy=left_dy,   left_iris_radius=left_iris_radius,   # type: ignore[arg-type]
            right_dx=right_dx, right_dy=right_dy, right_iris_radius=right_iris_radius, # type: ignore[arg-type]
            mirror=mirror,
        )

        asymmetry_score  = bav["asymmetry_score"]   # legacy scalar (retained)
        bav_nasal        = bav["bav_nasal"]
        bav_vertical     = bav["bav_vertical"]
        bav_magnitude    = bav["bav_magnitude"]      # drives severity
        dominant_eye     = bav["dominant_eye"]
        dominant_norm    = bav["dominant_norm"]

        flags.append("bav_mode")
        logger.info(
            f"[M5] BAV mode: nasal={bav_nasal:.4f}, vertical={bav_vertical:.4f}, "
            f"|BAV|={bav_magnitude:.4f}"
        )
    else:
        # ── Scalar fallback (no vector data supplied) ────────────────────────
        flags.append("scalar_asymmetry_mode")
        asym         = compute_asymmetry(left_displacement_norm, right_displacement_norm)
        asymmetry_score = asym["asymmetry_score"]
        bav_nasal    = 0.0
        bav_vertical = 0.0
        bav_magnitude = asymmetry_score
        dominant_eye  = asym["dominant_eye"]
        dominant_norm = asym["dominant_norm"]
        logger.warning("[M5] Falling back to scalar asymmetry — vector inputs missing")

    # Step 1: Severity from BAV magnitude (kappa-cancelled signal)
    #
    # Clinical rationale:
    #   |BAV| is the magnitude of (left_canonical − right_canonical) in anatomical space.
    #   Because kappa is anatomically symmetric, it appears identically in both canonical
    #   vectors and cancels on subtraction → |BAV| ≈ 0 for a fixating patient with any
    #   kappa offset.  Only genuine inter-ocular deviation contributes to |BAV|.
    #
    #   Scalar approach limitation: |‖L‖ − ‖R‖| loses direction — if one eye is displaced
    #   nasally and the other temporally by the same magnitude, the scalar gives 0 (appears
    #   symmetric) while |BAV| correctly gives 2× the displacement (maximally asymmetric).
    asym_angle = compute_angle(bav_magnitude)
    asymmetry_degrees = asym_angle["deviation_degrees"]

    # Step 2: Hirschberg angle of dominant eye (kept as reference display value only)
    angle = compute_angle(dominant_norm)

    # Step 3: Severity
    sev = compute_angle_severity(asymmetry_degrees)

    # Flag very low displacement in both eyes → almost certainly NORMAL
    if left_displacement_norm < 0.05 and right_displacement_norm < 0.05:
        flags.append("very_low_displacement_both")

    result = AsymmetryResult(
        asymmetry_score=round(asymmetry_score, 4),
        asymmetry_degrees=round(asymmetry_degrees, 2),
        bav_nasal=round(bav_nasal, 4),
        bav_vertical=round(bav_vertical, 4),
        dominant_eye=dominant_eye,
        deviation_degrees=round(angle["deviation_degrees"], 2),
        deviation_mm=round(angle["deviation_mm"], 3),
        severity=sev["severity"],
        flags=flags,
    )

    logger.info(
        f"[M5] |BAV|={bav_magnitude:.4f} → {asymmetry_degrees:.2f}° → {result.severity} | "
        f"dominant={dominant_eye}, abs_angle={result.deviation_degrees:.2f}°"
    )

    return result
