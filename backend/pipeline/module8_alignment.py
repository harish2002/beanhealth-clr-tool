"""
BeanHealth CLR Tool — Module 8: Pupil-vs-Corner Alignment
==========================================================

Responsibility:
    A *CLR-free* gross-alignment screening net.  Where the Hirschberg pipeline
    (Modules 3–6) measures the torch reflex displacement and needs a coaxial
    flash, this module measures only the geometric position of each pupil
    relative to its own eye corners (medial/lateral canthi) and compares the
    two eyes.

    This works on ordinary photos that have NO corneal light reflex — e.g.
    school ID photos — because it relies on eyelid/pupil geometry rather than
    a specular highlight.  It cannot produce a clinical angle (degrees) and is
    NOT a substitute for Hirschberg: it can only flag "these two eyes look
    meaningfully asymmetric — capture a torch image for a real measurement."

How it works (per eye):
    • Project the pupil centre onto the inner→outer canthus axis.
    • h_ratio = fractional position along that axis (0 = inner/nasal corner,
      1 = outer/temporal corner).  ~0.5 = centred between the corners.
    • v_ratio = signed perpendicular offset from the canthus line, normalised
      by the inter-canthus width (positive = below the line).
    Symmetric, aligned eyes give near-equal h_ratio and v_ratio between the two
    eyes.  A deviated eye shifts its pupil toward one corner, so the ratios
    diverge.

Pipeline position:  OPTIONAL — runs alongside Modules 4–6.
Failure behaviour:  Never raises.  Returns an AlignmentResult with
                    available=False when corner landmarks are missing.

Author: BeanHealth
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from utils.constants import (
    ALIGN_H_ALIGNED_MAX,
    ALIGN_H_BORDERLINE_MAX,
    ALIGN_V_ALIGNED_MAX,
    ALIGN_V_BORDERLINE_MAX,
)

logger = logging.getLogger(__name__)

Point = Tuple[float, float]
Corners = Optional[Tuple[Point, Point]]   # (inner, outer)

# Verdict constants
ALIGN_ALIGNED    = "ALIGNED"
ALIGN_BORDERLINE = "BORDERLINE"
ALIGN_ASYMMETRIC = "ASYMMETRIC"
ALIGN_UNAVAILABLE = "UNAVAILABLE"


@dataclass
class AlignmentResult:
    """Output of Module 8 — the corner-alignment screening net."""

    available: bool

    # Pupil position along inner→outer canthus axis (0=inner, 1=outer)
    left_h_ratio:  Optional[float] = None
    right_h_ratio: Optional[float] = None
    # Signed perpendicular offset from canthus line / inter-canthus width
    left_v_ratio:  Optional[float] = None
    right_v_ratio: Optional[float] = None

    # Inter-ocular asymmetry of those ratios
    h_asymmetry: Optional[float] = None
    v_asymmetry: Optional[float] = None

    verdict: str = ALIGN_UNAVAILABLE          # ALIGNED / BORDERLINE / ASYMMETRIC
    referral_flag: bool = False               # True → recommend torch capture
    interpretation: str = ""
    flags: List[str] = field(default_factory=list)


def _project_pupil(
    pupil_full: Point,
    inner: Point,
    outer: Point,
) -> Optional[Tuple[float, float]]:
    """
    Project a pupil point onto the inner→outer canthus axis.

    Returns (h_ratio, v_ratio):
        h_ratio = fractional position along the axis (0=inner, 1=outer)
        v_ratio = signed perpendicular distance / axis length
    or None if the canthi coincide (degenerate).
    """
    p = np.array(pupil_full, dtype=float)
    a = np.array(inner, dtype=float)
    b = np.array(outer, dtype=float)
    axis = b - a
    width = float(np.linalg.norm(axis))
    if width < 1e-3:
        return None
    rel = p - a
    h_ratio = float(np.dot(rel, axis) / (width * width))
    # 2-D cross product gives signed perpendicular distance
    cross = axis[0] * rel[1] - axis[1] * rel[0]
    v_ratio = float(cross / (width * width))
    return h_ratio, v_ratio


def compute_corner_alignment(
    left_pupil:  Point,
    right_pupil: Point,
    left_crop_box:  Tuple[int, int, int, int],
    right_crop_box: Tuple[int, int, int, int],
    left_eye_corners:  Corners,
    right_eye_corners: Corners,
    vertical_reliable: bool = True,
) -> AlignmentResult:
    """
    Compute the CLR-free corner-alignment screening verdict.

    Args:
        left_pupil / right_pupil:   pupil centres in CROP-local coords (Module 2).
        left_crop_box / right_crop_box: (x1,y1,x2,y2) crop origins in full image.
        left_eye_corners / right_eye_corners: (inner, outer) canthi in full-image
            coords (Module 1).  None when unavailable (eyes-only fallback).
        vertical_reliable: False when the canthi were estimated from the sclera
            extent (eyes-only fallback) — those corners are pinned to the iris
            centre height, so the vertical axis carries no real signal and must
            not drive the verdict.  The horizontal axis stays valid either way.

    Returns:
        AlignmentResult (never raises).
    """
    if left_eye_corners is None or right_eye_corners is None:
        return AlignmentResult(
            available=False,
            verdict=ALIGN_UNAVAILABLE,
            interpretation=(
                "Eye-corner landmarks were not available for this image "
                "(no full-face detection), so corner alignment could not be "
                "assessed. The Hirschberg reflex result above is unaffected."
            ),
            flags=["alignment_no_corners"],
        )

    # Map pupils from crop-local → full-image coords
    lp_full = (left_pupil[0]  + left_crop_box[0],  left_pupil[1]  + left_crop_box[1])
    rp_full = (right_pupil[0] + right_crop_box[0], right_pupil[1] + right_crop_box[1])

    left_proj  = _project_pupil(lp_full, left_eye_corners[0],  left_eye_corners[1])
    right_proj = _project_pupil(rp_full, right_eye_corners[0], right_eye_corners[1])

    if left_proj is None or right_proj is None:
        return AlignmentResult(
            available=False,
            verdict=ALIGN_UNAVAILABLE,
            interpretation="Eye-corner geometry was degenerate; alignment skipped.",
            flags=["alignment_degenerate"],
        )

    l_h, l_v = left_proj
    r_h, r_v = right_proj
    h_asym = abs(l_h - r_h)
    v_asym = abs(l_v - r_v)

    # When the canthi were estimated from sclera extent (eyes-only fallback),
    # both corners are pinned to the iris-centre height, so the vertical axis
    # carries no real signal — drop it from the verdict to avoid false flags.
    v_for_verdict = v_asym if vertical_reliable else 0.0

    # Verdict — worse of the two axes wins
    if h_asym >= ALIGN_H_BORDERLINE_MAX or v_for_verdict >= ALIGN_V_BORDERLINE_MAX:
        verdict = ALIGN_ASYMMETRIC
        referral = True
    elif h_asym >= ALIGN_H_ALIGNED_MAX or v_for_verdict >= ALIGN_V_ALIGNED_MAX:
        verdict = ALIGN_BORDERLINE
        referral = False
    else:
        verdict = ALIGN_ALIGNED
        referral = False

    interpretation = _interpret(
        verdict, l_h, r_h, l_v, r_v, h_asym, v_asym, vertical_reliable
    )

    logger.info(
        f"[M8] alignment verdict={verdict} h_asym={h_asym:.3f} v_asym={v_asym:.3f} "
        f"L(h={l_h:.2f},v={l_v:.2f}) R(h={r_h:.2f},v={r_v:.2f})"
    )

    return AlignmentResult(
        available=True,
        left_h_ratio=round(l_h, 4),
        right_h_ratio=round(r_h, 4),
        left_v_ratio=round(l_v, 4),
        right_v_ratio=round(r_v, 4),
        h_asymmetry=round(h_asym, 4),
        v_asymmetry=round(v_asym, 4),
        verdict=verdict,
        referral_flag=referral,
        interpretation=interpretation,
    )


def _interpret(
    verdict: str,
    l_h: float, r_h: float,
    l_v: float, r_v: float,
    h_asym: float, v_asym: float,
    vertical_reliable: bool = True,
) -> str:
    """Plain-English summary of the corner-alignment finding."""
    if verdict == ALIGN_ALIGNED:
        if vertical_reliable:
            return (
                "Both pupils sit at near-identical positions relative to their own "
                "eye corners. No gross horizontal or vertical misalignment is visible "
                "in this photo. (Geometric check only — does not rule out small-angle "
                "or intermittent deviations.)"
            )
        return (
            "Both pupils sit at near-identical horizontal positions relative to "
            "their own eye corners. No gross horizontal misalignment is visible in "
            "this photo. (Geometric check only, horizontal axis only — vertical "
            "alignment could not be assessed for this image, and small-angle or "
            "intermittent deviations are not ruled out.)"
        )

    # Identify which eye looks displaced and roughly which direction.
    # When the vertical axis is unreliable (estimated corners), force the
    # horizontal narrative regardless of which raw asymmetry is larger.
    parts: List[str] = []
    if not vertical_reliable or h_asym >= v_asym:
        # Horizontal divergence dominates
        if l_h < r_h:
            shifted, corner = "left", "nasally (inward)" if l_h < 0.5 else "temporally (outward)"
        else:
            shifted, corner = "right", "nasally (inward)" if r_h < 0.5 else "temporally (outward)"
        parts.append(
            f"The {shifted} pupil is shifted {corner} relative to its eye corners "
            f"compared with the fellow eye (horizontal asymmetry {h_asym:.0%} of "
            f"eye width)."
        )
    else:
        higher = "left" if l_v < r_v else "right"
        parts.append(
            f"The {higher} pupil sits higher relative to its eye corners than the "
            f"fellow eye (vertical asymmetry {v_asym:.0%} of eye width), which can "
            f"accompany a vertical deviation."
        )

    if verdict == ALIGN_ASYMMETRIC:
        parts.append(
            "This is a positive screening flag: capture a torch (CLR) image of "
            "this patient for a proper Hirschberg measurement and consider "
            "ophthalmology referral."
        )
    else:  # BORDERLINE
        parts.append(
            "This is a borderline finding. It may reflect head tilt, camera angle, "
            "or epicanthal folds rather than true strabismus — confirm with a "
            "torch (CLR) capture if there is any concern."
        )
    return " ".join(parts)
