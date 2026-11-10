"""
BeanHealth CLR Tool — Module 2b: Iris-edge centre
=================================================

Responsibility:
    Replace Module 2's pupil-centre estimate with the centre of a circle fitted
    to the iris edge (limbus), traced on the full-resolution photo.

Why:
    In a dark-brown iris the pupil cannot be told apart from the iris, so
    Module 2 falls back to the darkest patch near the iris landmarks. That
    patch follows lid shadow and framing: re-cropping one photo moved the
    reading by a median of 16 PD (up to 34 PD) on the clinic validation set.
    The iris edge, where dark iris meets white sclera, is high-contrast in
    every eye colour. Its fitted centre moved by under 2 PD under the same
    test.

    The edge is traced only on the left and right sides of the iris
    (±IRIS_EDGE_ARC_DEG of horizontal), where the lids rarely cover it.
    The top and bottom arcs are left out, so the horizontal centre is far
    better fixed than the vertical one. The pipeline therefore reports the
    horizontal deviation only (see Module 5, horizontal_only).

Refusals (never a guess):
    • An eye whose edge cannot be traced → PupilError("iris_edge_not_found_{eye}").
    • Two fitted radii more than IRIS_EDGE_MAX_RADIUS_MISMATCH apart →
      PupilError("iris_size_mismatch"). Both irises are the same physical size,
      so a mismatch means one edge is hidden (lid, turned eye, corneal haze)
      and its centre cannot be trusted.

Pipeline position:  between Module 2 and Module 3.

Author: BeanHealth
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pipeline.module1_detection import EyeDetectionResult
from pipeline.module2_pupil import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM, PupilResult
from utils.constants import (
    IRIS_EDGE_ARC_DEG,
    IRIS_EDGE_HIGH_RESIDUAL,
    IRIS_EDGE_MAX_CENTRE_SHIFT,
    IRIS_EDGE_MAX_RADIUS_MISMATCH,
    IRIS_EDGE_MAX_RESIDUAL,
    IRIS_EDGE_MIN_POINTS,
    IRIS_EDGE_RADIUS_RANGE,
)
from utils.exceptions import PupilError

logger = logging.getLogger(__name__)

_RANK = {CONFIDENCE_LOW: 0, CONFIDENCE_MEDIUM: 1, CONFIDENCE_HIGH: 2}


@dataclass
class IrisEdgeFit:
    """One eye's fitted iris edge, in full-resolution image pixels."""

    centre: Tuple[float, float]
    radius: float
    n_points: int
    residual: float            # median |distance − radius| of kept edge points, pixels


def _fit_circle(pts: np.ndarray) -> Tuple[float, float, float]:
    """Algebraic least-squares circle through (N, 2) points → (cx, cy, r)."""
    x, y = pts[:, 0], pts[:, 1]
    a = np.c_[2 * x, 2 * y, np.ones(len(x))]
    (cx, cy, c), *_ = np.linalg.lstsq(a, x ** 2 + y ** 2, rcond=None)
    return float(cx), float(cy), float(math.sqrt(max(c + cx ** 2 + cy ** 2, 0.0)))


def fit_iris_edge(
    gray: np.ndarray,
    centre_hint: Tuple[float, float],
    radius_hint: float,
) -> Optional[IrisEdgeFit]:
    """
    Trace the iris edge on the left and right arcs and fit a circle to it.

    Rays leave the hinted centre at angles within ±IRIS_EDGE_ARC_DEG of
    horizontal. On each ray the strongest dark→bright step between 0.6 and
    1.45 hinted radii is the edge. Outliers (eyelashes, a lid corner, a
    reflection on the sclera) are trimmed in three passes.

    Args:
        gray:        Full-resolution grayscale image (uint8).
        centre_hint: Iris centre from the face landmarks, same pixel frame.
        radius_hint: Iris radius from the face landmarks, pixels.

    Returns:
        IrisEdgeFit, or None when too few edge points survive.
    """
    r0 = float(radius_hint)
    if r0 < 3:
        return None
    h, w = gray.shape[:2]
    pad = int(math.ceil(1.6 * r0)) + 2
    x0, y0 = max(0, int(centre_hint[0]) - pad), max(0, int(centre_hint[1]) - pad)
    x1, y1 = min(w, int(centre_hint[0]) + pad), min(h, int(centre_hint[1]) + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    roi = cv2.GaussianBlur(gray[y0:y1, x0:x1], (0, 0), max(1.0, r0 / 25)).astype(np.float32)
    cx0, cy0 = centre_hint[0] - x0, centre_hint[1] - y0

    rs = np.linspace(0.6 * r0, 1.45 * r0, 120)
    pts: List[Tuple[float, float]] = []
    for side in (0.0, 180.0):
        for da in np.linspace(-IRIS_EDGE_ARC_DEG, IRIS_EDGE_ARC_DEG, 41):
            a = math.radians(side + da)
            xs = (cx0 + rs * math.cos(a)).astype(np.float32)
            ys = (cy0 + rs * math.sin(a)).astype(np.float32)
            v = cv2.remap(roi, xs[None], ys[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
            dv = np.gradient(v)
            i = int(np.argmax(dv))
            if dv[i] > 2:                                  # a real step, not noise
                pts.append((float(xs[i]), float(ys[i])))
    arr = np.array(pts)
    if len(arr) < IRIS_EDGE_MIN_POINTS:
        return None

    res = np.zeros(len(arr))
    for _ in range(3):
        cx, cy, r = _fit_circle(arr)
        res = np.abs(np.hypot(arr[:, 0] - cx, arr[:, 1] - cy) - r)
        keep = res < max(1.5, 2.5 * float(np.median(res)))
        arr = arr[keep]
        if len(arr) < IRIS_EDGE_MIN_POINTS:
            return None
    cx, cy, r = _fit_circle(arr)
    res = np.abs(np.hypot(arr[:, 0] - cx, arr[:, 1] - cy) - r)
    return IrisEdgeFit(centre=(cx + x0, cy + y0), radius=r, n_points=len(arr), residual=float(np.median(res)))


def _accept(fit: Optional[IrisEdgeFit], centre_hint: Tuple[float, float], radius_hint: float) -> Optional[str]:
    """Why a fit is rejected, or None when it is usable."""
    if fit is None:
        return "too few edge points"
    lo, hi = IRIS_EDGE_RADIUS_RANGE
    if not lo * radius_hint <= fit.radius <= hi * radius_hint:
        return f"radius {fit.radius:.1f}px is {fit.radius / radius_hint:.2f}× the landmark radius"
    if fit.residual > IRIS_EDGE_MAX_RESIDUAL * fit.radius:
        return f"edge points scatter {fit.residual / fit.radius:.3f} r from the circle"
    shift = math.hypot(fit.centre[0] - centre_hint[0], fit.centre[1] - centre_hint[1])
    if shift > IRIS_EDGE_MAX_CENTRE_SHIFT * radius_hint:
        return f"centre {shift / radius_hint:.2f} r from the landmark centre"
    return None


def refine_with_iris_edge(
    full_img: np.ndarray,
    scale: float,
    detection: EyeDetectionResult,
    pupil_result: PupilResult,
) -> PupilResult:
    """
    Re-centre both eyes on their fitted iris edge.

    Args:
        full_img:     RGB photo at the resolution the edge is traced on.
        scale:        full_img pixels per working-image pixel (≥ 1). Module 1's
                      crop boxes and landmarks are in the working image.
        detection:    Module 1 output (crop boxes, iris landmarks).
        pupil_result: Module 2 output (landmark centres and radii, crop coords).

    Returns:
        A PupilResult whose centres are the iris-edge centres and whose radii
        are one shared radius (both irises are the same physical size), in
        working-image crop coordinates. The Hough fields carry the fitted
        circle so the annotated picture draws the traced edge.

    Raises:
        PupilError("iris_edge_not_found_{eye}") or PupilError("iris_size_mismatch").
    """
    gray = cv2.cvtColor(full_img, cv2.COLOR_RGB2GRAY)
    fits = {}
    for eye in ("left", "right"):
        box = getattr(detection, f"{eye}_crop_box")
        lm = getattr(pupil_result, f"{eye}_landmark_centre")
        hint_c = ((lm[0] + box[0]) * scale, (lm[1] + box[1]) * scale)
        hint_r = getattr(pupil_result, f"{eye}_iris_radius") * scale
        fit = fit_iris_edge(gray, hint_c, hint_r)
        why = _accept(fit, hint_c, hint_r)
        if why:
            logger.warning(f"Module 2b [{eye}]: iris edge rejected — {why}")
            raise PupilError(f"iris_edge_not_found_{eye}")
        fits[eye] = fit
        logger.info(
            f"Module 2b [{eye}]: iris edge r={fit.radius:.1f}px, {fit.n_points} points, "
            f"residual {fit.residual / fit.radius:.3f} r"
        )

    r_l, r_r = fits["left"].radius, fits["right"].radius
    mismatch = abs(r_l - r_r) / max(r_l, r_r)
    if mismatch > IRIS_EDGE_MAX_RADIUS_MISMATCH:
        logger.warning(
            f"Module 2b: iris radii {r_r:.1f}px (right) and {r_l:.1f}px (left) differ by "
            f"{mismatch:.0%} — one edge is hidden or mis-traced"
        )
        raise PupilError("iris_size_mismatch")

    shared_r = (r_l + r_r) / 2 / scale
    out: dict = {}
    for eye in ("left", "right"):
        box = getattr(detection, f"{eye}_crop_box")
        f = fits[eye]
        out[f"{eye}_pupil"] = (f.centre[0] / scale - box[0], f.centre[1] / scale - box[1])
        out[f"{eye}_iris_radius"] = shared_r
        out[f"{eye}_hough_centre"] = out[f"{eye}_pupil"]
        out[f"{eye}_hough_radius"] = f.radius / scale
        edge_conf = CONFIDENCE_HIGH if f.residual <= IRIS_EDGE_HIGH_RESIDUAL * f.radius else CONFIDENCE_MEDIUM
        # Never raise a confidence Module 2 already lowered (the referral gate relies on it).
        out[f"{eye}_confidence"] = min(
            edge_conf, getattr(pupil_result, f"{eye}_confidence"), key=lambda c: _RANK.get(c, 0)
        )
    flags = [f for f in pupil_result.flags if not f.startswith(("pupil_disagreement_", "pupil_from_darkblob_"))]
    flags.append("centre_from_iris_edge")
    if mismatch > IRIS_EDGE_MAX_RADIUS_MISMATCH / 2:
        flags.append("iris_size_differs")
    return replace(pupil_result, flags=flags, **out)
