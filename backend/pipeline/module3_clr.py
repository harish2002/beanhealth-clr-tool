"""
BeanHealth CLR Tool — Module 3: CLR Bright Spot Detection
==========================================================

Responsibility:
    Locate the corneal light reflex (CLR) — the tiny bright reflection of
    the phone torch on the cornea surface — in each eye crop.

    This is the MOST CRITICAL module in the pipeline. A wrong CLR position
    produces a wrong triage result. Therefore:

      - If no torch is detected → INCONCLUSIVE immediately (no guessing)
      - If no valid blob is found → INCONCLUSIVE (never return a guessed position)
      - All 3 filter checks are MANDATORY — no shortcuts

    The method:
      1. Grayscale conversion
      2. Flash validation (max pixel < 240 → abort)
      3. Adaptive percentile threshold (top 3% brightest pixels)
      4. Binary mask → connected component analysis
      5. 3-way filter: location × area × circularity
      6. Select largest passing blob → its centroid is the CLR

Pipeline position:  THIRD — depends on Module 2 (needs iris_radius for area filter).
Failure behaviour:  Raises CLRError with specific code → caller returns INCONCLUSIVE.

Author: BeanHealth
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.constants import (
    CLR_LOCATION_MARGIN,
    CLR_MAX_AREA_RATIO,
    CLR_MIN_AREA_RATIO,
    CLR_MIN_CIRCULARITY,
    CLR_MIN_PEAK_BRIGHTNESS,
    CLR_PERCENTILE_THRESHOLD,
)
from utils.exceptions import CLRError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class BlobCandidate:
    """
    Represents a single connected component (bright blob) found in the
    binary threshold mask. Carries all attributes needed for the 3-way filter.
    """
    label:       int
    area:        float              # pixel area of the blob
    centroid_x:  float
    centroid_y:  float
    circularity: float              # 0.0–1.0  (1.0 = perfect circle)
    passed:      bool = False       # set to True after all 3 filters pass
    fail_reason: str  = ""         # which filter rejected it (for debug logs)


@dataclass
class CLRResult:
    """
    Output of Module 3.
    All coordinates are in crop-local pixel space (same as Module 2 output).
    """

    # CLR centroid in crop pixels
    left_clr:  Tuple[float, float]
    right_clr: Tuple[float, float]

    # Circularity score of the selected blob (0–1, higher = more circular)
    left_clr_confidence:  float
    right_clr_confidence: float

    # Non-fatal warnings
    flags: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Internal: Flash validation
# ─────────────────────────────────────────────────────────────

def _validate_flash(gray: np.ndarray, eye_label: str) -> None:
    """
    Check whether the torch/flash was active when the photo was taken.

    If the brightest pixel in the crop is below CLR_MIN_PEAK_BRIGHTNESS (240),
    there is no torch reflection to find. Returning a guessed position here
    would be dangerous — raise immediately.

    Args:
        gray:      Grayscale eye crop.
        eye_label: "left" or "right" for logging.

    Raises:
        CLRError("no_flash"): If peak brightness < threshold.
    """
    peak = int(np.max(gray))
    logger.debug(f"Module 3 [{eye_label}]: peak brightness = {peak}")

    if peak < CLR_MIN_PEAK_BRIGHTNESS:
        logger.warning(
            f"Module 3 [{eye_label}]: Peak brightness {peak} < {CLR_MIN_PEAK_BRIGHTNESS} "
            f"— torch not detected."
        )
        raise CLRError("no_flash")


# ─────────────────────────────────────────────────────────────
# Internal: Adaptive threshold → binary mask
# ─────────────────────────────────────────────────────────────

def _adaptive_threshold_mask(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Create a binary mask of the top CLR_PERCENTILE_THRESHOLD% brightest pixels.

    Uses np.percentile rather than a fixed value — adapts to ambient brightness.

    Args:
        gray: Single-channel uint8 grayscale image.

    Returns:
        (mask, threshold_value)
        mask: uint8 binary image (255 = above threshold, 0 = below)
        threshold_value: the actual pixel value used as the threshold
    """
    threshold = float(np.percentile(gray, CLR_PERCENTILE_THRESHOLD))
    _, mask = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )
    logger.debug(f"Module 3: adaptive threshold = {threshold:.1f}")
    return mask, threshold


# ─────────────────────────────────────────────────────────────
# Internal: Connected component analysis
# ─────────────────────────────────────────────────────────────

def _find_blobs(mask: np.ndarray) -> List[BlobCandidate]:
    """
    Find all connected white regions in the binary mask and compute
    area, centroid, and circularity for each.

    Circularity = (4π × area) / perimeter²
      → 1.0 for a perfect circle
      → < 0.5 for elongated or irregular shapes (eyelid highlights, etc.)

    The background label (0) is always skipped.

    Args:
        mask: Binary uint8 image from _adaptive_threshold_mask.

    Returns:
        List of BlobCandidate objects (unsorted, unfiltered).
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    blobs: List[BlobCandidate] = []

    for label in range(1, num_labels):   # skip label 0 = background
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 1:
            continue

        cx = float(centroids[label, 0])
        cy = float(centroids[label, 1])

        # Compute perimeter via contours for circularity
        blob_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        circularity = 0.0
        if contours:
            perimeter = cv2.arcLength(contours[0], closed=True)
            if perimeter > 0:
                circularity = (4.0 * math.pi * area) / (perimeter ** 2)
                circularity = min(circularity, 1.0)   # clamp rounding errors

        blobs.append(BlobCandidate(
            label=label,
            area=area,
            centroid_x=cx,
            centroid_y=cy,
            circularity=circularity,
        ))

    logger.debug(f"Module 3: {len(blobs)} raw blob(s) found before filtering")
    return blobs


# ─────────────────────────────────────────────────────────────
# Internal: 3-way filter
# ─────────────────────────────────────────────────────────────

def _apply_four_way_filter(
    blobs: List[BlobCandidate],
    crop_w: int,
    crop_h: int,
    iris_radius: float,
    pupil_centre: Optional[Tuple[float, float]],
    eye_label: str,
) -> List[BlobCandidate]:
    """
    Apply four mandatory filters to the blob list.

    Filter ①  Location: central 80% of crop.
    Filter ②  Area: acceptable size for CLR.
    Filter ③  Circularity: > 0.5.
    Filter ④  Iris Distance: if pupil_centre is known, distance from pupil center to CLR must be <= 1.2 × iris_radius.
              (The reflection must fall on the cornea, which roughly matches the iris). This rejects scleral glare.
    """
    iris_area  = math.pi * (iris_radius ** 2)
    min_area   = CLR_MIN_AREA_RATIO * iris_area
    max_area   = CLR_MAX_AREA_RATIO * iris_area

    margin     = CLR_LOCATION_MARGIN
    x_min = crop_w * margin
    x_max = crop_w * (1.0 - margin)
    y_min = crop_h * margin
    y_max = crop_h * (1.0 - margin)

    passing: List[BlobCandidate] = []

    for blob in blobs:
        # ── Filter ①: Location ──
        if not (x_min < blob.centroid_x < x_max and y_min < blob.centroid_y < y_max):
            blob.fail_reason = (
                f"location ({blob.centroid_x:.0f},{blob.centroid_y:.0f}) "
                f"outside [{x_min:.0f}–{x_max:.0f}, {y_min:.0f}–{y_max:.0f}]"
            )
            logger.debug(f"Module 3 [{eye_label}]: blob {blob.label} REJECTED — {blob.fail_reason}")
            continue

        # ── Filter ②: Area ──
        if not (min_area < blob.area < max_area):
            blob.fail_reason = (
                f"area {blob.area:.1f} outside [{min_area:.1f}–{max_area:.1f}] "
                f"(iris_area={iris_area:.1f})"
            )
            logger.debug(f"Module 3 [{eye_label}]: blob {blob.label} REJECTED — {blob.fail_reason}")
            continue

        # ── Filter ③: Circularity ──
        # Condition is strictly GREATER THAN — exactly 0.5 must fail
        if blob.circularity <= CLR_MIN_CIRCULARITY:
            blob.fail_reason = f"circularity {blob.circularity:.3f} < {CLR_MIN_CIRCULARITY}"
            logger.debug(f"Module 3 [{eye_label}]: blob {blob.label} REJECTED — {blob.fail_reason}")
            continue

        # ── Filter ④: Iris Distance ──
        if pupil_centre is not None:
            dist_to_pupil = math.hypot(blob.centroid_x - pupil_centre[0], blob.centroid_y - pupil_centre[1])
            max_dist = iris_radius * 1.25  # 25% margin in case pupil center is slightly off
            if dist_to_pupil > max_dist:
                blob.fail_reason = f"distance {dist_to_pupil:.1f}px > max {max_dist:.1f}px from pupil {pupil_centre}"
                logger.debug(f"Module 3 [{eye_label}]: blob {blob.label} REJECTED — {blob.fail_reason}")
                continue

        blob.passed = True
        passing.append(blob)
        logger.debug(
            f"Module 3 [{eye_label}]: blob {blob.label} PASSED — "
            f"area={blob.area:.1f}, circ={blob.circularity:.3f}, "
            f"pos=({blob.centroid_x:.1f},{blob.centroid_y:.1f})"
        )

    return passing


# ─────────────────────────────────────────────────────────────
# Internal: Select best blob
# ─────────────────────────────────────────────────────────────

def _select_clr_blob(
    passing: List[BlobCandidate],
    eye_label: str,
    flags: List[str],
) -> BlobCandidate:
    """
    From the blobs that passed all 3 filters, select the best CLR candidate.

    Selection rule: largest area (the corneal reflex is typically the
    brightest and largest valid circular blob in the crop).

    If multiple high-quality blobs exist (e.g. two corneal reflections from
    glasses + cornea), flag `ambiguous_reflex` but still return the largest.

    Args:
        passing:   Blobs that passed all 3 filters.
        eye_label: "left" or "right".
        flags:     Mutable list — flags appended here.

    Returns:
        The selected BlobCandidate.

    Raises:
        CLRError: If no blobs passed (no_reflex_{eye_label}).
    """
    if not passing:
        logger.warning(f"Module 3 [{eye_label}]: No blob passed all 3 filters → no_reflex")
        raise CLRError(f"no_reflex_{eye_label}")

    if len(passing) > 3:
        flags.append(f"ambiguous_reflex_{eye_label}")
        logger.warning(
            f"Module 3 [{eye_label}]: {len(passing)} blobs passed filters — "
            f"ambiguous, selecting largest."
        )

    # Select the largest passing blob
    best = max(passing, key=lambda b: b.area)
    logger.debug(
        f"Module 3 [{eye_label}]: Selected blob {best.label} — "
        f"area={best.area:.1f}, circ={best.circularity:.3f}, "
        f"pos=({best.centroid_x:.1f},{best.centroid_y:.1f})"
    )
    return best


# ─────────────────────────────────────────────────────────────
# Internal: Process one eye
# ─────────────────────────────────────────────────────────────

def _detect_clr_one_eye(
    crop_rgb:    np.ndarray,
    iris_radius: float,
    pupil_centre: Optional[Tuple[float, float]],
    eye_label:   str,
    flags:       List[str],
) -> Tuple[Tuple[float, float], float]:
    """
    Run the full CLR detection pipeline for one eye.

    Args:
        crop_rgb:    RGB eye crop from Module 1.
        iris_radius: Iris radius in crop pixels from Module 2.
        pupil_centre: Pupil center in crop pixels from Module 2 (if available).
        eye_label:   "left" or "right".
        flags:       Mutable list for non-fatal warnings.

    Returns:
        ((clr_x, clr_y), circularity_score)

    Raises:
        CLRError: On no_flash, no_reflex, or other detection failure.
    """
    h, w = crop_rgb.shape[:2]

    # ── Step 1: Grayscale ──
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    # ── Step 2: Flash validation ──
    _validate_flash(gray, eye_label)

    # ── Step 3: Adaptive threshold → binary mask ──
    mask, threshold_val = _adaptive_threshold_mask(gray)

    # ── Step 4: Connected component analysis ──
    blobs = _find_blobs(mask)

    if not blobs:
        logger.warning(f"Module 3 [{eye_label}]: No blobs found in mask after threshold.")
        raise CLRError(f"no_reflex_{eye_label}")

    # ── Step 5: 4-way filter ──
    passing = _apply_four_way_filter(blobs, w, h, iris_radius, pupil_centre, eye_label)

    # ── Step 6: Select best blob ──
    best = _select_clr_blob(passing, eye_label, flags)

    return (best.centroid_x, best.centroid_y), best.circularity


# ─────────────────────────────────────────────────────────────
# Module 3 Public Entry Point
# ─────────────────────────────────────────────────────────────

def detect_clr(
    left_crop:         np.ndarray,
    right_crop:        np.ndarray,
    left_iris_radius:  float,
    right_iris_radius: float,
    left_pupil:        Optional[Tuple[float, float]] = None,
    right_pupil:       Optional[Tuple[float, float]] = None,
    debug:             bool = False,
) -> CLRResult:
    """
    Detect the corneal light reflex in both eye crops.

    This is the single public function for Module 3.
    Call with the direct crops and iris radii from Modules 1 & 2.

    The function checks BOTH eyes before raising any error, so that
    partial results (one eye succeeded, one failed) can be logged fully.
    However, any CLRError still propagates to the caller as INCONCLUSIVE.

    Args:
        left_crop:         RGB eye crop, left eye.
        right_crop:        RGB eye crop, right eye.
        left_iris_radius:  Iris radius in crop pixels, left eye.
        right_iris_radius: Iris radius in crop pixels, right eye.
        left_pupil:        Pupil coordinate in crop pixels, left eye.
        right_pupil:       Pupil coordinate in crop pixels, right eye.
        debug:             Log extra detail if True.

    Returns:
        CLRResult with CLR positions, confidence scores, and flags.

    Raises:
        CLRError: If either eye's CLR cannot be located.
                  Caught by the API layer → INCONCLUSIVE response.
    """
    flags: List[str] = []
    left_error:  Optional[CLRError] = None
    right_error: Optional[CLRError] = None

    # ── Left eye ──
    left_clr:        Optional[Tuple[float, float]] = None
    left_confidence: float = 0.0
    try:
        left_clr, left_confidence = _detect_clr_one_eye(
            left_crop, left_iris_radius, left_pupil, "left", flags
        )
        logger.debug(
            f"Module 3 [left]: CLR at ({left_clr[0]:.1f},{left_clr[1]:.1f}), "
            f"confidence={left_confidence:.3f}"
        )
    except CLRError as e:
        left_error = e
        flags.append(e.code)
        logger.warning(f"Module 3 [left]: {e}")

    # ── Right eye ──
    right_clr:        Optional[Tuple[float, float]] = None
    right_confidence: float = 0.0
    try:
        right_clr, right_confidence = _detect_clr_one_eye(
            right_crop, right_iris_radius, right_pupil, "right", flags
        )
        logger.debug(
            f"Module 3 [right]: CLR at ({right_clr[0]:.1f},{right_clr[1]:.1f}), "
            f"confidence={right_confidence:.3f}"
        )
    except CLRError as e:
        right_error = e
        flags.append(e.code)
        logger.warning(f"Module 3 [right]: {e}")

    # ── Error propagation ──
    # Both failed — raise the more specific error
    if left_error and right_error:
        if "no_flash" in (left_error.code, right_error.code):
            raise CLRError("no_flash")
        raise CLRError("no_reflex_both")

    # One eye failed — still raise (cannot compute asymmetry with one eye)
    if left_error:
        raise left_error
    if right_error:
        raise right_error

    return CLRResult(
        left_clr=left_clr,
        right_clr=right_clr,
        left_clr_confidence=left_confidence,
        right_clr_confidence=right_confidence,
        flags=flags,
    )
