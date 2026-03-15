"""
BeanHealth CLR Tool — Module 2: Pupil Centre Localisation
==========================================================

Responsibility:
    Given an eye crop (from Module 1) and the 5 MediaPipe iris landmarks
    (in original image space), pinpoint the exact centre of the pupil using
    two independent methods:

      Primary   — MediaPipe iris landmark mean (always available)
      Secondary — Hough Circle Transform (edge-based, image-derived)

    The two estimates are cross-validated. Agreement distance determines the
    confidence tier (HIGH / MEDIUM / LOW), which propagates to the final report.

Pipeline position:  SECOND — depends on Module 1 output.
Failure behaviour:  Raises PupilError only if BOTH methods fail for one eye.
                    Single-method fallback (LOW confidence) is acceptable.

Key constraint:
    Iris landmarks from Module 1 are in ORIGINAL image pixel space.
    This module maps them into crop-local coordinates before use.

Author: BeanHealth
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.constants import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    GAUSSIAN_KERNEL,
    HOUGH_DP,
    HOUGH_MAX_RADIUS_RATIO,
    HOUGH_MIN_DIST,
    HOUGH_MIN_RADIUS_RATIO,
    HOUGH_PARAM1,
    HOUGH_PARAM2,
    PUPIL_AGREEMENT_HIGH_PX,
    PUPIL_AGREEMENT_MEDIUM_PX,
)
from utils.exceptions import PupilError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Confidence tiers
# ─────────────────────────────────────────────────────────────

CONFIDENCE_HIGH   = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW    = "LOW"


# ─────────────────────────────────────────────────────────────
# Data Structure
# ─────────────────────────────────────────────────────────────

@dataclass
class PupilResult:
    """
    Output of Module 2.
    All coordinates are in crop-local pixel space (not original image space).
    """

    # Final agreed pupil centre for each eye (x, y) in crop pixels
    left_pupil:  Tuple[float, float]
    right_pupil: Tuple[float, float]

    # Iris radius in crop pixels (used by Modules 3 and 4 for normalisation)
    left_iris_radius:  float
    right_iris_radius: float

    # Per-eye confidence tier
    left_confidence:  str   # HIGH / MEDIUM / LOW
    right_confidence: str

    # Intermediate estimates (kept for annotation in Module 7)
    left_landmark_centre:  Tuple[float, float]
    right_landmark_centre: Tuple[float, float]
    left_hough_centre:     Optional[Tuple[float, float]]   # None if Hough failed
    right_hough_centre:    Optional[Tuple[float, float]]
    left_hough_radius:     Optional[float]
    right_hough_radius:    Optional[float]

    # Non-fatal warnings
    flags: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Internal: Landmark → crop-space mapping
# ─────────────────────────────────────────────────────────────

def _map_landmarks_to_crop(
    iris_landmarks_orig: List[Tuple[float, float]],
    crop_box: Tuple[int, int, int, int],
) -> List[Tuple[float, float]]:
    """
    Convert iris landmark coordinates from original-image space to crop-local space.

    Module 1 returns landmarks in full-image pixels.
    Module 2 works in crop pixels — subtract the crop origin to convert.

    Args:
        iris_landmarks_orig: 5 (x, y) points in original image pixels.
        crop_box:            (x1, y1, x2, y2) of the crop in the original image.

    Returns:
        5 (x, y) points in crop-local pixel space.
    """
    x_offset, y_offset = crop_box[0], crop_box[1]
    return [
        (lm[0] - x_offset, lm[1] - y_offset)
        for lm in iris_landmarks_orig
    ]


def _iris_radius_in_crop(
    landmarks_crop: List[Tuple[float, float]],
) -> float:
    """
    Recompute the iris radius in crop-local pixels from mapped landmarks.

    MediaPipe iris landmark layout (index within the 5-point list):
      0 → centre
      1 → top boundary
      2 → right boundary
      3 → bottom boundary
      4 → left boundary

    Returns:
        Mean distance from centre to the 4 boundary points.
    """
    if len(landmarks_crop) < 5:
        return 0.0
    centre = np.array(landmarks_crop[0])
    boundary = np.array(landmarks_crop[1:])
    dists = np.linalg.norm(boundary - centre, axis=1)
    return float(np.mean(dists))


# ─────────────────────────────────────────────────────────────
# Internal: Primary estimate — MediaPipe landmark mean
# ─────────────────────────────────────────────────────────────

def _landmark_centre(
    landmarks_crop: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Compute the iris/pupil centre as the mean of all 5 iris landmarks.

    Using the mean of all 5 (centre + 4 boundary) rather than just index-0
    gives a more stable estimate — the centre landmark can sometimes drift
    slightly for sideways glances.

    Returns:
        (x, y) in crop pixels.
    """
    arr = np.array(landmarks_crop)
    mean = arr.mean(axis=0)
    return float(mean[0]), float(mean[1])


# ─────────────────────────────────────────────────────────────
# Internal: Secondary estimate — Hough Circle Transform
# ─────────────────────────────────────────────────────────────

def _preprocess_for_hough(crop_rgb: np.ndarray) -> np.ndarray:
    """
    Prepare an eye crop for Hough Circle detection.

    Steps:
      1. Grayscale conversion
      2. CLAHE contrast enhancement (boosts dark iris boundaries)
      3. Gaussian blur (removes eyelash / iris texture noise)

    Returns:
        Blurred single-channel uint8 image.
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, GAUSSIAN_KERNEL, 0)
    return blurred


def _hough_estimate(
    blurred: np.ndarray,
    crop_width: int,
    crop_height: int,
    iris_radius_hint: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Run Hough Circle Transform to find the pupil/iris circle.

    If iris_radius_hint is provided (from the MediaPipe landmark estimate),
    the search range is set tightly around it (±40%). This is more accurate
    than using a fixed fraction of the crop dimension, which can fail on
    compressed or landscape-aspect crops.

    Falls back to using min(crop_width, crop_height) if no hint is given.

    Args:
        blurred:          Pre-processed single-channel image.
        crop_width:       Width of the crop in pixels.
        crop_height:      Height of the crop in pixels.
        iris_radius_hint: Iris radius in crop pixels from landmark estimate.

    Returns:
        (cx, cy, radius) of the best detected circle, or None if not found.
    """
    if iris_radius_hint is not None and iris_radius_hint > 4:
        # Tight range around landmark estimate — more reliable on real photos
        min_r = max(4, int(iris_radius_hint * 0.60))
        max_r = max(min_r + 4, int(iris_radius_hint * 1.40))
    else:
        ref_dim = min(crop_width, crop_height)
        min_r = int(ref_dim * HOUGH_MIN_RADIUS_RATIO)
        max_r = int(ref_dim * HOUGH_MAX_RADIUS_RATIO)

    logger.debug(f"Hough radius search range: {min_r}–{max_r}px (hint={iris_radius_hint})")

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=12,          # tuned for real phone selfies — iris edge is partially occluded by eyelids
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None

    # HoughCircles returns shape (1, N, 3): [[cx, cy, r], ...]
    # Take the first (highest-vote) circle
    best = circles[0][0]
    return float(best[0]), float(best[1]), float(best[2])


# ─────────────────────────────────────────────────────────────
# Internal: Agreement check → final centre + confidence
# ─────────────────────────────────────────────────────────────

def _agree_and_fuse(
    landmark_centre: Tuple[float, float],
    hough_result: Optional[Tuple[float, float, float]],
    eye_label: str,
    flags: List[str],
) -> Tuple[Tuple[float, float], str, Optional[Tuple[float, float]], Optional[float]]:
    """
    Compare the two pupil estimates and return a fused centre + confidence tier.

    Agreement rules (from CLAUDE.md §4 Module 2):
      dist < PUPIL_AGREEMENT_HIGH_PX   → HIGH,   use averaged centre
      dist < PUPIL_AGREEMENT_MEDIUM_PX → MEDIUM, use averaged centre, flag disagreement
      Hough failed OR dist ≥ MEDIUM_PX → LOW,    use landmark only,  flag disagreement

    Args:
        landmark_centre: (x, y) from MediaPipe — always present.
        hough_result:    (cx, cy, radius) from Hough, or None if Hough failed.
        eye_label:       "left" or "right" — used for flag names.
        flags:           Mutable list — disagreement flags appended here.

    Returns:
        (final_centre, confidence, hough_centre_or_none, hough_radius_or_none)
    """
    if hough_result is None:
        # Hough failed entirely — landmark-only fallback
        flags.append(f"pupil_disagreement_{eye_label}")
        logger.debug(f"Module 2 [{eye_label}]: Hough failed — using landmark only (LOW confidence)")
        return landmark_centre, CONFIDENCE_LOW, None, None

    hx, hy, hr = hough_result
    hough_centre = (hx, hy)

    # Euclidean distance between the two estimates
    dist = float(np.linalg.norm(
        np.array(landmark_centre) - np.array(hough_centre)
    ))
    logger.debug(f"Module 2 [{eye_label}]: landmark-Hough distance = {dist:.2f}px")

    if dist < PUPIL_AGREEMENT_HIGH_PX:
        # HIGH confidence — average the two
        avg_x = (landmark_centre[0] + hx) / 2.0
        avg_y = (landmark_centre[1] + hy) / 2.0
        return (avg_x, avg_y), CONFIDENCE_HIGH, hough_centre, hr

    elif dist < PUPIL_AGREEMENT_MEDIUM_PX:
        # MEDIUM confidence — still average, but flag
        flags.append(f"pupil_disagreement_{eye_label}")
        avg_x = (landmark_centre[0] + hx) / 2.0
        avg_y = (landmark_centre[1] + hy) / 2.0
        logger.debug(f"Module 2 [{eye_label}]: MEDIUM confidence, dist={dist:.1f}px")
        return (avg_x, avg_y), CONFIDENCE_MEDIUM, hough_centre, hr

    else:
        # Disagreement too large — landmark wins
        flags.append(f"pupil_disagreement_{eye_label}")
        logger.debug(f"Module 2 [{eye_label}]: LOW confidence, dist={dist:.1f}px — using landmark")
        return landmark_centre, CONFIDENCE_LOW, hough_centre, hr


# ─────────────────────────────────────────────────────────────
# Internal: Process one eye
# ─────────────────────────────────────────────────────────────

def _localise_one_eye(
    crop_rgb: np.ndarray,
    iris_landmarks_orig: List[Tuple[float, float]],
    crop_box: Tuple[int, int, int, int],
    eye_label: str,
    flags: List[str],
) -> Tuple[
    Tuple[float, float],      # final pupil centre (crop coords)
    float,                    # iris radius (crop pixels)
    str,                      # confidence tier
    Tuple[float, float],      # landmark centre (crop coords)
    Optional[Tuple[float, float]],  # hough centre (crop coords) or None
    Optional[float],          # hough radius or None
]:
    """
    Run the full two-method localisation for a single eye.

    Args:
        crop_rgb:            RGB eye crop (H, W, 3).
        iris_landmarks_orig: 5 iris landmark points in original image coords.
        crop_box:            (x1, y1, x2, y2) crop origin in original image.
        eye_label:           "left" or "right" for logging/flags.
        flags:               Mutable list for appending non-fatal warnings.

    Returns:
        Tuple of (pupil_centre, iris_radius, confidence, lm_centre, hough_centre, hough_radius)

    Raises:
        PupilError: If both methods fail to locate the pupil.
    """
    h, w = crop_rgb.shape[:2]

    # ── Step 1: Map landmarks to crop space ──
    landmarks_crop = _map_landmarks_to_crop(iris_landmarks_orig, crop_box)

    # ── Step 2: Primary estimate — landmark mean ──
    lm_centre = _landmark_centre(landmarks_crop)

    # Sanity check: is the landmark centre inside the crop?
    lx, ly = lm_centre
    if not (0 <= lx <= w and 0 <= ly <= h):
        logger.warning(
            f"Module 2 [{eye_label}]: Landmark centre ({lx:.1f},{ly:.1f}) "
            f"is outside crop bounds {w}x{h}. Clamping."
        )
        lx = float(np.clip(lx, 0, w))
        ly = float(np.clip(ly, 0, h))
        lm_centre = (lx, ly)
        flags.append(f"landmark_outside_crop_{eye_label}")

    # ── Step 3: Iris radius in crop space ──
    iris_radius = _iris_radius_in_crop(landmarks_crop)
    if iris_radius < 3.0:
        flags.append(f"iris_radius_tiny_{eye_label}")
        logger.warning(f"Module 2 [{eye_label}]: Iris radius very small ({iris_radius:.1f}px) in crop.")

    # ── Step 4: Secondary estimate — Hough ──
    blurred = _preprocess_for_hough(crop_rgb)
    hough_result = _hough_estimate(blurred, w, h, iris_radius_hint=iris_radius)

    if hough_result is not None:
        hx, hy, hr = hough_result
        logger.debug(f"Module 2 [{eye_label}]: Hough result = ({hx:.1f},{hy:.1f}) r={hr:.1f}px")
    else:
        logger.debug(f"Module 2 [{eye_label}]: Hough found no circles.")

    # ── Step 5: Both methods failed → raise ──
    # (Landmark method cannot truly "fail" since it just takes a mean,
    #  but if the landmark is wildly out of bounds after clamping,
    #  treat it as unreliable only if Hough also failed.)
    if f"landmark_outside_crop_{eye_label}" in flags and hough_result is None:
        raise PupilError(f"pupil_not_found_{eye_label}")

    # ── Step 6: Agree and fuse ──
    final_centre, confidence, hough_centre, hough_radius = _agree_and_fuse(
        lm_centre, hough_result, eye_label, flags
    )

    logger.debug(
        f"Module 2 [{eye_label}]: final=({final_centre[0]:.1f},{final_centre[1]:.1f}) "
        f"confidence={confidence} iris_r={iris_radius:.1f}px"
    )

    return final_centre, iris_radius, confidence, lm_centre, hough_centre, hough_radius


# ─────────────────────────────────────────────────────────────
# Module 2 Public Entry Point
# ─────────────────────────────────────────────────────────────

def localise_pupils(
    left_crop:  np.ndarray,
    right_crop: np.ndarray,
    left_iris_landmarks_orig:  List[Tuple[float, float]],
    right_iris_landmarks_orig: List[Tuple[float, float]],
    left_crop_box:  Tuple[int, int, int, int],
    right_crop_box: Tuple[int, int, int, int],
    debug: bool = False,
) -> PupilResult:
    """
    Localise the pupil centre for both eyes.

    This is the single public function for Module 2.
    Call with the direct outputs of Module 1.

    Args:
        left_crop:                  RGB eye crop for the left eye.
        right_crop:                 RGB eye crop for the right eye.
        left_iris_landmarks_orig:   5 iris landmarks (original image coords).
        right_iris_landmarks_orig:  5 iris landmarks (original image coords).
        left_crop_box:              (x1,y1,x2,y2) of the left crop in original image.
        right_crop_box:             (x1,y1,x2,y2) of the right crop in original image.
        debug:                      Log extra detail if True.

    Returns:
        PupilResult with final centres, radii, confidence tiers, and flags.

    Raises:
        PupilError: If both methods fail for either eye. Treated as INCONCLUSIVE.
    """
    flags: List[str] = []

    # ── Left eye ──
    logger.debug("Module 2: Processing left eye...")
    (
        left_pupil,
        left_iris_radius,
        left_confidence,
        left_lm_centre,
        left_hough_centre,
        left_hough_radius,
    ) = _localise_one_eye(
        left_crop,
        left_iris_landmarks_orig,
        left_crop_box,
        "left",
        flags,
    )

    # ── Right eye ──
    logger.debug("Module 2: Processing right eye...")
    (
        right_pupil,
        right_iris_radius,
        right_confidence,
        right_lm_centre,
        right_hough_centre,
        right_hough_radius,
    ) = _localise_one_eye(
        right_crop,
        right_iris_landmarks_orig,
        right_crop_box,
        "right",
        flags,
    )

    result = PupilResult(
        left_pupil=left_pupil,
        right_pupil=right_pupil,
        left_iris_radius=left_iris_radius,
        right_iris_radius=right_iris_radius,
        left_confidence=left_confidence,
        right_confidence=right_confidence,
        left_landmark_centre=left_lm_centre,
        right_landmark_centre=right_lm_centre,
        left_hough_centre=left_hough_centre,
        right_hough_centre=right_hough_centre,
        left_hough_radius=left_hough_radius,
        right_hough_radius=right_hough_radius,
        flags=flags,
    )

    if debug:
        logger.debug(f"Module 2 result: {result}")

    return result
