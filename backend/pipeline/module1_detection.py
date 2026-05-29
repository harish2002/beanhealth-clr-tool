"""
BeanHealth CLR Tool — Module 1: Eye Detection & Crop
=====================================================

Responsibility:
    Given a raw input image (from phone camera), detect the face using
    Google MediaPipe Face Mesh, locate both eyes, extract a padded crop
    around each eye, and return iris landmark coordinates for use by
    Module 2 (pupil centre localisation).

Pipeline position:  FIRST — all downstream modules depend on this output.
Failure behaviour:  Raises DetectionError with a specific code.
                    The API layer catches this and returns INCONCLUSIVE.

Author: BeanHealth
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from utils.constants import (
    CROP_PAD_HORIZONTAL,
    CROP_PAD_VERTICAL,
    LEFT_EYE_BOUNDARY,
    LEFT_IRIS_INDICES,
    MIN_CROP_HEIGHT,
    MIN_CROP_WIDTH,
    MIN_FACE_CONFIDENCE,
    RIGHT_EYE_BOUNDARY,
    RIGHT_IRIS_INDICES,
)
from utils.exceptions import DetectionError
from utils.image_utils import crop_region, downscale_if_needed

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class EyeDetectionResult:
    """
    Output of Module 1.
    All coordinates are in the crop's local pixel space.
    iris_landmarks_* are in the original full-image pixel space
    (they are mapped to crop space at the start of Module 2).
    """

    # Cropped eye images (RGB, uint8)
    left_crop:  np.ndarray
    right_crop: np.ndarray

    # Bounding boxes of each crop in the original image (x1, y1, x2, y2)
    left_crop_box:  Tuple[int, int, int, int]
    right_crop_box: Tuple[int, int, int, int]

    # MediaPipe iris landmarks in ORIGINAL image pixel coords
    # 5 points each: [centre, top, right, bottom, left] (MediaPipe order)
    left_iris_landmarks:  List[Tuple[float, float]]
    right_iris_landmarks: List[Tuple[float, float]]

    # Rough iris radius (pixels) in original image — computed from landmarks
    left_iris_radius_orig:  float
    right_iris_radius_orig: float

    # Detection confidence from MediaPipe
    face_confidence: float

    # Non-fatal warnings accumulated during detection
    warnings: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# MediaPipe setup — created once at module load, reused per call
# ─────────────────────────────────────────────────────────────

_mp_face_mesh = mp.solutions.face_mesh


def _make_face_mesh() -> mp.solutions.face_mesh.FaceMesh:
    """
    Create a MediaPipe FaceMesh instance.
    refine_landmarks=True is required to get iris landmark indices 468–477.
    """
    return _mp_face_mesh.FaceMesh(
        static_image_mode=True,        # single image, not video stream
        max_num_faces=1,               # we only need one face per photo
        refine_landmarks=True,         # enables iris landmarks (468–477)
        min_detection_confidence=MIN_FACE_CONFIDENCE,
        min_tracking_confidence=0.5,
    )


# ─────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────

def _landmarks_to_pixels(
    landmarks,
    img_width: int,
    img_height: int,
    indices: List[int],
) -> List[Tuple[float, float]]:
    """
    Convert normalised MediaPipe landmark coords to pixel (x, y) tuples.

    MediaPipe returns x,y as fractions of image width/height (0.0–1.0).
    """
    return [
        (
            landmarks[i].x * img_width,
            landmarks[i].y * img_height,
        )
        for i in indices
    ]


def _bounding_box_from_landmarks(
    landmarks,
    img_width: int,
    img_height: int,
    indices: List[int],
) -> Tuple[int, int, int, int]:
    """
    Compute the axis-aligned bounding box around a set of landmarks.

    Returns:
        (x1, y1, x2, y2) in pixel coordinates, clamped to image bounds.
    """
    pts = _landmarks_to_pixels(landmarks, img_width, img_height, indices)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = max(0, int(min(xs)))
    y1 = max(0, int(min(ys)))
    x2 = min(img_width,  int(max(xs)))
    y2 = min(img_height, int(max(ys)))
    return x1, y1, x2, y2


def _iris_radius_from_landmarks(
    landmarks: List[Tuple[float, float]],
) -> float:
    """
    Estimate iris radius in pixels from the 5 MediaPipe iris landmarks.

    MediaPipe iris landmark order (relative to first landmark = centre):
      index 0 → iris centre
      index 1 → top boundary
      index 2 → right boundary
      index 3 → bottom boundary
      index 4 → left boundary

    Radius = mean distance from centre to the 4 boundary points.
    """
    if len(landmarks) < 5:
        return 0.0

    centre = np.array(landmarks[0])
    boundary = np.array(landmarks[1:])          # 4 boundary points
    dists = np.linalg.norm(boundary - centre, axis=1)
    return float(np.mean(dists))


def _check_eye_visibility(
    landmarks,
    img_width: int,
    img_height: int,
    iris_indices: List[int],
    eye_label: str,
    warnings: List[str],
) -> bool:
    """
    Heuristically check if an eye is open and facing the camera.

    Checks:
      1. Iris centre landmark has z < 0 (pointing toward camera)
      2. Iris landmarks are within reasonable image bounds

    Returns True if eye appears open and visible, False otherwise.
    Appends a warning to the list if the eye looks problematic.
    """
    cx = landmarks[iris_indices[0]].x * img_width
    cy = landmarks[iris_indices[0]].y * img_height
    cz = landmarks[iris_indices[0]].z

    if cz > 0.05:
        warnings.append(f"{eye_label}_eye_facing_away")
        return False

    margin = 0.05   # 5% from image edge
    if not (img_width * margin < cx < img_width * (1 - margin)):
        warnings.append(f"{eye_label}_eye_near_edge")
        return False

    if not (img_height * margin < cy < img_height * (1 - margin)):
        warnings.append(f"{eye_label}_eye_near_edge")
        return False

    return True


# ─────────────────────────────────────────────────────────────
# Module 1 Public Entry Point
# ─────────────────────────────────────────────────────────────

def _find_one_iris(
    gray_region: np.ndarray,
) -> Optional[Tuple[int, int, int]]:
    """
    Locate a single iris in a half-image region using Hough circles.

    Picks the candidate circle with the darkest interior (the iris is the
    dark disc of the eye).  Returns (cx, cy, r) in the region's local pixel
    coords, or None if no plausible circle is found.
    """
    h_r, w_r = gray_region.shape[:2]
    blur = cv2.medianBlur(gray_region, 5)

    # The iris edge is often partly occluded by eyelids, so the Hough
    # accumulator vote can be weak.  Try progressively lower thresholds and
    # accept the first that yields any candidate.
    circles = None
    for param2 in (28, 22, 18, 14):
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=w_r,                  # one eye per region — want few candidates
            param1=80,
            param2=param2,
            minRadius=int(w_r * 0.08),
            maxRadius=int(w_r * 0.45),
        )
        if circles is not None:
            break
    if circles is None:
        return None

    best: Optional[Tuple[int, int, int]] = None
    best_score = -1.0
    for cx, cy, r in np.round(circles[0]).astype(int):
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w_r, cx + r), min(h_r, cy + r)
        patch = gray_region[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        # Darker interior → more likely a real iris (255 - mean grey).
        score = 255.0 - float(np.mean(patch))
        if score > best_score:
            best_score = score
            best = (int(cx), int(cy), int(r))
    return best


def _synthetic_iris_landmarks(
    cx: float, cy: float, r: float,
) -> List[Tuple[float, float]]:
    """
    Build 5 iris landmarks in MediaPipe order [centre, top, right, bottom, left]
    from a circle, so downstream modules see the same shape they expect.
    """
    return [
        (cx, cy),          # centre
        (cx, cy - r),      # top
        (cx + r, cy),      # right
        (cx, cy + r),      # bottom
        (cx - r, cy),      # left
    ]


def detect_eyes_only_fallback(
    image_rgb: np.ndarray,
) -> EyeDetectionResult:
    """
    DEV/RESEARCH fallback for when MediaPipe FaceMesh cannot find a face —
    typically because the input is a tightly-cropped eyes-only photo (common
    in published strabismus reference images) with no nose/mouth/chin context.

    Splits the image into left and right halves, locates one iris in each via
    Hough circles, and synthesises the EyeDetectionResult that the rest of the
    pipeline consumes.  This path is LESS accurate than FaceMesh (no landmark
    cross-validation) and must only be used in test mode.

    Image-left eye  → the subject's RIGHT eye  → "right" in pipeline convention.
    Image-right eye → the subject's LEFT eye   → "left"  in pipeline convention.

    Raises:
        DetectionError("eyes_not_visible") if an iris cannot be found in either half.
    """
    image_rgb = downscale_if_needed(image_rgb)
    img_h, img_w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    mid = img_w // 2
    left_region  = gray[:, :mid]   # image-left  half
    right_region = gray[:, mid:]   # image-right half

    img_left_iris  = _find_one_iris(left_region)
    img_right_iris = _find_one_iris(right_region)

    if img_left_iris is None or img_right_iris is None:
        logger.warning("Module 1 (eye-only fallback): could not locate both irises.")
        raise DetectionError("eyes_not_visible")

    # Map half-local coords back to full-image coords.
    # image-left half iris  → subject's RIGHT eye
    r_cx, r_cy, r_r = img_left_iris
    # image-right half iris → subject's LEFT eye (offset x by `mid`)
    l_cx, l_cy, l_r = img_right_iris
    l_cx += mid

    left_iris_px  = _synthetic_iris_landmarks(l_cx, l_cy, l_r)
    right_iris_px = _synthetic_iris_landmarks(r_cx, r_cy, r_r)

    left_crop,  left_box  = crop_region(
        image_rgb, l_cx - l_r, l_cy - l_r, l_cx + l_r, l_cy + l_r,
        pad_h=CROP_PAD_HORIZONTAL, pad_v=CROP_PAD_VERTICAL,
    )
    right_crop, right_box = crop_region(
        image_rgb, r_cx - r_r, r_cy - r_r, r_cx + r_r, r_cy + r_r,
        pad_h=CROP_PAD_HORIZONTAL, pad_v=CROP_PAD_VERTICAL,
    )

    if (left_crop.shape[1]  < MIN_CROP_WIDTH or left_crop.shape[0]  < MIN_CROP_HEIGHT or
        right_crop.shape[1] < MIN_CROP_WIDTH or right_crop.shape[0] < MIN_CROP_HEIGHT):
        raise DetectionError("crop_too_small")

    logger.info(
        f"Module 1 (eye-only fallback): irises L=({l_cx},{l_cy},r{l_r}) "
        f"R=({r_cx},{r_cy},r{r_r})"
    )

    return EyeDetectionResult(
        left_crop=left_crop,
        right_crop=right_crop,
        left_crop_box=left_box,
        right_crop_box=right_box,
        left_iris_landmarks=left_iris_px,
        right_iris_landmarks=right_iris_px,
        left_iris_radius_orig=float(l_r),
        right_iris_radius_orig=float(r_r),
        face_confidence=0.5,                       # heuristic — no FaceMesh score
        warnings=["eye_only_fallback"],
    )


def detect_and_crop_eyes(
    image_rgb: np.ndarray,
    debug: bool = False,
) -> EyeDetectionResult:
    """
    Detect the face, locate both eyes, and return padded eye crops.

    This is the single public function for Module 1.
    Call this with a raw RGB image from the phone camera.

    Args:
        image_rgb:  RGB numpy array (H, W, 3), dtype uint8.
                    Can be any resolution — downscaled internally if too large.
        debug:      If True, logs extra detail about each landmark.

    Returns:
        EyeDetectionResult with crops, landmarks, and iris radii.

    Raises:
        DetectionError: With a specific code (no_face, eyes_closed, etc.)
                        whenever a reliable detection cannot be made.
                        The caller should treat all DetectionErrors as INCONCLUSIVE.
    """

    # ── Step 1: Downscale if needed (performance + MediaPipe stability) ──
    image_rgb = downscale_if_needed(image_rgb)
    img_h, img_w = image_rgb.shape[:2]
    logger.debug(f"Module 1 input image: {img_w}x{img_h}")

    # ── Step 2: Run MediaPipe Face Mesh ──
    warnings: List[str] = []

    with _make_face_mesh() as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        logger.warning("Module 1: No face detected.")
        raise DetectionError("no_face")

    # We only handle one face (max_num_faces=1)
    face_landmarks = results.multi_face_landmarks[0].landmark

    # MediaPipe doesn't expose a face-level confidence score in static mode.
    # We use the presence of iris landmarks as our confidence proxy.
    face_confidence = 1.0

    logger.debug(f"Module 1: Face detected. Total landmarks: {len(face_landmarks)}")

    # ── Step 3: Validate iris landmarks are present ──
    # Iris landmarks (468–477) only appear when refine_landmarks=True
    # and the eye is clearly visible. If they're missing, the eye is likely
    # closed or occluded.
    total_landmarks = len(face_landmarks)

    if total_landmarks < 478:
        # refine_landmarks didn't return iris points
        logger.warning(f"Module 1: Only {total_landmarks} landmarks found — iris landmarks missing.")
        raise DetectionError("eyes_not_visible")

    # ── Step 4: Check eye visibility / openness ──
    left_visible  = _check_eye_visibility(face_landmarks, img_w, img_h,
                                           LEFT_IRIS_INDICES,  "left",  warnings)
    right_visible = _check_eye_visibility(face_landmarks, img_w, img_h,
                                           RIGHT_IRIS_INDICES, "right", warnings)

    if not left_visible and not right_visible:
        raise DetectionError("eyes_closed")

    if not left_visible:
        warnings.append("left_eye_not_visible")
        logger.warning("Module 1: Left eye not clearly visible.")

    if not right_visible:
        warnings.append("right_eye_not_visible")
        logger.warning("Module 1: Right eye not clearly visible.")

    # ── Step 5: Extract iris landmarks in pixel space ──
    left_iris_px  = _landmarks_to_pixels(face_landmarks, img_w, img_h, LEFT_IRIS_INDICES)
    right_iris_px = _landmarks_to_pixels(face_landmarks, img_w, img_h, RIGHT_IRIS_INDICES)

    if debug:
        logger.debug(f"Left iris landmarks (px): {left_iris_px}")
        logger.debug(f"Right iris landmarks (px): {right_iris_px}")

    # ── Step 6: Compute iris radii from landmarks ──
    left_iris_radius  = _iris_radius_from_landmarks(left_iris_px)
    right_iris_radius = _iris_radius_from_landmarks(right_iris_px)

    logger.debug(f"Module 1: Left iris radius: {left_iris_radius:.1f}px, "
                 f"Right iris radius: {right_iris_radius:.1f}px")

    if left_iris_radius < 5 or right_iris_radius < 5:
        warnings.append("iris_radius_very_small")
        logger.warning("Module 1: Iris radius suspiciously small — image may be too far away.")

    # ── Step 7: Compute bounding boxes from eye boundary landmarks ──
    left_bb  = _bounding_box_from_landmarks(face_landmarks, img_w, img_h, LEFT_EYE_BOUNDARY)
    right_bb = _bounding_box_from_landmarks(face_landmarks, img_w, img_h, RIGHT_EYE_BOUNDARY)

    # ── Step 8: Crop with padding ──
    left_crop,  left_box  = crop_region(
        image_rgb, *left_bb,
        pad_h=CROP_PAD_HORIZONTAL,
        pad_v=CROP_PAD_VERTICAL,
    )
    right_crop, right_box = crop_region(
        image_rgb, *right_bb,
        pad_h=CROP_PAD_HORIZONTAL,
        pad_v=CROP_PAD_VERTICAL,
    )

    logger.debug(f"Module 1: Left crop size: {left_crop.shape[1]}x{left_crop.shape[0]}, "
                 f"Right crop size: {right_crop.shape[1]}x{right_crop.shape[0]}")

    # ── Step 9: Validate crop size ──
    if left_crop.shape[1]  < MIN_CROP_WIDTH or left_crop.shape[0]  < MIN_CROP_HEIGHT:
        raise DetectionError("crop_too_small")

    if right_crop.shape[1] < MIN_CROP_WIDTH or right_crop.shape[0] < MIN_CROP_HEIGHT:
        raise DetectionError("crop_too_small")

    # ── Step 10: Return result ──
    return EyeDetectionResult(
        left_crop=left_crop,
        right_crop=right_crop,
        left_crop_box=left_box,
        right_crop_box=right_box,
        left_iris_landmarks=left_iris_px,
        right_iris_landmarks=right_iris_px,
        left_iris_radius_orig=left_iris_radius,
        right_iris_radius_orig=right_iris_radius,
        face_confidence=face_confidence,
        warnings=warnings,
    )
