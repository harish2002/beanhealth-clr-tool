"""
BeanHealth CLR Tool — Module 7: Report Generation
==================================================

Responsibility:
    Assemble all upstream module outputs into a single structured JSON report
    and produce an annotated JPEG image with all detected landmarks overlaid.

    Handles three output states:
        SUCCESS      — full pipeline ran; result + annotated image returned
        INCONCLUSIVE — pipeline was halted by DetectionError or CLRError;
                       human-readable reason returned; NO triage result
        ERROR        — unexpected crash; 500-class response; no result

Report JSON schema (SUCCESS):
    {
        "status":               "SUCCESS",
        "patient":              {"name": str, "age": int},
        "result": {
            "urgency_tier":             str,
            "condition_name":           str,
            "icd10_code":               str,
            "deviation_degrees":        float,
            "asymmetry_score":          float,
            "severity":                 str,
            "referral_recommendation":  str,
            "timeframe":                str,
            "narrative":                str,
        },
        "technical": {
            "left_pupil":               [x, y],
            "right_pupil":              [x, y],
            "left_clr":                 [x, y],
            "right_clr":                [x, y],
            "left_displacement_norm":   float,
            "right_displacement_norm":  float,
            "left_direction":           str,
            "right_direction":          str,
            "deviation_mm":             float,
            "dominant_eye":             str,
            "confidence":               str,
            "flags":                    [str],
        },
        "annotated_image_b64":  str,   # base64 JPEG
        "timestamp":            str,   # ISO 8601
    }

Annotation legend (drawn on full image in original coordinates):
    • Blue filled dot  — pupil centre (both eyes)
    • Amber filled dot — CLR position (both eyes)
    • White line       — displacement vector (pupil → CLR)
    • Coloured border  — Red=URGENT, Orange=ROUTINE, Yellow=MONITOR, Green=NORMAL

Pipeline position:  SEVENTH — depends on all previous modules.
Failure behaviour:  Always returns a dict — never raises.

Author: BeanHealth
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from pipeline.module1_detection  import EyeDetectionResult
from pipeline.module2_pupil      import PupilResult
from pipeline.module3_clr        import CLRResult
from pipeline.module4_displacement import DisplacementResult
from pipeline.module5_asymmetry  import AsymmetryResult
from pipeline.module6_classify   import ClassificationResult
from utils.constants import (
    URGENCY_URGENT,
    URGENCY_ROUTINE,
    URGENCY_MONITOR,
    URGENCY_NORMAL,
)
from utils.exceptions import CLRPipelineError, DetectionError, CLRError
from utils.image_utils import combine_crops_to_base64

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Annotation colour constants  (BGR — OpenCV convention)
# ─────────────────────────────────────────────────────────────

_COLOUR_PUPIL    = (235, 100,  20)   # blue
_COLOUR_CLR      = ( 20, 180, 255)   # amber / orange-yellow
_COLOUR_VECTOR   = (255, 255, 255)   # white
_COLOUR_IRIS_RING = (100, 220, 100)  # soft green ring

# Border colours by urgency tier
_BORDER_COLOUR = {
    URGENCY_URGENT:  ( 20,  20, 220),   # red
    URGENCY_ROUTINE: ( 20, 140, 255),   # orange
    URGENCY_MONITOR: ( 20, 210, 255),   # yellow
    URGENCY_NORMAL:  ( 60, 200,  60),   # green
}

_BORDER_THICKNESS = 12   # px each side


# ─────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────

def _crop_to_full(
    pt: Tuple[float, float],
    box: Tuple[int, int, int, int],
) -> Tuple[int, int]:
    """
    Map a point in crop-local coordinates back to full-image coordinates.

    Args:
        pt:  (x, y) in crop space
        box: (x1, y1, x2, y2) bounding box of the crop in full image

    Returns:
        (x, y) in full-image pixel space (integers, clamped to image)
    """
    x1, y1 = box[0], box[1]
    return (int(round(pt[0] + x1)), int(round(pt[1] + y1)))


# ─────────────────────────────────────────────────────────────
# Image annotation
# ─────────────────────────────────────────────────────────────

def _draw_eye_annotations(
    img: np.ndarray,
    pupil_full: Tuple[int, int],
    clr_full:   Tuple[int, int],
    iris_radius_full: float,
) -> None:
    """
    Draw pupil dot, CLR dot, iris ring, and displacement vector for one eye.
    Mutates img in-place.
    """
    # Iris ring (subtle guide)
    cv2.circle(img, pupil_full, int(iris_radius_full), _COLOUR_IRIS_RING, 1, cv2.LINE_AA)

    # Displacement vector: pupil → CLR
    if pupil_full != clr_full:
        cv2.line(img, pupil_full, clr_full, _COLOUR_VECTOR, 1, cv2.LINE_AA)

    # CLR amber dot (draw before pupil so pupil is on top)
    cv2.circle(img, clr_full,   6, _COLOUR_CLR,   -1, cv2.LINE_AA)
    cv2.circle(img, clr_full,   7, (255,255,255),   1, cv2.LINE_AA)  # white outline

    # Pupil blue dot
    cv2.circle(img, pupil_full, 6, _COLOUR_PUPIL, -1, cv2.LINE_AA)
    cv2.circle(img, pupil_full, 7, (255,255,255),   1, cv2.LINE_AA)  # white outline


def _draw_zoomed_annotations(
    crop: np.ndarray,
    pupil: Optional[Tuple[float, float]] = None,
    clr: Optional[Tuple[float, float]] = None,
    iris_r: Optional[float] = None,
    draw_vector: bool = False,
    side_label: Optional[str] = None,
    measurement_label: Optional[str] = None,
) -> np.ndarray:
    """Helper to draw annotations directly onto an eye crop for the UI."""
    out = crop.copy()

    if iris_r and pupil:
        cv2.circle(out, (int(pupil[0]), int(pupil[1])), int(iris_r), _COLOUR_IRIS_RING, 1, cv2.LINE_AA)

    if draw_vector and pupil and clr:
        if pupil != clr:
            cv2.line(out, (int(pupil[0]), int(pupil[1])), (int(clr[0]), int(clr[1])), _COLOUR_VECTOR, 2, cv2.LINE_AA)

    if clr:
        cv2.circle(out, (int(clr[0]), int(clr[1])), 4, _COLOUR_CLR, -1, cv2.LINE_AA)
        cv2.circle(out, (int(clr[0]), int(clr[1])), 5, (255, 255, 255), 1, cv2.LINE_AA)

    if pupil:
        cv2.circle(out, (int(pupil[0]), int(pupil[1])), 4, _COLOUR_PUPIL, -1, cv2.LINE_AA)
        cv2.circle(out, (int(pupil[0]), int(pupil[1])), 5, (255, 255, 255), 1, cv2.LINE_AA)

    # Side label (L / R) — top-left corner
    if side_label:
        _draw_label(out, side_label, (5, 14), colour=(220, 220, 220), scale=0.45, thickness=1)

    # Measurement label — bottom-left corner
    if measurement_label:
        h = out.shape[0]
        _draw_label(out, measurement_label, (4, h - 5), colour=(255, 235, 80), scale=0.40, thickness=1)

    return out


def _generate_grayscale_clahe_crops(
    left_crop: np.ndarray,
    right_crop: np.ndarray,
) -> str:
    """
    Convert both eye crops to grayscale and apply CLAHE contrast enhancement.
    Returns side-by-side crops as base64 JPEG.

    This preprocessing step boosts the bright CLR spot against the iris so the
    percentile threshold in Module 3 can isolate it reliably.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    left_gray      = cv2.cvtColor(left_crop,  cv2.COLOR_RGB2GRAY)
    right_gray     = cv2.cvtColor(right_crop, cv2.COLOR_RGB2GRAY)
    left_enhanced  = clahe.apply(left_gray)
    right_enhanced = clahe.apply(right_gray)

    left_rgb  = cv2.cvtColor(left_enhanced,  cv2.COLOR_GRAY2RGB)
    right_rgb = cv2.cvtColor(right_enhanced, cv2.COLOR_GRAY2RGB)

    # Add L / R labels
    _draw_label(left_rgb,  "L", (5, 14), colour=(220, 220, 220), scale=0.45)
    _draw_label(right_rgb, "R", (5, 14), colour=(220, 220, 220), scale=0.45)

    return combine_crops_to_base64(left_rgb, right_rgb)


def _generate_result_overlay(
    annotated_img:  np.ndarray,
    asymmetry:      AsymmetryResult,
    classification: ClassificationResult,
    displacement:   DisplacementResult,
) -> str:
    """
    Final step image: full annotated image with a semi-transparent banner
    showing deviation angle, asymmetry score, condition, ICD-10, and
    per-eye displacement — everything a judge needs at a glance.
    """
    result = annotated_img.copy()
    h, w   = result.shape[:2]

    # Semi-transparent dark bar at the bottom
    bar_h   = 58
    overlay = result.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, result, 0.28, 0, result)

    # Line 1 — deviation angle + asymmetry score (large, yellow)
    line1 = f"{asymmetry.deviation_degrees:.1f} deg  |  asymmetry {asymmetry.asymmetry_score:.3f}"
    _draw_label(result, line1, (12, h - bar_h + 17), colour=(255, 235, 80), scale=0.50, thickness=1)

    # Line 2 — condition + ICD-10 + urgency tier
    line2 = f"{classification.condition_name}  |  {classification.icd10_code}  |  {classification.urgency_tier}"
    _draw_label(result, line2, (12, h - bar_h + 34), colour=(200, 200, 200), scale=0.42, thickness=1)

    # Line 3 — per-eye displacement in iris radii
    line3 = (
        f"L: {displacement.left_displacement_norm:.3f} iris-r  "
        f"R: {displacement.right_displacement_norm:.3f} iris-r"
    )
    _draw_label(result, line3, (12, h - bar_h + 50), colour=(160, 160, 160), scale=0.38, thickness=1)

    return _image_to_base64_jpeg(result)


def _draw_urgency_border(img: np.ndarray, urgency_tier: str) -> np.ndarray:
    """
    Add a coloured border rectangle around the full image to indicate urgency.
    Returns a new image (does not mutate original).
    """
    colour = _BORDER_COLOUR.get(urgency_tier, _BORDER_COLOUR[URGENCY_NORMAL])
    h, w = img.shape[:2]
    t = _BORDER_THICKNESS

    bordered = img.copy()
    # Top / bottom / left / right bars
    bordered[:t, :, :]   = colour
    bordered[h-t:, :, :] = colour
    bordered[:, :t, :]   = colour
    bordered[:, w-t:, :] = colour
    return bordered


def _draw_label(
    img:     np.ndarray,
    text:    str,
    pos:     Tuple[int, int],
    colour:  Tuple[int, int, int] = (255, 255, 255),
    scale:   float = 0.45,
    thickness: int = 1,
) -> None:
    """Draw a text label with a dark drop-shadow for readability."""
    shadow_offset = 1
    shadow_pos = (pos[0] + shadow_offset, pos[1] + shadow_offset)
    cv2.putText(img, text, shadow_pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, thickness, cv2.LINE_AA)


def _annotate_image(
    original_img:    np.ndarray,
    detection:       EyeDetectionResult,
    pupil_result:    PupilResult,
    clr_result:      CLRResult,
    displacement:    DisplacementResult,
    classification:  ClassificationResult,
) -> np.ndarray:
    """
    Produce the fully-annotated image.

    All drawing is done on a copy of the original RGB image.
    Returns the annotated RGB image.
    """
    annotated = original_img.copy()

    # ── Map crop-local coords → full-image coords ──────────────

    left_box  = detection.left_crop_box
    right_box = detection.right_crop_box

    # Scale iris radius from crop space to full image
    # (boxes are in full coords; crop width == box width)
    left_crop_w  = left_box[2]  - left_box[0]
    right_crop_w = right_box[2] - right_box[0]
    left_iris_r_full  = pupil_result.left_iris_radius  * (left_crop_w  / detection.left_crop.shape[1])
    right_iris_r_full = pupil_result.right_iris_radius * (right_crop_w / detection.right_crop.shape[1])

    left_pupil_full  = _crop_to_full(pupil_result.left_pupil,  left_box)
    right_pupil_full = _crop_to_full(pupil_result.right_pupil, right_box)
    left_clr_full    = _crop_to_full(clr_result.left_clr,      left_box)
    right_clr_full   = _crop_to_full(clr_result.right_clr,     right_box)

    # ── Draw per-eye annotations ──────────────────────────────

    _draw_eye_annotations(annotated, left_pupil_full,  left_clr_full,  left_iris_r_full)
    _draw_eye_annotations(annotated, right_pupil_full, right_clr_full, right_iris_r_full)

    # ── Eye side labels ───────────────────────────────────────

    lx1, ly1 = left_box[0],  left_box[1]
    rx1, ry1 = right_box[0], right_box[1]
    _draw_label(annotated, "L", (lx1 + 4, ly1 + 16), colour=(200, 200, 200))
    _draw_label(annotated, "R", (rx1 + 4, ry1 + 16), colour=(200, 200, 200))

    # ── Urgency banner at top ─────────────────────────────────

    urgency   = classification.urgency_tier
    condition = classification.condition_name
    banner    = f"{urgency}  |  {condition}  |  ICD {classification.icd10_code}"
    _draw_label(annotated, banner, (16, 28),
                colour=_BORDER_COLOUR.get(urgency, (255,255,255)),
                scale=0.5, thickness=1)

    # ── Urgency border ────────────────────────────────────────

    annotated = _draw_urgency_border(annotated, urgency)

    return annotated


def _image_to_base64_jpeg(img_rgb: np.ndarray, quality: int = 90) -> str:
    """
    Encode an RGB numpy image to a base64 JPEG string.

    Args:
        img_rgb:  H×W×3 uint8 array in RGB colour order
        quality:  JPEG quality (0–100)

    Returns:
        base64-encoded JPEG bytes as a UTF-8 string
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─────────────────────────────────────────────────────────────
# Report builders
# ─────────────────────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _confidence_label(pupil_result: PupilResult) -> str:
    """
    Derive overall confidence from the two pupil confidence levels.
    Worst-case wins: HIGH+LOW → LOW.
    """
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    worst = max(
        pupil_result.left_confidence,
        pupil_result.right_confidence,
        key=lambda c: order.get(c, 2),
    )
    return worst


def build_success_report(
    patient_name:   str,
    patient_age:    int,
    original_img:   np.ndarray,
    detection:      EyeDetectionResult,
    pupil_result:   PupilResult,
    clr_result:     CLRResult,
    displacement:   DisplacementResult,
    asymmetry:      AsymmetryResult,
    classification: ClassificationResult,
) -> Dict[str, Any]:
    """
    Assemble a SUCCESS report from all upstream module outputs.

    Args:
        patient_name:    From the API request.
        patient_age:     From the API request.
        original_img:    The full RGB image passed to Module 1.
        detection:       Module 1 output.
        pupil_result:    Module 2 output.
        clr_result:      Module 3 output.
        displacement:    Module 4 output.
        asymmetry:       Module 5 output.
        classification:  Module 6 output.

    Returns:
        dict matching the Module 7 report schema.
    """
    # Merge all flags from every module (deduplicated, order preserved)
    seen: set = set()
    all_flags: List[str] = []
    for flag in (
        detection.warnings
        + pupil_result.flags
        + clr_result.flags
        + displacement.flags
        + asymmetry.flags
        + classification.flags
    ):
        if flag not in seen:
            seen.add(flag)
            all_flags.append(flag)

    # Annotated image
    annotated_img = _annotate_image(
        original_img, detection, pupil_result,
        clr_result, displacement, classification,
    )
    img_b64 = _image_to_base64_jpeg(annotated_img)

    # ── Intermediate pipeline images (6 steps) ───────────────────

    m1_left  = detection.left_crop
    m1_right = detection.right_crop

    # Step 1 — raw eye crops with L/R labels
    m1_left_labelled  = _draw_zoomed_annotations(m1_left,  side_label="L")
    m1_right_labelled = _draw_zoomed_annotations(m1_right, side_label="R")
    m1_b64 = combine_crops_to_base64(m1_left_labelled, m1_right_labelled)

    # Step 2 — grayscale + CLAHE contrast enhancement
    m2_b64 = _generate_grayscale_clahe_crops(m1_left, m1_right)

    # Step 3 — pupil centre localisation (blue dot + iris ring)
    m3_left  = _draw_zoomed_annotations(m1_left,  pupil=pupil_result.left_pupil,  iris_r=pupil_result.left_iris_radius,  side_label="L")
    m3_right = _draw_zoomed_annotations(m1_right, pupil=pupil_result.right_pupil, iris_r=pupil_result.right_iris_radius, side_label="R")
    m3_b64   = combine_crops_to_base64(m3_left, m3_right)

    # Step 4 — CLR bright spot detection (amber dot)
    m4_left  = _draw_zoomed_annotations(m1_left,  clr=clr_result.left_clr,  side_label="L")
    m4_right = _draw_zoomed_annotations(m1_right, clr=clr_result.right_clr, side_label="R")
    m4_b64   = combine_crops_to_base64(m4_left, m4_right)

    # Step 5 — displacement vector with per-eye measurements
    l_disp_norm = round(displacement.left_displacement_norm,  3)
    r_disp_norm = round(displacement.right_displacement_norm, 3)
    m5_left  = _draw_zoomed_annotations(
        m1_left,
        pupil=pupil_result.left_pupil,  clr=clr_result.left_clr,
        iris_r=pupil_result.left_iris_radius,
        draw_vector=True,
        side_label="L",
        measurement_label=f"{l_disp_norm}r",
    )
    m5_right = _draw_zoomed_annotations(
        m1_right,
        pupil=pupil_result.right_pupil, clr=clr_result.right_clr,
        iris_r=pupil_result.right_iris_radius,
        draw_vector=True,
        side_label="R",
        measurement_label=f"{r_disp_norm}r",
    )
    m5_b64 = combine_crops_to_base64(m5_left, m5_right)

    # Step 6 — final annotated full image with clinical measurements overlay
    m6_b64 = _generate_result_overlay(annotated_img, asymmetry, classification, displacement)

    report = {
        "status": "SUCCESS",
        "patient": {
            "name": patient_name,
            "age":  patient_age,
        },
        "result": {
            "urgency_tier":            classification.urgency_tier,
            "condition_name":          classification.condition_name,
            "icd10_code":              classification.icd10_code,
            "deviation_degrees":       round(asymmetry.deviation_degrees, 2),
            "asymmetry_score":         round(asymmetry.asymmetry_score,   4),
            "asymmetry_degrees":       round(asymmetry.asymmetry_degrees, 2),
            "severity":                asymmetry.severity,
            "referral_recommendation": classification.referral_recommendation,
            "timeframe":               classification.timeframe,
            "narrative":               classification.narrative,
        },
        "technical": {
            "left_pupil":               list(pupil_result.left_pupil),
            "right_pupil":              list(pupil_result.right_pupil),
            "left_clr":                 list(clr_result.left_clr),
            "right_clr":                list(clr_result.right_clr),
            "left_displacement_norm":   round(displacement.left_displacement_norm,  4),
            "right_displacement_norm":  round(displacement.right_displacement_norm, 4),
            "left_direction":           displacement.left_direction,
            "right_direction":          displacement.right_direction,
            "deviation_mm":             round(asymmetry.deviation_mm, 3),
            "dominant_eye":             asymmetry.dominant_eye,
            "confidence":               _confidence_label(pupil_result),
            "flags":                    all_flags,
        },
        "intermediate_images": {
            "module1_crops":  m1_b64,
            "module2_clahe":  m2_b64,
            "module3_pupil":  m3_b64,
            "module4_clr":    m4_b64,
            "module5_vector": m5_b64,
            "module6_result": m6_b64,
        },
        "annotated_image_b64": img_b64,
        "timestamp":           _timestamp(),
    }

    logger.info(
        f"[M7] Report built: status=SUCCESS | "
        f"urgency={classification.urgency_tier} | "
        f"condition={classification.condition_name} | "
        f"angle={asymmetry.deviation_degrees:.1f}° | "
        f"flags={all_flags}"
    )

    return report


def build_inconclusive_report(
    error:        CLRPipelineError,
    patient_name: Optional[str] = None,
    patient_age:  Optional[int] = None,
    extra_flags:  Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Assemble an INCONCLUSIVE report from a pipeline error.

    This is returned when a DetectionError or CLRError halts the pipeline.
    No triage result is included — only the reason and human explanation.

    Args:
        error:        The exception that caused the halt.
        patient_name: From the API request (may be None in early failure).
        patient_age:  From the API request (may be None in early failure).
        extra_flags:  Any flags accumulated before the halt.

    Returns:
        dict with status=INCONCLUSIVE.
    """
    flags = list(extra_flags or [])
    if error.code not in flags:
        flags.append(error.code)

    report: Dict[str, Any] = {
        "status":       "INCONCLUSIVE",
        "reason":       error.code,
        "reason_human": error.human_message,
        "flags":        flags,
        "timestamp":    _timestamp(),
    }

    if patient_name is not None:
        report["patient"] = {"name": patient_name, "age": patient_age}

    logger.warning(
        f"[M7] INCONCLUSIVE: code={error.code} | reason={error.human_message}"
    )

    return report


def build_error_report(
    exc:          Exception,
    patient_name: Optional[str] = None,
    patient_age:  Optional[int] = None,
) -> Dict[str, Any]:
    """
    Assemble an ERROR report for unexpected crashes.

    The raw traceback is NOT included in the response (logged server-side).

    Args:
        exc:          The unexpected exception.
        patient_name: From the API request.
        patient_age:  From the API request.

    Returns:
        dict with status=ERROR.
    """
    report: Dict[str, Any] = {
        "status":    "ERROR",
        "message":   "An unexpected error occurred. Please retry.",
        "timestamp": _timestamp(),
    }

    if patient_name is not None:
        report["patient"] = {"name": patient_name, "age": patient_age}

    logger.exception(f"[M7] Unexpected pipeline crash: {exc}")

    return report


# ─────────────────────────────────────────────────────────────
# Public convenience function — wraps the full pipeline
# ─────────────────────────────────────────────────────────────

def generate_report(
    patient_name:   str,
    patient_age:    int,
    original_img:   Optional[np.ndarray],
    detection:      Optional[EyeDetectionResult]  = None,
    pupil_result:   Optional[PupilResult]         = None,
    clr_result:     Optional[CLRResult]           = None,
    displacement:   Optional[DisplacementResult]  = None,
    asymmetry:      Optional[AsymmetryResult]     = None,
    classification: Optional[ClassificationResult] = None,
    error:          Optional[Exception]            = None,
) -> Dict[str, Any]:
    """
    Generate the final report.

    If `error` is provided, builds INCONCLUSIVE or ERROR report accordingly.
    If all module outputs are provided, builds a SUCCESS report.

    Args:
        patient_name:    Patient name from request.
        patient_age:     Patient age from request.
        original_img:    Full RGB image (required for SUCCESS report).
        detection:       Module 1 output (required for SUCCESS).
        pupil_result:    Module 2 output (required for SUCCESS).
        clr_result:      Module 3 output (required for SUCCESS).
        displacement:    Module 4 output (required for SUCCESS).
        asymmetry:       Module 5 output (required for SUCCESS).
        classification:  Module 6 output (required for SUCCESS).
        error:           Exception that halted the pipeline (if any).

    Returns:
        Report dict (always — never raises).
    """
    try:
        if error is not None:
            if isinstance(error, (DetectionError, CLRError)):
                return build_inconclusive_report(
                    error, patient_name, patient_age,
                )
            if isinstance(error, CLRPipelineError):
                return build_inconclusive_report(
                    error, patient_name, patient_age,
                )
            return build_error_report(error, patient_name, patient_age)

        # All modules must have run successfully
        if any(x is None for x in [
            original_img, detection, pupil_result, clr_result,
            displacement, asymmetry, classification,
        ]):
            raise ValueError(
                "generate_report called with missing module outputs but no error set."
            )

        return build_success_report(
            patient_name, patient_age,
            original_img, detection, pupil_result,
            clr_result, displacement, asymmetry, classification,
        )

    except (DetectionError, CLRError, CLRPipelineError) as e:
        return build_inconclusive_report(e, patient_name, patient_age)
    except Exception as e:
        return build_error_report(e, patient_name, patient_age)
