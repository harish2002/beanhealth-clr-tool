"""
BeanHealth CLR Tool — Module Aggregate: Multi-Frame Analysis
=============================================================

Responsibility:
    Accept the per-frame pipeline results from N frames captured during a
    streaming session, reject bad frames (blinks, outliers, low confidence),
    and produce a single averaged result with a statistical confidence score.

    This module is the key improvement over single-frame analysis:
    - Single frame variance:  ±3–5°  (unreliable)
    - 10-frame averaged:      ±0.3–0.8°  (clinically useful)

Outlier rejection rules (frame is rejected if ANY of):
    1. status != "SUCCESS"          → blink / no flash / no face
    2. deviation is None / non-finite / negative / > 60°
    3. deviation is a statistical outlier  → > 1.5× IQR from median of batch
    (LOW pupil confidence frames are now accepted — IQR handles noise)

Confidence tier from standard deviation:
    std < 1.0°   → HIGH
    std < 3.0°   → MEDIUM
    std >= 3.0°  → LOW  (tell user to retry — too much movement)

Author: BeanHealth
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Confidence thresholds
# ─────────────────────────────────────────────────────────────

STD_HIGH_THRESHOLD   = 1.0   # degrees — std below this → HIGH confidence
STD_MEDIUM_THRESHOLD = 3.0   # degrees — std below this → MEDIUM confidence
IQR_OUTLIER_FACTOR   = 1.5   # frames outside 1.5×IQR are rejected
MIN_ACCEPTED_FRAMES  = 3     # need at least 3 good frames to report a result


# ─────────────────────────────────────────────────────────────
# Frame-level validation
# ─────────────────────────────────────────────────────────────

def _is_frame_usable(frame_report: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Decide whether a single-frame pipeline result is usable for averaging.

    Accepts SUCCESS frames at any confidence level — LOW confidence frames
    (Hough/landmark disagreement) still contain usable deviation readings and
    are far better than nothing.  Statistical outlier rejection in Step 2 will
    remove any wildly off readings regardless of confidence.

    Only hard-rejects:
        • Non-SUCCESS frames (blink / no flash / no face detected)
        • Frames with a mathematically invalid deviation value

    Returns:
        (True, "ok") if the frame passes all checks.
        (False, reason_code) if the frame should be rejected.
    """
    if frame_report.get("status") != "SUCCESS":
        return False, frame_report.get("reason", "pipeline_failed")

    deviation = frame_report.get("result", {}).get("deviation_degrees")
    if deviation is None or not math.isfinite(deviation) or deviation < 0:
        return False, "invalid_deviation"

    # Sanity bound — anything above 60° is almost certainly a detection error
    if deviation > 60.0:
        return False, "deviation_out_of_range"

    return True, "ok"


# ─────────────────────────────────────────────────────────────
# Statistical outlier rejection
# ─────────────────────────────────────────────────────────────

def _reject_statistical_outliers(
    deviations: List[float],
    indices: List[int],
) -> Tuple[List[float], List[int], List[int]]:
    """
    Remove statistical outliers using the IQR method.

    Args:
        deviations: List of deviation_degrees values from accepted frames.
        indices:    Corresponding original frame indices.

    Returns:
        (clean_deviations, clean_indices, rejected_indices)
    """
    if len(deviations) < 3:
        # Too few frames to apply IQR — keep all
        return deviations, indices, []

    arr = np.array(deviations)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr    = q3 - q1
    lower  = q1 - IQR_OUTLIER_FACTOR * iqr
    upper  = q3 + IQR_OUTLIER_FACTOR * iqr

    clean_devs, clean_idx, rejected_idx = [], [], []
    for dev, idx in zip(deviations, indices):
        if lower <= dev <= upper:
            clean_devs.append(dev)
            clean_idx.append(idx)
        else:
            rejected_idx.append(idx)
            logger.debug(f"[Aggregate] Frame {idx} rejected as outlier: {dev:.2f}° (IQR range [{lower:.2f}, {upper:.2f}])")

    return clean_devs, clean_idx, rejected_idx


# ─────────────────────────────────────────────────────────────
# Main aggregation function
# ─────────────────────────────────────────────────────────────

def aggregate_frame_results(
    frame_reports: List[Dict[str, Any]],
    best_frame_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Aggregate N per-frame pipeline reports into a single averaged result.

    Args:
        frame_reports:     List of report dicts, one per captured frame.
                           Each is the output of generate_report() from module7.
        best_frame_report: Optional — the single highest-quality frame report
                           to use for intermediate_images and annotated image.
                           If None, the median-closest accepted frame is used.

    Returns:
        Aggregated result dict. status is one of:
            "SUCCESS"      — enough accepted frames, result is reliable
            "INCONCLUSIVE" — too few accepted frames (< MIN_ACCEPTED_FRAMES)
    """
    total_frames = len(frame_reports)
    logger.info(f"[Aggregate] Processing {total_frames} frames")

    # ── Step 1: Filter usable frames ─────────────────────────

    accepted_deviations: List[float]      = []
    accepted_asymmetries: List[float]     = []
    accepted_indices: List[int]           = []
    rejected_frames: List[Dict[str, Any]] = []

    per_frame_readings: List[Optional[float]] = []

    for i, report in enumerate(frame_reports):
        usable, reason = _is_frame_usable(report)
        if usable:
            dev = report["result"]["deviation_degrees"]
            asym = report["result"]["asymmetry_score"]
            accepted_deviations.append(dev)
            accepted_asymmetries.append(asym)
            accepted_indices.append(i)
            per_frame_readings.append(round(dev, 2))
            logger.debug(f"[Aggregate] Frame {i}: ACCEPTED deviation={dev:.2f}°")
        else:
            rejected_frames.append({"frame": i, "reason": reason})
            per_frame_readings.append(None)
            logger.debug(f"[Aggregate] Frame {i}: REJECTED reason={reason}")

    frames_after_quality = len(accepted_deviations)

    # ── Step 2: Statistical outlier rejection ─────────────────

    if frames_after_quality >= 4:
        clean_devs, clean_idx, outlier_idx = _reject_statistical_outliers(
            accepted_deviations, accepted_indices
        )
        for idx in outlier_idx:
            per_frame_readings[idx] = None   # mark as rejected in strip
            rejected_frames.append({"frame": idx, "reason": "statistical_outlier"})

        accepted_deviations = clean_devs
        accepted_asymmetries = [accepted_asymmetries[accepted_indices.index(i)]
                                 for i in clean_idx]
        accepted_indices     = clean_idx
    else:
        outlier_idx = []

    frames_accepted = len(accepted_deviations)
    frames_rejected = total_frames - frames_accepted

    logger.info(
        f"[Aggregate] Accepted {frames_accepted}/{total_frames} frames "
        f"({frames_after_quality - frames_accepted} outliers removed)"
    )

    # ── Step 3: Check minimum frames ─────────────────────────

    if frames_accepted < MIN_ACCEPTED_FRAMES:
        return {
            "status":       "INCONCLUSIVE",
            "reason":       "insufficient_frames",
            "reason_human": (
                f"Only {frames_accepted} of {total_frames} frames were usable "
                f"(minimum {MIN_ACCEPTED_FRAMES} required). "
                "Please hold the phone steadier, ensure torch is on, and keep eyes open."
            ),
            "frames_total":    total_frames,
            "frames_accepted": frames_accepted,
            "frames_rejected": frames_rejected,
            "per_frame_readings": per_frame_readings,
            "flags": [r["reason"] for r in rejected_frames],
        }

    # ── Step 4: Compute statistics ───────────────────────────

    dev_array  = np.array(accepted_deviations)
    asym_array = np.array(accepted_asymmetries)

    dev_mean  = float(np.mean(dev_array))
    dev_std   = float(np.std(dev_array, ddof=1)) if frames_accepted > 1 else 0.0
    dev_min   = float(np.min(dev_array))
    dev_max   = float(np.max(dev_array))
    asym_mean = float(np.mean(asym_array))

    # ── Step 5: Confidence tier from std dev ─────────────────

    if dev_std < STD_HIGH_THRESHOLD:
        agg_confidence = "HIGH"
    elif dev_std < STD_MEDIUM_THRESHOLD:
        agg_confidence = "MEDIUM"
    else:
        agg_confidence = "LOW"

    # ── Step 6: Pick best frame for images ───────────────────

    # Best frame = the accepted frame whose deviation is closest to the mean
    if best_frame_report is None and accepted_indices:
        closest_idx = accepted_indices[
            int(np.argmin(np.abs(dev_array - dev_mean)))
        ]
        best_frame_report = frame_reports[closest_idx]

    # Extract per-eye directions from best frame for condition mapping
    best_technical = best_frame_report.get("technical", {}) if best_frame_report else {}
    best_result    = best_frame_report.get("result",    {}) if best_frame_report else {}

    # ── Step 7: Re-classify using averaged deviation ─────────

    from pipeline.module6_classify import classify
    from utils.constants import (
        SEVERITY_NORMAL, SEVERITY_MILD, SEVERITY_MODERATE, SEVERITY_SEVERE,
        SEVERITY_MILD_DEG, SEVERITY_MODERATE_DEG, SEVERITY_SEVERE_DEG,
    )

    # Derive severity tier from averaged deviation angle
    if dev_mean < SEVERITY_MILD_DEG:
        avg_severity = SEVERITY_NORMAL
    elif dev_mean < SEVERITY_MODERATE_DEG:
        avg_severity = SEVERITY_MILD
    elif dev_mean < SEVERITY_SEVERE_DEG:
        avg_severity = SEVERITY_MODERATE
    else:
        avg_severity = SEVERITY_SEVERE

    dominant_dir  = best_technical.get("dominant_eye", "left")
    dominant_dir_label = best_technical.get("left_direction", "nasal") \
        if dominant_dir != "right" else best_technical.get("right_direction", "nasal")

    classification = classify(dominant_dir_label, avg_severity)
    avg_urgency   = classification["urgency_tier"]
    avg_condition = classification["condition_name"]
    avg_icd10     = classification["icd10_code"]
    avg_referral  = classification["referral_recommendation"]
    avg_timeframe = classification["timeframe"]
    avg_narrative = classification["narrative"]

    # Collect all flags seen across accepted frames
    all_flags = list({
        f
        for i in accepted_indices
        for f in frame_reports[i].get("technical", {}).get("flags", [])
    })

    return {
        "status":         "SUCCESS",
        # ── Aggregated measurements ──
        "frames_total":    total_frames,
        "frames_accepted": frames_accepted,
        "frames_rejected": frames_rejected,
        "per_frame_readings": [round(v, 2) if v is not None else None
                                for v in per_frame_readings],
        "deviation_avg_deg":  round(dev_mean,  2),
        "deviation_std_deg":  round(dev_std,   2),
        "deviation_min_deg":  round(dev_min,   2),
        "deviation_max_deg":  round(dev_max,   2),
        "asymmetry_avg":      round(asym_mean, 4),
        "aggregate_confidence": agg_confidence,
        # ── Re-derived clinical result ──
        "result": {
            "urgency_tier":            avg_urgency,
            "condition_name":          avg_condition,
            "icd10_code":              avg_icd10,
            "deviation_degrees":       round(dev_mean, 2),
            "deviation_std_deg":       round(dev_std,  2),
            "asymmetry_score":         round(asym_mean, 4),
            "severity":                avg_severity,
            "referral_recommendation": avg_referral,
            "timeframe":               avg_timeframe,
            "narrative":               avg_narrative,
        },
        # ── Technical detail from best frame ──
        "technical": {
            **best_technical,
            "confidence":         agg_confidence,
            "flags":              all_flags,
        },
        # ── Images from best frame ──
        "intermediate_images":   best_frame_report.get("intermediate_images")  if best_frame_report else None,
        "annotated_image_b64":   best_frame_report.get("annotated_image_b64") if best_frame_report else None,
        "patient":               best_frame_report.get("patient")             if best_frame_report else None,
    }
