"""
BeanHealth CLR Tool — Session-Level Device Auto-Calibration
============================================================

Phase 1: Extract a CLR fingerprint from the first 3 accepted frames of a
streaming session and derive adjusted Module 3 thresholds for the remainder
of that session.

No database or network call required.  All calibration is in-memory,
session-scoped, and derived from the CLR measurements themselves.

Key insight
-----------
Every phone has a different torch brightness, lens-to-torch offset, and ISP
processing pipeline.  These differences manifest as:

  • bloom_factor — how big the corneal reflex blob appears relative to the
                   iris (larger on phones with brighter / wider-angle torches)
  • peak_brightness — absolute peak pixel intensity in the eye crop
                      (lower on phones with weaker torches)

By measuring these from the first 3 frames we can:

  1. Widen or narrow the area filter in Module 3 so the right blob is always
     accepted regardless of torch brightness.
  2. Optionally lower the flash detection threshold for genuinely weak
     torches that still produce a valid reflex above ~200.

The clinical measurement (asymmetry) is NOT modified.  Only blob-detection
thresholds change.

Author: BeanHealth
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from typing import List, Optional

from utils.constants import (
    CLR_MAX_AREA_RATIO,
    CLR_MIN_AREA_RATIO,
    CLR_MIN_PEAK_BRIGHTNESS,
)

logger = logging.getLogger(__name__)


# ─── Calibration constants ────────────────────────────────────────────────────

# Geometric mean of the default [min, max] area ratio — serves as the expected
# CLR blob size for a "typical" device.
BASELINE_AREA_RATIO: float = math.sqrt(CLR_MIN_AREA_RATIO * CLR_MAX_AREA_RATIO)  # ≈ 0.027

# Clamping range for the bloom scale factor
BLOOM_SCALE_MIN: float = 0.3   # never shrink thresholds below 30 % of default
BLOOM_SCALE_MAX: float = 3.0   # never expand thresholds beyond 300 % of default

# Hard floors / ceilings for final threshold values
SESSION_MIN_AREA_FLOOR:   float = 0.001
SESSION_MAX_AREA_CEILING: float = 0.40

# Flash detection — we'll lower the threshold proportionally for weak torches
# but never below this absolute floor
MIN_ALLOWED_PEAK_THRESHOLD: float = 200.0
# If peak_avg is at or above this the torch is "normal" and threshold stays put
NORMAL_PEAK_BRIGHTNESS: float = 245.0


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CalibrationMeasurement:
    """
    CLR measurements extracted from one successfully processed calibration frame.
    Both eyes are measured independently and stored separately.
    """
    peak_brightness_left:  float   # max pixel in grayscale left-eye crop
    peak_brightness_right: float
    blob_area_ratio_left:  float   # CLR blob area / iris area, left eye
    blob_area_ratio_right: float


@dataclass
class SessionCalibration:
    """
    Session-level calibration profile derived from the first 3 accepted frames.
    Carry this into every subsequent call to detect_clr() within the same
    /analyse-stream request.
    """
    device_model:   str
    calibrated:     bool          # False if < 1 usable measurement
    calibration_frames: int       # number of frames that contributed

    # ── Adjusted Module 3 thresholds ──────────────────────────
    session_min_area_ratio:      float
    session_max_area_ratio:      float
    session_min_peak_brightness: float

    # ── Diagnostic values ─────────────────────────────────────
    bloom_factor:        float    # median_area_ratio / BASELINE_AREA_RATIO
    peak_brightness_avg: float    # mean peak across all cal-frame eyes


# ─── Public API ───────────────────────────────────────────────────────────────

def calibrate_from_measurements(
    measurements: List[CalibrationMeasurement],
    device_model: str,
) -> SessionCalibration:
    """
    Compute a SessionCalibration from 1–3 CalibrationMeasurement objects.

    If no valid measurements are available the returned SessionCalibration
    carries calibrated=False and uses the default constants unchanged —
    so callers can always use the result without branching.

    Args:
        measurements: CLR measurements from the first accepted frames.
        device_model: Human-readable device string from device_fingerprint.

    Returns:
        SessionCalibration ready to be passed to detect_clr().
    """
    _defaults = SessionCalibration(
        device_model=device_model,
        calibrated=False,
        calibration_frames=0,
        session_min_area_ratio=CLR_MIN_AREA_RATIO,
        session_max_area_ratio=CLR_MAX_AREA_RATIO,
        session_min_peak_brightness=CLR_MIN_PEAK_BRIGHTNESS,
        bloom_factor=1.0,
        peak_brightness_avg=0.0,
    )

    if not measurements:
        logger.info("[Calibrate] No calibration measurements — using defaults")
        return _defaults

    # ── Collect per-eye readings ───────────────────────────────
    area_ratios: List[float] = []
    peak_values: List[float] = []

    for m in measurements:
        if m.blob_area_ratio_left  > 0:
            area_ratios.append(m.blob_area_ratio_left)
        if m.blob_area_ratio_right > 0:
            area_ratios.append(m.blob_area_ratio_right)
        peak_values.extend([m.peak_brightness_left, m.peak_brightness_right])

    if not area_ratios:
        logger.info(
            f"[Calibrate] device='{device_model}' — no valid area ratios, using defaults"
        )
        peak_avg = float(statistics.mean(peak_values)) if peak_values else 0.0
        return SessionCalibration(
            **{**_defaults.__dict__, "calibration_frames": len(measurements),
               "peak_brightness_avg": peak_avg}
        )

    # ── Bloom factor ───────────────────────────────────────────
    # How large is the torch blob relative to what we expect?
    median_area = statistics.median(area_ratios)
    bloom = median_area / BASELINE_AREA_RATIO
    bloom = max(BLOOM_SCALE_MIN, min(BLOOM_SCALE_MAX, bloom))

    # ── Adjusted area bounds ───────────────────────────────────
    # Scale both bounds by the bloom factor so the filter window
    # stays centred on what this device actually produces.
    s_min = max(SESSION_MIN_AREA_FLOOR,    CLR_MIN_AREA_RATIO * bloom)
    s_max = min(SESSION_MAX_AREA_CEILING,  CLR_MAX_AREA_RATIO * bloom)

    # Safety guard — ensure min < max after clamping
    if s_min >= s_max:
        s_min = CLR_MIN_AREA_RATIO
        s_max = CLR_MAX_AREA_RATIO

    # ── Peak brightness threshold ──────────────────────────────
    peak_avg = float(statistics.mean(peak_values)) if peak_values else 0.0

    if MIN_ALLOWED_PEAK_THRESHOLD <= peak_avg < NORMAL_PEAK_BRIGHTNESS:
        # Linearly interpolate the threshold downward for weak-torch phones
        # 245 → keep at 240,  200 → lower to 200
        span = NORMAL_PEAK_BRIGHTNESS - MIN_ALLOWED_PEAK_THRESHOLD
        t = (peak_avg - MIN_ALLOWED_PEAK_THRESHOLD) / span      # 0.0–1.0
        s_peak = MIN_ALLOWED_PEAK_THRESHOLD + t * (CLR_MIN_PEAK_BRIGHTNESS - MIN_ALLOWED_PEAK_THRESHOLD)
        s_peak = max(MIN_ALLOWED_PEAK_THRESHOLD, s_peak)
    else:
        s_peak = CLR_MIN_PEAK_BRIGHTNESS

    cal = SessionCalibration(
        device_model=device_model,
        calibrated=True,
        calibration_frames=len(measurements),
        session_min_area_ratio=s_min,
        session_max_area_ratio=s_max,
        session_min_peak_brightness=s_peak,
        bloom_factor=bloom,
        peak_brightness_avg=peak_avg,
    )

    logger.info(
        f"[Calibrate] device='{device_model}' "
        f"frames={len(measurements)} "
        f"bloom={bloom:.2f}× "
        f"area=[{s_min:.4f}, {s_max:.4f}] "
        f"peak_avg={peak_avg:.1f} "
        f"peak_thr={s_peak:.1f}"
    )

    return cal
