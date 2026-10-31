"""
BeanHealth CLR — Tier 2 saccade detector (brief §3)
===================================================

Velocity-threshold (I-VT) detection on one eye's gaze track (deg) sampled at
~240 fps:

  1. Speed per sample from a 5-point central difference of each axis (less
     noise-sensitive than a two-point difference), combined as
     √(ḣ² + v̇²). Samples next to a gap (untracked) get no speed.
  2. Threshold = max(saccade_velocity_floor_deg_s,
                     saccade_noise_factor × median speed of the trace).
  3. Runs of supra-threshold samples separated by one sub-threshold sample
     are merged; runs shorter than saccade_min_duration_ms are dropped.
  4. Each saccade's onset / offset are extended outward while the speed
     keeps falling, to the local minimum, so the amplitude covers the whole
     movement and not only its fast core.
  5. The amplitude (mean of 3 samples at each end) must clear k × its own
     noise (decision_margin_k; position noise from the sample-to-sample
     differences): a velocity spike that goes nowhere is not a saccade.

Durations read long by about two samples at each end (the derivative's
half-width plus the extension to the speed minima, ≈ 15 ms at 240 fps);
onsets and amplitudes are not biased this way. Every setting comes from the
KB (capture.yaml). Tested on synthetic traces
(exam.tier2.sim) only — not yet on IR captures.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from exam.kb import load_kb
from exam.kb.loader import KnowledgeBase
from exam.models import Strict

SACCADES_VERSION = "0.1.0"


class Saccade(Strict):
    onset_ms: float
    offset_ms: float
    duration_ms: float
    amplitude_deg: float
    horizontal_deg: float          # + = toward the patient's right
    vertical_deg: float            # + = up
    peak_velocity_deg_s: float


class SaccadeTrace(Strict):
    threshold_deg_s: float
    saccades: List[Saccade]
    saccades_version: str = SACCADES_VERSION


def _speed(t_s: np.ndarray, h: np.ndarray, v: np.ndarray) -> np.ndarray:
    """5-point central-difference speed; NaN where a neighbour is missing or spacing is uneven."""
    n = len(t_s)
    speed = np.full(n, np.nan)
    for i in range(2, n - 2):
        window = slice(i - 2, i + 3)
        if np.isnan(h[window]).any() or np.isnan(v[window]).any():
            continue
        dt = (t_s[i + 2] - t_s[i - 2]) / 4.0
        if dt <= 0:
            continue
        dh = (-h[i + 2] + 8 * h[i + 1] - 8 * h[i - 1] + h[i - 2]) / (12 * dt)
        dv = (-v[i + 2] + 8 * v[i + 1] - 8 * v[i - 1] + v[i - 2]) / (12 * dt)
        speed[i] = math.hypot(dh, dv)
    return speed


def detect_saccades(t_ms: Sequence[float], gaze: Sequence[Optional[Tuple[float, float]]],
                    kb: Optional[KnowledgeBase] = None) -> SaccadeTrace:
    kb = kb or load_kb()
    floor = float(kb.scalar("saccade_velocity_floor_deg_s").value)      # type: ignore[arg-type]
    factor = float(kb.scalar("saccade_noise_factor").value)              # type: ignore[arg-type]
    min_ms = float(kb.scalar("saccade_min_duration_ms").value)           # type: ignore[arg-type]
    if len(t_ms) != len(gaze):
        raise ValueError("t_ms and gaze need the same length")
    t = np.asarray(t_ms, dtype=float)
    h = np.array([np.nan if g is None else g[0] for g in gaze])
    v = np.array([np.nan if g is None else g[1] for g in gaze])
    speed = _speed(t / 1000.0, h, v)
    finite = speed[np.isfinite(speed)]
    if finite.size == 0:
        return SaccadeTrace(threshold_deg_s=floor, saccades=[])
    threshold = max(floor, factor * float(np.median(finite)))
    k = float(kb.threshold("decision_margin_k").value)                    # type: ignore[arg-type]
    steps = np.hypot(np.diff(h), np.diff(v))
    steps = steps[np.isfinite(steps)]
    # Position noise per axis from the step sizes (Rayleigh median = 1.1774 σ√2 for white noise).
    sigma = float(np.median(steps)) / (1.1774 * math.sqrt(2)) if steps.size else 0.0
    amp_noise = sigma * math.sqrt(2.0 / 3.0) * math.sqrt(2.0)          # 3-sample means, two ends, 2 axes

    def end_mean(idx: int, side: int) -> Optional[Tuple[float, float]]:
        pick = [j for j in (idx, idx + side, idx + 2 * side) if 0 <= j < len(h) and not np.isnan(h[j])]
        return (float(np.mean(h[pick])), float(np.mean(v[pick]))) if pick else None

    above = np.where(np.isfinite(speed), speed > threshold, False)
    runs: List[List[int]] = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and (above[j + 1] or (j + 2 < len(above) and above[j + 2])):
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1

    out: List[Saccade] = []
    for a, b in runs:
        if t[b] - t[a] < min_ms - 1e-9:
            continue
        # Extend to the speed minima on either side.
        while a > 0 and np.isfinite(speed[a - 1]) and speed[a - 1] < speed[a]:
            a -= 1
        while b + 1 < len(speed) and np.isfinite(speed[b + 1]) and speed[b + 1] < speed[b]:
            b += 1
        start, end = end_mean(a, -1), end_mean(b, 1)
        if start is None or end is None:
            continue
        dh, dv = end[0] - start[0], end[1] - start[1]
        if math.hypot(dh, dv) <= k * amp_noise:
            continue
        out.append(Saccade(onset_ms=float(t[a]), offset_ms=float(t[b]), duration_ms=float(t[b] - t[a]),
                           amplitude_deg=math.hypot(dh, dv), horizontal_deg=dh, vertical_deg=dv,
                           peak_velocity_deg_s=float(np.nanmax(speed[a:b + 1]))))
    return SaccadeTrace(threshold_deg_s=threshold, saccades=out)
