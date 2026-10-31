"""
BeanHealth CLR — Synthetic Tier 2 traces (test fixture, not physiology)
=======================================================================

Gaze tracks at 240 fps for the saccade detector and the cover-test
movement module. Saccades follow a minimum-jerk position profile; their
duration grows with amplitude by a fixture rule (20 ms + 2.5 ms/deg) chosen
only to give realistic-looking shapes — it is not a clinical value and must
never be read into the KB.

A cover trial: the fixing eye holds the target throughout; the covered eye
drifts by `under_cover_deg` while covered (a latent deviation), and on
uncover either recovers (moves back) or, with `manifest=True`, the fellow
eye takes up fixation instead.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from exam.capture.session import DeviceInfo
from exam.tier2.session import (CoverTestHeader, CoverTestSession, CoverTestSettings, CoverTrial, GazeSample)


def saccade_duration_ms(amplitude_deg: float) -> float:
    return 20.0 + 2.5 * abs(amplitude_deg)


def min_jerk(s: np.ndarray) -> np.ndarray:
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def trace(t_ms: np.ndarray, start: Tuple[float, float],
          moves: Sequence[Tuple[float, Tuple[float, float]]]) -> np.ndarray:
    """Position (n × 2, deg) starting at `start`, with a saccade of (dh, dv) starting at each onset (ms)."""
    pos = np.tile(np.asarray(start, dtype=float), (len(t_ms), 1))
    for onset, (dh, dv) in moves:
        dur = saccade_duration_ms(float(np.hypot(dh, dv)))
        s = np.clip((t_ms - onset) / dur, 0.0, 1.0)
        pos += min_jerk(s)[:, None] * np.array([dh, dv])
    return pos


def gaze_track(t_ms: np.ndarray, start: Tuple[float, float], moves: Sequence[Tuple[float, Tuple[float, float]]],
               noise_deg: float = 0.05, seed: Optional[int] = 0) -> List[Tuple[float, float]]:
    pos = trace(t_ms, start, moves)
    rng = np.random.default_rng(seed)
    pos = pos + rng.normal(0.0, noise_deg, pos.shape)
    return [(float(a), float(b)) for a, b in pos]


def cover_trial(covered_eye: str = "L", under_cover_deg: Tuple[float, float] = (0.0, 0.0), manifest: bool = False,
                fps: float = 240.0, cover_s: float = 3.0, after_s: float = 1.2, lead_s: float = 0.5,
                noise_deg: float = 0.05, seed: int = 0, gaps: Sequence[Tuple[float, float]] = ()) -> CoverTrial:
    """One cover / uncover of `covered_eye`; the movement under cover starts 300 ms after the cover."""
    cover_on = lead_s * 1000
    cover_off = cover_on + cover_s * 1000
    end = cover_off + after_s * 1000
    t = np.arange(0.0, end + 1e-9, 1000.0 / fps)
    fellow = "R" if covered_eye == "L" else "L"
    dh, dv = under_cover_deg
    moves: Dict[str, List[Tuple[float, Tuple[float, float]]]] = {covered_eye: [], fellow: []}
    if dh or dv:
        moves[covered_eye].append((cover_on + 300, (dh, dv)))
        if manifest:
            # The fellow eye stays deviated after the uncover: the covered eye keeps the target only
            # while the fellow is covered, so on uncover the fellow refixates (moves by −deviation).
            moves[fellow].append((cover_off + 150, (-dh, -dv)))
            moves[covered_eye].append((cover_off + 150, (-dh, -dv)))
        else:
            moves[covered_eye].append((cover_off + 150, (-dh, -dv)))
    tracks = {e: gaze_track(t, (0.0, 0.0), moves[e], noise_deg, seed + (0 if e == "R" else 1)) for e in ("R", "L")}
    samples = []
    for i, ti in enumerate(t):
        missing = any(a <= ti <= b for a, b in gaps)
        samples.append(GazeSample(t_ms=float(ti), eyes={e: None if missing else tracks[e][i] for e in ("R", "L")}))
    return CoverTrial(covered_eye=covered_eye, cover_on_ms=cover_on, cover_off_ms=cover_off, end_ms=float(t[-1]),
                      samples=samples)


def cover_session(trials: Sequence[CoverTrial], fps: float = 240.0, height_px: int = 1080) -> CoverTestSession:
    return CoverTestSession(
        header=CoverTestHeader(session_id="sim-cover", device=DeviceInfo(model="Simulated phone + IR clip"),
                               settings=CoverTestSettings(fps=fps, width_px=int(height_px * 16 / 9),
                                                          height_px=height_px),
                               distance_cm=36.0, started_at="2026-10-31T10:00:00Z"),
        trials=list(trials))
