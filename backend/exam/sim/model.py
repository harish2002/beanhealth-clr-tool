"""
BeanHealth CLR — Synthetic Muscle Model
========================================

Generates examination measurements from a simple linear model of the six
extraocular muscles per eye, so the rules engine can be tested before the CV
pipeline exists (brief §9).

THIS IS A TEST FIXTURE, NOT PHYSIOLOGY. Its gains and sizes are arbitrary
numbers chosen to make patterns clearly visible; nothing here is a clinical
value, and nothing here may be read into the KB.

Model, per eye, for a gaze target (h, v) in degrees (h + = patient's right,
v + = up). Each muscle has a strength (1 = normal, 0 = paralysed).

  • Abduction is the eye-relative horizontal (+ = out): the right eye
    abducts on right gaze, the left eye on left gaze.
  • Horizontal movement is driven by LR (out) or MR (in).
  • Vertical movement is shared: the vertical recti work best in abduction,
    the obliques in adduction. Weight w = 0.5 + out / (2·A), clipped to
    [0, 1]: SR / IR carry w of it, IO / SO carry 1 − w.
  • A weak muscle also lets the eye drift from its unopposed antagonist
    (tonic imbalance), scaled by the same weights — so a vertical palsy is
    largest in the muscle's field of action (Parks step 2).
  • Head tilt: the eye on the tilt side intorts (SR + SO), the other
    extorts (IR + IO). In each pair the vertical actions cancel unless one
    muscle is weak (Bielschowsky, Parks step 3).

Deviation (the sign convention of exam.models):
  horizontal  eso = −(outward error of R + outward error of L)
  vertical    right hyper = upward error of R − upward error of L
Both eyes' errors count, so it does not matter which eye fixes. Secondary
deviation (fixing with the weak eye) is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from exam.models import (
    PROTOCOL_POSITIONS,
    EyeMeasures,
    ExamMeasurements,
    GazeCell,
    GazeGrid,
    Measured,
    TiltMeasurements,
    degrees_to_pd,
)

MUSCLES = ("MR", "LR", "SR", "IR", "SO", "IO")

GAZE_DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "primary": (0, 0), "right": (1, 0), "left": (-1, 0), "up": (0, 1), "down": (0, -1),
    "up_right": (1, 1), "up_left": (-1, 1), "down_right": (1, -1), "down_left": (-1, -1),
}


@dataclass(frozen=True)
class SimCase:
    """A synthetic patient. Strengths default to 1.0 (normal)."""
    name: str
    weak: Dict[str, float] = field(default_factory=dict)       # e.g. {"R-LR": 0.2}
    amplitude_deg: float = 25.0           # gaze target eccentricity
    tonic_h_deg: float = 10.0             # horizontal drift per unit weakness
    tonic_v_deg: float = 6.0              # vertical drift per unit weakness
    tilt_deg: float = 4.0                 # vertical shift on head tilt per unit weakness
    # Added after the muscle model, in PD:
    comitant_pd: Tuple[float, float] = (0.0, 0.0)   # (horizontal, vertical)
    h_per_up_pd: float = 0.0              # eso added per unit of up-gaze (A / V patterns)
    # Lids and pupils (mm / ratios); defaults are arbitrary "normal" fixture values.
    fissure_mm: Dict[str, float] = field(default_factory=lambda: {"R": 10.0, "L": 10.0})
    fissure_ratio_adduction: Dict[str, float] = field(default_factory=lambda: {"R": 1.0, "L": 1.0})
    pupil_mm: Dict[str, float] = field(default_factory=lambda: {"R": 4.0, "L": 4.0})
    scleral_show_mm: Dict[str, float] = field(default_factory=lambda: {"R": 0.0, "L": 0.0})


@dataclass(frozen=True)
class Noise:
    """
    Reported standard errors, and whether to perturb values by them.

    seed=None gives exact values (still reported with these errors), which
    keeps logic tests deterministic; a seed draws Gaussian noise of that size.
    """
    pd: float = 1.0
    fraction: float = 0.03
    mm: float = 0.2
    ratio: float = 0.03
    seed: Optional[int] = None
    confidence: str = "HIGH"


def strength(case: SimCase, eye: str, muscle: str) -> float:
    return case.weak.get(f"{eye}-{muscle}", 1.0)


def eye_position(case: SimCase, eye: str, h: float, v: float) -> Tuple[float, float]:
    """(outward, upward) eye-relative position in degrees for gaze target (h, v)."""
    s = {m: strength(case, eye, m) for m in MUSCLES}
    A = case.amplitude_deg
    out = h if eye == "R" else -h
    w = min(1.0, max(0.0, 0.5 + out / (2 * A)))

    out_move = out * (s["LR"] if out > 0 else s["MR"])
    if v > 0:
        up_move = v * (w * s["SR"] + (1 - w) * s["IO"])
    else:
        up_move = v * (w * s["IR"] + (1 - w) * s["SO"])

    out_drift = case.tonic_h_deg * ((1 - s["MR"]) - (1 - s["LR"]))
    up_drift = case.tonic_v_deg * (w * (1 - s["IR"]) - w * (1 - s["SR"])
                                   + (1 - w) * (1 - s["SO"]) - (1 - w) * (1 - s["IO"]))
    return out_move + out_drift, up_move + up_drift


def tilt_shift(case: SimCase, eye: str, tilt_side: str) -> float:
    """Upward shift of this eye on head tilt toward tilt_side."""
    s = {m: strength(case, eye, m) for m in MUSCLES}
    eye_side = "right" if eye == "R" else "left"
    if eye_side == tilt_side:   # intorting: SR elevates, SO depresses
        return case.tilt_deg * ((1 - s["SO"]) - (1 - s["SR"]))
    return case.tilt_deg * ((1 - s["IR"]) - (1 - s["IO"]))   # extorting: IO elevates, IR depresses


def true_deviation(case: SimCase, gaze: str) -> Tuple[float, float]:
    """Noise-free (horizontal_pd, vertical_pd) for one gaze."""
    dh, dv = GAZE_DIRECTIONS[gaze]
    A = case.amplitude_deg
    h, v = dh * A, dv * A
    err = {}
    for eye in ("R", "L"):
        out_target = h if eye == "R" else -h
        out_pos, up_pos = eye_position(case, eye, h, v)
        err[eye] = (out_pos - out_target, up_pos - v)
    eso_deg = -(err["R"][0] + err["L"][0])
    hyper_deg = err["R"][1] - err["L"][1]
    horizontal = degrees_to_pd(eso_deg) + case.comitant_pd[0] + case.h_per_up_pd * dv
    vertical = degrees_to_pd(hyper_deg) + case.comitant_pd[1]
    return horizontal, vertical


def true_tilt(case: SimCase, side: str) -> float:
    _, primary_v = true_deviation(case, "primary")
    shift = tilt_shift(case, "R", side) - tilt_shift(case, "L", side)
    return primary_v + degrees_to_pd(shift)


def true_duction(case: SimCase, eye: str, direction: str) -> float:
    """Excursion from primary as a fraction of the gaze amplitude."""
    A = case.amplitude_deg
    toward_eye_side = 1 if eye == "R" else -1
    gaze = {
        "abduction": (toward_eye_side * A, 0.0),
        "adduction": (-toward_eye_side * A, 0.0),
        "elevation": (0.0, A),
        "depression": (0.0, -A),
    }[direction]
    p_out, p_up = eye_position(case, eye, 0.0, 0.0)
    g_out, g_up = eye_position(case, eye, *gaze)
    excursion = {
        "abduction": g_out - p_out, "adduction": p_out - g_out,
        "elevation": g_up - p_up, "depression": p_up - g_up,
    }[direction]
    return excursion / A


def simulate(case: SimCase, noise: Noise = Noise(), protocol: str = "nine_gaze",
             tilt: bool = True, insufficient: Tuple[str, ...] = ()) -> ExamMeasurements:
    """Build ExamMeasurements for a case. `insufficient` marks positions with too few frames."""
    rng = np.random.default_rng(noise.seed) if noise.seed is not None else None

    def measure(value: float, error: float) -> Measured:
        drawn = value + (rng.normal(0.0, error) if rng is not None else 0.0)
        return Measured(value=round(drawn, 4), error=error, confidence=noise.confidence)  # type: ignore[arg-type]

    cells = {}
    for gaze in PROTOCOL_POSITIONS[protocol]:
        if gaze in insufficient:
            cells[gaze] = GazeCell(status="INSUFFICIENT_DATA", valid_frames=3)
            continue
        h, v = true_deviation(case, gaze)
        cells[gaze] = GazeCell(horizontal_pd=measure(h, noise.pd), vertical_pd=measure(v, noise.pd),
                               valid_frames=60)

    eyes = {
        eye: EyeMeasures(
            ductions={d: measure(true_duction(case, eye, d), noise.fraction)  # type: ignore[misc]
                      for d in ("abduction", "adduction", "elevation", "depression")},
            fissure_height_mm=measure(case.fissure_mm[eye], noise.mm),
            fissure_ratio_adduction=measure(case.fissure_ratio_adduction[eye], noise.ratio),
            pupil_diameter_mm=measure(case.pupil_mm[eye], noise.mm),
            scleral_show_mm=measure(case.scleral_show_mm[eye], noise.mm),
        )
        for eye in ("R", "L")
    }
    tilts = TiltMeasurements(right=measure(true_tilt(case, "right"), noise.pd),
                             left=measure(true_tilt(case, "left"), noise.pd)) if tilt else None
    return ExamMeasurements(grid=GazeGrid(protocol=protocol, cells=cells), tilt=tilts, eyes=eyes)  # type: ignore[arg-type]
