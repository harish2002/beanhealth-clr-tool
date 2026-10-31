"""
BeanHealth CLR — Per-gaze measurement (brief §4.5)
===================================================

QC-valid frames + calibration → the measurements the rules engine reads:
the 9-gaze (or 5-gaze) deviation grid, the head-tilt deviations, and per-eye
ductions, fissures, scleral show and pupils. Every value is a `Measured`
with a one-sigma error and a confidence level.

Deviation, per frame (exam.calibration.eye_rotation_pd gives each eye's
nasal / up rotation from the light, kappa removed):

    horizontal_pd = R_nasal + L_nasal        + = eso, − = exo
    vertical_pd   = R_up − L_up              + = right hyper, − = left hyper

Fusion across frames: the median, with a MAD-based standard error over the
effective number of frames (successive frames are correlated: fixation
drifts slowly, so 60 frames are not 60 independent looks). The error of a
cell then adds, in quadrature:

    frame scatter   MAD-based SE of the median (n_eff from lag-1 autocorrelation)
    kappa           each eye's kappa SE × ratio — the same in every cell
    Hirschberg      |value| × relative error of the ratio — the same in every cell
    method          measurement_method_error_pd (KB): landmark bias scatter cannot show

Kappa and Hirschberg errors are systematic: they shift every cell together
and cancel in a difference between cells. The engine combines cell errors in
quadrature as if independent, which overstates the error of a difference
(comitance, A / V, Parks steps 2–3): those tests are conservative — they
return INCONCLUSIVE more often, never a pattern the data does not support.

Confidence of a cell: HIGH; MODERATE when the KB default Hirschberg ratio is
in use or the session has a capture-quality flag (distance, frame rate,
exposure); LOW when QC rejected more than measurement_low_confidence_valid_fraction
of the position's frames (fixation was unsteady). LOW never refers.

Ductions (eye-in-orbit excursion vs what the head rotation asks for). While
the patient fixates the phone, an eye's position in the orbit is its
direction relative to the camera minus the head's: turning the head 25° to
the left asks for 25° of right gaze. A duction is measured as excursion
from primary — fraction = 1 + (change in the eye's direction relative to the
camera) / (head rotation from primary) — so kappa, the eye's offset from the
camera axis and any constant head-pose bias cancel. Binocular viewing: the
fixing eye reaches the target by definition, so a limited duction shows only
in the non-fixing eye, and a paretic eye that fixes shows as overaction of
its fellow (Tier 1 cannot tell which eye fixes; the cover test is Tier 2).

Lids and pupils come from primary gaze (fissure ratio from the adducting
gaze), in mm by each frame's own iris scale.

Known limitations (brief §11): near fixation only; no phoria or intermittent
deviation detection; torsion not measured; vertical deviations use the
horizontally fitted Hirschberg ratio; fixation compliance in children is
not guaranteed (LOW confidence on unsteady positions is the only guard).
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from pydantic import Field

from exam.calibration import CALIBRATION_VERSION, Calibration, calibrate, eye_rotation_pd
from exam.capture.protocol import CAPTURE_PROTOCOL_VERSION, PositionPlan, build_plan
from exam.capture.session import CaptureSession, HeadPose
from exam.kb import load_kb
from exam.kb.loader import KnowledgeBase
from exam.kb.schema import Eye
from exam.models import (
    PROTOCOL_POSITIONS,
    Confidence,
    DuctionDirection,
    EyeMeasures,
    ExamMeasurements,
    GazeCell,
    GazeGrid,
    Measured,
    Strict,
    TiltMeasurements,
    lowest_confidence,
    pd_to_degrees,
)
from exam.qc import QC_VERSION, PositionQC, SessionQC, run_qc

MEASUREMENT_VERSION = "1.0.0"
EYES: Tuple[Eye, Eye] = ("R", "L")

# Normal-consistency factor × √(π/2): standard error of a median from the MAD.
_MAD_SE = 1.4826 * math.sqrt(math.pi / 2)
# Largest lag-1 autocorrelation credited when estimating n_eff.
_MAX_RHO = 0.9
# PD → degrees near zero (d deg / d PD at 0).
_DEG_PER_PD = 180.0 / (math.pi * 100.0)
# Session QC flags that lower every cell to MODERATE.
QUALITY_FLAGS = ("distance_out_of_range", "distance_not_estimated", "low_fps", "exposure_too_long")

# The gaze position each duction is read from, per eye.
DUCTION_GAZE: Dict[Eye, Dict[DuctionDirection, str]] = {
    "R": {"abduction": "right", "adduction": "left", "elevation": "up", "depression": "down"},
    "L": {"abduction": "left", "adduction": "right", "elevation": "up", "depression": "down"},
}
ADDUCTING_GAZE: Dict[Eye, str] = {"R": "left", "L": "right"}


class EyeRotation(Strict):
    """One eye at one position (median over the frames used)."""
    nasal_pd: Measured                 # rotation from the light, kappa removed
    up_pd: Measured
    # Eye-in-orbit angle: direction relative to the camera minus the head pose
    # (deg; + = the patient's right / up).
    in_orbit_h_deg: Measured
    in_orbit_v_deg: Measured


class PositionMeasurement(Strict):
    position: str
    kind: Literal["gaze", "tilt"]
    status: Literal["OK", "INSUFFICIENT_DATA"]
    reason: Optional[Literal["not_captured", "too_few_valid_frames", "kappa_not_measured"]] = None
    frames_total: int
    frames_valid: int                  # passed frame QC
    frames_used: int                   # passed QC and measurable in both eyes
    min_valid_frames: int
    horizontal_pd: Optional[Measured] = None
    vertical_pd: Optional[Measured] = None
    horizontal_deg: Optional[float] = None
    vertical_deg: Optional[float] = None
    head_pose: Optional[HeadPose] = None          # median
    eyes: Dict[Eye, EyeRotation] = Field(default_factory=dict)
    # Why the confidence is below HIGH (MeasurementFlag codes).
    confidence_reasons: List[str] = Field(default_factory=list)


class MeasurementFlag(Strict):
    code: Literal["kappa_not_measured", "kb_default_hirschberg", "session_quality", "unsteady_fixation",
                  "duction_not_measured", "lid_not_measured", "pupil_not_measured"]
    position: Optional[str] = None
    eye: Optional[Eye] = None
    detail: str


class GridMeasurement(Strict):
    session_id: str
    protocol: str
    positions: List[PositionMeasurement]
    measurements: ExamMeasurements                # what the rules engine reads
    flags: List[MeasurementFlag] = Field(default_factory=list)
    measurement_version: str = MEASUREMENT_VERSION
    calibration_version: str = CALIBRATION_VERSION
    qc_version: str = QC_VERSION
    capture_protocol_version: str = CAPTURE_PROTOCOL_VERSION
    kb_version: str
    kb_sha256: str

    def position(self, name: str) -> Optional[PositionMeasurement]:
        return next((p for p in self.positions if p.position == name), None)


# ─────────────────────────────────────────────────────────────
# Fusion
# ─────────────────────────────────────────────────────────────

def effective_n(values: Sequence[float]) -> float:
    """Frames' worth of independent information: n·(1 − ρ)/(1 + ρ), ρ the lag-1 autocorrelation."""
    a = np.asarray(values, dtype=float)
    n = len(a)
    if n < 3:
        return float(n)
    d = a - a.mean()
    denom = float(d @ d)
    rho = 0.0 if denom == 0 else float(d[:-1] @ d[1:]) / denom
    rho = min(_MAX_RHO, max(0.0, rho))
    return max(1.0, n * (1.0 - rho) / (1.0 + rho))


def fuse(values: Sequence[float]) -> Tuple[float, float]:
    """(median, MAD-based standard error over the effective number of frames)."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        raise ValueError("no values")
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return med, _MAD_SE * mad / math.sqrt(effective_n(values))


def _measured(value: float, *errors: float, confidence: Confidence = "HIGH") -> Measured:
    err = math.sqrt(sum(e * e for e in errors))
    return Measured(value=value, error=max(err, 1e-6), confidence=confidence)


def _required(kb: KnowledgeBase, entry_id: str) -> float:
    value = kb.scalar(entry_id).value
    if value is None:
        raise ValueError(f"KB entry {entry_id!r} has no value; measurement needs it")
    return value


# ─────────────────────────────────────────────────────────────
# One position
# ─────────────────────────────────────────────────────────────

class _Series:
    """Per-frame values of one position, in frame order."""

    def __init__(self) -> None:
        self.h: List[float] = []
        self.v: List[float] = []
        self.yaw: List[float] = []
        self.pitch: List[float] = []
        self.roll: List[float] = []
        self.nasal: Dict[Eye, List[float]] = {e: [] for e in EYES}
        self.up: Dict[Eye, List[float]] = {e: [] for e in EYES}
        self.cam_h: Dict[Eye, List[float]] = {e: [] for e in EYES}   # deg, + = patient's right
        self.cam_v: Dict[Eye, List[float]] = {e: [] for e in EYES}
        self.orbit_h: Dict[Eye, List[float]] = {e: [] for e in EYES}
        self.orbit_v: Dict[Eye, List[float]] = {e: [] for e in EYES}


class _Errors:
    def __init__(self, kb: KnowledgeBase, cal: Calibration):
        self.method_pd = _required(kb, "measurement_method_error_pd")
        self.method_mm = _required(kb, "measurement_method_error_mm")
        self.head_deg = _required(kb, "head_pose_excursion_error_deg")
        self.low_fraction = _required(kb, "measurement_low_confidence_valid_fraction")
        r = cal.hirschberg.pd_per_mm
        self.ratio = r.value
        self.ratio_rel = r.error / r.value
        self.ratio_confidence: Confidence = r.confidence

    def kappa_pd(self, cal: Calibration, eye: Eye, axis: str) -> float:
        k = cal.kappa[eye]
        assert k is not None
        return self.ratio * (k.temporal_mm.error if axis == "h" else k.inferior_mm.error)


def _frame_confidence(pqc: Optional[PositionQC], base: Tuple[Confidence, List[str]],
                      errs: _Errors) -> Tuple[Confidence, List[str]]:
    """A position's confidence before the Hirschberg ratio: the session's, LOW if fixation was unsteady."""
    if pqc is not None and pqc.frames_total and pqc.frames_valid < errs.low_fraction * pqc.frames_total:
        return "LOW", [*base[1], "unsteady_fixation"]
    return base


def _series(session: CaptureSession, pqc: PositionQC, cal: Calibration) -> _Series:
    capture = session.capture(pqc.position)
    s = _Series()
    if capture is None:
        return s
    valid = {v.index for v in pqc.frames if v.valid}
    for f in capture.frames:
        if f.index not in valid or f.head_pose is None:
            continue
        rot = {e: eye_rotation_pd(f.eyes[e], e, f.head_pose.roll_deg, cal) if e in f.eyes else None
               for e in EYES}
        if any(r is None for r in rot.values()):
            continue
        hp = f.head_pose
        s.h.append(rot["R"][0] + rot["L"][0])            # type: ignore[index]
        s.v.append(rot["R"][1] - rot["L"][1])            # type: ignore[index]
        s.yaw.append(hp.yaw_deg)
        s.pitch.append(hp.pitch_deg)
        s.roll.append(hp.roll_deg)
        for e in EYES:
            nasal, up = rot[e]                           # type: ignore[misc]
            # Nasal is toward the patient's left for the right eye, toward the right for the left eye.
            cam_h = pd_to_degrees(-nasal if e == "R" else nasal)
            cam_v = pd_to_degrees(up)
            s.nasal[e].append(nasal)
            s.up[e].append(up)
            s.cam_h[e].append(cam_h)
            s.cam_v[e].append(cam_v)
            s.orbit_h[e].append(cam_h - hp.yaw_deg)
            s.orbit_v[e].append(cam_v - hp.pitch_deg)
    return s


def _position(session: CaptureSession, plan: PositionPlan, pqc: PositionQC, cal: Calibration,
              errs: _Errors, base: Tuple[Confidence, List[str]]) -> Tuple[PositionMeasurement, _Series]:
    head = dict(position=plan.position, kind=plan.kind, frames_total=pqc.frames_total,
                frames_valid=pqc.frames_valid, min_valid_frames=plan.min_valid_frames)
    if pqc.status != "OK":
        return PositionMeasurement(status="INSUFFICIENT_DATA", reason=pqc.reason, frames_used=0,
                                   **head), _Series()          # type: ignore[arg-type]
    if any(cal.kappa.get(e) is None for e in EYES):
        return PositionMeasurement(status="INSUFFICIENT_DATA", reason="kappa_not_measured", frames_used=0,
                                   **head), _Series()          # type: ignore[arg-type]
    s = _series(session, pqc, cal)
    if len(s.h) < plan.min_valid_frames:
        return PositionMeasurement(status="INSUFFICIENT_DATA", reason="too_few_valid_frames",
                                   frames_used=len(s.h), **head), s   # type: ignore[arg-type]

    confidence, reasons = _frame_confidence(pqc, base, errs)
    dev_conf = lowest_confidence([confidence, errs.ratio_confidence])
    dev_reasons = reasons + (["kb_default_hirschberg"] if errs.ratio_confidence != "HIGH" else [])

    def deviation(values: List[float], axis: str) -> Measured:
        med, se = fuse(values)
        return _measured(med, se, math.hypot(errs.kappa_pd(cal, "R", axis), errs.kappa_pd(cal, "L", axis)),
                         abs(med) * errs.ratio_rel, errs.method_pd, confidence=dev_conf)

    def calibration_pd(e: Eye, axis: str, value: float) -> float:
        """One eye's calibration and method error (PD) — everything but the frame scatter."""
        return math.sqrt(errs.kappa_pd(cal, e, axis) ** 2 + (value * errs.ratio_rel) ** 2
                         + errs.method_pd ** 2 / 2)

    def per_eye(values: List[float], e: Eye, axis: str) -> Measured:
        med, se = fuse(values)
        return _measured(med, se, calibration_pd(e, axis, med), confidence=dev_conf)

    def orbit(values: List[float], e: Eye, axis: str, rotation: Measured) -> Measured:
        med, se = fuse(values)
        return _measured(med, se, calibration_pd(e, axis, rotation.value) * _DEG_PER_PD, errs.head_deg,
                         confidence=dev_conf)

    h, v = deviation(s.h, "h"), deviation(s.v, "v")
    eyes = {}
    for e in EYES:
        nasal, up = per_eye(s.nasal[e], e, "h"), per_eye(s.up[e], e, "v")
        eyes[e] = EyeRotation(nasal_pd=nasal, up_pd=up,
                              in_orbit_h_deg=orbit(s.orbit_h[e], e, "h", nasal),
                              in_orbit_v_deg=orbit(s.orbit_v[e], e, "v", up))
    pose = HeadPose(yaw_deg=float(np.median(s.yaw)), pitch_deg=float(np.median(s.pitch)),
                    roll_deg=float(np.median(s.roll)))
    return PositionMeasurement(
        status="OK", frames_used=len(s.h), horizontal_pd=h, vertical_pd=v,
        horizontal_deg=pd_to_degrees(h.value), vertical_deg=pd_to_degrees(v.value),
        head_pose=pose, eyes=eyes, confidence_reasons=list(dict.fromkeys(dev_reasons)), **head,
    ), s                                                   # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────
# Ductions, lids, pupils
# ─────────────────────────────────────────────────────────────

def _duction(eye: Eye, direction: DuctionDirection, pm: Dict[str, PositionMeasurement],
             series: Dict[str, _Series], errs: _Errors) -> Optional[Measured]:
    """Excursion from primary as a fraction of the head rotation from primary; None if unmeasurable."""
    gaze = DUCTION_GAZE[eye][direction]
    P, G = pm.get("primary"), pm.get(gaze)
    if P is None or G is None or P.status != "OK" or G.status != "OK":
        return None
    horizontal = direction in ("abduction", "adduction")
    # + along the direction the gaze asks for: right / up gaze positive, left / down negative.
    sign = 1.0 if gaze in ("right", "up") else -1.0
    cam = series[gaze].cam_h[eye] if horizontal else series[gaze].cam_v[eye]
    cam0 = series["primary"].cam_h[eye] if horizontal else series["primary"].cam_v[eye]
    head = series[gaze].yaw if horizontal else series[gaze].pitch
    head0 = series["primary"].yaw if horizontal else series["primary"].pitch
    (g, g_se), (g0, g0_se) = fuse(cam), fuse(cam0)
    (y, y_se), (y0, y0_se) = fuse(head), fuse(head0)
    moved = sign * (g - g0)                  # the eye's own excursion relative to the camera
    asked = sign * (y0 - y)                  # what the head rotation asks of a fixing eye
    asked_err = math.sqrt(y_se ** 2 + y0_se ** 2 + errs.head_deg ** 2)
    if asked <= 2.0 * asked_err:
        return None
    moved_err = math.sqrt(g_se ** 2 + g0_se ** 2 + (moved * errs.ratio_rel) ** 2
                          + (errs.method_pd * _DEG_PER_PD) ** 2)
    fraction = 1.0 + moved / asked
    err = math.hypot(moved_err / asked, moved * asked_err / asked ** 2)
    confidence = lowest_confidence([P.horizontal_pd.confidence, G.horizontal_pd.confidence])  # type: ignore[union-attr]
    return _measured(fraction, err, confidence=confidence)


def _lid_values(session: CaptureSession, pqc: Optional[PositionQC], eye: Eye,
                iris_mm: float) -> Dict[str, List[float]]:
    """Per-frame fissure height, scleral show and pupil diameter (mm) of one eye at one position."""
    out: Dict[str, List[float]] = {"fissure": [], "scleral": [], "pupil": []}
    capture = None if pqc is None else session.capture(pqc.position)
    if capture is None or pqc is None or pqc.status != "OK":
        return out
    valid = {v.index for v in pqc.frames if v.valid}
    for f in capture.frames:
        obs = f.eyes.get(eye)
        if f.index not in valid or obs is None or obs.limbus is None:
            continue
        mm_per_px = iris_mm / (2.0 * obs.limbus.semi_major_px)
        for key, px in (("fissure", obs.fissure_height_px), ("scleral", obs.scleral_show_px),
                        ("pupil", obs.pupil_diameter_px)):
            if px is not None:
                out[key].append(px * mm_per_px)
    return out


# ─────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────

def measure(session: CaptureSession, qc: Optional[SessionQC] = None, cal: Optional[Calibration] = None,
            kb: Optional[KnowledgeBase] = None) -> GridMeasurement:
    """The engine's measurements from a capture session (QC and calibration are run if not given)."""
    kb = kb or load_kb()
    qc = qc or run_qc(session, kb=kb)
    cal = cal or calibrate(session, qc, kb)
    errs = _Errors(kb, cal)
    iris_mm = _required(kb, "iris_diameter_mm")
    protocol = session.header.protocol
    plan = build_plan(protocol, include_tilt=True, kb=kb, include_calibration=False)
    qc_by_name = {p.position: p for p in qc.positions}
    flags: List[MeasurementFlag] = []

    for e in EYES:
        if cal.kappa.get(e) is None:
            flags.append(MeasurementFlag(code="kappa_not_measured", eye=e,
                                         detail="no kappa for this eye: no position can be measured; "
                                                "retake its calibration"))
    if cal.hirschberg.source != "fitted":
        flags.append(MeasurementFlag(code="kb_default_hirschberg",
                                     detail="KB default Hirschberg ratio in use; deviations at MODERATE confidence"))
    quality = [f.code for f in qc.session_flags if f.code in QUALITY_FLAGS]
    base: Tuple[Confidence, List[str]] = ("HIGH", [])
    if quality:
        base = ("MODERATE", ["session_quality"])
        flags.append(MeasurementFlag(code="session_quality",
                                     detail="capture quality flags " + ", ".join(quality) +
                                            "; every value at MODERATE confidence or below"))

    positions: List[PositionMeasurement] = []
    series: Dict[str, _Series] = {}
    for p in plan.positions:
        pqc = qc_by_name.get(p.position) or PositionQC(
            position=p.position, kind=p.kind, status="INSUFFICIENT_DATA", reason="not_captured",
            frames_total=0, frames_valid=0, min_valid_frames=p.min_valid_frames)
        pm, s = _position(session, p, pqc, cal, errs, base)
        positions.append(pm)
        series[p.position] = s
        if "unsteady_fixation" in pm.confidence_reasons:
            flags.append(MeasurementFlag(code="unsteady_fixation", position=p.position,
                                         detail=f"QC kept {pm.frames_valid} of {pm.frames_total} frames; "
                                                f"cell at LOW confidence"))
    by_name = {p.position: p for p in positions}

    cells = {}
    for gaze in PROTOCOL_POSITIONS[protocol]:
        pm = by_name[gaze]
        cells[gaze] = (GazeCell(horizontal_pd=pm.horizontal_pd, vertical_pd=pm.vertical_pd,
                                valid_frames=pm.frames_used)
                       if pm.status == "OK" else
                       GazeCell(status="INSUFFICIENT_DATA", valid_frames=pm.frames_used))
    tilt_r, tilt_l = by_name["tilt_right"], by_name["tilt_left"]
    tilt = None
    if tilt_r.status == "OK" or tilt_l.status == "OK":
        tilt = TiltMeasurements(right=tilt_r.vertical_pd, left=tilt_l.vertical_pd)

    lid_conf = _frame_confidence(qc_by_name.get("primary"), base, errs)[0]
    min_frames = by_name["primary"].min_valid_frames
    eyes: Dict[Eye, EyeMeasures] = {}
    for e in EYES:
        ductions = {}
        for direction in DUCTION_GAZE[e]:
            d = _duction(e, direction, by_name, series, errs)            # type: ignore[arg-type]
            if d is None:
                flags.append(MeasurementFlag(code="duction_not_measured", eye=e,
                                             position=DUCTION_GAZE[e][direction],  # type: ignore[index]
                                             detail=f"{direction}: the position or primary is missing, "
                                                    f"or the head did not turn far enough"))
            else:
                ductions[direction] = d
        lids = _lid_values(session, qc_by_name.get("primary"), e, iris_mm)
        lids_add = _lid_values(session, qc_by_name.get(ADDUCTING_GAZE[e]), e, iris_mm)

        def mm(values: List[float], what: str, code: str = "lid_not_measured") -> Optional[Measured]:
            if len(values) < min_frames:
                flags.append(MeasurementFlag(code=code, eye=e, position="primary",  # type: ignore[arg-type]
                                             detail=f"{what} found in {len(values)} frames "
                                                    f"(needs {min_frames})"))
                return None
            med, se = fuse(values)
            return _measured(med, se, errs.method_mm, confidence=lid_conf)

        fissure = mm(lids["fissure"], "fissure")
        ratio = None
        if fissure is not None and len(lids_add["fissure"]) >= min_frames and fissure.value > 0:
            add_med, add_se = fuse(lids_add["fissure"])
            add = math.hypot(add_se, errs.method_mm)
            r = add_med / fissure.value
            add_conf = _frame_confidence(qc_by_name.get(ADDUCTING_GAZE[e]), base, errs)[0]
            ratio = _measured(r, r * math.hypot(add / add_med if add_med else 1.0, fissure.error / fissure.value),
                              confidence=lowest_confidence([lid_conf, add_conf]))
        eyes[e] = EyeMeasures(
            ductions=ductions, fissure_height_mm=fissure, fissure_ratio_adduction=ratio,
            pupil_diameter_mm=mm(lids["pupil"], "pupil", "pupil_not_measured"),
            scleral_show_mm=mm(lids["scleral"], "scleral show"),
        )

    measurements = ExamMeasurements(grid=GazeGrid(protocol=protocol, cells=cells), tilt=tilt, eyes=eyes)  # type: ignore[arg-type]
    return GridMeasurement(session_id=session.header.session_id, protocol=protocol, positions=positions,
                           measurements=measurements, flags=flags, **kb.version_info())
