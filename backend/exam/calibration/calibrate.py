"""
BeanHealth CLR — Calibration (brief §4.4)
==========================================

Turns reflex positions into eye rotations for one patient.

Scale. Each frame's mm per pixel comes from that eye's limbus: the KB
iris diameter over the fitted major axis (the axis a turned eye does not
foreshorten).

Decentration. The reflex relative to the limbus centre, rotated into the
head's frame by the frame's head roll (so a tilted head does not turn a
horizontal offset into a vertical one) and expressed per eye as
(temporal, inferior) mm. An eye turned in shows its reflex temporally; an
eye turned up shows it inferiorly.

Kappa. With the fellow eye covered, the fixing eye looks at the light by the
lens. Its decentration then is its own offset between the line of sight and
the reflex — angle kappa plus the constant pupil-to-limbus-centre offset —
and is subtracted from every later reading of that eye.

Hirschberg ratio. The same eye then fixes card targets a known distance to
either side of the lens; each asks for 100·offset/distance PD of rotation.
A weighted straight-line fit of temporal decentration against the rotation
asked for (one intercept per eye, one shared slope) gives the patient's
PD per mm. It is used only when it is plausible (KB range) and more precise
than the KB default; otherwise the default is used and the reason flagged.
A fitted ratio also absorbs the KB iris-diameter assumption, since scale and
ratio are fitted together.

    rotation (PD) = ratio × (decentration − kappa)       nasal +, up +

Limitations: the fit is horizontal only and assumes the same ratio
vertically; it ignores the eye's sideways offset from the lens (≈1%); the
distance comes from the session header and its error is carried into the
ratio's error.
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import Field

from exam.capture.protocol import CALIBRATION_TARGETS, PositionPlan, build_plan
from exam.capture.session import CaptureSession, EyeObservation
from exam.kb import load_kb
from exam.kb.loader import KnowledgeBase
from exam.kb.schema import Eye
from exam.models import Measured, Strict, degrees_to_pd
from exam.qc import QC_VERSION, SessionQC, run_qc

CALIBRATION_VERSION = "1.0.0"

# Normal-consistency factor × √(π/2): standard error of a median from the MAD.
_MAD_SE = 1.4826 * math.sqrt(math.pi / 2)
# Smallest standard error reported (mm): keeps noiseless input a valid Measured.
MIN_SE_MM = 1e-4


class EyeKappa(Strict):
    """Reflex decentration while the eye fixes the light (mm; + = temporal / inferior)."""
    temporal_mm: Measured
    inferior_mm: Measured
    frames: int


class CalibrationPoint(Strict):
    position: str
    eye: Eye
    rotation_asked_pd: float          # nasal rotation the target asks of the eye (0 at the lens)
    temporal_mm: Measured
    inferior_mm: Measured
    frames: int


class HirschbergRatio(Strict):
    pd_per_mm: Measured               # the ratio in use
    source: Literal["fitted", "kb_default"]
    fitted: Optional[Measured] = None  # the per-patient fit, kept even when not used
    residual_rms_mm: Optional[float] = None
    points: int = 0


class CalibrationFlag(Strict):
    code: Literal["kappa_not_measured", "distance_not_estimated", "hirschberg_not_fitted",
                  "hirschberg_implausible", "hirschberg_imprecise"]
    eye: Optional[Eye] = None
    detail: str


class Calibration(Strict):
    session_id: str
    iris_diameter_mm: float
    hirschberg: HirschbergRatio
    kappa: Dict[Eye, Optional[EyeKappa]]
    points: List[CalibrationPoint] = Field(default_factory=list)
    flags: List[CalibrationFlag] = Field(default_factory=list)
    calibration_version: str = CALIBRATION_VERSION
    qc_version: str = QC_VERSION
    kb_version: str
    kb_sha256: str


# ─────────────────────────────────────────────────────────────
# Per-frame geometry
# ─────────────────────────────────────────────────────────────

def decentration_mm(obs: EyeObservation, eye: Eye, roll_deg: float,
                    iris_diameter_mm: float) -> Optional[Tuple[float, float]]:
    """(temporal, inferior) mm of the reflex from the limbus centre, in the head's frame; None if unmeasured."""
    if obs.reflex_px is None or obs.limbus is None:
        return None
    mm_per_px = iris_diameter_mm / (2.0 * obs.limbus.semi_major_px)
    dx = (obs.reflex_px[0] - obs.limbus.centre_px[0]) * mm_per_px
    dy = (obs.reflex_px[1] - obs.limbus.centre_px[1]) * mm_per_px
    # Head axes in the image: x from the right eye to the left eye, y down.
    r = math.radians(roll_deg)
    x_head = dx * math.cos(r) - dy * math.sin(r)
    y_head = dx * math.sin(r) + dy * math.cos(r)
    temporal = -x_head if eye == "R" else x_head      # the right eye's temporal side is the patient's right
    return temporal, y_head


def eye_rotation_pd(obs: EyeObservation, eye: Eye, roll_deg: float,
                    cal: Calibration) -> Optional[Tuple[float, float]]:
    """(nasal, up) rotation of the eye from the light, PD, after kappa; None if unmeasured or no kappa."""
    d = decentration_mm(obs, eye, roll_deg, cal.iris_diameter_mm)
    kappa = cal.kappa.get(eye)
    if d is None or kappa is None:
        return None
    ratio = cal.hirschberg.pd_per_mm.value
    return ratio * (d[0] - kappa.temporal_mm.value), ratio * (d[1] - kappa.inferior_mm.value)


def robust_location(values: List[float]) -> Measured:
    """Median with a MAD-based standard error (brief §4.5)."""
    if not values:
        raise ValueError("no values")
    a = np.asarray(values, dtype=float)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return Measured(value=med, error=max(MIN_SE_MM, _MAD_SE * mad / math.sqrt(len(a))))


# ─────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────

def _required(kb: KnowledgeBase, entry_id: str) -> float:
    value = kb.scalar(entry_id).value
    if value is None:
        raise ValueError(f"KB entry {entry_id!r} has no value; calibration needs it")
    return value


def _point(session: CaptureSession, qc: SessionQC, plan: PositionPlan, distance_cm: Optional[float],
           iris_mm: float) -> Optional[CalibrationPoint]:
    pqc = next((p for p in qc.positions if p.position == plan.position), None)
    capture = session.capture(plan.position)
    if pqc is None or pqc.status != "OK" or capture is None or plan.fixing_eye is None:
        return None
    if plan.target_offset_cm and distance_cm is None:
        return None
    eye = plan.fixing_eye
    valid = {v.index for v in pqc.frames if v.valid}
    temporal, inferior = [], []
    for f in capture.frames:
        if f.index not in valid or f.head_pose is None:
            continue
        d = decentration_mm(f.eyes[eye], eye, f.head_pose.roll_deg, iris_mm)
        if d is not None:
            temporal.append(d[0])
            inferior.append(d[1])
    if not temporal:
        return None
    target_pd = 100.0 * plan.target_offset_cm / distance_cm if plan.target_offset_cm else 0.0  # type: ignore[operator]
    # A target to the patient's right turns the right eye out (temporal) and the left eye in.
    asked = -target_pd if eye == "R" else target_pd
    return CalibrationPoint(position=plan.position, eye=eye, rotation_asked_pd=asked,
                            temporal_mm=robust_location(temporal), inferior_mm=robust_location(inferior),
                            frames=len(temporal))


def _fit_ratio(points: List[CalibrationPoint], distance: Measured
               ) -> Optional[Tuple[Measured, float]]:
    """Weighted fit temporal = kappa_eye + rotation / ratio; (ratio, rms residual mm) or None."""
    eyes = sorted({p.eye for p in points if any(q.eye == p.eye and q.rotation_asked_pd != p.rotation_asked_pd
                                                for q in points)})
    pts = [p for p in points if p.eye in eyes]
    if len(pts) < 3:
        return None
    x = np.array([p.rotation_asked_pd for p in pts])
    y = np.array([p.temporal_mm.value for p in pts])
    sd = np.array([p.temporal_mm.error for p in pts])
    a = np.column_stack([x] + [[1.0 if p.eye == e else 0.0 for p in pts] for e in eyes])
    aw, yw = a / sd[:, None], y / sd
    coef, *_ = np.linalg.lstsq(aw, yw, rcond=None)
    cov = np.linalg.inv(aw.T @ aw)
    resid = yw - aw @ coef
    dof = len(pts) - a.shape[1]
    if dof > 0:
        cov = cov * max(1.0, float(resid @ resid) / dof)
    slope, slope_se = float(coef[0]), math.sqrt(float(cov[0, 0]))
    if slope <= 0:
        return None
    ratio = 1.0 / slope
    rel = math.hypot(slope_se / slope, distance.error / distance.value)
    rms = float(np.sqrt(np.mean((y - a @ coef) ** 2)))
    return Measured(value=ratio, error=max(MIN_SE_MM, ratio * rel)), rms


def calibrate(session: CaptureSession, qc: Optional[SessionQC] = None,
              kb: Optional[KnowledgeBase] = None) -> Calibration:
    """Kappa per eye and the Hirschberg ratio from a session's calibration captures."""
    kb = kb or load_kb()
    qc = qc or run_qc(session, kb=kb)
    iris_mm = _required(kb, "iris_diameter_mm")
    plan = build_plan(session.header.protocol, include_tilt=False, kb=kb)
    calib_plans = [p for p in plan.positions if p.position in CALIBRATION_TARGETS]
    distance = session.header.distance_cm
    flags: List[CalibrationFlag] = []

    points = [pt for p in calib_plans
              if (pt := _point(session, qc, p, distance.value if distance else None, iris_mm))]
    kappa: Dict[Eye, Optional[EyeKappa]] = {}
    for eye in ("R", "L"):
        lens = next((p for p in points if p.eye == eye and p.rotation_asked_pd == 0.0), None)
        kappa[eye] = (EyeKappa(temporal_mm=lens.temporal_mm, inferior_mm=lens.inferior_mm, frames=lens.frames)
                      if lens else None)
        if lens is None:
            flags.append(CalibrationFlag(code="kappa_not_measured", eye=eye,
                                         detail=f"calib_{eye}_lens has too few valid frames; "
                                                f"this eye's rotations cannot be measured"))

    default = Measured(value=degrees_to_pd(_required(kb, "hirschberg_ratio_deg_per_mm")),
                       error=_required(kb, "hirschberg_ratio_default_sd_pd_per_mm"), confidence="MODERATE")
    ratio = HirschbergRatio(pd_per_mm=default, source="kb_default", points=len(points))
    if distance is None:
        flags.append(CalibrationFlag(code="distance_not_estimated",
                                     detail="no camera distance; target angles unknown, KB default ratio used"))
    else:
        fit = _fit_ratio(points, distance)
        if fit is None:
            flags.append(CalibrationFlag(code="hirschberg_not_fitted",
                                         detail="too few usable calibration targets (or no reflex shift); "
                                                "KB default ratio used"))
        else:
            fitted, rms = fit
            lo, hi = kb.scalar("hirschberg_ratio_plausible_pd_per_mm").plausible_range  # type: ignore[misc]
            use = False
            if not lo <= fitted.value <= hi:
                flags.append(CalibrationFlag(code="hirschberg_implausible",
                                             detail=f"fitted {fitted.value:.1f} ± {fitted.error:.1f} PD/mm outside "
                                                    f"{lo:g}–{hi:g}; KB default ratio used"))
            elif fitted.error >= default.error:
                flags.append(CalibrationFlag(code="hirschberg_imprecise",
                                             detail=f"fitted ± {fitted.error:.1f} PD/mm is no better than the "
                                                    f"default's ± {default.error:.1f}; KB default ratio used"))
            else:
                use = True
            ratio = HirschbergRatio(pd_per_mm=fitted if use else default,
                                    source="fitted" if use else "kb_default",
                                    fitted=fitted, residual_rms_mm=rms, points=len(points))

    return Calibration(session_id=session.header.session_id, iris_diameter_mm=iris_mm, hirschberg=ratio,
                       kappa=kappa, points=points, flags=flags, **kb.version_info())
