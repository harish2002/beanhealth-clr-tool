"""
BeanHealth CLR — Frame QC (brief §4.2)
=======================================

Decides which frames of a capture session can be measured, and whether each
position has enough of them.

A frame is rejected — with every reason logged — for:
  no_face               face mesh found no face (head_pose null)
  eye_not_found         an eye has no observation
  missing_observation   a value a check needs is null (a check that cannot
                        run counts as failed, never as passed) — including
                        the limbus, the reference every reflex is read against
  blink                 fissure height ÷ iris diameter below the blink ratio
  lid_covers_limbus     lids hide more of the limbus than allowed
  limbus_size_mismatch  the two limbus fits differ in size more than two irises
                        can (one fit caught a lid or the pupil)
  reflex_not_found      no corneal light reflex in an eye
  motion_blur           eye-region sharpness well below the position's own
                        sharp frames (relative: no fixed value works across
                        phones and lighting — see capture.yaml)
  head_pose_off_target  head yaw / pitch (and roll, on tilts) outside the target
  saccade               gaze moving faster than the saccade velocity
  saccade_unverifiable  no neighbouring frame close enough in time to tell

Both eyes must pass: the deviation is the difference between the two eyes.
On a calibration position one eye is covered and only the fixing eye is
checked. The pupil is not required (it is often invisible in dark irises).

Saccade test: each eye's gaze direction relative to the camera is its
reflex-to-limbus-centre vector, converted to degrees with the iris-diameter scale
and the Hirschberg ratio from the KB. While the patient fixates the phone this
vector stays put even as the head turns (the eyes counter-rotate), so a fast
change means the eyes jumped. Velocity is taken to both neighbours and the
larger counts, so the frames at either edge of a saccade are rejected too.
The threshold is the larger of a fixed floor and a multiple of the
position's median speed, because landmark jitter alone produces apparent
speeds of tens of degrees per second at 30 fps.

A position is OK with at least min_valid_frames_per_position valid frames,
otherwise INSUFFICIENT_DATA (the engine then treats that cell as missing).
Every threshold is read from the KB (capture.yaml, thresholds.yaml).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
from pydantic import Field

from exam.capture.protocol import ProtocolPlan, PositionPlan, build_plan
from exam.capture.session import CaptureSession, FrameObservation, PositionCapture
from exam.kb import load_kb
from exam.kb.loader import KnowledgeBase
from exam.kb.schema import Eye
from exam.models import Strict

QC_VERSION = "1.1.0"
EYES: Tuple[Eye, Eye] = ("R", "L")
# A neighbour further away than this many frame periods is too far apart to
# measure a saccade against.
MAX_NEIGHBOUR_GAP_FRAMES = 3

RejectCode = Literal[
    "no_face", "eye_not_found", "missing_observation", "blink", "lid_covers_limbus",
    "limbus_size_mismatch", "reflex_not_found", "motion_blur", "head_pose_off_target", "saccade",
    "saccade_unverifiable",
]


class Rejection(Strict):
    code: RejectCode
    eye: Optional[Eye] = None
    detail: str


class FrameVerdict(Strict):
    index: int
    valid: bool
    reasons: List[Rejection] = Field(default_factory=list)


class PositionQC(Strict):
    position: str
    kind: Literal["calibration", "gaze", "tilt"]
    status: Literal["OK", "INSUFFICIENT_DATA"]
    reason: Optional[Literal["not_captured", "too_few_valid_frames"]] = None
    frames_total: int
    frames_valid: int
    min_valid_frames: int
    reject_counts: Dict[str, int] = Field(default_factory=dict)
    frames: List[FrameVerdict] = Field(default_factory=list)


class SessionFlag(Strict):
    code: Literal["distance_out_of_range", "distance_not_estimated", "low_fps",
                  "exposure_too_long", "exposure_unknown", "below_recommended_resolution"]
    detail: str


class SessionQC(Strict):
    session_id: str
    protocol: str
    positions: List[PositionQC]
    session_flags: List[SessionFlag] = Field(default_factory=list)
    insufficient_positions: List[str] = Field(default_factory=list)
    qc_version: str = QC_VERSION
    kb_version: str
    kb_sha256: str


# ─────────────────────────────────────────────────────────────
# Sharpness — the reference definition the phone must reproduce
# ─────────────────────────────────────────────────────────────

def sharpness(gray: np.ndarray, iris_centre: Tuple[float, float], iris_radius_px: float,
              kb: Optional[KnowledgeBase] = None) -> float:
    """
    Motion-blur measure for one eye: variance of the 3×3 Laplacian
    [[0,1,0],[1,-4,1],[0,1,0]] over a square of side 4 × iris radius centred
    on the iris, after rescaling so the iris radius is
    qc_sharpness_canonical_iris_radius_px (area interpolation when shrinking,
    bilinear when enlarging). Scale-free, so comparable across phones and
    distances.
    """
    if gray.ndim != 2:
        raise ValueError("sharpness needs a single-channel image")
    if iris_radius_px <= 0:
        raise ValueError("iris_radius_px must be positive")
    kb = kb or load_kb()
    canonical = _required(kb, "qc_sharpness_canonical_iris_radius_px")
    cx, cy = iris_centre
    half = 2.0 * iris_radius_px
    h, w = gray.shape
    x0, x1 = max(0, int(round(cx - half))), min(w, int(round(cx + half)))
    y0, y1 = max(0, int(round(cy - half))), min(h, int(round(cy + half)))
    if x1 - x0 < 3 or y1 - y0 < 3:
        raise ValueError("eye region falls outside the image")
    crop = gray[y0:y1, x0:x1]
    scale = canonical / iris_radius_px
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interp)
    return float(cv2.Laplacian(crop, cv2.CV_64F, ksize=1).var())


# ─────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────

def _required(kb: KnowledgeBase, entry_id: str) -> float:
    value = kb.scalar(entry_id).value
    if value is None:
        raise ValueError(f"KB entry {entry_id!r} has no value; frame QC needs it")
    return value


class _Gates:
    def __init__(self, kb: KnowledgeBase):
        self.blink_ratio = _required(kb, "qc_blink_openness_ratio")
        self.max_covered = _required(kb, "qc_max_limbus_covered_fraction")
        self.max_size_difference = _required(kb, "qc_max_limbus_size_difference")
        self.sharpness_percentile = _required(kb, "qc_sharpness_reference_percentile")
        self.min_relative_sharpness = _required(kb, "qc_min_relative_sharpness")
        self.saccade_deg_s = _required(kb, "qc_saccade_velocity_deg_s")
        self.saccade_noise_factor = _required(kb, "qc_saccade_noise_factor")
        # Reflex shift (in iris radii) → degrees of gaze.
        iris_radius_mm = _required(kb, "iris_diameter_mm") / 2.0
        self.deg_per_iris_radius = iris_radius_mm * _required(kb, "hirschberg_ratio_deg_per_mm")


# ─────────────────────────────────────────────────────────────
# Per-frame checks
# ─────────────────────────────────────────────────────────────

def _gaze_deg(frame: FrameObservation, eye: Eye, gates: _Gates) -> Optional[Tuple[float, float]]:
    """The eye's gaze direction relative to the camera, in degrees, if measurable."""
    obs = frame.eyes.get(eye)
    if obs is None or obs.reflex_px is None or obs.limbus is None:
        return None
    (cx, cy), k = obs.limbus.centre_px, gates.deg_per_iris_radius / obs.limbus.semi_major_px
    return ((obs.reflex_px[0] - cx) * k, (obs.reflex_px[1] - cy) * k)


def _static_checks(frame: FrameObservation, plan: PositionPlan, gates: _Gates,
                   min_sharpness: Optional[float]) -> List[Rejection]:
    if frame.head_pose is None:
        return [Rejection(code="no_face", detail="no face found")]
    out: List[Rejection] = []

    for eye in plan.observed_eyes:
        obs = frame.eyes.get(eye)
        if obs is None:
            out.append(Rejection(code="eye_not_found", eye=eye, detail=f"{eye} eye not observed"))
            continue
        if obs.limbus is None:
            out.append(Rejection(code="missing_observation", eye=eye,
                                 detail="limbus not fitted; no reference for the reflex"))
        if obs.fissure_height_px is None or obs.iris_radius_px is None:
            out.append(Rejection(code="missing_observation", eye=eye,
                                 detail="fissure height or iris radius missing; blink check cannot run"))
        else:
            ratio = obs.fissure_height_px / (2.0 * obs.iris_radius_px)
            if ratio < gates.blink_ratio:
                out.append(Rejection(code="blink", eye=eye,
                                     detail=f"openness {ratio:.2f} < {gates.blink_ratio:g}"))
        if obs.limbus_covered_fraction is None:
            out.append(Rejection(code="missing_observation", eye=eye,
                                 detail="limbus coverage missing; lid check cannot run"))
        elif obs.limbus_covered_fraction > gates.max_covered:
            out.append(Rejection(code="lid_covers_limbus", eye=eye,
                                 detail=f"{obs.limbus_covered_fraction:.0%} of limbus covered "
                                        f"> {gates.max_covered:.0%}"))
        if obs.reflex_px is None:
            out.append(Rejection(code="reflex_not_found", eye=eye, detail=f"no corneal reflex in {eye} eye"))

    limbi = [frame.eyes[e].limbus for e in plan.observed_eyes if e in frame.eyes]
    if len(plan.observed_eyes) == 2 and len(limbi) == 2 and all(limbi):
        big, small = sorted((l.semi_major_px for l in limbi), reverse=True)   # type: ignore[union-attr]
        if (big - small) / big > gates.max_size_difference:
            out.append(Rejection(code="limbus_size_mismatch",
                                 detail=f"limbus radii {big:.1f} and {small:.1f} px differ by "
                                        f"{(big - small) / big:.0%} > {gates.max_size_difference:.0%}"))

    if frame.sharpness is None:
        out.append(Rejection(code="missing_observation", detail="sharpness missing; blur check cannot run"))
    elif min_sharpness is not None and frame.sharpness < min_sharpness:
        out.append(Rejection(code="motion_blur",
                             detail=f"sharpness {frame.sharpness:.1f} < {min_sharpness:.1f} "
                                    f"({gates.min_relative_sharpness:.0%} of this position's reference)"))

    pose, target = frame.head_pose, plan.head
    off = []
    if abs(pose.yaw_deg - target.yaw_deg) > target.tolerance_deg:
        off.append(f"yaw {pose.yaw_deg:.1f}° (target {target.yaw_deg:g}° ± {target.tolerance_deg:g}°)")
    if abs(pose.pitch_deg - target.pitch_deg) > target.tolerance_deg:
        off.append(f"pitch {pose.pitch_deg:.1f}° (target {target.pitch_deg:g}° ± {target.tolerance_deg:g}°)")
    if target.roll_deg is not None and not target.roll_deg.min <= pose.roll_deg <= target.roll_deg.max:
        off.append(f"roll {pose.roll_deg:.1f}° (target {target.roll_deg.min:g}° to {target.roll_deg.max:g}°)")
    if off:
        out.append(Rejection(code="head_pose_off_target", detail="; ".join(off)))
    return out


def _saccade_checks(frames: List[FrameObservation], gates: _Gates, fps: float,
                    eyes: Tuple[Eye, ...] = EYES) -> List[List[Rejection]]:
    max_gap_ms = MAX_NEIGHBOUR_GAP_FRAMES * 1000.0 / fps
    gaze = [{eye: _gaze_deg(f, eye, gates) for eye in eyes} for f in frames]
    measurable = [all(g[eye] is not None for eye in eyes) for g in gaze]

    # Fastest gaze speed to either neighbour, per frame (None: no neighbour).
    fastest: List[Optional[Tuple[float, Eye]]] = []
    all_speeds: List[float] = []
    for i, f in enumerate(frames):
        speeds: List[Tuple[float, Eye]] = []
        for j in (i - 1, i + 1):
            if not measurable[i] or not 0 <= j < len(frames) or not measurable[j]:
                continue
            dt_s = abs(frames[j].t_ms - f.t_ms) / 1000.0
            if dt_s * 1000.0 > max_gap_ms:
                continue
            for eye in eyes:
                (x0, y0), (x1, y1) = gaze[i][eye], gaze[j][eye]   # type: ignore[misc]
                speeds.append((math.hypot(x1 - x0, y1 - y0) / dt_s, eye))
        all_speeds += [v for v, _ in speeds]
        fastest.append(max(speeds) if speeds else None)

    threshold = gates.saccade_deg_s
    if all_speeds:
        threshold = max(threshold, gates.saccade_noise_factor * float(np.median(all_speeds)))

    out: List[List[Rejection]] = []
    for i, top in enumerate(fastest):
        if not measurable[i]:
            out.append([])   # already rejected for the missing reflex / limbus
        elif top is None:
            out.append([Rejection(code="saccade_unverifiable",
                                  detail=f"no measurable neighbouring frame within {max_gap_ms:.0f} ms")])
        elif top[0] > threshold:
            out.append([Rejection(code="saccade", eye=top[1],
                                  detail=f"gaze moving {top[0]:.0f}°/s > {threshold:.0f}°/s")])
        else:
            out.append([])
    return out


# ─────────────────────────────────────────────────────────────
# Positions and session
# ─────────────────────────────────────────────────────────────

def qc_position(capture: Optional[PositionCapture], plan: PositionPlan, fps: float,
                gates: _Gates) -> PositionQC:
    if capture is None or not capture.frames:
        return PositionQC(position=plan.position, kind=plan.kind, status="INSUFFICIENT_DATA",
                          reason="not_captured", frames_total=0, frames_valid=0,
                          min_valid_frames=plan.min_valid_frames)
    saccades = _saccade_checks(capture.frames, gates, fps, plan.observed_eyes)
    scores = [f.sharpness for f in capture.frames if f.sharpness is not None and f.head_pose is not None]
    min_sharpness = (gates.min_relative_sharpness * float(np.percentile(scores, gates.sharpness_percentile))
                     if scores else None)
    verdicts = []
    counts: Counter = Counter()
    for frame, extra in zip(capture.frames, saccades):
        reasons = _static_checks(frame, plan, gates, min_sharpness) + extra
        counts.update({r.code for r in reasons})
        verdicts.append(FrameVerdict(index=frame.index, valid=not reasons, reasons=reasons))
    valid = sum(v.valid for v in verdicts)
    ok = valid >= plan.min_valid_frames
    return PositionQC(position=plan.position, kind=plan.kind,
                      status="OK" if ok else "INSUFFICIENT_DATA",
                      reason=None if ok else "too_few_valid_frames",
                      frames_total=len(verdicts), frames_valid=valid,
                      min_valid_frames=plan.min_valid_frames,
                      reject_counts=dict(sorted(counts.items())), frames=verdicts)


def _session_flags(session: CaptureSession, plan: ProtocolPlan) -> List[SessionFlag]:
    h, s = session.header, plan.settings
    flags: List[SessionFlag] = []
    lo, hi = s.distance_cm.min - s.distance_tolerance_cm, s.distance_cm.max + s.distance_tolerance_cm
    if h.distance_cm is None:
        flags.append(SessionFlag(code="distance_not_estimated", detail="no camera distance estimate"))
    elif not lo <= h.distance_cm.value <= hi:
        flags.append(SessionFlag(code="distance_out_of_range",
                                 detail=f"{h.distance_cm.value:.0f} cm, outside {lo:g}–{hi:g} cm"))
    if h.settings.fps < s.min_fps:
        flags.append(SessionFlag(code="low_fps", detail=f"{h.settings.fps:g} fps < {s.min_fps:g} fps"))
    if h.settings.exposure_s is None:
        flags.append(SessionFlag(code="exposure_unknown",
                                 detail="exposure not reported; relying on per-frame sharpness"))
    elif h.settings.exposure_s > s.max_exposure_s:
        flags.append(SessionFlag(code="exposure_too_long",
                                 detail=f"1/{1 / h.settings.exposure_s:.0f} s slower than "
                                        f"1/{1 / s.max_exposure_s:.0f} s"))
    short_side = min(h.settings.width_px, h.settings.height_px)
    if short_side < s.recommended_frame_height_px:
        flags.append(SessionFlag(code="below_recommended_resolution",
                                 detail=f"{short_side}p < {s.recommended_frame_height_px}p"))
    return flags


def run_qc(session: CaptureSession, kb: Optional[KnowledgeBase] = None,
           include_tilt: bool = True, include_calibration: bool = True) -> SessionQC:
    """QC every planned position of a session (a planned position not captured is INSUFFICIENT_DATA)."""
    kb = kb or load_kb()
    plan = build_plan(session.header.protocol, include_tilt=include_tilt, kb=kb,
                      include_calibration=include_calibration)
    gates = _Gates(kb)
    positions = [qc_position(session.capture(p.position), p, session.header.settings.fps, gates)
                 for p in plan.positions]
    return SessionQC(
        session_id=session.header.session_id, protocol=session.header.protocol,
        positions=positions, session_flags=_session_flags(session, plan),
        insufficient_positions=[p.position for p in positions if p.status != "OK"],
        **kb.version_info(),
    )
