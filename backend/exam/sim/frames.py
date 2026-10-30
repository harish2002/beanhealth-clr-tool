"""
BeanHealth CLR — Synthetic frame tracks (test fixture, not physiology).

Builds the per-frame observations a phone would upload for each planned
position — steady fixation with landmark jitter — and lets a test inject the
events frame QC must catch: blinks, blur, glances away, head off target, a
missing reflex, a lost face, dropped frames.

Geometry: a 4K landscape frame, the patient's right eye on the image left
(unmirrored). Each eye's rotation relative to the light (prism dioptres) is
turned into a reflex-to-limbus offset through the patient's Hirschberg ratio
and kappa and the KB iris diameter — the inverse of exam.calibration — and
rotated by the head roll. A glance is added in degrees with the KB's
default ratio, the inverse of what frame QC does.

The simulated patient fixes with the right eye; the left eye carries the
deviation. On a calibration position the fellow eye is covered and absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np

from exam.capture.protocol import CAPTURE_PROTOCOL_VERSION, PositionPlan, ProtocolPlan, build_plan
from exam.capture.session import (
    CaptureSession,
    CaptureSettings,
    DeviceInfo,
    Ellipse,
    EyeObservation,
    FrameObservation,
    HeadPose,
    PositionCapture,
    SessionHeader,
)
from exam.kb import load_kb
from exam.kb.loader import KnowledgeBase
from exam.models import Measured, Protocol, degrees_to_pd

EYE_CENTRE_PX = {"R": (1700.0, 1080.0), "L": (2140.0, 1080.0)}


@dataclass(frozen=True)
class FrameEvents:
    """Frame indices (within one position) where something goes wrong."""
    blink: FrozenSet[int] = frozenset()
    blur: FrozenSet[int] = frozenset()
    head_off_target: FrozenSet[int] = frozenset()
    no_reflex_right: FrozenSet[int] = frozenset()
    no_face: FrozenSet[int] = frozenset()
    lid_low: FrozenSet[int] = frozenset()
    # Frames where both eyes glance away by `glance_deg` (conjugate).
    glance: FrozenSet[int] = frozenset()
    glance_deg: float = 8.0
    # Frames the phone never delivered.
    dropped: FrozenSet[int] = frozenset()


def events(**kwargs: Iterable[int]) -> FrameEvents:
    """FrameEvents(blink=[3, 4]) without spelling out frozenset()."""
    return FrameEvents(**{k: (frozenset(v) if not isinstance(v, (int, float)) else v)
                          for k, v in kwargs.items()})


@dataclass(frozen=True)
class SimEyes:
    """The patient's optics and alignment (true values the calibration should recover)."""
    hirschberg_pd_per_mm: Optional[float] = None        # None → the KB default ratio
    # Reflex decentration (temporal, inferior) in mm while each eye fixes the light.
    kappa_mm: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Deviation on binocular positions: (horizontal, vertical) PD, + = eso / right hyper.
    deviation_pd: Tuple[float, float] = (0.0, 0.0)


def default_hirschberg_pd_per_mm(kb: KnowledgeBase) -> float:
    """The KB default ratio (deg/mm) as PD per mm of reflex shift, linearised at 1 mm."""
    return degrees_to_pd(kb.scalar("hirschberg_ratio_deg_per_mm").value)   # type: ignore[arg-type]


def _rotations_pd(plan: PositionPlan, sim: SimEyes, distance_cm: float) -> Dict[str, Tuple[float, float]]:
    """Each observed eye's (nasal, up) rotation relative to the light, PD."""
    if plan.fixing_eye is not None:
        target_pd = 100.0 * plan.target_offset_cm / distance_cm     # + = target to the patient's right
        # Looking to the patient's right turns the right eye out and the left eye in.
        return {plan.fixing_eye: (-target_pd if plan.fixing_eye == "R" else target_pd, 0.0)}
    h, v = sim.deviation_pd
    return {"R": (0.0, 0.0), "L": (h, -v)}


def _to_image(temporal_mm: float, inferior_mm: float, eye: str, roll_deg: float) -> Tuple[float, float]:
    """Head-frame reflex decentration (mm; + = temporal / inferior) → image axes, rotated by the head roll."""
    x_h = -temporal_mm if eye == "R" else temporal_mm    # head x axis runs from the right eye to the left
    r = np.radians(roll_deg)
    return (x_h * np.cos(r) + inferior_mm * np.sin(r), -x_h * np.sin(r) + inferior_mm * np.cos(r))


@dataclass(frozen=True)
class FrameNoise:
    landmark_px: float = 0.3
    head_deg: float = 1.0
    sharpness: float = 2.0
    seed: Optional[int] = 0


def _deg_per_px(kb: KnowledgeBase, iris_radius_px: float) -> float:
    iris_radius_mm = kb.scalar("iris_diameter_mm").value / 2.0          # type: ignore[operator]
    return iris_radius_mm * kb.scalar("hirschberg_ratio_deg_per_mm").value / iris_radius_px  # type: ignore[operator]


def simulate_position(plan: PositionPlan, n_frames: int = 60, fps: float = 30.0,
                      ev: FrameEvents = FrameEvents(), noise: FrameNoise = FrameNoise(),
                      iris_radius_px: float = 86.0, kb: Optional[KnowledgeBase] = None,
                      sim: SimEyes = SimEyes(), distance_cm: float = 36.0) -> PositionCapture:
    kb = kb or load_kb()
    rng = np.random.default_rng(noise.seed)
    px_per_deg = 1.0 / _deg_per_px(kb, iris_radius_px)
    px_per_mm = iris_radius_px / (kb.scalar("iris_diameter_mm").value / 2.0)   # type: ignore[operator]
    ratio = sim.hirschberg_pd_per_mm or default_hirschberg_pd_per_mm(kb)
    rotations = _rotations_pd(plan, sim, distance_cm)
    roll = 0.0 if plan.head.roll_deg is None else (plan.head.roll_deg.min + plan.head.roll_deg.max) / 2
    frames: List[FrameObservation] = []
    for i in range(n_frames):
        if i in ev.dropped:
            continue
        t_ms = i * 1000.0 / fps
        if i in ev.no_face:
            frames.append(FrameObservation(index=i, t_ms=t_ms))
            continue
        yaw = plan.head.yaw_deg + rng.normal(0, noise.head_deg) + (12.0 if i in ev.head_off_target else 0.0)
        pitch = plan.head.pitch_deg + rng.normal(0, noise.head_deg)
        glance = ev.glance_deg if i in ev.glance else 0.0
        frame_roll = roll + rng.normal(0, noise.head_deg)
        eyes = {}
        for eye, (cx, cy) in EYE_CENTRE_PX.items():
            if eye not in rotations:
                continue                                  # covered
            centre = (cx + rng.normal(0, noise.landmark_px), cy + rng.normal(0, noise.landmark_px))
            nasal_pd, up_pd = rotations[eye]
            kt, ki = sim.kappa_mm.get(eye, (0.0, 0.0))
            # An eye turned in (nasal) shows its reflex temporally; turned up, inferiorly.
            dx, dy = _to_image(kt + nasal_pd / ratio, ki + up_pd / ratio, eye, frame_roll)
            reflex = (centre[0] + dx * px_per_mm + glance * px_per_deg + rng.normal(0, noise.landmark_px),
                      centre[1] + dy * px_per_mm + rng.normal(0, noise.landmark_px))
            open_ratio = 0.1 if i in ev.blink else 0.8
            eyes[eye] = EyeObservation(
                reflex_px=None if (eye == "R" and i in ev.no_reflex_right) else reflex,
                limbus=Ellipse(centre_px=centre, semi_major_px=iris_radius_px,
                               semi_minor_px=iris_radius_px, angle_deg=0.0),
                pupil_px=centre, pupil_diameter_px=0.35 * iris_radius_px,
                iris_radius_px=iris_radius_px,
                fissure_height_px=open_ratio * 2 * iris_radius_px,
                scleral_show_px=0.05 * iris_radius_px,
                limbus_covered_fraction=0.6 if i in ev.lid_low else 0.15,
            )
        sharp = max(0.0, 40.0 + rng.normal(0, noise.sharpness)) * (0.4 if i in ev.blur else 1.0)
        frames.append(FrameObservation(index=i, t_ms=t_ms,
                                       head_pose=HeadPose(yaw_deg=yaw, pitch_deg=pitch, roll_deg=frame_roll),
                                       sharpness=sharp, eyes=eyes))
    return PositionCapture(position=plan.position, frames=frames)


@dataclass(frozen=True)
class SessionSpec:
    protocol: Protocol = "nine_gaze"
    include_tilt: bool = True
    include_calibration: bool = True
    n_frames: int = 60
    fps: float = 30.0
    width_px: int = 3840
    height_px: int = 2160
    exposure_s: Optional[float] = 1 / 500
    distance_cm: Optional[float] = 36.0
    distance_error_cm: float = 1.0
    events: Dict[str, FrameEvents] = field(default_factory=dict)
    skip: Tuple[str, ...] = ()
    noise: FrameNoise = FrameNoise()
    sim: SimEyes = SimEyes()
    # Distance the patient really sat at (the header reports distance_cm ± distance_error_cm).
    true_distance_cm: Optional[float] = None


def simulate_session(spec: SessionSpec = SessionSpec(), kb: Optional[KnowledgeBase] = None) -> CaptureSession:
    kb = kb or load_kb()
    plan: ProtocolPlan = build_plan(spec.protocol, include_tilt=spec.include_tilt, kb=kb,
                                    include_calibration=spec.include_calibration)
    true_distance = spec.true_distance_cm or spec.distance_cm or 36.0
    captures = []
    for k, p in enumerate(plan.positions):
        if p.position in spec.skip:
            continue
        noise = FrameNoise(spec.noise.landmark_px, spec.noise.head_deg, spec.noise.sharpness,
                           None if spec.noise.seed is None else spec.noise.seed + k)
        captures.append(simulate_position(p, spec.n_frames, spec.fps,
                                          spec.events.get(p.position, FrameEvents()), noise, kb=kb,
                                          sim=spec.sim, distance_cm=true_distance))
    header = SessionHeader(
        session_id="sim-session", protocol_version=CAPTURE_PROTOCOL_VERSION, protocol=spec.protocol,
        device=DeviceInfo(model="Simulated phone", lens="main"),
        settings=CaptureSettings(width_px=spec.width_px, height_px=spec.height_px, fps=spec.fps,
                                 exposure_s=spec.exposure_s),
        distance_cm=None if spec.distance_cm is None else Measured(value=spec.distance_cm, error=spec.distance_error_cm),
        started_at="2026-10-30T10:00:00Z",
    )
    return CaptureSession(header=header, captures=captures)
