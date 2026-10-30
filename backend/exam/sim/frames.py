"""
BeanHealth CLR — Synthetic frame tracks (test fixture, not physiology).

Builds the per-frame observations a phone would upload for each planned
position — steady fixation with landmark jitter — and lets a test inject the
events frame QC must catch: blinks, blur, glances away, head off target, a
missing reflex, a lost face, dropped frames.

Geometry: a 4K landscape frame, the patient's right eye on the image left
(unmirrored). Gaze is turned into a reflex-to-pupil offset with the KB's
iris-diameter scale and Hirschberg ratio, the inverse of what frame QC does.
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
    EyeObservation,
    FrameObservation,
    HeadPose,
    PositionCapture,
    SessionHeader,
)
from exam.kb import load_kb
from exam.kb.loader import KnowledgeBase
from exam.models import Measured, Protocol

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
                      iris_radius_px: float = 86.0, kb: Optional[KnowledgeBase] = None) -> PositionCapture:
    kb = kb or load_kb()
    rng = np.random.default_rng(noise.seed)
    px_per_deg = 1.0 / _deg_per_px(kb, iris_radius_px)
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
        eyes = {}
        for eye, (cx, cy) in EYE_CENTRE_PX.items():
            pupil = (cx + rng.normal(0, noise.landmark_px), cy + rng.normal(0, noise.landmark_px))
            reflex = (pupil[0] + glance * px_per_deg + rng.normal(0, noise.landmark_px),
                      pupil[1] + rng.normal(0, noise.landmark_px))
            open_ratio = 0.1 if i in ev.blink else 0.8
            eyes[eye] = EyeObservation(
                reflex_px=None if (eye == "R" and i in ev.no_reflex_right) else reflex,
                pupil_px=pupil, iris_radius_px=iris_radius_px,
                fissure_height_px=open_ratio * 2 * iris_radius_px,
                limbus_covered_fraction=0.6 if i in ev.lid_low else 0.15,
            )
        sharp = max(0.0, 40.0 + rng.normal(0, noise.sharpness)) * (0.4 if i in ev.blur else 1.0)
        frames.append(FrameObservation(index=i, t_ms=t_ms,
                                       head_pose=HeadPose(yaw_deg=yaw, pitch_deg=pitch,
                                                          roll_deg=roll + rng.normal(0, noise.head_deg)),
                                       sharpness=sharp, eyes=eyes))
    return PositionCapture(position=plan.position, frames=frames)


@dataclass(frozen=True)
class SessionSpec:
    protocol: Protocol = "nine_gaze"
    include_tilt: bool = True
    n_frames: int = 60
    fps: float = 30.0
    width_px: int = 3840
    height_px: int = 2160
    exposure_s: Optional[float] = 1 / 500
    distance_cm: Optional[float] = 36.0
    events: Dict[str, FrameEvents] = field(default_factory=dict)
    skip: Tuple[str, ...] = ()
    noise: FrameNoise = FrameNoise()


def simulate_session(spec: SessionSpec = SessionSpec(), kb: Optional[KnowledgeBase] = None) -> CaptureSession:
    kb = kb or load_kb()
    plan: ProtocolPlan = build_plan(spec.protocol, include_tilt=spec.include_tilt, kb=kb)
    captures = []
    for k, p in enumerate(plan.positions):
        if p.position in spec.skip:
            continue
        noise = FrameNoise(spec.noise.landmark_px, spec.noise.head_deg, spec.noise.sharpness,
                           None if spec.noise.seed is None else spec.noise.seed + k)
        captures.append(simulate_position(p, spec.n_frames, spec.fps,
                                          spec.events.get(p.position, FrameEvents()), noise, kb=kb))
    header = SessionHeader(
        session_id="sim-session", protocol_version=CAPTURE_PROTOCOL_VERSION, protocol=spec.protocol,
        device=DeviceInfo(model="Simulated phone", lens="main"),
        settings=CaptureSettings(width_px=spec.width_px, height_px=spec.height_px, fps=spec.fps,
                                 exposure_s=spec.exposure_s),
        distance_cm=None if spec.distance_cm is None else Measured(value=spec.distance_cm, error=1.0),
        started_at="2026-10-30T10:00:00Z",
    )
    return CaptureSession(header=header, captures=captures)
