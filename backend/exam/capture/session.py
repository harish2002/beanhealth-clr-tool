"""
BeanHealth CLR — Capture Session Format (brief §4.1)
=====================================================

What the phone uploads after a gaze exam. Landmarks are extracted on the
phone (approved design), so a session carries per-frame observations — not
video. Raw frames never leave the phone unless the patient consents (brief §7).

    CaptureSession
      header    protocol version, device, camera settings, distance estimate
      captures  one PositionCapture per position: its frames, in order

Coordinates are pixels in the full, unmirrored frame (x right, y down).
"R" / "L" are the PATIENT's eyes; head pose follows exam.capture.protocol.
Any observation the phone could not make is null — the QC treats a missing
value as a failed check, never as a pass.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import Field, model_validator

from exam.capture.protocol import CAPTURE_PROTOCOL_VERSION, TILT_POSITIONS
from exam.kb.schema import Eye
from exam.models import PROTOCOL_POSITIONS, Measured, Protocol, Strict

Point = Tuple[float, float]
# 20 s at 60 fps — bounds the upload; a position is held for about 2 s.
MAX_FRAMES_PER_POSITION = 1200


class HeadPose(Strict):
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


class EyeObservation(Strict):
    reflex_px: Optional[Point] = None          # corneal light reflex centroid; null = not found
    pupil_px: Optional[Point] = None           # pupil centre
    iris_radius_px: Optional[float] = Field(default=None, gt=0)
    fissure_height_px: Optional[float] = Field(default=None, ge=0)
    # Fraction of the limbus perimeter hidden by the lids.
    limbus_covered_fraction: Optional[float] = Field(default=None, ge=0, le=1)


class FrameObservation(Strict):
    index: int = Field(ge=0)
    t_ms: float = Field(ge=0)
    head_pose: Optional[HeadPose] = None       # null = no face found
    # exam.qc.sharpness over the eye regions (the lower of the two eyes).
    sharpness: Optional[float] = Field(default=None, ge=0)
    eyes: Dict[Eye, EyeObservation] = Field(default_factory=dict)


class PositionCapture(Strict):
    position: str
    frames: List[FrameObservation] = Field(default_factory=list, max_length=MAX_FRAMES_PER_POSITION)

    @model_validator(mode="after")
    def _frames_in_order(self) -> "PositionCapture":
        for a, b in zip(self.frames, self.frames[1:]):
            if b.index <= a.index or b.t_ms <= a.t_ms:
                raise ValueError(f"{self.position}: frames must have increasing index and t_ms")
        return self


class DeviceInfo(Strict):
    model: str = Field(min_length=1)
    lens: Optional[str] = None
    # The existing per-phone calibration mechanism's profile (brief §4.4).
    calibration_profile_id: Optional[str] = None


class CaptureSettings(Strict):
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    fps: float = Field(gt=0)
    exposure_s: Optional[float] = Field(default=None, gt=0)   # most browsers cannot report it


class SessionHeader(Strict):
    session_id: str = Field(min_length=1)
    protocol_version: str
    protocol: Protocol
    tier: Literal[1] = 1
    device: DeviceInfo
    settings: CaptureSettings
    distance_cm: Optional[Measured] = None
    started_at: str

    @model_validator(mode="after")
    def _known_protocol_version(self) -> "SessionHeader":
        if self.protocol_version != CAPTURE_PROTOCOL_VERSION:
            raise ValueError(
                f"protocol_version {self.protocol_version!r} is not supported "
                f"(this server speaks {CAPTURE_PROTOCOL_VERSION!r})"
            )
        return self


class CaptureSession(Strict):
    header: SessionHeader
    captures: List[PositionCapture] = Field(default_factory=list)

    @model_validator(mode="after")
    def _positions_belong_to_protocol(self) -> "CaptureSession":
        allowed = set(PROTOCOL_POSITIONS[self.header.protocol]) | set(TILT_POSITIONS)
        seen = set()
        for c in self.captures:
            if c.position not in allowed:
                raise ValueError(f"{c.position!r} is not a position of the {self.header.protocol} protocol")
            if c.position in seen:
                raise ValueError(f"{c.position!r} captured twice")
            seen.add(c.position)
        return self

    def capture(self, position: str) -> Optional[PositionCapture]:
        return next((c for c in self.captures if c.position == position), None)
