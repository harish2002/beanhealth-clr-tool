"""
BeanHealth CLR — Capture protocol and session format (brief §4.1).

    from exam.capture import build_plan
    plan = build_plan("nine_gaze")        # what the phone should capture
"""

from exam.capture.protocol import (
    CAPTURE_PROTOCOL_VERSION,
    GAZE_DIRECTION,
    TILT_POSITIONS,
    HeadPoseTarget,
    PositionPlan,
    ProtocolPlan,
    build_plan,
)
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

__all__ = [
    "CAPTURE_PROTOCOL_VERSION", "GAZE_DIRECTION", "TILT_POSITIONS", "CaptureSession",
    "CaptureSettings", "DeviceInfo", "EyeObservation", "FrameObservation", "HeadPose",
    "HeadPoseTarget", "PositionCapture", "PositionPlan", "ProtocolPlan", "SessionHeader",
    "build_plan",
]
