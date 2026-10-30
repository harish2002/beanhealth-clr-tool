"""
BeanHealth CLR — Capture protocol and session format (brief §4.1).

    from exam.capture import build_plan
    plan = build_plan("nine_gaze")        # what the phone should capture
"""

from exam.capture.protocol import (
    CALIBRATION_POSITIONS,
    CALIBRATION_TARGETS,
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
    Ellipse,
    EyeObservation,
    FrameObservation,
    HeadPose,
    PositionCapture,
    SessionHeader,
)

__all__ = [
    "CALIBRATION_POSITIONS", "CALIBRATION_TARGETS", "CAPTURE_PROTOCOL_VERSION", "GAZE_DIRECTION", "TILT_POSITIONS", "CaptureSession",
    "CaptureSettings", "DeviceInfo", "Ellipse", "EyeObservation", "FrameObservation", "HeadPose",
    "HeadPoseTarget", "PositionCapture", "PositionPlan", "ProtocolPlan", "SessionHeader",
    "build_plan",
]
