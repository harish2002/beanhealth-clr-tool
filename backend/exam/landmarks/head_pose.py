"""
BeanHealth CLR — Head pose from face-mesh landmarks (brief §4.3)
=================================================================

Yaw, pitch and roll in the capture protocol's convention:
  yaw    + = face turned toward the PATIENT's right
  pitch  + = chin up
  roll   + = head tilted toward the patient's right shoulder

Method: build the face's own axes from the 3D face-mesh landmarks (x, y in
pixels, z in the same scale as x, larger z = farther from the camera, as
MediaPipe reports them), then decompose the rotation as
yaw (about the vertical) · pitch (about the horizontal) · roll (about the
line of sight):
  x axis  right-eye centre → left-eye centre (both eye contours averaged)
  y axis  forehead → chin, made perpendicular to x
  z axis  x × y (into the head)

Geometric rather than a fitted 3D model, so the phone computes exactly the
same numbers without OpenCV. Limitations: face-mesh depth is approximate, so
yaw and pitch are good to a few degrees; the QC tolerance (capture.yaml)
reflects that.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from exam.capture.session import HeadPose

# Face-mesh indices. The "right" set is the PATIENT's right eye (image left).
RIGHT_EYE_RING = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
LEFT_EYE_RING = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
FOREHEAD = (10, 151, 9)
CHIN = (152, 175, 199)

Vec = Tuple[float, float, float]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("degenerate face landmarks")
    return v / n


def face_axes(points: np.ndarray) -> np.ndarray:
    """3×3 matrix whose columns are the face's x, y, z axes in camera coordinates."""
    eye_r = points[list(RIGHT_EYE_RING)].mean(axis=0)
    eye_l = points[list(LEFT_EYE_RING)].mean(axis=0)
    top = points[list(FOREHEAD)].mean(axis=0)
    chin = points[list(CHIN)].mean(axis=0)
    ex = _unit(eye_l - eye_r)
    down = chin - top
    ey = _unit(down - np.dot(down, ex) * ex)
    ez = np.cross(ex, ey)
    return np.stack([ex, ey, ez], axis=1)


def pose_from_axes(m: np.ndarray) -> HeadPose:
    """Decompose M = Ry(yaw) · Rx(−pitch) · Rz(−roll) (see module docstring for the signs)."""
    ex, ez = m[:, 0], m[:, 2]
    yaw = math.atan2(ez[0], ez[2])
    pitch = math.asin(max(-1.0, min(1.0, ez[1])))
    # Undo yaw and pitch; what is left of the x axis is the roll in the image plane.
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry_t = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])          # Ry(yaw)ᵀ
    rx_t = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])         # Rx(−pitch)ᵀ
    x_rolled = rx_t @ ry_t @ ex
    roll = math.atan2(-x_rolled[1], x_rolled[0])
    return HeadPose(yaw_deg=math.degrees(yaw), pitch_deg=math.degrees(pitch), roll_deg=math.degrees(roll))


def pose_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """The rotation pose_from_axes inverts (used to synthesise faces in tests)."""
    y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(-p), -math.sin(-p)], [0, math.sin(-p), math.cos(-p)]])
    rz = np.array([[math.cos(-r), -math.sin(-r), 0], [math.sin(-r), math.cos(-r), 0], [0, 0, 1]])
    return ry @ rx @ rz


def head_pose(points: Sequence[Vec]) -> HeadPose:
    """Head pose from ≥ 468 face-mesh landmarks as (x_px, y_px, z) with z scaled like x."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 468:
        raise ValueError("head_pose needs at least 468 landmarks of (x, y, z)")
    return pose_from_axes(face_axes(pts))
