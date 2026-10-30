"""
BeanHealth CLR — Gaze examination API (Phase 3)
================================================

    GET  /exam/protocol/{protocol}   the capture plan the phone follows
    POST /exam/qc                    frame QC of an uploaded capture session
    POST /exam/calibrate             kappa and Hirschberg ratio from its calibration captures
    POST /exam/landmarks             one frame's landmarks, computed on the server

Sessions carry landmark tracks only, never images (brief §7). /exam/landmarks
is the reference the phone's own extraction is checked against: the frame is
processed in memory and discarded with the request.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from exam.calibration import Calibration, calibrate
from exam.capture import CaptureSession, FrameObservation, ProtocolPlan, build_plan
from exam.landmarks.eye import LANDMARKS_VERSION
from exam.landmarks.extract import extract_frame
from exam.models import Protocol, Strict
from exam.qc import SessionQC, run_qc
from utils.image_utils import downscale_if_needed, load_image_from_bytes

router = APIRouter(prefix="/exam", tags=["Gaze examination"])

_TILT = "include the head-tilt captures"
_CALIBRATION = "include the calibration captures (kappa and Hirschberg ratio)"
# Longest side processed; larger frames are downscaled first (CLAUDE.md §10).
_MAX_FRAME_SIDE = 4000


class FrameLandmarks(Strict):
    width_px: int
    height_px: int
    frame: FrameObservation
    landmarks_version: str = LANDMARKS_VERSION


@router.get("/protocol/{protocol}", response_model=ProtocolPlan)
def get_protocol(protocol: Protocol, tilt: bool = Query(True, description=_TILT),
                 calibration: bool = Query(True, description=_CALIBRATION)) -> ProtocolPlan:
    """Positions in capture order, each with its head-pose target, hold time and frame minimum."""
    return build_plan(protocol, include_tilt=tilt, include_calibration=calibration)


@router.post("/qc", response_model=SessionQC)
def post_qc(session: CaptureSession, tilt: bool = Query(True, description=_TILT),
            calibration: bool = Query(True, description=_CALIBRATION)) -> SessionQC:
    """Per-frame rejection reasons and per-position OK / INSUFFICIENT_DATA."""
    return run_qc(session, include_tilt=tilt, include_calibration=calibration)


@router.post("/calibrate", response_model=Calibration)
def post_calibrate(session: CaptureSession) -> Calibration:
    """Kappa per eye and the Hirschberg ratio in use (fitted or KB default, with the reason)."""
    return calibrate(session)


@router.post("/landmarks", response_model=FrameLandmarks)
async def post_landmarks(image: UploadFile = File(..., description="one unmirrored video frame (JPEG/PNG)"),
                         index: int = Query(0, ge=0), t_ms: float = Query(0.0, ge=0)) -> FrameLandmarks:
    """Per-eye landmarks and head pose of one frame, in the coordinates of the frame as processed."""
    try:
        rgb = load_image_from_bytes(await image.read())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Uploaded file is not a valid image. Please upload a JPEG or PNG.")
    rgb = downscale_if_needed(rgb, _MAX_FRAME_SIDE)
    h, w = rgb.shape[:2]
    return FrameLandmarks(width_px=w, height_px=h, frame=extract_frame(rgb, index, t_ms))
