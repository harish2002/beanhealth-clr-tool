"""
BeanHealth CLR — Gaze examination API (Phase 3)
================================================

    GET  /exam/protocol/{protocol}   the capture plan the phone follows
    POST /exam/qc                    frame QC of an uploaded capture session
    POST /exam/calibrate             kappa and Hirschberg ratio from its calibration captures
    POST /exam/measure               the per-gaze grid with errors (QC and calibration included)
    POST /exam/evaluate              the whole exam: QC, calibration, grid and rules-engine findings
    POST /exam/report[/html|/pdf]    the clinician report (JSON, HTML page or PDF)
    POST /exam/sessions              examine and store a session (pseudonymized; needs consent and EXAM_DB_PATH)
    POST /exam/landmarks             one frame's landmarks, computed on the server

Sessions carry landmark tracks only, never images (brief §7). /exam/landmarks
is the reference the phone's own extraction is checked against: the frame is
processed in memory and discarded with the request.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import HTMLResponse, Response

from exam.calibration import Calibration, calibrate
from exam.capture import CaptureSession, FrameObservation, ProtocolPlan, build_plan
from exam.db import PatientRecord, StoredSession, connect, store_exam
from exam.examine import ExamResult, examine
from exam.landmarks.eye import LANDMARKS_VERSION
from exam.landmarks.extract import extract_frame
from exam.measurement import GridMeasurement, measure
from exam.models import Protocol, Strict
from exam.report import ExamReport, ReportPatient, build_report, render_html, render_pdf
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


@router.post("/measure", response_model=GridMeasurement)
def post_measure(session: CaptureSession) -> GridMeasurement:
    """Deviation (PD ± error, confidence) per gaze and head tilt; ductions, lids and pupils per eye."""
    return measure(session)


@router.post("/evaluate", response_model=ExamResult)
def post_evaluate(session: CaptureSession, tilt: bool = Query(True, description=_TILT)) -> ExamResult:
    """QC → calibration → measurement → rules engine: findings, referral decision and every version."""
    return examine(session, include_tilt=tilt)


class ReportRequest(Strict):
    session: CaptureSession
    # Printed on the report only; never stored (brief §7).
    patient: Optional[ReportPatient] = None


def _report(req: ReportRequest, tilt: bool) -> ExamReport:
    return build_report(req.session, examine(req.session, include_tilt=tilt), patient=req.patient)


@router.post("/report", response_model=ExamReport)
def post_report(req: ReportRequest, tilt: bool = Query(True, description=_TILT)) -> ExamReport:
    """The clinician report as JSON: red flags, referral, grid, findings, quality, calibration, versions."""
    return _report(req, tilt)


@router.post("/report/html", response_class=HTMLResponse)
def post_report_html(req: ReportRequest, tilt: bool = Query(True, description=_TILT)) -> HTMLResponse:
    """The clinician report as one self-contained HTML page (print it to PDF from the browser)."""
    return HTMLResponse(render_html(_report(req, tilt)))


@router.post("/report/pdf", response_class=Response,
             responses={200: {"content": {"application/pdf": {}}}})
def post_report_pdf(req: ReportRequest, tilt: bool = Query(True, description=_TILT)) -> Response:
    """The clinician report as an A4 PDF."""
    report = _report(req, tilt)
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in report.session.session_id)[:60]
    return Response(render_pdf(report), media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="gaze-exam-{name}.pdf"'})


class StoreRequest(Strict):
    session: CaptureSession
    patient: PatientRecord


class StoreResponse(Strict):
    stored: StoredSession
    result: ExamResult


@router.post("/sessions", response_model=StoreResponse, status_code=status.HTTP_201_CREATED)
def post_session(req: StoreRequest, tilt: bool = Query(True, description=_TILT)) -> StoreResponse:
    """Examine a session and store it with its patient's pseudo ID (no names, no images)."""
    path = os.environ.get("EXAM_DB_PATH")
    if not path:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Session storage is not configured on this server.")
    if not req.patient.consent.store_measurements:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="The patient has not consented to storing measurements.")
    result = examine(req.session, include_tilt=tilt)
    conn = connect(path)
    try:
        stored = store_exam(conn, req.session, result, req.patient)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    finally:
        conn.close()
    return StoreResponse(stored=stored, result=result)


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
