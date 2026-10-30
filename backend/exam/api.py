"""
BeanHealth CLR — Gaze examination API (Phase 3)
================================================

    GET  /exam/protocol/{protocol}   the capture plan the phone follows
    POST /exam/qc                    frame QC of an uploaded capture session

The session body carries landmark tracks only, never images (brief §7).
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from exam.capture import CaptureSession, ProtocolPlan, build_plan
from exam.models import Protocol
from exam.qc import SessionQC, run_qc

router = APIRouter(prefix="/exam", tags=["Gaze examination"])


@router.get("/protocol/{protocol}", response_model=ProtocolPlan)
def get_protocol(protocol: Protocol, tilt: bool = Query(True, description="include the head-tilt captures")) -> ProtocolPlan:
    """Positions in capture order, each with its head-pose target, hold time and frame minimum."""
    return build_plan(protocol, include_tilt=tilt)


@router.post("/qc", response_model=SessionQC)
def post_qc(session: CaptureSession, tilt: bool = Query(True, description="expect the head-tilt captures")) -> SessionQC:
    """Per-frame rejection reasons and per-position OK / INSUFFICIENT_DATA."""
    return run_qc(session, include_tilt=tilt)
