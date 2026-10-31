"""
BeanHealth CLR — The gaze examination, end to end (brief §4)
=============================================================

    capture session → frame QC → calibration → per-gaze measurement → rules engine

    result = examine(session)

One KB is loaded once and used by every stage, so the result is traceable to
a single kb_version / kb_sha256. Each stage's own version travels with its
output; `versions` repeats them in one place for the report (milestone 7)
and the database (milestone 8).
"""

from __future__ import annotations

from typing import Dict, Optional

from exam.calibration import Calibration, calibrate
from exam.capture.session import CaptureSession
from exam.engine import ENGINE_VERSION, POSITIONING, evaluate
from exam.engine.findings import EngineResult
from exam.kb import load_kb
from exam.kb.loader import KnowledgeBase
from exam.measurement import GridMeasurement, measure
from exam.models import Strict
from exam.qc import SessionQC, run_qc


class ExamResult(Strict):
    session_id: str
    positioning: str
    qc: SessionQC
    calibration: Calibration
    measurement: GridMeasurement
    engine: EngineResult
    versions: Dict[str, str]


def examine(session: CaptureSession, kb: Optional[KnowledgeBase] = None,
            include_tilt: bool = True) -> ExamResult:
    """include_tilt=False: an exam planned without head tilts (Parks step 3 then cannot run)."""
    kb = kb or load_kb()
    qc = run_qc(session, kb=kb, include_tilt=include_tilt)
    cal = calibrate(session, qc, kb)
    grid = measure(session, qc, cal, kb)
    engine = evaluate(grid.measurements, kb)
    versions = {
        "capture_protocol": grid.capture_protocol_version,
        "qc": qc.qc_version,
        "calibration": cal.calibration_version,
        "measurement": grid.measurement_version,
        "engine": ENGINE_VERSION,
        "kb": kb.version,
        "kb_sha256": kb.sha256,
    }
    return ExamResult(session_id=session.header.session_id, positioning=POSITIONING, qc=qc,
                      calibration=cal, measurement=grid, engine=engine, versions=versions)
