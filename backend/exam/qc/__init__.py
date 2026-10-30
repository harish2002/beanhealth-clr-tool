"""
BeanHealth CLR — Frame QC (brief §4.2).

    from exam.qc import run_qc
    report = run_qc(session)      # per-frame reasons, per-position OK / INSUFFICIENT_DATA
"""

from exam.qc.frame_qc import (
    QC_VERSION,
    FrameVerdict,
    PositionQC,
    Rejection,
    SessionFlag,
    SessionQC,
    run_qc,
    sharpness,
)

__all__ = ["QC_VERSION", "FrameVerdict", "PositionQC", "Rejection", "SessionFlag", "SessionQC",
           "run_qc", "sharpness"]
