"""
BeanHealth CLR — Calibration (brief §4.4): scale, per-patient Hirschberg ratio, angle kappa.

    from exam.calibration import calibrate, eye_rotation_pd
    cal = calibrate(session)                       # from the session's calibration captures
    nasal_pd, up_pd = eye_rotation_pd(obs, "R", roll_deg, cal)
"""

from exam.calibration.calibrate import (
    CALIBRATION_VERSION,
    Calibration,
    CalibrationFlag,
    CalibrationPoint,
    EyeKappa,
    HirschbergRatio,
    calibrate,
    decentration_mm,
    eye_rotation_pd,
    robust_location,
)

__all__ = ["CALIBRATION_VERSION", "Calibration", "CalibrationFlag", "CalibrationPoint", "EyeKappa",
           "HirschbergRatio", "calibrate", "decentration_mm", "eye_rotation_pd", "robust_location"]
