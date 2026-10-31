"""
BeanHealth CLR — Tier 2 (phone + IR clip): interfaces and stubs (brief §3).

    from exam.tier2 import analyse_cover_test, detect_saccades
    result = analyse_cover_test(session)     # movements per eye and window; no interpretation yet
"""

from exam.tier2.movement import MOVEMENT_VERSION, CoverTestResult, EyeMovement, analyse_cover_test, movements_by
from exam.tier2.saccades import SACCADES_VERSION, Saccade, SaccadeTrace, detect_saccades
from exam.tier2.session import (COVER_TEST_PROTOCOL_VERSION, CoverTestSession, CoverTrial, GazeSample,
                                SessionProblem, check_session)

__all__ = ["COVER_TEST_PROTOCOL_VERSION", "MOVEMENT_VERSION", "SACCADES_VERSION", "CoverTestResult",
           "CoverTestSession", "CoverTrial", "EyeMovement", "GazeSample", "Saccade", "SaccadeTrace",
           "SessionProblem", "analyse_cover_test", "check_session", "detect_saccades", "movements_by"]
