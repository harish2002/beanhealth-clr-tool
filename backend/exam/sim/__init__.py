"""
BeanHealth CLR — Synthetic case generator (test fixture, not physiology).

    from exam.sim import CASES, simulate
    measurements = simulate(CASES["r_cn4"])
"""

from exam.sim.frames import FrameEvents, FrameNoise, SessionSpec, events, simulate_position, simulate_session
from exam.sim.cases import CASE_NOISE, CASES, CYCLOVERTICAL, parks_case
from exam.sim.model import Noise, SimCase, simulate, true_deviation, true_duction, true_tilt

__all__ = ["CASE_NOISE", "CASES", "CYCLOVERTICAL", "FrameEvents", "FrameNoise", "Noise", "SessionSpec",
           "SimCase", "events", "parks_case", "simulate", "simulate_position", "simulate_session",
           "true_deviation", "true_duction", "true_tilt"]
