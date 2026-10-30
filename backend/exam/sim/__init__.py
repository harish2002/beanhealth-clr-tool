"""
BeanHealth CLR — Synthetic case generator (test fixture, not physiology).

    from exam.sim import CASES, simulate
    measurements = simulate(CASES["r_cn4"])
"""

from exam.sim.cases import CASE_NOISE, CASES, CYCLOVERTICAL, parks_case
from exam.sim.model import Noise, SimCase, simulate, true_deviation, true_duction, true_tilt

__all__ = ["CASE_NOISE", "CASES", "CYCLOVERTICAL", "Noise", "SimCase", "parks_case",
           "simulate", "true_deviation", "true_duction", "true_tilt"]
