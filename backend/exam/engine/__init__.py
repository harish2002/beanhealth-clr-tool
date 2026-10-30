"""
BeanHealth CLR — Rules Engine.

Deterministic, KB-driven: measurements (9-gaze grid, head tilt, lids,
pupils, ductions) → findings with evidence, confidence and versions →
referral decision.
"""

from exam.engine.engine import ENGINE_VERSION, LIMITATIONS, POSITIONING, decide_referral, evaluate

__all__ = ["ENGINE_VERSION", "LIMITATIONS", "POSITIONING", "decide_referral", "evaluate"]
