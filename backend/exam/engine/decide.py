"""
BeanHealth CLR — Error-Aware Decisions
=======================================

Every comparison the engine makes has three outcomes:

    PRESENT       clear of the threshold by more than k × error
    ABSENT        clear of it the other way
    INCONCLUSIVE  within k × error of it, or the data or threshold is missing

There are two kinds of test:

  at_least(q, T)   "q ≥ T". PRESENT if q − k·e ≥ T, ABSENT if q + k·e < T,
                   otherwise INCONCLUSIVE. Used whenever there is a threshold,
                   and for "which side is worse" (T = 0), where a result
                   inside the error cannot pick a side.
  beyond_error(q)  "is anything there at all?" PRESENT if |q| > k·e (or, one-
                   sided, q > k·e), otherwise
                   ABSENT — a value indistinguishable from zero is not
                   detected. Its confidence carries how much that is worth:
                   imprecise (LOW) data cannot support a referral.

Conditions combine features with three-valued (Kleene) logic: one ABSENT
requirement rules a condition out; otherwise any INCONCLUSIVE one makes it
INCONCLUSIVE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

from exam.engine.findings import Evidence, Reason
from exam.models import Confidence, Measured, lowest_confidence

State = Literal["PRESENT", "ABSENT", "INCONCLUSIVE"]
_NORMAL = NormalDist()

# Reason codes. `repeatable`: a new capture could resolve it (decides
# REPEAT_TEST advice); the others need a clinician, a KB value or a later
# milestone, so repeating the test would not help. `within_error` is
# repeatable only when the value leans toward the feature being present —
# a result that is probably normal but not proven is not worth a repeat.
REASONS = {
    "primary_insufficient":     True,
    "insufficient_data":        True,
    "within_error":             True,
    "tilt_not_captured":        True,
    "parks_step2_within_error": True,
    "parks_step3_within_error": True,
    "parks_steps_conflict":     False,
    "not_in_protocol":          False,
    "not_measured":             False,
    "threshold_not_set":        False,
}


def reason(code: str, detail: str, repeatable: Optional[bool] = None) -> Reason:
    return Reason(code=code, detail=detail, repeatable=REASONS[code] if repeatable is None else repeatable)


def family_k(k: float, n: int) -> float:
    """
    k for one of n tests, so that "any of n beyond k_n × error" has the same
    false-alarm rate as one test at k (Šidák). Scanning 9 gaze cells at plain
    k = 2 would flag a stray cell about a third of the time.
    """
    if n <= 1:
        return k
    p_one = 2.0 * (1.0 - _NORMAL.cdf(k))
    p_each = 1.0 - (1.0 - p_one) ** (1.0 / n)
    return _NORMAL.inv_cdf(1.0 - p_each / 2.0)


@dataclass(frozen=True)
class Decision:
    state: State
    detail: str
    evidence: Tuple[Evidence, ...] = ()
    reasons: Tuple[Reason, ...] = ()
    kb_ids: Tuple[str, ...] = ()
    margin_sigma: Optional[float] = None

    @property
    def confidence(self) -> Confidence:
        return lowest_confidence([e.confidence for e in self.evidence])


def inconclusive(code: str, detail: str, kb_ids: Sequence[str] = (),
                 evidence: Sequence[Evidence] = ()) -> Decision:
    return Decision("INCONCLUSIVE", detail, tuple(evidence), (reason(code, detail),), tuple(kb_ids))


@dataclass(frozen=True)
class Quantity:
    """A derived quantity (a cell value, or a difference of cells) with its combined error."""
    name: str
    value: float
    error: float
    unit: str
    confidence: Confidence

    @staticmethod
    def of(name: str, m: Measured, unit: str, sign: float = 1.0) -> "Quantity":
        return Quantity(name, sign * m.value, m.error, unit, m.confidence)

    @staticmethod
    def diff(name: str, a: Measured, b: Measured, unit: str, sign: float = 1.0) -> "Quantity":
        """sign·(a − b), errors added in quadrature."""
        return Quantity(name, sign * (a.value - b.value), math.hypot(a.error, b.error), unit,
                        lowest_confidence([a.confidence, b.confidence]))

    def evidence(self, threshold: Optional[float] = None, threshold_id: Optional[str] = None,
                 margin: Optional[float] = None) -> Evidence:
        return Evidence(quantity=self.name, value=round(self.value, 3), error=round(self.error, 3),
                        unit=self.unit, confidence=self.confidence, threshold=threshold,
                        threshold_id=threshold_id,
                        margin_sigma=None if margin is None else round(margin, 3))


def at_least(q: Quantity, threshold: float, k: float, threshold_id: Optional[str] = None,
             kb_ids: Sequence[str] = ()) -> Decision:
    margin = (q.value - threshold) / q.error
    ev = (q.evidence(threshold, threshold_id, margin),)
    ids = tuple(kb_ids)
    bound = f"{q.name} = {q.value:.1f} ± {q.error:.1f} {q.unit} vs {threshold:g} (k = {k:g})"
    if q.value - k * q.error >= threshold:
        return Decision("PRESENT", bound, ev, (), ids, abs(margin))
    if q.value + k * q.error < threshold:
        return Decision("ABSENT", bound, ev, (), ids, abs(margin))
    leans_present = q.value >= threshold
    return Decision("INCONCLUSIVE", bound, ev,
                    (reason("within_error", f"within k × error: {bound}", leans_present),), ids, abs(margin))


def at_most(q: Quantity, threshold: float, k: float, threshold_id: Optional[str] = None,
            kb_ids: Sequence[str] = ()) -> Decision:
    """`q ≤ threshold`, the mirror of at_least."""
    flipped = Quantity(q.name, -q.value, q.error, q.unit, q.confidence)
    d = at_least(flipped, -threshold, k, threshold_id, kb_ids)
    margin = (q.value - threshold) / q.error
    bound = f"{q.name} = {q.value:.1f} ± {q.error:.1f} {q.unit} vs {threshold:g} (k = {k:g})"
    reasons = tuple(reason("within_error", f"within k × error: {bound}", r.repeatable) for r in d.reasons)
    return Decision(d.state, bound, (q.evidence(threshold, threshold_id, margin),), reasons,
                    d.kb_ids, d.margin_sigma)


def beyond_error(q: Quantity, k: float, two_sided: bool = True, kb_ids: Sequence[str] = ()) -> Decision:
    """PRESENT if q clears zero by more than k·e (either way, or only upward if not two_sided)."""
    margin = q.value / q.error
    ev = (q.evidence(0.0, None, margin),)
    detail = f"{q.name} = {q.value:.1f} ± {q.error:.1f} {q.unit} (k = {k:g})"
    clears = abs(q.value) if two_sided else q.value
    state: State = "PRESENT" if clears > k * q.error else "ABSENT"
    return Decision(state, detail, ev, (), tuple(kb_ids), abs(margin))


def all_of(decisions: Sequence[Decision], detail: str) -> Decision:
    """Kleene AND."""
    return _combine(decisions, detail, conjunctive=True)


def any_of(decisions: Sequence[Decision], detail: str) -> Decision:
    """Kleene OR."""
    return _combine(decisions, detail, conjunctive=False)


def _combine(decisions: Sequence[Decision], detail: str, conjunctive: bool) -> Decision:
    states = [d.state for d in decisions]
    decisive, other = ("ABSENT", "PRESENT") if conjunctive else ("PRESENT", "ABSENT")
    if decisive in states:
        state: State = decisive  # type: ignore[assignment]
        used = [d for d in decisions if d.state == decisive]
    elif "INCONCLUSIVE" in states:
        state = "INCONCLUSIVE"
        used = list(decisions)
    else:
        state = other  # type: ignore[assignment]
        used = list(decisions)
    margins = [d.margin_sigma for d in used if d.margin_sigma is not None]
    return Decision(
        state, detail,
        tuple(e for d in decisions for e in d.evidence),
        tuple(r for d in decisions if d.state == "INCONCLUSIVE" for r in d.reasons)
        if state == "INCONCLUSIVE" else (),
        tuple(dict.fromkeys(i for d in decisions for i in d.kb_ids)),
        min(margins) if margins else None,
    )


def unique_reasons(reasons: Iterable[Reason]) -> List[Reason]:
    seen, out = set(), []
    for r in reasons:
        if (r.code, r.detail) not in seen:
            seen.add((r.code, r.detail))
            out.append(r)
    return out
