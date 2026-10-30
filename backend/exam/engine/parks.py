"""
BeanHealth CLR — Parks–Bielschowsky 3-Step Test
================================================

    Step 1  Which eye is hyper in primary? (vertical beyond k × error; if
            there is none the test does not apply)
    Step 2  Is the hyper worse on right or left gaze?
    Step 3  Is it worse on right or left head tilt?

The (hyper, gaze, tilt) triple is looked up in parks_3_step.yaml. A step whose
difference is within k × error cannot pick a side → INCONCLUSIVE.

"Steps conflict" is read as: the hyper eye reverses (the other eye becomes
hyper beyond error) in some measured gaze or tilt. A single weak muscle
leaves one eye hyper or level, never the other one hyper, so a reversal does
not fit the single-muscle table → INCONCLUSIVE with the KB's reason. This
reading needs clinician review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from exam.engine.decide import Decision, Quantity, at_least, beyond_error, family_k, reason
from exam.engine.features import PD, Context, Unavailable

PARKS_INCONCLUSIVE_ID = "parks_inconclusive"


@dataclass(frozen=True)
class ParksResult:
    applicable: bool
    decision: Decision
    muscle: Optional[str] = None
    hyper_eye: Optional[str] = None
    worse_on_gaze: Optional[str] = None
    worse_on_tilt: Optional[str] = None


def _pick_side(d: Decision) -> Optional[str]:
    """at_least(right − left, 0): PRESENT → right, ABSENT → left."""
    return {"PRESENT": "right", "ABSENT": "left"}.get(d.state)


def run_parks(ctx: Context) -> ParksResult:
    kb = ctx.kb
    k = ctx.k
    vp = ctx.v("primary")
    step1 = beyond_error(Quantity.of("V[primary]", vp, PD), k)
    if step1.state != "PRESENT":
        return ParksResult(False, step1)

    hyper = "right" if vp.value > 0 else "left"
    s = 1.0 if hyper == "right" else -1.0
    ids: List[str] = [f"parks_step1_{hyper}_hyper"]
    kb_reason = kb.parks.inconclusive.value.reason
    evidence = list(step1.evidence)
    problems = []

    # Conflict: the other eye hyper beyond error anywhere (k corrected for
    # scanning every cell and both tilts).
    scanned = []
    for gaze in ctx.m.grid.cells:
        try:
            scanned.append(Quantity.of(f"{hyper} hyper in {gaze} gaze", ctx.v(gaze), PD, s))
        except Unavailable:
            pass
    for side in ("right", "left"):
        try:
            scanned.append(Quantity.of(f"{hyper} hyper on tilt {side}", ctx.tilt(side), PD, s))
        except Unavailable:
            pass
    k_scan = family_k(k, len(scanned))
    reversed_in = [q for q in scanned if q.value < -k_scan * q.error]
    if reversed_in:
        detail = (f"the hyper eye reverses ({', '.join(q.name for q in reversed_in)}). {kb_reason}")
        return ParksResult(True, Decision(
            "INCONCLUSIVE", detail, tuple([*evidence, *(q.evidence(0.0) for q in reversed_in)]),
            (reason("parks_steps_conflict", detail),), tuple([*ids, PARKS_INCONCLUSIVE_ID])), hyper_eye=hyper)

    margins = [step1.margin_sigma]
    sides = {}
    for step, code, (a, b) in (
        ("gaze", "parks_step2_within_error", ("right", "left")),
        ("tilt", "parks_step3_within_error", ("right", "left")),
    ):
        try:
            if step == "gaze":
                q = Quantity.diff(f"{hyper} hyper: V[right] − V[left]", ctx.v(a), ctx.v(b), PD, s)
            else:
                q = Quantity.diff(f"{hyper} hyper: V[tilt right] − V[tilt left]", ctx.tilt(a), ctx.tilt(b), PD, s)
        except Unavailable as u:
            problems.extend(u.decision.reasons)
            continue
        d = at_least(q, 0.0, k)
        evidence.extend(d.evidence)
        margins.append(d.margin_sigma)
        side = _pick_side(d)
        if side is None:
            problems.append(reason(code, f"step {2 if step == 'gaze' else 3}: {d.detail}. {kb_reason}"))
        sides[step] = side

    if problems:
        return ParksResult(True, Decision("INCONCLUSIVE", "3-step test inconclusive", tuple(evidence),
                                          tuple(problems), tuple([*ids, PARKS_INCONCLUSIVE_ID])),
                           hyper_eye=hyper, worse_on_gaze=sides.get("gaze"), worse_on_tilt=sides.get("tilt"))

    row = next(r for r in kb.parks.rows
               if (r.value.hyper_eye, r.value.worse_on_gaze, r.value.worse_on_tilt)
               == (hyper, sides["gaze"], sides["tilt"]))
    ids.append(row.id)
    detail = (f"{hyper} hyper, worse on {sides['gaze']} gaze and {sides['tilt']} head tilt "
              f"→ {row.value.muscle}")
    return ParksResult(True, Decision("PRESENT", detail, tuple(evidence), (), tuple(ids),
                                      min(m for m in margins if m is not None)),
                       muscle=row.value.muscle, hyper_eye=hyper,
                       worse_on_gaze=sides["gaze"], worse_on_tilt=sides["tilt"])
