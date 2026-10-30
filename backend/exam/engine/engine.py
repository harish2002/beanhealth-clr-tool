"""
BeanHealth CLR — Rules Engine
==============================

    evaluate(measurements, kb) → EngineResult

A pure function: the same measurements and KB always give the same result.
Every clinical value comes from the KB; this file holds only the order of
evaluation and how findings are assembled.

Order (brief §6):
  1. data sufficiency   primary gaze must be measured, or nothing else runs
  2. direction          eso / exo and right / left hyper in primary
  3. comitance          horizontal deviation stable across gazes?
  4. patterns           conditions.yaml patterns — eye-agnostic ones (A / V,
                        comitant) first, then palsy patterns, each per eye
  5. Parks 3-step       vertical deviations only
  6. red flags          conditions.yaml red flags, per eye

Referral (brief §6): only REPORTED abnormal findings at MODERATE or HIGH
confidence refer, URGENT red flags included. LOW confidence never refers — it
asks for a repeat test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from exam.engine.decide import Decision, Quantity, all_of, any_of, beyond_error, reason, unique_reasons
from exam.engine.features import FEATURES, PD, Context, Unavailable, comitance, evaluate_feature
from exam.engine.findings import EngineResult, FeatureOutcome, Finding, KBRef, NotEvaluated, Quality, Referral
from exam.engine.parks import ParksResult, run_parks
from exam.kb import KnowledgeBase, load_kb
from exam.kb.schema import Condition, FeatureRef, Scalar
from exam.models import ExamMeasurements, lowest_confidence, pd_to_degrees

ENGINE_VERSION = "0.1.0"

POSITIONING = ("Measures and characterizes ocular deviation; suggests likely patterns. "
               "Decision support — not a diagnosis.")

LIMITATIONS = [
    "Near fixation only (33–40 cm); no 6 m distance measurement.",
    "No phoria or intermittent deviation detection in Tier 1.",
    "Torsion is not measured.",
    "Clinical thresholds are defined at distance and are not validated at near.",
    "Paralytic and restrictive causes cannot be separated in Tier 1.",
    "Fixation compliance in children is not guaranteed.",
]


def evaluate(measurements: ExamMeasurements, kb: Optional[KnowledgeBase] = None) -> EngineResult:
    kb = kb or load_kb()
    _check_kb_supported(kb)
    k = kb.threshold("decision_margin_k").value
    assert k is not None  # _check_kb_supported
    ctx = Context(measurements, kb, float(k))
    b = _Builder(kb)

    grid = measurements.grid
    tilt = measurements.tilt
    quality = Quality(
        protocol=grid.protocol,
        insufficient_positions=[g for g, c in grid.cells.items() if c.status != "OK"],
        tilt_captured=bool(tilt and tilt.right and tilt.left),
    )

    # 1. Data sufficiency
    if grid.cells["primary"].status != "OK":  # type: ignore[index]
        detail = "primary gaze has too few valid frames; nothing else can be evaluated"
        b.add("data_sufficiency", "Primary gaze not measured", "measurement",
              Decision("INCONCLUSIVE", detail, (), (reason("primary_insufficient", detail),)),
              status="INCONCLUSIVE", abnormal=False, wording=detail)
        return b.result(quality)

    # 2. Direction
    _direction(ctx, b)

    # 3. Comitance
    try:
        d = comitance(ctx)
    except Unavailable as u:
        d = u.decision
    ids = ("incomitance_horizontal_pd", "decision_margin_k")
    if d.state == "PRESENT":
        b.add("comitance", "Horizontal deviation comitant", "measurement", d, abnormal=False, kb_ids=ids,
              wording=f"Horizontal deviation stable across gazes ({d.detail}).")
    elif d.state == "ABSENT":
        b.add("comitance", "Incomitant horizontal deviation", "measurement", d, status="REPORTED",
              abnormal=True, kb_ids=ids, wording=f"Horizontal deviation changes with gaze ({d.detail}).")
    else:
        b.add("comitance", "Horizontal comitance undecided", "measurement", d, abnormal=False, kb_ids=ids,
              wording="Cannot tell whether the horizontal deviation changes with gaze.")

    # 5 is needed by 4 (CN IV cites the 3-step result), so run it first.
    parks = run_parks(ctx)
    ctx = Context(measurements, kb, ctx.k, parks=parks)

    # 4. Patterns
    patterns = [c for c in kb.conditions_file.conditions if c.value.kind == "pattern"]
    for entry in sorted(patterns, key=lambda c: _eye_specific(c.value)):
        _condition(ctx, b, entry.id, entry.value)

    # 5. Parks 3-step
    if parks.applicable:
        _parks_finding(parks, b)

    # 6. Red flags
    for entry in kb.conditions_file.conditions:
        if entry.value.kind == "red_flag":
            _condition(ctx, b, entry.id, entry.value)

    return b.result(quality)


# ─────────────────────────────────────────────────────────────
# Rules
# ─────────────────────────────────────────────────────────────

def _direction(ctx: Context, b: "_Builder") -> None:
    primary = ctx.m.grid.cells["primary"]  # type: ignore[index]
    for axis, m in (("horizontal", primary.horizontal_pd), ("vertical", primary.vertical_pd)):
        assert m is not None
        d = beyond_error(Quantity.of(f"{'H' if axis == 'horizontal' else 'V'}[primary]", m, PD), ctx.k)
        size = f"{abs(m.value):.1f} ± {m.error:.1f} PD ({abs(pd_to_degrees(m.value)):.1f}°)"
        if d.state == "PRESENT":
            if axis == "horizontal":
                kind = "Eso" if m.value > 0 else "Exo"
            else:
                kind = "Right hyper" if m.value > 0 else "Left hyper"
            b.add(f"direction_{axis}", f"{kind} deviation in primary", "measurement", d,
                  status="REPORTED", abnormal=True, kb_ids=("decision_margin_k",),
                  wording=f"{kind} deviation of {size} in primary position.")
        else:
            b.add(f"direction_{axis}", f"No {axis} deviation beyond measurement error", "measurement", d,
                  status="REPORTED", abnormal=False, kb_ids=("decision_margin_k",),
                  wording=f"{axis.capitalize()} deviation in primary {m.value:.1f} ± {m.error:.1f} PD "
                          f"— not distinguishable from zero at k = {ctx.k:g}.")


def _eye_specific(cond: Condition) -> bool:
    refs = [*cond.required, *cond.supporting, *(r for g in cond.any_of for r in g)]
    return any(v in ("affected", "fellow") for r in refs for v in r.params.values())


def _condition(ctx: Context, b: "_Builder", cond_id: str, cond: Condition) -> None:
    eyes: Sequence[Optional[str]] = ("R", "L") if _eye_specific(cond) else (None,)
    for eye in eyes:
        c = Context(ctx.m, ctx.kb, ctx.k, affected=eye, parks=ctx.parks)
        outcomes: List[FeatureOutcome] = []

        def run(ref: FeatureRef, role: str) -> Decision:
            d = evaluate_feature(c, ref.feature, ref.params)
            outcomes.append(FeatureOutcome(feature=ref.feature, params=dict(ref.params), role=role,  # type: ignore[arg-type]
                                           state=d.state, detail=d.detail))
            return d

        required = [run(r, "required") for r in cond.required]
        groups = [[run(r, "any_of") for r in g] for g in cond.any_of]
        supporting = [run(r, "supporting") for r in cond.supporting]
        overall = all_of([*required, *(any_of(g, "any of") for g in groups)], cond.label)
        if overall.state == "ABSENT":
            continue
        considered = [*required, *(d for g in groups for d in g)]
        if (overall.state == "INCONCLUSIVE" and not any(d.state == "PRESENT" for d in considered)
                and not any(r.repeatable for r in overall.reasons)):
            b.not_evaluated.append(NotEvaluated(rule_id=cond_id, affected_eye=eye,  # type: ignore[arg-type]
                                                reasons=unique_reasons(overall.reasons)))
            continue

        keep = ("PRESENT",) if overall.state == "PRESENT" else ("PRESENT", "INCONCLUSIVE")
        decisive = [d for d in required if d.state in keep] + [d for g in groups for d in g if d.state in keep]
        margins = [d.margin_sigma for d in decisive if d.state == "PRESENT" and d.margin_sigma is not None]
        evidence = [e for d in decisive for e in d.evidence] + \
                   [e for d in supporting if d.state == "PRESENT" for e in d.evidence]
        decided = Decision(
            overall.state, overall.detail, tuple(evidence), overall.reasons,
            tuple(dict.fromkeys([cond_id, *overall.kb_ids, *(i for d in supporting for i in d.kb_ids)])),
            min(margins) if margins else None,
        )
        confidence = lowest_confidence([e.confidence for d in decisive for e in d.evidence])
        b.add(cond_id, cond.label, cond.kind, decided, affected_eye=eye, abnormal=True,
              urgency=cond.urgency, wording=cond.report_wording, features=outcomes,
              confidence=confidence)


def _parks_finding(parks: ParksResult, b: "_Builder") -> None:
    d = parks.decision
    if d.state == "PRESENT":
        b.add("parks_3_step", f"3-step test isolates {parks.muscle}", "pattern", d, abnormal=True,
              muscle=parks.muscle, kb_ids=("decision_margin_k",),
              wording=f"Parks 3-step: {d.detail} — consistent with an isolated {parks.muscle} weakness.")
    else:
        b.add("parks_3_step", "3-step test inconclusive", "pattern", d, abnormal=True,
              kb_ids=("decision_margin_k",),
              wording="Parks 3-step test could not isolate a single muscle.")


# ─────────────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────────────

class _Builder:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.findings: List[Finding] = []
        self.not_evaluated: List[NotEvaluated] = []
        self._index: Dict[str, Any] = {e.id: e for _, e in kb.entries()}

    def _kb_ref(self, entry_id: str) -> KBRef:
        e = self._index[entry_id]
        return KBRef(id=e.id, value=e.value if isinstance(e, Scalar) else None, review_status=e.review_status)

    def add(self, rule_id: str, label: str, kind: str, d: Decision, *, abnormal: bool, wording: str,
            status: Optional[str] = None, affected_eye: Optional[str] = None, urgency: Optional[str] = None,
            features: Sequence[FeatureOutcome] = (), muscle: Optional[str] = None,
            kb_ids: Tuple[str, ...] = (), confidence: Optional[str] = None) -> None:
        status = status or ("REPORTED" if d.state == "PRESENT" else "INCONCLUSIVE")
        ids = [i for i in dict.fromkeys([*kb_ids, *d.kb_ids]) if i in self._index]
        self.findings.append(Finding(
            rule_id=rule_id, label=label, kind=kind, affected_eye=affected_eye,  # type: ignore[arg-type]
            status=status, abnormal=abnormal and status == "REPORTED", urgency=urgency,  # type: ignore[arg-type]
            confidence=confidence or d.confidence,  # type: ignore[arg-type]
            margin_over_threshold=None if d.margin_sigma is None else round(d.margin_sigma, 3),
            wording=wording,
            reasons=unique_reasons(d.reasons) if status == "INCONCLUSIVE" else [],
            evidence=list(d.evidence), features=list(features), muscle=muscle,
            kb_entries=[self._kb_ref(i) for i in ids],
            kb_version=self.kb.version, kb_sha256=self.kb.sha256, engine_version=ENGINE_VERSION,
        ))

    def result(self, quality: Quality) -> EngineResult:
        return EngineResult(
            findings=self.findings, not_evaluated=self.not_evaluated,
            referral=decide_referral(self.findings), quality=quality,
            positioning=POSITIONING, limitations=LIMITATIONS,
            kb_version=self.kb.version, kb_sha256=self.kb.sha256, engine_version=ENGINE_VERSION,
        )


def _tag(f: Finding) -> str:
    return f"{f.rule_id}:{f.affected_eye}" if f.affected_eye else f.rule_id


def decide_referral(findings: Sequence[Finding]) -> Referral:
    reported = [f for f in findings if f.status == "REPORTED" and f.abnormal]
    confident = [f for f in reported if f.confidence != "LOW"]
    low = [f for f in reported if f.confidence == "LOW"]
    repeatable = [f for f in findings if f.status == "INCONCLUSIVE" and any(r.repeatable for r in f.reasons)]

    urgent = any(f.urgency == "URGENT" for f in confident)
    urgent_unconfirmed = not urgent and any(f.urgency == "URGENT" for f in [*low, *repeatable])

    if confident:
        advice = ("URGENT referral to ophthalmology — see red flags." if urgent
                  else "Refer to ophthalmology with this report.")
        return Referral(decision="REFER", urgent=urgent, urgent_unconfirmed=urgent_unconfirmed,
                        because=[_tag(f) for f in confident], advice=advice)
    if low or repeatable:
        advice = ("Repeat the examination. The findings are LOW confidence or borderline, "
                  "which cannot support a referral.")
        if urgent_unconfirmed:
            advice += " A possible URGENT red flag was seen — repeat without delay."
        return Referral(decision="REPEAT_TEST", urgent_unconfirmed=urgent_unconfirmed,
                        because=[_tag(f) for f in [*low, *repeatable]], advice=advice)
    return Referral(decision="NO_REFERRAL", advice="No deviation or pattern detected beyond measurement error.")


def _check_kb_supported(kb: KnowledgeBase) -> None:
    problems = []
    kb_features = {f.id for f in kb.conditions_file.features}
    if kb_features - set(FEATURES):
        problems.append(f"no evaluator for KB features {sorted(kb_features - set(FEATURES))}")
    if set(FEATURES) - kb_features:
        problems.append(f"evaluators for features not in the KB: {sorted(set(FEATURES) - kb_features)}")
    if kb.threshold("decision_margin_k").value is None:
        problems.append("KB threshold decision_margin_k has no value")
    if problems:
        raise ValueError(f"KB {kb.version} not supported by engine {ENGINE_VERSION}: " + "; ".join(problems))
