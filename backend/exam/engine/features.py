"""
BeanHealth CLR — Feature Evaluators
====================================

One evaluator per feature in conditions.yaml. Each turns measurements into a
three-valued Decision (see decide.py) and reads every threshold from the KB;
a threshold with no value makes the feature INCONCLUSIVE ("threshold not
set") rather than guessing.

`affected` / `fellow` parameters resolve against the eye a condition is being
evaluated for. Sides follow the patient: the right eye's side is "right".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from exam.engine.decide import (
    Decision,
    Quantity,
    all_of,
    any_of,
    at_least,
    at_most,
    beyond_error,
    family_k,
    inconclusive,
)
from exam.kb import KnowledgeBase
from exam.kb.derive import fellow, side_of
from exam.models import ExamMeasurements, GazeCell, Measured

PD = "PD"
OBLIQUE_GAZES = ("up_right", "up_left", "down_right", "down_left")


class Unavailable(Exception):
    """Raised inside an evaluator when it cannot run; carries the INCONCLUSIVE decision."""

    def __init__(self, decision: Decision):
        self.decision = decision


@dataclass(frozen=True)
class Context:
    m: ExamMeasurements
    kb: KnowledgeBase
    k: float
    affected: Optional[str] = None      # "R" / "L" for eye-specific conditions
    parks: Optional["ParksResult"] = None  # noqa: F821 — set by the engine

    # ── Eyes ──
    def eye(self, who: str) -> str:
        if self.affected is None:
            raise ValueError(f"'{who}' needs an affected eye")
        return self.affected if who == "affected" else fellow(self.affected)

    def hyper_sign(self) -> float:
        """+1 if the affected eye is the right one: right hyper is positive vertical."""
        return 1.0 if self.eye("affected") == "R" else -1.0

    # ── Data access; raise Unavailable when missing ──
    def cell(self, gaze: str) -> GazeCell:
        cells = self.m.grid.cells
        if gaze not in cells:
            raise Unavailable(inconclusive(
                "not_in_protocol", f"{gaze} gaze is not part of the {self.m.grid.protocol} protocol"))
        c = cells[gaze]  # type: ignore[index]
        if c.status != "OK":
            raise Unavailable(inconclusive(
                "insufficient_data", f"{gaze} gaze has too few valid frames ({c.valid_frames})"))
        return c

    def h(self, gaze: str) -> Measured:
        return self.cell(gaze).horizontal_pd  # type: ignore[return-value]

    def v(self, gaze: str) -> Measured:
        return self.cell(gaze).vertical_pd  # type: ignore[return-value]

    def tilt(self, side: str) -> Measured:
        t = self.m.tilt
        value = None if t is None else getattr(t, side)
        if value is None:
            raise Unavailable(inconclusive("tilt_not_captured", f"no head tilt to the {side} captured"))
        return value

    def eye_measure(self, eye: str, name: str, key: Optional[str] = None) -> Measured:
        em = self.m.eyes.get(eye)  # type: ignore[call-overload]
        value = None if em is None else getattr(em, name)
        if key is not None and value is not None:
            value = value.get(key)
        if value is None:
            what = f"{name}[{key}]" if key else name
            raise Unavailable(inconclusive("not_measured", f"{what} not measured for the {side_of(eye)} eye"))
        return value

    def threshold(self, threshold_id: str) -> float:
        value = self.kb.threshold(threshold_id).value
        if value is None:
            raise Unavailable(inconclusive(
                "threshold_not_set", f"KB threshold {threshold_id} has no value yet", [threshold_id]))
        return float(value)


Evaluator = Callable[[Context, Dict[str, str]], Decision]
FEATURES: Dict[str, Evaluator] = {}


def feature(feature_id: str) -> Callable[[Evaluator], Evaluator]:
    def register(fn: Evaluator) -> Evaluator:
        FEATURES[feature_id] = fn
        return fn
    return register


def evaluate_feature(ctx: Context, feature_id: str, params: Dict[str, str]) -> Decision:
    try:
        d = FEATURES[feature_id](ctx, params)
    except Unavailable as u:
        d = u.decision
    ids = tuple(dict.fromkeys([feature_id, *ctx.kb.feature(feature_id).uses_thresholds, *d.kb_ids]))
    return Decision(d.state, d.detail, d.evidence, d.reasons, ids, d.margin_sigma)


# ─────────────────────────────────────────────────────────────
# Shared measurements
# ─────────────────────────────────────────────────────────────

def comitance(ctx: Context, axis: str = "horizontal") -> Decision:
    """
    Is the deviation on this axis the same in every gaze (within its
    incomitance threshold)? PRESENT = comitant, ABSENT = incomitant.

    Incomitant if any pair of gazes differs by ≥ T beyond k × error; comitant
    only if every pair is below T beyond error, with every position measured.
    """
    threshold_id = f"incomitance_{axis}_pd"
    T = ctx.threshold(threshold_id)
    sym = "H" if axis == "horizontal" else "V"
    read = ctx.h if axis == "horizontal" else ctx.v
    measured: Dict[str, Measured] = {}
    missing: List[Decision] = []
    for gaze in ctx.m.grid.cells:
        try:
            measured[gaze] = read(gaze)
        except Unavailable as u:
            missing.append(u.decision)

    pairs = []
    names = list(measured)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            hi, lo = (a, b) if measured[a].value >= measured[b].value else (b, a)
            q = Quantity.diff(f"{sym}[{hi}] − {sym}[{lo}]", measured[hi], measured[lo], PD)
            pairs.append(at_least(q, T, ctx.k, threshold_id))

    incomitant = [d for d in pairs if d.state == "PRESENT"]
    if incomitant:
        widest = max(incomitant, key=lambda d: d.evidence[0].value)
        return Decision("ABSENT", f"{axis} incomitant: {widest.detail}", widest.evidence, (),
                        widest.kb_ids, widest.margin_sigma)
    unsure = [d for d in pairs if d.state == "INCONCLUSIVE"]
    if unsure or missing:
        worst = [max(unsure, key=lambda d: d.evidence[0].value)] if unsure else []
        return Decision("INCONCLUSIVE", f"{axis} comitance undecided",
                        tuple(e for d in worst for e in d.evidence),
                        tuple(r for d in [*worst, *missing] for r in d.reasons),
                        (threshold_id,), worst[0].margin_sigma if worst else None)
    if not pairs:
        raise Unavailable(inconclusive("insufficient_data", "fewer than two gaze positions measured"))
    widest = max(pairs, key=lambda d: d.evidence[0].value)
    return Decision("PRESENT", f"{axis} comitant: {widest.detail}", widest.evidence, (),
                    widest.kb_ids, widest.margin_sigma)


# ─────────────────────────────────────────────────────────────
# Features — one per conditions.yaml feature
# ─────────────────────────────────────────────────────────────

@feature("deviation_in_primary_present")
def _deviation_in_primary_present(ctx: Context, p: Dict[str, str]) -> Decision:
    h = beyond_error(Quantity.of("H[primary]", ctx.h("primary"), PD), ctx.k)
    v = beyond_error(Quantity.of("V[primary]", ctx.v("primary"), PD), ctx.k)
    return any_of([h, v], "deviation in primary position")


@feature("deviation_stable_across_gazes")
def _deviation_stable_across_gazes(ctx: Context, p: Dict[str, str]) -> Decision:
    return comitance(ctx)


@feature("vertical_deviation_stable_across_gazes")
def _vertical_stable_across_gazes(ctx: Context, p: Dict[str, str]) -> Decision:
    return comitance(ctx, "vertical")


@feature("duction_limited")
def _duction_limited(ctx: Context, p: Dict[str, str]) -> Decision:
    T = ctx.threshold("duction_limit_fraction")
    eye = ctx.eye(p["eye"])
    m = ctx.eye_measure(eye, "ductions", p["direction"])
    q = Quantity.of(f"{eye} {p['direction']} fraction", m, "fraction")
    return at_most(q, T, ctx.k, "duction_limit_fraction")


@feature("horizontal_deviation_increases_toward_gaze")
def _horizontal_increases_toward_gaze(ctx: Context, p: Dict[str, str]) -> Decision:
    T = ctx.threshold("incomitance_horizontal_pd")
    gaze = side_of(ctx.eye(p["side"]))
    sign = 1.0 if p["deviation"] == "eso" else -1.0
    kind = p["deviation"]
    there = beyond_error(Quantity.of(f"{kind} in {gaze} gaze", ctx.h(gaze), PD, sign), ctx.k, two_sided=False)
    grows = at_least(Quantity.diff(f"{kind}: H[{gaze}] − H[primary]", ctx.h(gaze), ctx.h("primary"), PD, sign),
                     T, ctx.k, "incomitance_horizontal_pd")
    return all_of([there, grows], f"{kind} increases toward {gaze} gaze")


@feature("vertical_deviation_absent")
def _vertical_deviation_absent(ctx: Context, p: Dict[str, str]) -> Decision:
    found, missing, checked = [], [], []
    k = family_k(ctx.k, len(ctx.m.grid.cells))
    for gaze in ctx.m.grid.cells:
        try:
            d = beyond_error(Quantity.of(f"V[{gaze}]", ctx.v(gaze), PD), k)
        except Unavailable as u:
            missing.append(u.decision)
            continue
        (found if d.state == "PRESENT" else checked).append(d)
    if found:
        return Decision("ABSENT", "vertical deviation present", tuple(e for d in found for e in d.evidence),
                        (), (), min(d.margin_sigma for d in found))  # type: ignore[type-var]
    if missing:
        return Decision("INCONCLUSIVE", "not every gaze measured", (),
                        tuple(r for d in missing for r in d.reasons))
    return Decision("PRESENT", "no vertical deviation beyond error in any gaze",
                    tuple(e for d in checked for e in d.evidence))


@feature("hypertropia_in_primary")
def _hypertropia_in_primary(ctx: Context, p: Dict[str, str]) -> Decision:
    eye = ctx.eye(p["eye"])
    sign = 1.0 if eye == "R" else -1.0
    q = Quantity.of(f"{side_of(eye)} hyper in primary", ctx.v("primary"), PD, sign)
    return beyond_error(q, ctx.k, two_sided=False)


@feature("vertical_deviation_worse_on_gaze")
def _vertical_worse_on_gaze(ctx: Context, p: Dict[str, str]) -> Decision:
    gaze = {
        "toward_affected_side": side_of(ctx.eye("affected")),
        "toward_fellow_side": side_of(ctx.eye("fellow")),
    }.get(p["gaze"], p["gaze"])
    s = ctx.hyper_sign()
    q = Quantity.diff(f"{side_of(ctx.eye('affected'))} hyper: V[{gaze}] − V[primary]",
                      ctx.v(gaze), ctx.v("primary"), PD, s)
    return at_least(q, 0.0, ctx.k)


@feature("vertical_deviation_worse_on_head_tilt")
def _vertical_worse_on_tilt(ctx: Context, p: Dict[str, str]) -> Decision:
    toward = side_of(ctx.eye(p["side"]))
    away = "left" if toward == "right" else "right"
    s = ctx.hyper_sign()
    q = Quantity.diff(f"{side_of(ctx.eye('affected'))} hyper: V[tilt {toward}] − V[tilt {away}]",
                      ctx.tilt(toward), ctx.tilt(away), PD, s)
    return at_least(q, 0.0, ctx.k)


@feature("parks_isolates_muscle")
def _parks_isolates_muscle(ctx: Context, p: Dict[str, str]) -> Decision:
    parks = ctx.parks
    want = f"{ctx.eye('affected')}-{p['muscle']}"
    if parks is None or not parks.applicable:
        return Decision("ABSENT", "3-step test not applicable: no hypertropia in primary")
    d = parks.decision
    if d.state != "PRESENT":
        return Decision("INCONCLUSIVE", f"3-step test inconclusive", d.evidence, d.reasons, d.kb_ids)
    state = "PRESENT" if parks.muscle == want else "ABSENT"
    return Decision(state, f"3-step test isolates {parks.muscle}", d.evidence, (), d.kb_ids, d.margin_sigma)


@feature("eye_position_in_primary")
def _eye_position_in_primary(ctx: Context, p: Dict[str, str]) -> Decision:
    eye = ctx.eye("affected")
    h_sign = 1.0 if p["horizontal"] == "eso" else -1.0
    # The affected eye hyper ⇔ positive vertical for the right eye, negative for the left.
    v_sign = (1.0 if p["vertical"] == "hyper" else -1.0) * (1.0 if eye == "R" else -1.0)
    h = beyond_error(Quantity.of(f"{p['horizontal']} in primary", ctx.h("primary"), PD, h_sign),
                     ctx.k, two_sided=False)
    v = beyond_error(Quantity.of(f"{side_of(eye)} {p['vertical']} in primary", ctx.v("primary"), PD, v_sign),
                     ctx.k, two_sided=False)
    return all_of([h, v], f"{side_of(eye)} eye {p['horizontal']} and {p['vertical']} in primary")


@feature("ptosis")
def _ptosis(ctx: Context, p: Dict[str, str]) -> Decision:
    T = ctx.threshold("ptosis_fissure_difference_mm")
    eye = ctx.eye(p["eye"])
    q = Quantity.diff(f"fissure {fellow(eye)} − {eye}", ctx.eye_measure(fellow(eye), "fissure_height_mm"),
                      ctx.eye_measure(eye, "fissure_height_mm"), "mm")
    return at_least(q, T, ctx.k, "ptosis_fissure_difference_mm")


@feature("anisocoria_beyond_tolerance")
def _anisocoria(ctx: Context, p: Dict[str, str]) -> Decision:
    T = ctx.threshold("anisocoria_tolerance_mm")
    eye = ctx.eye(p["larger_pupil"])
    q = Quantity.diff(f"pupil {eye} − {fellow(eye)}", ctx.eye_measure(eye, "pupil_diameter_mm"),
                      ctx.eye_measure(fellow(eye), "pupil_diameter_mm"), "mm")
    return at_least(q, T, ctx.k, "anisocoria_tolerance_mm")


@feature("fissure_narrowing_on_adduction")
def _fissure_narrowing(ctx: Context, p: Dict[str, str]) -> Decision:
    T = ctx.threshold("duane_fissure_narrowing_ratio")
    eye = ctx.eye(p["eye"])
    q = Quantity.of(f"{eye} fissure ratio in adduction", ctx.eye_measure(eye, "fissure_ratio_adduction"), "ratio")
    return at_most(q, T, ctx.k, "duane_fissure_narrowing_ratio")


@feature("small_primary_deviation")
def _small_primary_deviation(ctx: Context, p: Dict[str, str]) -> Decision:
    T = ctx.threshold("duane_small_primary_deviation_pd")
    parts = []
    for name, m in (("|H[primary]|", ctx.h("primary")), ("|V[primary]|", ctx.v("primary"))):
        q = Quantity.of(name, m, PD, 1.0 if m.value >= 0 else -1.0)
        parts.append(at_most(q, T, ctx.k, "duane_small_primary_deviation_pd"))
    return all_of(parts, "small primary deviation")


@feature("upper_lid_retraction")
def _upper_lid_retraction(ctx: Context, p: Dict[str, str]) -> Decision:
    T = ctx.threshold("scleral_show_mm")
    eye = ctx.eye(p["eye"])
    q = Quantity.of(f"{eye} scleral show", ctx.eye_measure(eye, "scleral_show_mm"), "mm")
    return at_least(q, T, ctx.k, "scleral_show_mm")


@feature("a_pattern")
def _a_pattern(ctx: Context, p: Dict[str, str]) -> Decision:
    T = ctx.threshold("a_pattern_pd")
    q = Quantity.diff("H[up] − H[down]", ctx.h("up"), ctx.h("down"), PD)
    return at_least(q, T, ctx.k, "a_pattern_pd")


@feature("v_pattern")
def _v_pattern(ctx: Context, p: Dict[str, str]) -> Decision:
    T = ctx.threshold("v_pattern_pd")
    q = Quantity.diff("H[down] − H[up]", ctx.h("down"), ctx.h("up"), PD)
    return at_least(q, T, ctx.k, "v_pattern_pd")


@feature("oblique_overaction_in_oblique_gaze")
def _oblique_overaction(ctx: Context, p: Dict[str, str]) -> Decision:
    # Crude indicator, supporting only: a vertical deviation in an oblique
    # gaze that primary does not show. Refine with clinician input.
    decisions = []
    k = family_k(ctx.k, len(OBLIQUE_GAZES))
    for gaze in OBLIQUE_GAZES:
        try:
            q = Quantity.diff(f"V[{gaze}] − V[primary]", ctx.v(gaze), ctx.v("primary"), PD)
            decisions.append(beyond_error(q, k))
        except Unavailable as u:
            decisions.append(u.decision)
    return any_of(decisions, "vertical deviation in oblique gaze")
