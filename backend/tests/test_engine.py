"""
BeanHealth CLR Tool — Rules Engine Tests (Phase 3, milestone 3)
===============================================================

  • Error-aware decision helpers (three-valued, Kleene logic, family-wise k).
  • One test per rule, on synthetic cases.
  • All 8 Parks 3-step rows, both from the simulator and from hand-set values.
  • Every INCONCLUSIVE path, by reason code.
  • Referral: LOW confidence never refers (regression), URGENT handling.
  • Every finding carries its rule ID, confidence, evidence, KB entries and
    the KB and engine versions; the engine reads every threshold from the KB.

Patterns whose thresholds the shipped KB has not set (duction limits, ptosis,
anisocoria, Duane, lid retraction, vertical incomitance) are tested twice:
with the shipped KB they must come out INCONCLUSIVE / not evaluated, and with
`test_kb` — a copy carrying the TEST-ONLY values below — they must be found.
Those values are fixtures for exercising logic, not clinical thresholds.

Run:
    cd backend && pytest tests/test_engine.py -v
"""

import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from exam.engine import ENGINE_VERSION, POSITIONING, decide_referral, evaluate
from exam.engine.decide import Quantity, all_of, any_of, at_least, at_most, beyond_error, family_k
from exam.engine.features import FEATURES, Context, evaluate_feature
from exam.engine.findings import Finding
from exam.kb import DEFAULT_KB_DIR, load_kb
from exam.models import ExamMeasurements, GazeCell, Measured, TiltMeasurements
from exam.sim import CASE_NOISE, CASES, Noise, parks_case, simulate

pytestmark = pytest.mark.unit

# TEST-ONLY threshold values — not clinical. See module docstring.
TEST_THRESHOLDS = {
    "duction_limit_fraction": 0.7,
    "anisocoria_tolerance_mm": 1.0,
    "duane_fissure_narrowing_ratio": 0.85,
    "duane_small_primary_deviation_pd": 10,
    "scleral_show_mm": 1.0,
    "ptosis_fissure_difference_mm": 2.0,
    "incomitance_vertical_pd": 5,
}


@pytest.fixture(scope="module")
def kb():
    return load_kb()


@pytest.fixture(scope="module")
def test_kb(tmp_path_factory):
    dst = tmp_path_factory.mktemp("kb") / "kb"
    shutil.copytree(DEFAULT_KB_DIR, dst)
    path = dst / "thresholds.yaml"
    data = yaml.safe_load(path.read_text())
    for e in data["entries"]:
        if e["id"] in TEST_THRESHOLDS:
            e["value"] = TEST_THRESHOLDS[e["id"]]
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return load_kb(dst, verify_hash=False)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def run(case_name: str, kb, **kw):
    return evaluate(simulate(CASES[case_name], CASE_NOISE.get(case_name, Noise()), **kw), kb)


def find(result, rule_id: str, eye: Optional[str] = None) -> List[Finding]:
    return [f for f in result.findings if f.rule_id == rule_id and (eye is None or f.affected_eye == eye)]


def one(result, rule_id: str, eye: Optional[str] = None) -> Finding:
    hits = find(result, rule_id, eye)
    assert len(hits) == 1, f"{rule_id}:{eye} → {[(f.rule_id, f.affected_eye, f.status) for f in result.findings]}"
    return hits[0]


def reported(result) -> Dict[str, Finding]:
    return {(f"{f.rule_id}:{f.affected_eye}" if f.affected_eye else f.rule_id): f
            for f in result.findings if f.status == "REPORTED" and f.abnormal}


def codes(f: Finding) -> set:
    return {r.code for r in f.reasons}


def M(value: float, error: float = 1.0, confidence: str = "HIGH") -> Measured:
    return Measured(value=value, error=error, confidence=confidence)


def set_vertical(m: ExamMeasurements, **values: float) -> ExamMeasurements:
    """Replace the vertical value of the named gaze cells (and tilt_right / tilt_left)."""
    cells = dict(m.grid.cells)
    tilt = m.tilt
    for name, v in values.items():
        if name.startswith("tilt_"):
            tilt = tilt.model_copy(update={name[5:]: M(v)})
        else:
            cells[name] = cells[name].model_copy(update={"vertical_pd": M(v)})
    return m.model_copy(update={"grid": m.grid.model_copy(update={"cells": cells}), "tilt": tilt})


# ─────────────────────────────────────────────────────────────
# Decision helpers
# ─────────────────────────────────────────────────────────────

class TestDecide:
    def q(self, v, e=1.0):
        return Quantity("x", v, e, "PD", "HIGH")

    def test_at_least_three_ways(self):
        assert at_least(self.q(13), 10, 2).state == "PRESENT"      # 13 − 2 ≥ 10
        assert at_least(self.q(7.9), 10, 2).state == "ABSENT"      # 7.9 + 2 < 10
        d = at_least(self.q(10.5), 10, 2)
        assert d.state == "INCONCLUSIVE" and d.reasons[0].code == "within_error"

    def test_within_error_is_repeatable_only_when_leaning_present(self):
        assert at_least(self.q(10.5), 10, 2).reasons[0].repeatable
        assert not at_least(self.q(9.5), 10, 2).reasons[0].repeatable
        assert at_most(self.q(0.69, 0.03), 0.7, 2).reasons[0].repeatable
        assert not at_most(self.q(0.71, 0.03), 0.7, 2).reasons[0].repeatable

    def test_at_most_mirrors(self):
        assert at_most(self.q(0.2, 0.03), 0.7, 2).state == "PRESENT"
        assert at_most(self.q(1.0, 0.03), 0.7, 2).state == "ABSENT"

    def test_beyond_error(self):
        assert beyond_error(self.q(2.1), 2).state == "PRESENT"
        assert beyond_error(self.q(-2.1), 2).state == "PRESENT"
        assert beyond_error(self.q(-2.1), 2, two_sided=False).state == "ABSENT"
        assert beyond_error(self.q(1.9), 2).state == "ABSENT"

    def test_margin_is_in_errors(self):
        d = at_least(self.q(16, 2), 10, 2)
        assert d.margin_sigma == pytest.approx(3.0)
        assert d.evidence[0].margin_sigma == pytest.approx(3.0)

    def test_kleene_logic(self):
        P, A, I = (at_least(self.q(v), 10, 2) for v in (20, 0, 10))
        assert all_of([P, P], "").state == "PRESENT"
        assert all_of([P, I], "").state == "INCONCLUSIVE"
        assert all_of([I, A], "").state == "ABSENT"         # one ABSENT rules it out
        assert any_of([A, I], "").state == "INCONCLUSIVE"
        assert any_of([I, P], "").state == "PRESENT"
        assert any_of([A, A], "").state == "ABSENT"

    def test_family_k(self):
        assert family_k(2, 1) == 2
        assert family_k(2, 9) == pytest.approx(2.80, abs=0.01)
        assert family_k(2, 9) > family_k(2, 4) > 2


# ─────────────────────────────────────────────────────────────
# KB coupling
# ─────────────────────────────────────────────────────────────

class TestKBCoupling:
    def test_every_kb_feature_has_an_evaluator_and_vice_versa(self, kb):
        assert set(FEATURES) == {f.id for f in kb.conditions_file.features}

    def test_engine_refuses_a_kb_with_an_unknown_feature(self, tmp_path):
        dst = tmp_path / "kb"
        shutil.copytree(DEFAULT_KB_DIR, dst)
        path = dst / "conditions.yaml"
        data = yaml.safe_load(path.read_text())
        data["features"].append({"id": "new_sign", "value": {"description": "x", "measured_from": "x"},
                                 "source": "test", "review_status": "draft"})
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        with pytest.raises(ValueError, match="no evaluator for KB features \\['new_sign'\\]"):
            evaluate(simulate(CASES["normal"]), load_kb(dst, verify_hash=False))

    def test_thresholds_come_from_the_kb(self, tmp_path):
        """Raising the A-pattern threshold in the KB turns the A pattern off."""
        dst = tmp_path / "kb"
        shutil.copytree(DEFAULT_KB_DIR, dst)
        path = dst / "thresholds.yaml"
        data = yaml.safe_load(path.read_text())
        for e in data["entries"]:
            if e["id"] == "a_pattern_pd":
                e["value"] = 40
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        assert "a_pattern_strabismus" in reported(run("a_pattern", load_kb()))
        assert "a_pattern_strabismus" not in reported(run("a_pattern", load_kb(dst, verify_hash=False)))

    def test_every_output_is_stamped(self, kb):
        r = run("r_cn4", kb)
        assert (r.kb_version, r.kb_sha256, r.engine_version) == (kb.version, kb.sha256, ENGINE_VERSION)
        assert r.positioning == POSITIONING and r.limitations
        for f in r.findings:
            assert (f.kb_version, f.kb_sha256, f.engine_version) == (kb.version, kb.sha256, ENGINE_VERSION)
            assert f.rule_id and f.confidence in ("HIGH", "MODERATE", "LOW")
            if f.status == "REPORTED":
                assert f.evidence, f.rule_id
                assert all(e.error > 0 for e in f.evidence)
            assert "decision_margin_k" in {e.id for e in f.kb_entries} or f.rule_id == "data_sufficiency"

    def test_kb_entries_carry_review_status(self, kb):
        f = one(run("a_pattern", kb), "a_pattern_strabismus")
        refs = {e.id: e for e in f.kb_entries}
        assert refs["a_pattern_pd"].value == 10
        assert refs["a_pattern_pd"].review_status == "needs_clinician_review"
        assert "a_pattern_strabismus" in refs

    def test_deterministic(self, kb):
        m = simulate(CASES["r_cn4"], Noise(seed=3))
        assert evaluate(m, kb) == evaluate(m, kb)


# ─────────────────────────────────────────────────────────────
# Rules, one by one
# ─────────────────────────────────────────────────────────────

class TestDirection:
    @pytest.mark.parametrize("h, v, h_label, v_label", [
        (20, 0, "Eso deviation in primary", "No vertical deviation beyond measurement error"),
        (-20, 0, "Exo deviation in primary", "No vertical deviation beyond measurement error"),
        (0, 6, "No horizontal deviation beyond measurement error", "Right hyper deviation in primary"),
        (0, -6, "No horizontal deviation beyond measurement error", "Left hyper deviation in primary"),
    ])
    def test_sign_convention(self, kb, h, v, h_label, v_label):
        r = evaluate(simulate(CASES["normal"].__class__("x", comitant_pd=(h, v))), kb)
        assert one(r, "direction_horizontal").label == h_label
        assert one(r, "direction_vertical").label == v_label

    def test_within_error_is_not_a_deviation(self, kb):
        m = set_vertical(simulate(CASES["normal"]), primary=1.9)
        f = one(evaluate(m, kb), "direction_vertical")
        assert f.status == "REPORTED" and not f.abnormal

    def test_degrees_are_reported_too(self, kb):
        f = one(run("comitant_eso_20", kb), "direction_horizontal")
        assert "20.0 ± 1.0 PD (11.3°)" in f.wording


class TestComitance:
    def test_comitant(self, kb):
        f = one(run("comitant_eso_20", kb), "comitance")
        assert f.status == "REPORTED" and not f.abnormal and f.label == "Horizontal deviation comitant"

    def test_incomitant(self, kb):
        f = one(run("r_cn6", kb), "comitance")
        assert f.status == "REPORTED" and f.abnormal
        assert f.evidence[0].quantity.startswith("H[right] − ")      # eso peaks on right gaze

    def test_borderline_is_inconclusive(self, kb):
        f = one(run("noisy_borderline", kb), "comitance")
        assert f.status == "INCONCLUSIVE" and codes(f) == {"within_error"}


class TestPatterns:
    def test_normal_reports_nothing_abnormal(self, kb, test_kb):
        for k in (kb, test_kb):
            r = run("normal", k)
            assert reported(r) == {}
            assert r.referral.decision == "NO_REFERRAL"
            assert not [f for f in r.findings if f.status == "INCONCLUSIVE"]

    def test_comitant_strabismus(self, kb):
        assert set(reported(run("comitant_eso_20", kb))) == {
            "direction_horizontal", "comitant_strabismus"}

    def test_comitant_needs_a_stable_vertical_too(self, kb, test_kb):
        """CN IV: horizontal stable, vertical not — never 'comitant strabismus'."""
        assert one(run("r_cn4", kb), "comitant_strabismus").status == "INCONCLUSIVE"   # threshold not set
        assert not find(run("r_cn4", test_kb), "comitant_strabismus")                  # ruled out

    def test_a_pattern(self, kb):
        r = run("a_pattern", kb)
        assert "a_pattern_strabismus" in reported(r) and "v_pattern_strabismus" not in reported(r)

    def test_v_pattern(self, kb):
        r = run("v_pattern", kb)
        assert "v_pattern_strabismus" in reported(r) and "a_pattern_strabismus" not in reported(r)

    def test_cn4_reported_with_the_shipped_kb(self, kb):
        r = run("r_cn4", kb)
        f = one(r, "cn4_palsy", "R")
        assert f.status == "REPORTED" and f.kind == "pattern"
        assert {o.feature for o in f.features if o.state == "PRESENT"} >= {
            "hypertropia_in_primary", "vertical_deviation_worse_on_gaze",
            "vertical_deviation_worse_on_head_tilt", "parks_isolates_muscle"}
        assert not find(r, "cn4_palsy", "L")

    def test_cn6(self, kb, test_kb):
        f = one(run("r_cn6", kb), "cn6_palsy", "R")
        assert f.status == "INCONCLUSIVE" and codes(f) == {"threshold_not_set"}
        assert "duction_limit_fraction" in f.reasons[0].detail
        r = run("r_cn6", test_kb)
        assert one(r, "cn6_palsy", "R").status == "REPORTED"
        assert not find(r, "cn6_palsy", "L")

    def test_cn6_left_is_the_mirror(self, test_kb):
        r = evaluate(simulate(parks_case("L-LR")), test_kb)
        assert one(r, "cn6_palsy", "L").status == "REPORTED"
        assert not find(r, "cn6_palsy", "R")

    def test_cn3_complete(self, kb, test_kb):
        f = one(run("l_cn3_complete", kb), "cn3_palsy_complete", "L")
        assert f.status == "INCONCLUSIVE" and codes(f) == {"threshold_not_set"}
        r = run("l_cn3_complete", test_kb)
        assert one(r, "cn3_palsy_complete", "L").status == "REPORTED"
        assert not find(r, "cn3_pupil_involving")       # pupils equal


class TestRedFlags:
    def test_unset_thresholds_are_listed_once_not_reported(self, kb):
        r = run("normal", kb)
        assert {(n.rule_id, n.affected_eye) for n in r.not_evaluated} == {
            (c, e) for c in ("duane_like", "ted_like", "cn3_pupil_involving") for e in ("R", "L")}
        assert all(rr.code == "threshold_not_set" for n in r.not_evaluated for rr in n.reasons)

    def test_duane_like(self, test_kb):
        f = one(run("r_duane_like", test_kb), "duane_like", "R")
        assert f.status == "REPORTED" and f.kind == "red_flag" and f.urgency is None

    def test_ted_like(self, test_kb):
        assert one(run("r_ted_like", test_kb), "ted_like", "R").status == "REPORTED"

    def test_pupil_involving_cn3_is_urgent(self, test_kb):
        r = run("l_cn3_pupil_involving", test_kb)
        f = one(r, "cn3_pupil_involving", "L")
        assert f.status == "REPORTED" and f.urgency == "URGENT"
        assert r.referral.decision == "REFER" and r.referral.urgent
        assert "cn3_pupil_involving:L" in r.referral.because

    def test_supporting_feature_adds_evidence(self, test_kb):
        f = one(run("l_cn3_pupil_involving", test_kb), "cn3_palsy_complete", "L")
        sup = [o for o in f.features if o.role == "supporting"]
        assert sup and sup[0].state == "PRESENT"
        assert any(e.quantity == "pupil L − R" for e in f.evidence)


# ─────────────────────────────────────────────────────────────
# Parks 3-step
# ─────────────────────────────────────────────────────────────

ROWS = [  # brief §5.3: hyper, worse gaze, worse tilt → muscle
    ("right", "left", "right", "R-SO"), ("right", "left", "left", "L-SR"),
    ("right", "right", "right", "L-IO"), ("right", "right", "left", "R-IR"),
    ("left", "right", "left", "L-SO"), ("left", "right", "right", "R-SR"),
    ("left", "left", "left", "R-IO"), ("left", "left", "right", "L-IR"),
]


class TestParks:
    @pytest.mark.parametrize("hyper, gaze, tilt, muscle", ROWS)
    def test_simulated_palsy(self, kb, hyper, gaze, tilt, muscle):
        f = one(evaluate(simulate(parks_case(muscle)), kb), "parks_3_step")
        assert (f.status, f.muscle) == ("REPORTED", muscle)
        assert f"parks_{muscle.lower().replace('-', '_')}" in {e.id for e in f.kb_entries}

    @pytest.mark.parametrize("hyper, gaze, tilt, muscle", ROWS)
    def test_hand_set_values(self, kb, hyper, gaze, tilt, muscle):
        """The table lookup alone, independent of the simulator."""
        s = 1 if hyper == "right" else -1
        other = {"right": "left", "left": "right"}
        values = {"primary": 6 * s, gaze: 12 * s, other[gaze]: 3 * s,
                  f"tilt_{tilt}": 14 * s, f"tilt_{other[tilt]}": 4 * s}
        m = set_vertical(simulate(CASES["normal"]), **values)
        f = one(evaluate(m, kb), "parks_3_step")
        assert (f.status, f.muscle) == ("REPORTED", muscle)

    def test_not_applicable_without_a_hyper(self, kb):
        assert not find(run("normal", kb), "parks_3_step")
        assert not find(run("r_cn6", kb), "parks_3_step")

    def test_step2_within_error(self, kb):
        m = set_vertical(simulate(parks_case("R-SO")), right=6.0, left=6.5)
        f = one(evaluate(m, kb), "parks_3_step")
        assert f.status == "INCONCLUSIVE" and codes(f) == {"parks_step2_within_error"}
        assert "single-muscle palsy" in f.reasons[0].detail          # the KB's reason
        assert "parks_inconclusive" in {e.id for e in f.kb_entries}

    def test_step3_within_error(self, kb):
        m = set_vertical(simulate(parks_case("R-SO")), tilt_right=6.0, tilt_left=6.5)
        f = one(evaluate(m, kb), "parks_3_step")
        assert f.status == "INCONCLUSIVE" and codes(f) == {"parks_step3_within_error"}

    def test_tilt_not_captured(self, kb):
        r = evaluate(simulate(parks_case("R-SO"), tilt=False), kb)
        assert codes(one(r, "parks_3_step")) == {"tilt_not_captured"}
        assert one(r, "cn4_palsy", "R").status == "INCONCLUSIVE"
        assert not r.quality.tilt_captured

    def test_steps_conflict_when_the_hyper_reverses(self, kb):
        m = set_vertical(simulate(parks_case("R-SO")), down=-8.0)
        f = one(evaluate(m, kb), "parks_3_step")
        assert codes(f) == {"parks_steps_conflict"}
        assert not f.reasons[0].repeatable

    def test_multiple_palsies_do_not_fit(self, kb):
        """A complete CN III weakens four muscles; the table must not name one."""
        f = one(run("l_cn3_complete", kb), "parks_3_step")
        assert f.status == "INCONCLUSIVE" and codes(f) == {"parks_steps_conflict"}

    def test_cn4_cites_parks(self, kb):
        r = evaluate(simulate(parks_case("R-SR")), kb)        # 3-step says R-SR, not SO
        assert not find(r, "cn4_palsy")


# ─────────────────────────────────────────────────────────────
# Every INCONCLUSIVE path
# ─────────────────────────────────────────────────────────────

class TestInconclusivePaths:
    def test_primary_insufficient_stops_everything(self, kb):
        r = run("r_cn4", kb, insufficient=("primary",))
        assert [f.rule_id for f in r.findings] == ["data_sufficiency"]
        assert codes(r.findings[0]) == {"primary_insufficient"}
        assert r.referral.decision == "REPEAT_TEST"
        assert r.quality.insufficient_positions == ["primary"]

    def test_insufficient_cell(self, kb):
        f = one(run("a_pattern", kb, insufficient=("up",)), "a_pattern_strabismus")
        assert f.status == "INCONCLUSIVE" and codes(f) == {"insufficient_data"}

    def test_insufficient_cell_does_not_hide_a_clear_incomitance(self, kb):
        f = one(run("r_cn6", kb, insufficient=("up",)), "comitance")
        assert f.status == "REPORTED" and f.abnormal

    def test_within_error(self, kb):
        r = run("noisy_borderline", kb)
        f = one(r, "a_pattern_strabismus")
        assert f.status == "INCONCLUSIVE" and codes(f) == {"within_error"}
        assert r.referral.decision == "REPEAT_TEST"

    def test_threshold_not_set(self, kb):
        assert codes(one(run("r_cn6", kb), "cn6_palsy", "R")) == {"threshold_not_set"}

    def test_not_measured(self, test_kb):
        m = simulate(CASES["r_cn6"]).model_copy(update={"eyes": {}})
        f = one(evaluate(m, test_kb), "cn6_palsy", "R")
        assert f.status == "INCONCLUSIVE" and codes(f) == {"not_measured"}

    def test_not_in_protocol(self, kb):
        m = simulate(CASES["normal"], protocol="five_gaze")
        d = evaluate_feature(Context(m, kb, 2.0), "oblique_overaction_in_oblique_gaze", {})
        assert d.state == "INCONCLUSIVE" and {r.code for r in d.reasons} == {"not_in_protocol"}

    def test_tilt_not_captured(self, kb):
        f = one(evaluate(simulate(parks_case("R-SO"), tilt=False), kb), "cn4_palsy", "R")
        assert "tilt_not_captured" in codes(f)

    def test_kids_mode_still_characterises(self, test_kb):
        assert one(run("r_cn6", test_kb, protocol="five_gaze"), "cn6_palsy", "R").status == "REPORTED"
        assert one(run("r_cn4", test_kb, protocol="five_gaze"), "cn4_palsy", "R").status == "REPORTED"


# ─────────────────────────────────────────────────────────────
# Referral
# ─────────────────────────────────────────────────────────────

class TestReferral:
    def test_confident_finding_refers(self, kb):
        r = run("comitant_eso_20", kb)
        assert r.referral.decision == "REFER" and not r.referral.urgent
        assert "comitant_strabismus" in r.referral.because

    @pytest.mark.parametrize("case", ["comitant_eso_20", "r_cn4", "a_pattern", "l_cn3_complete"])
    def test_low_confidence_never_refers(self, kb, test_kb, case):
        """Regression (brief §2): LOW confidence must never issue a referral."""
        for k in (kb, test_kb):
            r = evaluate(simulate(CASES[case], Noise(confidence="LOW")), k)
            assert reported(r), "the findings are still reported…"
            assert all(f.confidence == "LOW" for f in reported(r).values())
            assert r.referral.decision == "REPEAT_TEST", "…but only as a reason to repeat"
            assert not r.referral.urgent

    def test_low_confidence_urgent_flag_is_surfaced_not_referred(self, test_kb):
        r = evaluate(simulate(CASES["l_cn3_pupil_involving"], Noise(confidence="LOW")), test_kb)
        assert one(r, "cn3_pupil_involving", "L").status == "REPORTED"
        assert r.referral.decision == "REPEAT_TEST"
        assert not r.referral.urgent and r.referral.urgent_unconfirmed
        assert "URGENT" in r.referral.advice

    def test_moderate_confidence_refers(self, kb):
        r = evaluate(simulate(CASES["comitant_eso_20"], Noise(confidence="MODERATE")), kb)
        assert r.referral.decision == "REFER"

    def test_one_low_cell_lowers_the_finding(self, kb):
        m = simulate(CASES["a_pattern"])
        cells = dict(m.grid.cells)
        up = cells["up"]
        cells["up"] = up.model_copy(update={"horizontal_pd": up.horizontal_pd.model_copy(update={"confidence": "LOW"})})
        m = m.model_copy(update={"grid": m.grid.model_copy(update={"cells": cells})})
        r = evaluate(m, kb)
        assert one(r, "a_pattern_strabismus").confidence == "LOW"

    def test_threshold_gaps_do_not_ask_for_a_repeat(self, kb):
        """Repeating the test cannot set a KB threshold."""
        r = run("normal", kb)
        assert r.not_evaluated and r.referral.decision == "NO_REFERRAL"

    def test_probably_normal_but_unproven_does_not_ask_for_a_repeat(self, kb):
        # Comitance unprovable at this error, but every difference points to comitant.
        r = evaluate(simulate(CASES["normal"], Noise(pd=3.0)), kb)
        assert one(r, "comitance").status == "INCONCLUSIVE"
        assert r.referral.decision == "NO_REFERRAL"

    def test_decide_referral_on_no_findings(self):
        assert decide_referral([]).decision == "NO_REFERRAL"


# ─────────────────────────────────────────────────────────────
# Noise
# ─────────────────────────────────────────────────────────────

class TestNoise:
    @pytest.mark.parametrize("case, rule, eye", [
        ("r_cn6", "cn6_palsy", "R"), ("r_cn4", "cn4_palsy", "R"), ("a_pattern", "a_pattern_strabismus", None),
        ("l_cn3_complete", "cn3_palsy_complete", "L"), ("r_duane_like", "duane_like", "R"),
    ])
    def test_patterns_survive_noise(self, test_kb, case, rule, eye):
        hits = sum(
            any(f.rule_id == rule and f.affected_eye == eye and f.status == "REPORTED"
                for f in evaluate(simulate(CASES[case], Noise(pd=0.5, seed=s)), test_kb).findings)
            for s in range(100)
        )
        assert hits >= 88, f"{rule} found in {hits}/100 noisy runs"

    def test_one_stray_cell_does_not_rule_out_cn6(self, test_kb):
        """'No vertical deviation in any gaze' scans 9 cells with a family-wise k."""
        m = set_vertical(simulate(CASES["r_cn6"]), up_left=2.5)    # 2.5 errors: past k=2, not k≈2.8
        assert one(evaluate(m, test_kb), "cn6_palsy", "R").status == "REPORTED"
