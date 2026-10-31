"""
BeanHealth CLR Tool — Measurement Grid Tests (Phase 3, milestone 6)
====================================================================

  • Fusion: median, MAD-based error, effective frame count.
  • The grid from simulated sessions: values and signs recovered in every
    cell and on head tilt; errors that cover the truth; each confidence rule
    (fitted vs default ratio, capture flags, unsteady fixation); every way a
    cell becomes INSUFFICIENT_DATA.
  • Ductions, fissures, scleral show and pupils against the case's truth.
  • End to end: every synthetic engine case (and all 8 Parks rows) rendered
    as frame tracks gives the same findings through QC → calibration →
    measurement → engine as the case's grid does directly.
  • LOW confidence from unsteady fixation never refers.
  • examine() and the /exam/measure, /exam/evaluate endpoints.

Sessions come from exam.sim.frames — test fixtures, not physiology.

Run:
    cd backend && pytest tests/test_measurement.py -v
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from exam.engine import evaluate  # noqa: E402
from exam.examine import examine  # noqa: E402
from exam.kb import load_kb  # noqa: E402
from exam.kb.loader import DEFAULT_KB_DIR  # noqa: E402
from exam.measurement import effective_n, fuse, measure  # noqa: E402
from exam.models import PROTOCOL_POSITIONS  # noqa: E402
from exam.sim import (  # noqa: E402
    CASE_NOISE,
    CASES,
    CYCLOVERTICAL,
    FrameNoise,
    Noise,
    SessionSpec,
    SimEyes,
    events,
    parks_case,
    simulate,
    simulate_session,
    true_deviation,
    true_duction,
    true_tilt,
)

KB = load_kb()
KAPPA = {"R": (0.3, 0.1), "L": (0.25, -0.05)}
PATIENT = SimEyes(hirschberg_pd_per_mm=16.0, kappa_mm=KAPPA, deviation_pd=(20.0, -6.0))


def session(**kw):
    return simulate_session(SessionSpec(**kw), KB)


def case_session(case, **kw):
    return session(sim=SimEyes(hirschberg_pd_per_mm=16.0, kappa_mm=KAPPA, case=case), **kw)


def kb_with(**values):
    """A copy of the KB with measurement.yaml entries replaced."""
    def build(tmp: Path):
        dst = tmp / "kb"
        shutil.copytree(DEFAULT_KB_DIR, dst)
        path = dst / "measurement.yaml"
        data = yaml.safe_load(path.read_text())
        for e in data["entries"]:
            if e["id"] in values:
                e["value"] = values[e["id"]]
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        return load_kb(dst, verify_hash=False)
    return build


@pytest.fixture(scope="module")
def grid():
    return measure(session(sim=PATIENT), kb=KB)


# ─────────────────────────────────────────────────────────────
# Fusion
# ─────────────────────────────────────────────────────────────

class TestFusion:
    def test_independent_frames_count_fully(self):
        x = np.random.default_rng(0).normal(size=2000)
        assert 1800 < effective_n(x) <= 2000

    def test_correlated_frames_count_less(self):
        # AR(1) with ρ = 0.8: n·(1 − ρ)/(1 + ρ) ≈ n / 9.
        rng = np.random.default_rng(1)
        x = np.zeros(4000)
        for i in range(1, len(x)):
            x[i] = 0.8 * x[i - 1] + rng.normal()
        assert 4000 / 12 < effective_n(x) < 4000 / 7

    def test_median_resists_outliers(self):
        med, se = fuse([1.0] * 20 + [50.0, -40.0])
        assert med == 1.0 and se == 0.0

    def test_mad_error_shrinks_with_frames(self):
        rng = np.random.default_rng(2)
        _, few = fuse(rng.normal(size=20))
        _, many = fuse(rng.normal(size=2000))
        assert many < few / 5
        # ≈ 1.2533 / √n for a normal sample (the median's standard error).
        assert many == pytest.approx(1.2533 / np.sqrt(2000), rel=0.15)

    def test_empty(self):
        with pytest.raises(ValueError):
            fuse([])


# ─────────────────────────────────────────────────────────────
# The grid
# ─────────────────────────────────────────────────────────────

class TestGrid:
    def test_every_cell_recovers_the_deviation(self, grid):
        for gaze, cell in grid.measurements.grid.cells.items():
            assert cell.status == "OK" and cell.valid_frames == 60
            assert abs(cell.horizontal_pd.value - 20.0) < 2 * cell.horizontal_pd.error, gaze
            assert abs(cell.vertical_pd.value + 6.0) < 2 * cell.vertical_pd.error, gaze

    @pytest.mark.parametrize("dev", [(20.0, -6.0), (-15.0, 8.0)])
    def test_signs_eso_and_right_hyper_positive(self, dev):
        g = measure(session(sim=SimEyes(hirschberg_pd_per_mm=16.0, kappa_mm=KAPPA, deviation_pd=dev),
                            include_tilt=False, n_frames=20), kb=KB)
        p = g.position("primary")
        assert np.sign(p.horizontal_pd.value) == np.sign(dev[0])
        assert np.sign(p.vertical_pd.value) == np.sign(dev[1])
        assert p.horizontal_deg == pytest.approx(np.degrees(np.arctan(p.horizontal_pd.value / 100)))

    def test_tilt(self, grid):
        tilt = grid.measurements.tilt
        assert abs(tilt.right.value + 6.0) < 2 * tilt.right.error
        assert abs(tilt.left.value + 6.0) < 2 * tilt.left.error

    def test_no_tilt_captured(self):
        g = measure(session(sim=PATIENT, include_tilt=False, n_frames=20), kb=KB)
        assert g.measurements.tilt is None
        assert g.position("tilt_right").reason == "not_captured"

    @pytest.mark.parametrize("protocol", ["nine_gaze", "five_gaze"])
    def test_grid_matches_the_protocol(self, protocol):
        g = measure(session(sim=PATIENT, protocol=protocol, include_tilt=False, n_frames=20), kb=KB)
        assert list(g.measurements.grid.cells) == PROTOCOL_POSITIONS[protocol]

    def test_every_output_is_stamped(self, grid):
        assert grid.kb_version == KB.version and grid.kb_sha256 == KB.sha256
        assert grid.measurement_version == "1.0.0" and grid.calibration_version == "1.0.0"
        assert grid.qc_version == "1.1.0" and grid.capture_protocol_version == "1.1.0"

    def test_error_includes_the_method_error(self, grid):
        method = KB.scalar("measurement_method_error_pd").value
        for cell in grid.measurements.grid.cells.values():
            assert cell.horizontal_pd.error >= method and cell.vertical_pd.error >= method

    def test_default_ratio_widens_large_deviations_and_lowers_confidence(self, grid):
        g = measure(session(sim=PATIENT, distance_cm=None, include_tilt=False, n_frames=30), kb=KB)
        fitted, default = grid.position("primary"), g.position("primary")
        assert default.horizontal_pd.error > 2 * fitted.horizontal_pd.error
        assert fitted.horizontal_pd.confidence == "HIGH"
        assert default.horizontal_pd.confidence == "MODERATE"
        assert "kb_default_hirschberg" in default.confidence_reasons
        assert "kb_default_hirschberg" in {f.code for f in g.flags}
        # The default ratio (12.3 PD/mm) under-reads a patient whose true ratio is 16.
        assert default.horizontal_pd.value < fitted.horizontal_pd.value

    def test_capture_flags_lower_confidence(self):
        g = measure(session(sim=PATIENT, fps=20.0, include_tilt=False, n_frames=30), kb=KB)
        assert g.position("primary").horizontal_pd.confidence == "MODERATE"
        assert "session_quality" in {f.code for f in g.flags}

    def test_unsteady_fixation_is_low_confidence(self):
        g = measure(session(sim=PATIENT, include_tilt=False,
                            events={"up": events(blink=range(0, 70, 2), dropped=range(1, 35, 2))}), kb=KB)
        up = g.position("up")
        assert up.status == "OK" and up.frames_valid < 0.5 * up.frames_total
        assert up.horizontal_pd.confidence == "LOW" and "unsteady_fixation" in up.confidence_reasons
        assert g.position("primary").horizontal_pd.confidence == "HIGH"
        assert ("unsteady_fixation", "up") in {(f.code, f.position) for f in g.flags}


class TestInsufficient:
    def test_position_not_captured(self):
        g = measure(session(sim=PATIENT, include_tilt=False, n_frames=20, skip=("up_left",)), kb=KB)
        cell = g.measurements.grid.cells["up_left"]
        assert cell.status == "INSUFFICIENT_DATA" and cell.horizontal_pd is None
        assert g.position("up_left").reason == "not_captured"

    def test_too_few_valid_frames(self):
        g = measure(session(sim=PATIENT, include_tilt=False, n_frames=20,
                            events={"down": events(blink=range(0, 15))}), kb=KB)
        assert g.position("down").status == "INSUFFICIENT_DATA"
        assert g.position("down").reason == "too_few_valid_frames"

    def test_no_kappa_measures_nothing(self):
        """Never rotate without kappa: an eye without one leaves every cell unmeasured."""
        s = session(sim=PATIENT, include_tilt=False, n_frames=20, skip=("calib_L_lens",))
        g = measure(s, kb=KB)
        assert all(p.status == "INSUFFICIENT_DATA" and p.reason == "kappa_not_measured"
                   for p in g.positions if p.kind == "gaze")
        assert ("kappa_not_measured", "L") in {(f.code, f.eye) for f in g.flags}
        result = evaluate(g.measurements, KB)
        assert result.findings[0].rule_id == "data_sufficiency"
        assert result.referral.decision == "REPEAT_TEST"


# ─────────────────────────────────────────────────────────────
# Ductions, lids, pupils
# ─────────────────────────────────────────────────────────────

class TestEyeMeasures:
    @pytest.mark.parametrize("name", ["normal", "r_cn6", "l_cn3_complete", "r_ted_like"])
    def test_ductions_match_the_case(self, name):
        case = CASES[name]
        g = measure(case_session(case, include_tilt=False, n_frames=30), kb=KB)
        for eye, em in g.measurements.eyes.items():
            assert set(em.ductions) == {"abduction", "adduction", "elevation", "depression"}
            for d, m in em.ductions.items():
                assert abs(m.value - true_duction(case, eye, d)) < 2 * m.error + 0.01, (eye, d)

    def test_limited_abduction_is_seen(self):
        g = measure(case_session(CASES["r_cn6"], include_tilt=False, n_frames=30), kb=KB)
        abd = g.measurements.eyes["R"].ductions["abduction"]
        assert abd.value + 2 * abd.error < 0.9
        normal = g.measurements.eyes["L"].ductions["abduction"]
        assert abs(normal.value - 1.0) < 2 * normal.error

    def test_lids_and_pupils_in_mm(self):
        case = CASES["l_cn3_pupil_involving"]
        g = measure(case_session(case, include_tilt=False, n_frames=30), kb=KB)
        for eye in ("R", "L"):
            em = g.measurements.eyes[eye]
            assert em.fissure_height_mm.value == pytest.approx(case.fissure_mm[eye], abs=0.05)
            assert em.pupil_diameter_mm.value == pytest.approx(case.pupil_mm[eye], abs=0.05)
            assert em.scleral_show_mm.value == pytest.approx(case.scleral_show_mm[eye], abs=0.05)
            assert em.fissure_height_mm.error >= KB.scalar("measurement_method_error_mm").value

    def test_fissure_ratio_on_adduction(self):
        g = measure(case_session(CASES["r_duane_like"], include_tilt=False, n_frames=30), kb=KB)
        assert g.measurements.eyes["R"].fissure_ratio_adduction.value == pytest.approx(0.7, abs=0.01)
        assert g.measurements.eyes["L"].fissure_ratio_adduction.value == pytest.approx(1.0, abs=0.01)

    def test_pupil_not_found_is_flagged_not_guessed(self):
        s = case_session(CASES["normal"], include_tilt=False, n_frames=20)
        caps = []
        for c in s.captures:
            if c.position == "primary":
                frames = [f.model_copy(update={"eyes": {e: o.model_copy(update={"pupil_diameter_px": None})
                                                        for e, o in f.eyes.items()}}) for f in c.frames]
                c = c.model_copy(update={"frames": frames})
            caps.append(c)
        g = measure(s.model_copy(update={"captures": caps}), kb=KB)
        assert g.measurements.eyes["R"].pupil_diameter_mm is None
        assert ("pupil_not_measured", "R") in {(f.code, f.eye) for f in g.flags}

    def test_duction_needs_the_position(self):
        g = measure(case_session(CASES["normal"], include_tilt=False, n_frames=20, skip=("right",)), kb=KB)
        assert "abduction" not in g.measurements.eyes["R"].ductions
        assert "adduction" not in g.measurements.eyes["L"].ductions
        assert "abduction" in g.measurements.eyes["L"].ductions


class TestErrorsAreHonest:
    def test_errors_cover_the_truth(self, tmp_path):
        """
        Without the method-error term (which models bias the simulator does not
        add), measured − true over the reported error behaves like a standard
        normal: about 5% beyond 2.
        """
        kb = kb_with(measurement_method_error_pd=1e-6)(tmp_path)
        z = []
        for seed, name in enumerate(["normal", "comitant_eso_20", "r_cn4", "a_pattern", "l_cn3_complete"]):
            case = CASES[name]
            g = measure(simulate_session(SessionSpec(noise=FrameNoise(seed=10 * seed), sim=SimEyes(
                hirschberg_pd_per_mm=16.0, kappa_mm=KAPPA, case=case)), kb), kb=kb)
            for gaze, c in g.measurements.grid.cells.items():
                th, tv = true_deviation(case, gaze)
                z += [(c.horizontal_pd.value - th) / c.horizontal_pd.error,
                      (c.vertical_pd.value - tv) / c.vertical_pd.error]
            for side in ("right", "left"):
                t = getattr(g.measurements.tilt, side)
                z.append((t.value - true_tilt(case, side)) / t.error)
        z = np.abs(z)
        assert np.mean(z > 2) < 0.12 and z.max() < 4


# ─────────────────────────────────────────────────────────────
# End to end: frames → engine
# ─────────────────────────────────────────────────────────────

def _tags(result):
    return sorted(f"{f.rule_id}:{f.affected_eye}:{f.status}" for f in result.findings
                  if f.abnormal or f.status == "INCONCLUSIVE")


E2E = {**CASES, **{c.name: c for c in (parks_case(f"{e}-{m}") for e in "RL" for m in CYCLOVERTICAL)}}


class TestEndToEnd:
    @pytest.mark.parametrize("name", [n for n in E2E if n != "noisy_borderline"])
    def test_frames_give_the_same_findings_as_the_grid(self, name):
        case = E2E[name]
        via_frames = examine(case_session(case), KB).engine
        direct = evaluate(simulate(case, CASE_NOISE.get(name, Noise())), KB)
        assert _tags(via_frames) == _tags(direct)
        assert via_frames.referral.decision == direct.referral.decision

    def test_parks_rows_from_frames(self):
        for e in "RL":
            for m in CYCLOVERTICAL:
                r = examine(case_session(parks_case(f"{e}-{m}")), KB).engine
                parks = next(f for f in r.findings if f.rule_id == "parks_3_step")
                assert parks.status == "REPORTED" and parks.muscle == f"{e}-{m}"

    def test_borderline_a_pattern_stays_inconclusive(self):
        r = examine(case_session(CASES["noisy_borderline"]), KB).engine
        a = [f for f in r.findings if f.rule_id == "a_pattern_strabismus"]
        assert len(a) == 1 and a[0].status == "INCONCLUSIVE"

    def test_normal_is_not_referred(self):
        assert examine(case_session(CASES["normal"]), KB).engine.referral.decision == "NO_REFERRAL"

    def test_low_confidence_never_refers(self):
        """Unsteady fixation everywhere: a clear 20 PD eso asks for a repeat, never a referral."""
        unsteady = {p: events(blink=range(0, 70, 2), dropped=range(1, 35, 2))
                    for p in PROTOCOL_POSITIONS["nine_gaze"]}
        result = examine(case_session(CASES["comitant_eso_20"], include_tilt=False, events=unsteady), KB)
        primary = result.measurement.position("primary")
        assert primary.horizontal_pd.confidence == "LOW" and primary.horizontal_pd.value > 15
        assert result.engine.referral.decision == "REPEAT_TEST"


class TestExamine:
    def test_versions_in_one_place(self):
        r = examine(case_session(CASES["normal"], include_tilt=False, n_frames=20), KB)
        assert r.versions == {
            "capture_protocol": "1.1.0", "qc": "1.1.0", "calibration": "1.0.0", "measurement": "1.0.0",
            "engine": r.engine.engine_version, "kb": KB.version, "kb_sha256": KB.sha256,
        }
        assert r.positioning.startswith("Measures and characterizes ocular deviation")

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_measure_endpoint(self, client):
        s = session(sim=PATIENT, include_tilt=False, n_frames=20)
        r = client.post("/exam/measure", content=s.model_dump_json(), headers={"content-type": "application/json"})
        assert r.status_code == 200
        cell = r.json()["measurements"]["grid"]["cells"]["primary"]
        assert cell["status"] == "OK" and abs(cell["horizontal_pd"]["value"] - 20) < 3

    def test_evaluate_endpoint(self, client):
        s = case_session(CASES["comitant_eso_20"], include_tilt=False, n_frames=20)
        r = client.post("/exam/evaluate", content=s.model_dump_json(), headers={"content-type": "application/json"})
        assert r.status_code == 200
        body = r.json()
        assert body["engine"]["referral"]["decision"] == "REFER"
        assert body["versions"]["kb"] == KB.version

    def test_evaluate_without_tilt(self, client):
        s = case_session(CASES["normal"], include_tilt=False, n_frames=20)
        r = client.post("/exam/evaluate", content=s.model_dump_json(), params={"tilt": False},
                        headers={"content-type": "application/json"})
        assert r.status_code == 200
        assert r.json()["qc"]["insufficient_positions"] == []

    def test_evaluate_rejects_a_malformed_session(self, client):
        r = client.post("/exam/evaluate", json={"header": {}, "captures": []})
        assert r.status_code == 422
