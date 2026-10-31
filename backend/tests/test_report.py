"""
BeanHealth CLR Tool — Clinician Report Tests (Phase 3, milestone 7)
====================================================================

  • The JSON report: red flags first, referral, grid drawn as the examiner
    faces the patient, cell classes from k × error, findings with evidence,
    INCONCLUSIVE reasons, unapproved KB entries, every version.
  • Golden files (brief §9): the JSON report of fixed synthetic sessions and
    one HTML page must not change unnoticed. After an intended change,
    regenerate with
        UPDATE_GOLDEN=1 pytest tests/test_report.py
    and review the diff.
  • HTML: self-contained, escaped, disclaimer present. PDF: a valid document.
  • The /exam/report, /exam/report/html and /exam/report/pdf endpoints.

Run:
    cd backend && pytest tests/test_report.py -v
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from exam.examine import examine  # noqa: E402
from exam.kb import load_kb  # noqa: E402
from exam.kb.loader import DEFAULT_KB_DIR  # noqa: E402
from exam.report import ReportPatient, build_report, grid_layout, render_html, render_pdf  # noqa: E402
from exam.sim import CASES, SessionSpec, SimEyes, events, simulate_session  # noqa: E402

KB = load_kb()
GOLDEN = Path(__file__).parent / "golden"
KAPPA = {"R": (0.3, 0.1), "L": (0.25, -0.05)}
WHEN = "2026-10-31T12:00:00Z"
# The thresholds the engine tests use (tests/test_engine.py), so red-flag rules can fire.
TEST_THRESHOLDS = {
    "duction_limit_fraction": 0.7, "anisocoria_tolerance_mm": 1.0, "duane_fissure_narrowing_ratio": 0.85,
    "duane_small_primary_deviation_pd": 10, "scleral_show_mm": 1.0, "ptosis_fissure_difference_mm": 2.0,
    "incomitance_vertical_pd": 5,
}


def session(case: str, **kw):
    return simulate_session(SessionSpec(sim=SimEyes(hirschberg_pd_per_mm=16.0, kappa_mm=KAPPA, case=CASES[case]), **kw),
                            KB)


def report(case: str, kb=KB, patient=None, **kw):
    s = session(case, **kw)
    return build_report(s, examine(s, kb), kb=kb, patient=patient, generated_at=WHEN)


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


@pytest.fixture(scope="module")
def cn4():
    return report("r_cn4", skip=("down_left",))


# ─────────────────────────────────────────────────────────────
# JSON report
# ─────────────────────────────────────────────────────────────

class TestReport:
    def test_layout_faces_the_patient(self):
        assert grid_layout("nine_gaze") == [["up_right", "up", "up_left"], ["right", "primary", "left"],
                                            ["down_right", "down", "down_left"]]
        assert grid_layout("five_gaze") == [[None, "up", None], ["right", "primary", "left"], [None, "down", None]]

    def test_cells(self, cn4):
        assert cn4.grid["down_left"].cls == "insufficient" and cn4.grid["down_left"].reason == "not_captured"
        assert cn4.grid["down"].cls == "beyond_error" and cn4.grid["down"].vertical_label == "R hyper"
        assert cn4.grid["right"].cls == "within_error"
        assert cn4.grid["right"].horizontal_label == "within error"

    def test_referral_names_its_findings(self, cn4):
        assert cn4.referral.decision == "REFER"
        assert "3-step test isolates R-SO" in cn4.referral.because
        assert "CN IV (trochlear) palsy pattern (right eye)" in cn4.referral.because

    def test_findings_carry_evidence_and_kb_status(self, cn4):
        parks = next(f for f in cn4.findings if f.rule_id == "parks_3_step")
        assert parks.muscle == "R-SO" and parks.evidence
        assert any(f.unapproved_kb for f in cn4.findings)
        assert cn4.unapproved_kb_entries > 0

    def test_inconclusive_reasons(self, cn4):
        comitance = next(f for f in cn4.inconclusive if f.rule_id == "comitance")
        assert any("down_left" in r for r in comitance.reasons)

    def test_versions_and_positioning(self, cn4):
        assert cn4.versions["report"] == "1.0.0" and cn4.versions["kb"] == KB.version
        assert {"capture_protocol", "qc", "calibration", "measurement", "engine", "kb_sha256"} <= set(cn4.versions)
        assert cn4.positioning.endswith("Decision support — not a diagnosis.")
        assert len(cn4.limitations) == 6

    def test_red_flags_come_first(self, test_kb):
        r = report("l_cn3_pupil_involving", kb=test_kb, include_tilt=False)
        assert r.red_flags and r.red_flags[0].urgency == "URGENT"
        assert r.referral.urgent and r.referral.decision == "REFER"
        html = render_html(r)
        assert html.index("Red flags") < html.index("Deviation by gaze")

    def test_repeat_when_no_kappa(self):
        r = report("comitant_eso_20", include_tilt=False, n_frames=20, skip=("calib_R_lens",))
        assert r.referral.decision == "REPEAT_TEST"
        assert all(c.cls == "insufficient" and c.reason == "kappa_not_measured" for c in r.grid.values())
        assert r.calibration.kappa["R"] is None and any("calib_R_lens" in f for f in r.calibration.flags)


# ─────────────────────────────────────────────────────────────
# Golden files
# ─────────────────────────────────────────────────────────────

GOLDEN_CASES = {
    "normal": dict(case="normal", include_tilt=False),
    "r_cn4": dict(case="r_cn4", skip=("down_left",)),
    "eso_unsteady": dict(case="comitant_eso_20", include_tilt=False,
                         events={"primary": events(blink=range(0, 70, 2), dropped=range(1, 35, 2))}),
}


def _check_golden(name: str, text: str) -> None:
    path = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(text)
    assert path.exists(), f"missing {path}; run UPDATE_GOLDEN=1 pytest {__file__}"
    assert text == path.read_text(), f"{name} changed; review, then UPDATE_GOLDEN=1 pytest tests/test_report.py"


class TestGolden:
    @pytest.mark.parametrize("name", list(GOLDEN_CASES))
    def test_json(self, name):
        spec = dict(GOLDEN_CASES[name])
        r = report(spec.pop("case"), **spec)
        _check_golden(f"report_{name}.json", json.dumps(r.model_dump(mode="json"), indent=1, ensure_ascii=False) + "\n")

    def test_html(self):
        r = report("r_cn4", skip=("down_left",), patient=ReportPatient(label="Golden patient", age_years=7))
        _check_golden("report_r_cn4.html", render_html(r) + "\n")


# ─────────────────────────────────────────────────────────────
# HTML and PDF
# ─────────────────────────────────────────────────────────────

class TestRendering:
    def test_html_is_self_contained_and_escaped(self):
        r = report("normal", include_tilt=False, n_frames=20,
                   patient=ReportPatient(label="<script>alert(1)</script>", age_years=5))
        html = render_html(r)
        assert html.startswith("<!doctype html>")
        assert "<script>" not in html and "&lt;script&gt;" in html
        assert "src=" not in html and "<link" not in html
        assert "Decision support — not a diagnosis." in html
        assert all(c.name in html for c in r.grid.values())

    def test_pdf(self, cn4):
        pdf = render_pdf(cn4)
        assert pdf.startswith(b"%PDF") and pdf.rstrip().endswith(b"%%EOF")
        assert pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page ") >= 2      # the report runs over two pages


class TestAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    @pytest.fixture(scope="class")
    def body(self):
        s = session("r_cn4", include_tilt=True, n_frames=20)
        return {"session": json.loads(s.model_dump_json()), "patient": {"label": "API patient", "age_years": 9}}

    def test_json(self, client, body):
        r = client.post("/exam/report", json=body)
        assert r.status_code == 200
        assert r.json()["referral"]["decision"] == "REFER" and r.json()["patient"]["label"] == "API patient"

    def test_html(self, client, body):
        r = client.post("/exam/report/html", json=body)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        assert "API patient" in r.text

    def test_pdf(self, client, body):
        r = client.post("/exam/report/pdf", json=body)
        assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
        assert r.content.startswith(b"%PDF") and "gaze-exam-sim-session.pdf" in r.headers["content-disposition"]

    def test_bad_patient(self, client, body):
        r = client.post("/exam/report", json={**body, "patient": {"age_years": 300}})
        assert r.status_code == 422
