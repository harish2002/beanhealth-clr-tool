"""
BeanHealth CLR Tool — Database & Validation Tests (Phase 3, milestone 8)
=========================================================================

  • Storage: one examined session lands in every table, with its versions;
    consent is required; pseudo IDs only (no name column anywhere); no
    video (storage_uri NULL); a session cannot be stored twice; erasure
    removes a patient and everything derived from them.
  • Statistics against hand-computed values: Bland–Altman, Cohen's κ,
    sensitivity / specificity with INCONCLUSIVE calls held out, Wilson CI.
  • Validation end to end: simulated visits with an "orthoptist" reference
    (truth + noise) — Bland–Altman per position, pooled and per device,
    pattern agreement; patients without research consent are left out.
  • The CLI and POST /exam/sessions.

Run:
    cd backend && pytest tests/test_db_validation.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exam.db import (  # noqa: E402
    Consent,
    PatientRecord,
    add_reference_diagnosis,
    add_reference_measurement,
    age_band,
    connect,
    delete_patient,
    new_pseudo_id,
    store_exam,
)
from exam.examine import examine  # noqa: E402
from exam.kb import load_kb  # noqa: E402
from exam.sim import CASES, FrameNoise, SessionSpec, SimEyes, simulate_session, true_deviation  # noqa: E402
from exam.validation import agreement, bland_altman, cohen_kappa, summary, validate, wilson  # noqa: E402
from exam.validation.__main__ import main as cli  # noqa: E402

KB = load_kb()
KAPPA = {"R": (0.3, 0.1), "L": (0.25, -0.05)}
CONSENT = Consent(store_measurements=True, research=True)


def make_session(case: str, session_id: str, seed: int = 0, model: str = "Simulated phone", **kw):
    kw = {"include_tilt": False, "n_frames": 30, **kw}
    s = simulate_session(SessionSpec(noise=FrameNoise(seed=seed),
                                     sim=SimEyes(hirschberg_pd_per_mm=16.0, kappa_mm=KAPPA, case=CASES[case]), **kw), KB)
    header = s.header.model_copy(update={"session_id": session_id,
                                         "device": s.header.device.model_copy(update={"model": model})})
    return s.model_copy(update={"header": header})


def patient(research: bool = True, store: bool = True) -> PatientRecord:
    return PatientRecord(pseudo_id=new_pseudo_id(), age_band=age_band(7),
                         consent=Consent(store_measurements=store, research=research))


@pytest.fixture(scope="module")
def stored():
    conn = connect()
    s = make_session("r_cn6", "store-1")
    result = examine(s, KB, include_tilt=False)
    p = patient()
    info = store_exam(conn, s, result, p)
    return conn, s, result, p, info


def count(conn, table, where="", args=()):
    return conn.execute(f"SELECT COUNT(*) FROM {table} {where}", args).fetchone()[0]


# ─────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────

class TestStore:
    def test_every_table_is_filled(self, stored):
        conn, s, result, p, info = stored
        assert info.captures == len(s.captures) == count(conn, "captures")
        assert count(conn, "frame_qc") == sum(len(c.frames) for c in s.captures)
        assert count(conn, "measurements") == len([c for c in s.captures if not c.position.startswith("calib")])
        assert count(conn, "eye_measures") == 2
        assert count(conn, "findings") == info.findings == len(result.engine.findings)
        row = conn.execute("SELECT * FROM sessions").fetchone()
        assert row["referral_decision"] == result.engine.referral.decision
        assert row["kb_sha256"] == KB.sha256 and row["hirschberg_source"] == "fitted"

    def test_versions(self, stored):
        conn, *_ = stored
        assert conn.execute("SELECT kb_version FROM kb_versions").fetchone()[0] == KB.version
        ev = conn.execute("SELECT * FROM engine_versions").fetchone()
        assert ev["measurement_version"] == "1.0.0" and ev["qc_version"] == "1.1.0"

    def test_measurements_round_trip(self, stored):
        conn, _, result, *_ = stored
        h = conn.execute("SELECT m.horizontal_pd, m.horizontal_error FROM measurements m JOIN captures c "
                         "ON c.id = m.capture_id WHERE c.position = 'right'").fetchone()
        cell = result.measurement.measurements.grid.cells["right"]
        assert h[0] == pytest.approx(cell.horizontal_pd.value) and h[1] == pytest.approx(cell.horizontal_pd.error)

    def test_frame_rejections_are_kept(self):
        from exam.sim import events
        conn = connect()
        s = make_session("normal", "qc-1", events={"up": events(blink=[3, 4])})
        store_exam(conn, s, examine(s, KB, include_tilt=False), patient())
        reasons = [r[0] for r in conn.execute(
            "SELECT reject_reason FROM frame_qc f JOIN captures c ON c.id = f.capture_id "
            "WHERE c.position = 'up' AND f.valid = 0")]
        assert reasons and all("blink" in r for r in reasons)

    def test_pseudonymized_and_no_video(self, stored):
        conn, *_ = stored
        columns = {r[1] for t in [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                   for r in conn.execute(f"PRAGMA table_info({t})")}
        assert not {"name", "patient_name", "label", "dob", "birth_date", "image", "video"} & columns
        assert count(conn, "captures", "WHERE storage_uri IS NOT NULL") == 0

    def test_consent_required(self):
        conn = connect()
        s = make_session("normal", "consent-1")
        with pytest.raises(PermissionError):
            store_exam(conn, s, examine(s, KB, include_tilt=False), patient(store=False))
        assert count(conn, "sessions") == 0

    @pytest.mark.parametrize("bad", ["John Smith", "short", "x" * 80, "name@example"])
    def test_pseudo_id_shape(self, bad):
        with pytest.raises(ValueError):
            PatientRecord(pseudo_id=bad, consent=CONSENT)

    def test_age_bands(self):
        assert [age_band(a) for a in (0, 2, 3, 6, 12, 18, 40, 70)] == \
            ["0-2", "0-2", "3-5", "6-11", "12-17", "18-39", "40-64", "65+"]

    def test_no_double_store(self, stored):
        conn, s, result, p, _ = stored
        with pytest.raises(ValueError):
            store_exam(conn, s, result, p)

    def test_erasure(self):
        conn = connect()
        p = patient()
        for i in range(2):
            s = make_session("normal", f"erase-{i}", seed=i)
            store_exam(conn, s, examine(s, KB, include_tilt=False), p)
        add_reference_measurement(conn, "erase-0", "primary", 0.0, 0.0)
        add_reference_diagnosis(conn, "erase-0", "direction_horizontal", False)
        assert delete_patient(conn, p.pseudo_id) == 2
        for table in ("patients", "sessions", "captures", "frame_qc", "measurements", "eye_measures", "findings",
                      "reference_measurements", "reference_diagnoses"):
            assert count(conn, table) == 0, table


# ─────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────

class TestStats:
    def test_bland_altman(self):
        b = bland_altman([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
        assert b.bias == 3 and b.sd == pytest.approx(math.sqrt(2.5))
        assert b.loa_low == pytest.approx(3 - 1.959964 * math.sqrt(2.5), abs=1e-5)
        assert b.bias_ci95[0] < 3 < b.bias_ci95[1] and b.mean_abs_error == 3
        assert bland_altman([], []).n == 0 and bland_altman([1.0], [0.0]).sd is None

    def test_cohen_kappa(self):
        # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no: po 0.70, pe 0.50 → κ 0.40.
        a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
        b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
        assert cohen_kappa(a, b) == pytest.approx(0.4)
        assert cohen_kappa([1, 1], [1, 1]) == 1.0 and cohen_kappa([], []) is None

    def test_agreement_holds_inconclusive_out(self):
        a = agreement([True, True, False, False, None, None], [True, False, False, True, True, False])
        assert (a.tp, a.fp, a.tn, a.fn, a.inconclusive) == (1, 1, 1, 1, 2)
        assert a.sensitivity == 0.5 and a.specificity == 0.5 and a.inconclusive_rate == pytest.approx(1 / 3)

    def test_wilson(self):
        lo, hi = wilson(9, 10)
        assert lo == pytest.approx(0.5958, abs=1e-3) and hi == pytest.approx(0.9821, abs=1e-3)
        assert wilson(0, 0) is None


# ─────────────────────────────────────────────────────────────
# Validation end to end
# ─────────────────────────────────────────────────────────────

VISITS = ["normal", "comitant_eso_20", "r_cn4", "r_cn6", "a_pattern"] * 2


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "exam.sqlite"
    conn = connect(path)
    rng = np.random.default_rng(7)
    for i, name in enumerate(VISITS):
        sid = f"val-{i}"
        s = make_session(name, sid, seed=100 + i, model="Pixel 8" if i % 2 else "iPhone 15")
        store_exam(conn, s, examine(s, KB, include_tilt=False), patient())
        for gaze in ("primary", "right", "left", "up", "down"):
            th, tv = true_deviation(CASES[name], gaze)
            add_reference_measurement(conn, sid, gaze, th + rng.normal(0, 1.5), tv + rng.normal(0, 1.5),
                                      examiner="orthoptist-A")
        add_reference_diagnosis(conn, sid, "direction_horizontal", name in ("comitant_eso_20", "r_cn6"))
        add_reference_diagnosis(conn, sid, "cn4_palsy", name == "r_cn4", affected_eye="R")
    # A patient without research consent: stored, but never in a validation.
    s = make_session("comitant_eso_20", "no-research", seed=999)
    store_exam(conn, s, examine(s, KB, include_tilt=False), patient(research=False))
    add_reference_measurement(conn, "no-research", "primary", -50.0, 0.0)
    add_reference_diagnosis(conn, "no-research", "direction_horizontal", False)
    return path, validate(conn)


class TestValidation:
    def test_research_consent_only(self, study):
        _, r = study
        assert r.sessions == len(VISITS) and r.paired_measurements == 5 * len(VISITS)

    def test_bland_altman_recovers_the_reference_noise(self, study):
        _, r = study
        h = r.pooled["horizontal"]
        assert h.n == 5 * len(VISITS) and abs(h.bias) < 1.0 and 0.8 < h.sd < 2.5
        assert set(r.bland_altman) == {"primary", "right", "left", "up", "down"}

    def test_per_device(self, study):
        _, r = study
        assert set(r.per_device) == {"Pixel 8", "iPhone 15"}
        assert r.per_device["Pixel 8"]["horizontal"].n == r.per_device["iPhone 15"]["horizontal"].n == 25

    def test_pattern_agreement(self, study):
        _, r = study
        d = r.patterns["direction_horizontal"]
        assert d.n == len(VISITS) and d.sensitivity == 1.0 and d.specificity == 1.0 and d.kappa == 1.0
        cn4 = r.patterns["cn4_palsy:R"]
        assert cn4.tp + cn4.fn + (cn4.inconclusive if cn4.inconclusive else 0) >= 1
        assert cn4.fp == 0

    def test_summary_and_cli(self, study, tmp_path, capsys):
        path, r = study
        assert "Bland–Altman" in summary(r)
        out = tmp_path / "v.json"
        assert cli(["--db", str(path), "--json", str(out)]) == 0
        assert "Pattern agreement" in capsys.readouterr().out and out.exists()
        assert cli(["--db", str(tmp_path / "missing.sqlite")]) == 2


class TestStoreAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def body(self, sid, **consent):
        s = make_session("normal", sid, n_frames=20)
        return {"session": s.model_dump(mode="json"),
                "patient": {"pseudo_id": new_pseudo_id(), "age_band": "6-11",
                            "consent": {"store_measurements": True, **consent}}}

    def test_not_configured(self, client, monkeypatch):
        monkeypatch.delenv("EXAM_DB_PATH", raising=False)
        assert client.post("/exam/sessions", params={"tilt": False}, json=self.body("api-0")).status_code == 503

    def test_store(self, client, monkeypatch, tmp_path):
        monkeypatch.setenv("EXAM_DB_PATH", str(tmp_path / "api.sqlite"))
        r = client.post("/exam/sessions", params={"tilt": False}, json=self.body("api-1"))
        assert r.status_code == 201 and r.json()["stored"]["session_id"] == "api-1"
        again = client.post("/exam/sessions", params={"tilt": False}, json=self.body("api-1"))
        assert again.status_code == 409
        refused = client.post("/exam/sessions", params={"tilt": False},
                              json=self.body("api-2", store_measurements=False))
        assert refused.status_code == 403
