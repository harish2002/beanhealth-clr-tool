"""
BeanHealth CLR Tool — Referral Safety Gate Regression Tests
============================================================

LOW confidence must never issue a referral (ROUTINE / URGENT). Before the gate,
three routes could:

  • /analyse-stream — aggregate confidence went LOW from deviation variance
    while asymmetry variance sat under the 2.5° gate, and ROUTINE/URGENT was
    returned as SUCCESS.
  • /analyse        — a frame whose pupil confidence was LOW still returned
    its tier; there was no confidence check at all.
  • /analyse-batch  — any non-NORMAL Hirschberg tier raised the referral flag,
    whatever the confidence.

Each route now returns INCONCLUSIVE ("repeat the test") instead.

Run:
    cd backend && pytest tests/test_referral_gate.py -v
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import main
from pipeline.module6_classify import ClassificationResult, referral_withheld
from pipeline.module_aggregate import aggregate_frame_results

SUCCESS_REFERENCE = Path(__file__).parent / "test_images" / "success_reference" / "clr_both_eyes.jpg"

needs_reference = pytest.mark.skipif(
    not SUCCESS_REFERENCE.exists(),
    reason=f"Reference image not present: {SUCCESS_REFERENCE.name} (test images are git-ignored)",
)


# ─────────────────────────────────────────────────────────────
# The policy
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tier,confidence,withheld", [
    ("URGENT",  "LOW",    True),
    ("ROUTINE", "LOW",    True),
    ("MONITOR", "LOW",    False),   # re-screen in 3 months is not a referral
    ("NORMAL",  "LOW",    False),
    ("URGENT",  "MEDIUM", False),
    ("URGENT",  "HIGH",   False),
    ("ROUTINE", "HIGH",   False),
    ("URGENT",  None,     False),
    (None,      "LOW",    False),
])
def test_referral_withheld(tier, confidence, withheld):
    assert referral_withheld(tier, confidence) is withheld


# ─────────────────────────────────────────────────────────────
# /analyse-stream — aggregation
# ─────────────────────────────────────────────────────────────

def _frames(asym_degs, dev_degs):
    return [
        {
            "status": "SUCCESS",
            "result": {
                "deviation_degrees": dev,
                "asymmetry_score":   asym / (5.75 * 7.0),
                "asymmetry_degrees": asym,
            },
            "technical": {
                "dominant_eye":   "left",
                "left_direction": "nasal",
                "left_displacement_norm":  0.3,
                "right_displacement_norm": 0.1,
                "flags": [],
            },
        }
        for asym, dev in zip(asym_degs, dev_degs)
    ]


class TestAggregateGate:
    """
    Three frames keep the IQR and drift filters out of the way. Asymmetry
    σ = 2.0° (under the 2.5° variance gate, over the 1.5° HIGH shortcut);
    deviation σ ≈ 6° → aggregate confidence LOW.
    """

    def test_low_confidence_routine_is_withheld(self):
        out = aggregate_frame_results(_frames([18, 20, 22], [10, 15, 22]))
        assert out["status"] == "INCONCLUSIVE"
        assert out["reason"] == "low_confidence_referral_withheld"
        assert "result" not in out
        assert out["asymmetry_avg_deg"] == 20.0   # measurements kept for the clinician

    def test_low_confidence_urgent_is_withheld(self):
        out = aggregate_frame_results(_frames([32, 34, 36], [10, 15, 22]))
        assert out["status"] == "INCONCLUSIVE"
        assert out["reason"] == "low_confidence_referral_withheld"

    def test_confident_routine_still_refers(self):
        out = aggregate_frame_results(_frames([18, 20, 22], [14, 15, 16]))
        assert out["status"] == "SUCCESS"
        assert out["aggregate_confidence"] != "LOW"
        assert out["result"]["urgency_tier"] == "ROUTINE"

    def test_low_confidence_monitor_is_not_a_referral(self):
        out = aggregate_frame_results(_frames([6, 8, 10], [10, 15, 22]))
        assert out["status"] == "SUCCESS"
        assert out["aggregate_confidence"] == "LOW"
        assert out["result"]["urgency_tier"] == "MONITOR"

    def test_no_success_result_is_ever_a_low_confidence_referral(self):
        """Sweep the variance space: whatever is returned, never LOW + referral."""
        rng = np.random.default_rng(0)
        for _ in range(300):
            asym_mean = rng.uniform(0, 45)
            asyms = list(asym_mean + rng.normal(0, rng.uniform(0, 3), 3))
            devs  = list(np.abs(rng.uniform(0, 40) + rng.normal(0, rng.uniform(0, 8), 3)))
            out = aggregate_frame_results(_frames([max(a, 0.0) for a in asyms], devs))
            if out["status"] == "SUCCESS":
                assert not referral_withheld(out["result"]["urgency_tier"], out["aggregate_confidence"])


# ─────────────────────────────────────────────────────────────
# /analyse and /analyse-batch — single photo
# ─────────────────────────────────────────────────────────────

def _force(monkeypatch, *, confidence, tier):
    """Run the real pipeline on the reference photo, forcing pupil confidence and tier."""
    real_localise = main.localise_pupils

    def localise(**kwargs):
        result = real_localise(**kwargs)
        return dataclasses.replace(result, left_confidence=confidence, right_confidence=confidence)

    def classify(**kwargs):
        return ClassificationResult(
            condition_name="Esotropia", icd10_code="H50.01", urgency_tier=tier,
            referral_recommendation="Refer to ophthalmology within 1 week",
            timeframe="1 week", narrative="", flags=list(kwargs.get("upstream_flags") or []),
        )

    monkeypatch.setattr(main, "localise_pupils", localise)
    monkeypatch.setattr(main, "classify_strabismus", classify)


def _analyse():
    with TestClient(main.app) as client:
        r = client.post(
            "/analyse",
            files={"image": ("ref.jpg", SUCCESS_REFERENCE.read_bytes(), "image/jpeg")},
            data={"patient_name": "Gate Test", "patient_age": "8"},
        )
    assert r.status_code == 200
    return r.json()


def _batch_item():
    img = main.np.array(main.Image.open(SUCCESS_REFERENCE).convert("RGB"))
    return main._process_batch_item(img, 0, "ref", main._batch_calibration())


@needs_reference
class TestSinglePhotoGate:

    def test_analyse_low_confidence_urgent_is_withheld(self, monkeypatch):
        _force(monkeypatch, confidence="LOW", tier="URGENT")
        body = _analyse()
        assert body["status"] == "INCONCLUSIVE"
        assert body["reason"] == "low_confidence_referral_withheld"
        assert body.get("result") is None
        assert "alignment" in body          # the torch-free check still reports

    def test_analyse_confident_urgent_still_refers(self, monkeypatch):
        _force(monkeypatch, confidence="HIGH", tier="URGENT")
        body = _analyse()
        assert body["status"] == "SUCCESS"
        assert body["result"]["urgency_tier"] == "URGENT"

    def test_batch_low_confidence_does_not_flag(self, monkeypatch):
        _force(monkeypatch, confidence="LOW", tier="URGENT")
        item = _batch_item()
        assert item["hirschberg"]["confidence"] == "LOW"
        assert item["referral_flag"] is (item["alignment"]["referral_flag"] is True)

    def test_batch_confident_urgent_flags(self, monkeypatch):
        _force(monkeypatch, confidence="HIGH", tier="URGENT")
        assert _batch_item()["referral_flag"] is True
