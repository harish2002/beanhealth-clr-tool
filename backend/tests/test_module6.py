"""
BeanHealth CLR Tool — Module 6 Test Suite
==========================================

Pure unit tests for clinical classification.
No images needed — all tests use synthetic direction + severity inputs.

Run:
    cd backend && pytest tests/test_module6.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.module6_classify import (
    ClassificationResult,
    classify,
    classify_strabismus,
)
from utils.constants import (
    DIRECTION_INFERIOR,
    DIRECTION_NASAL,
    DIRECTION_SUPERIOR,
    DIRECTION_TEMPORAL,
    SEVERITY_MILD,
    SEVERITY_MODERATE,
    SEVERITY_NORMAL,
    SEVERITY_SEVERE,
    URGENCY_MONITOR,
    URGENCY_NORMAL,
    URGENCY_ROUTINE,
    URGENCY_URGENT,
)


# ─────────────────────────────────────────────────────────────
# classify — core direction → condition mapping
# ─────────────────────────────────────────────────────────────

class TestClassifyConditionMapping:

    @pytest.mark.parametrize("direction,expected_condition,expected_icd", [
        (DIRECTION_NASAL,    "Esotropia",   "H50.01"),
        (DIRECTION_TEMPORAL, "Exotropia",   "H50.11"),
        (DIRECTION_SUPERIOR, "Hypertropia", "H50.21"),
        (DIRECTION_INFERIOR, "Hypotropia",  "H50.22"),
    ])
    def test_condition_and_icd_mapping(self, direction, expected_condition, expected_icd):
        """Each direction maps to the correct clinical condition and ICD-10 code."""
        r = classify(direction=direction, severity=SEVERITY_MODERATE)
        assert r["condition_name"] == expected_condition, (
            f"direction={direction}: expected {expected_condition}, got {r['condition_name']}"
        )
        assert r["icd10_code"] == expected_icd, (
            f"direction={direction}: expected {expected_icd}, got {r['icd10_code']}"
        )

    def test_normal_severity_is_orthophoria_regardless_of_direction(self):
        """NORMAL severity → Orthophoria (H50.40) no matter what direction is given."""
        for direction in [DIRECTION_NASAL, DIRECTION_TEMPORAL, DIRECTION_SUPERIOR, DIRECTION_INFERIOR]:
            r = classify(direction=direction, severity=SEVERITY_NORMAL)
            assert r["condition_name"] == "Orthophoria", (
                f"direction={direction}: expected Orthophoria at NORMAL severity, got {r['condition_name']}"
            )
            assert r["icd10_code"] == "H50.40"

    def test_unknown_direction_fallback(self):
        """Unknown direction → fallback condition 'Strabismus, unspecified'."""
        r = classify(direction="unknown_direction", severity=SEVERITY_MILD)
        assert "strabismus" in r["condition_name"].lower() or r["condition_name"] == "Strabismus, unspecified"


# ─────────────────────────────────────────────────────────────
# classify — severity → urgency mapping
# ─────────────────────────────────────────────────────────────

class TestClassifyUrgencyMapping:

    @pytest.mark.parametrize("severity,expected_urgency", [
        (SEVERITY_SEVERE,   URGENCY_URGENT),
        (SEVERITY_MODERATE, URGENCY_ROUTINE),
        (SEVERITY_MILD,     URGENCY_MONITOR),
        (SEVERITY_NORMAL,   URGENCY_NORMAL),
    ])
    def test_urgency_tier_mapping(self, severity, expected_urgency):
        """Each severity tier maps to the correct urgency tier."""
        r = classify(direction=DIRECTION_NASAL, severity=severity)
        assert r["urgency_tier"] == expected_urgency, (
            f"severity={severity}: expected {expected_urgency}, got {r['urgency_tier']}"
        )

    def test_severe_referral_text(self):
        """SEVERE → referral text contains '1 week'."""
        r = classify(direction=DIRECTION_NASAL, severity=SEVERITY_SEVERE)
        assert "1 week" in r["referral_recommendation"]
        assert r["timeframe"] == "1 week"

    def test_moderate_referral_text(self):
        """MODERATE → referral text contains '4 weeks'."""
        r = classify(direction=DIRECTION_NASAL, severity=SEVERITY_MODERATE)
        assert "4 weeks" in r["referral_recommendation"]
        assert r["timeframe"] == "4 weeks"

    def test_mild_referral_text(self):
        """MILD → monitor / re-screen in 3 months."""
        r = classify(direction=DIRECTION_NASAL, severity=SEVERITY_MILD)
        assert "3 months" in r["referral_recommendation"]
        assert r["timeframe"] == "3 months"

    def test_normal_no_referral(self):
        """NORMAL → 'No referral required', timeframe = 'N/A'."""
        r = classify(direction=DIRECTION_NASAL, severity=SEVERITY_NORMAL)
        assert "No referral" in r["referral_recommendation"]
        assert r["timeframe"] == "N/A"


# ─────────────────────────────────────────────────────────────
# classify — narrative generation
# ─────────────────────────────────────────────────────────────

class TestClassifyNarrative:

    def test_narrative_is_non_empty_string(self):
        """Narrative should always be a non-empty string."""
        for severity in [SEVERITY_NORMAL, SEVERITY_MILD, SEVERITY_MODERATE, SEVERITY_SEVERE]:
            r = classify(direction=DIRECTION_NASAL, severity=severity)
            assert isinstance(r["narrative"], str)
            assert len(r["narrative"]) > 20

    def test_urgent_narrative_contains_condition(self):
        """URGENT narrative should mention the condition name."""
        r = classify(direction=DIRECTION_NASAL, severity=SEVERITY_SEVERE)
        assert "Esotropia" in r["narrative"]

    def test_routine_narrative_contains_condition(self):
        """ROUTINE narrative should mention the condition name."""
        r = classify(direction=DIRECTION_TEMPORAL, severity=SEVERITY_MODERATE)
        assert "Exotropia" in r["narrative"]

    def test_normal_narrative_mentions_no_asymmetry(self):
        """NORMAL narrative should indicate normal alignment."""
        r = classify(direction=DIRECTION_NASAL, severity=SEVERITY_NORMAL)
        assert "normal" in r["narrative"].lower() or "no significant" in r["narrative"].lower()


# ─────────────────────────────────────────────────────────────
# classify — return schema completeness
# ─────────────────────────────────────────────────────────────

class TestClassifyReturnSchema:

    def test_all_keys_present(self):
        """classify() must return all required keys."""
        r = classify(direction=DIRECTION_NASAL, severity=SEVERITY_MODERATE)
        required_keys = [
            "condition_name", "icd10_code", "urgency_tier",
            "referral_recommendation", "timeframe", "narrative",
        ]
        for key in required_keys:
            assert key in r, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────
# classify_strabismus — full public API
# ─────────────────────────────────────────────────────────────

class TestClassifyStrabismus:

    def test_returns_classification_result(self):
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_MODERATE,
            asymmetry_score=0.3,
        )
        assert isinstance(r, ClassificationResult)

    def test_flags_propagated_from_upstream(self):
        """Upstream flags must appear in the result flags."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_MODERATE,
            asymmetry_score=0.3,
            upstream_flags=["pupil_disagreement_left", "large_displacement_right"],
        )
        assert "pupil_disagreement_left"     in r.flags
        assert "large_displacement_right"    in r.flags

    def test_borderline_asymmetry_flag(self):
        """NORMAL severity but asymmetry_score > 0.05 → 'borderline_asymmetry' flag."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_NORMAL,
            asymmetry_score=0.06,
        )
        assert "borderline_asymmetry" in r.flags

    def test_no_borderline_flag_when_asymmetry_within_normal(self):
        """NORMAL severity, asymmetry_score ≤ 0.05 → no borderline flag."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_NORMAL,
            asymmetry_score=0.03,
        )
        assert "borderline_asymmetry" not in r.flags

    def test_no_borderline_flag_when_not_normal_severity(self):
        """Borderline flag should only apply when severity is NORMAL."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_MILD,
            asymmetry_score=0.1,
        )
        assert "borderline_asymmetry" not in r.flags

    def test_all_fields_present_in_result(self):
        """ClassificationResult must have all required fields."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_TEMPORAL,
            severity=SEVERITY_MILD,
            asymmetry_score=0.15,
        )
        for field_name in [
            "condition_name", "icd10_code", "urgency_tier",
            "referral_recommendation", "timeframe", "narrative", "flags",
        ]:
            assert hasattr(r, field_name), f"Missing field: {field_name}"

    def test_esotropia_full_pipeline(self):
        """End-to-end: nasal + SEVERE → Esotropia, H50.01, URGENT."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_SEVERE,
            asymmetry_score=0.8,
        )
        assert r.condition_name == "Esotropia"
        assert r.icd10_code     == "H50.01"
        assert r.urgency_tier   == URGENCY_URGENT

    def test_exotropia_full_pipeline(self):
        """End-to-end: temporal + MODERATE → Exotropia, H50.11, ROUTINE."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_TEMPORAL,
            severity=SEVERITY_MODERATE,
            asymmetry_score=0.4,
        )
        assert r.condition_name == "Exotropia"
        assert r.icd10_code     == "H50.11"
        assert r.urgency_tier   == URGENCY_ROUTINE

    def test_hypertropia_full_pipeline(self):
        """End-to-end: superior + MILD → Hypertropia, H50.21, MONITOR."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_SUPERIOR,
            severity=SEVERITY_MILD,
            asymmetry_score=0.1,
        )
        assert r.condition_name == "Hypertropia"
        assert r.icd10_code     == "H50.21"
        assert r.urgency_tier   == URGENCY_MONITOR

    def test_hypotropia_full_pipeline(self):
        """End-to-end: inferior + MILD → Hypotropia, H50.22, MONITOR."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_INFERIOR,
            severity=SEVERITY_MILD,
            asymmetry_score=0.1,
        )
        assert r.condition_name == "Hypotropia"
        assert r.icd10_code     == "H50.22"
        assert r.urgency_tier   == URGENCY_MONITOR

    def test_orthophoria_full_pipeline(self):
        """End-to-end: any direction + NORMAL → Orthophoria, H50.40, NORMAL."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_NORMAL,
            asymmetry_score=0.02,
        )
        assert r.condition_name == "Orthophoria"
        assert r.icd10_code     == "H50.40"
        assert r.urgency_tier   == URGENCY_NORMAL

    def test_flags_is_list(self):
        """flags field should always be a list, even with no upstream flags."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_MODERATE,
            asymmetry_score=0.3,
        )
        assert isinstance(r.flags, list)


# ─────────────────────────────────────────────────────────────
# Direction reliability — suppression logic
# ─────────────────────────────────────────────────────────────

class TestDirectionReliability:
    """
    When pupil localisation disagrees (pupil_disagreement flag) at low severity,
    or when asymmetry is too low to trust the direction, Module 6 must fall back
    to "Strabismus, Unspecified" (H50.9) and add the direction_unreliable flag.

    CRITICAL: urgency_tier must NEVER change — only the condition label changes.
    """

    def test_pupil_disagreement_plus_mild_suppresses_direction(self):
        """pupil_disagreement_left flag + MILD severity → direction suppressed."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_MILD,
            asymmetry_score=0.15,
            upstream_flags=["pupil_disagreement_left"],
        )
        assert r.condition_name == "Strabismus, Unspecified"
        assert r.icd10_code     == "H50.9"
        assert "direction_unreliable" in r.flags

    def test_urgency_unchanged_when_direction_suppressed(self):
        """MONITOR urgency must be preserved even when direction is suppressed."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_TEMPORAL,
            severity=SEVERITY_MILD,
            asymmetry_score=0.15,
            upstream_flags=["pupil_disagreement_right"],
        )
        assert r.urgency_tier == URGENCY_MONITOR
        assert r.condition_name == "Strabismus, Unspecified"

    def test_low_asymmetry_suppresses_direction(self):
        """asymmetry_score < 0.05 → direction suppressed regardless of flags."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_MILD,
            asymmetry_score=0.016,   # from test01 PDF
        )
        assert r.condition_name == "Strabismus, Unspecified"
        assert r.icd10_code     == "H50.9"
        assert "direction_unreliable" in r.flags

    def test_low_asymmetry_exact_boundary_suppressed(self):
        """asymmetry_score exactly 0.04 (< 0.05) → suppressed."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_TEMPORAL,
            severity=SEVERITY_MILD,
            asymmetry_score=0.04,
        )
        assert r.condition_name == "Strabismus, Unspecified"

    def test_asymmetry_above_threshold_not_suppressed(self):
        """asymmetry_score >= 0.05, no disagreement → direction preserved."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_MILD,
            asymmetry_score=0.12,
        )
        assert r.condition_name == "Esotropia"
        assert r.icd10_code     == "H50.01"
        assert "direction_unreliable" not in r.flags

    def test_pupil_disagreement_moderate_not_suppressed(self):
        """pupil_disagreement + MODERATE severity → direction NOT suppressed.
        At 15-30°, displacement is large enough that direction is reliable."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_MODERATE,
            asymmetry_score=0.3,
            upstream_flags=["pupil_disagreement_left"],
        )
        assert r.condition_name == "Esotropia"
        assert "direction_unreliable" not in r.flags

    def test_pupil_disagreement_severe_not_suppressed(self):
        """pupil_disagreement + SEVERE severity → direction NOT suppressed."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_TEMPORAL,
            severity=SEVERITY_SEVERE,
            asymmetry_score=0.8,
            upstream_flags=["pupil_disagreement_right"],
        )
        assert r.condition_name == "Exotropia"
        assert "direction_unreliable" not in r.flags

    def test_normal_severity_not_affected(self):
        """NORMAL severity → already Orthophoria; suppression does not apply."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_NORMAL,
            asymmetry_score=0.01,
            upstream_flags=["pupil_disagreement_left"],
        )
        assert r.condition_name == "Orthophoria"
        assert r.icd10_code     == "H50.40"
        assert "direction_unreliable" not in r.flags

    def test_both_triggers_simultaneously(self):
        """Both triggers active → direction suppressed, flag added once."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_SUPERIOR,
            severity=SEVERITY_MILD,
            asymmetry_score=0.02,
            upstream_flags=["pupil_disagreement_left"],
        )
        assert r.condition_name == "Strabismus, Unspecified"
        assert r.flags.count("direction_unreliable") == 1

    def test_test01_scenario_suppressed(self):
        """Reproduce test01 PDF: nasal+MILD+asymmetry=0.016 → was Esotropia, now Unspecified."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_MILD,
            asymmetry_score=0.016,
            upstream_flags=["pupil_disagreement_left"],
        )
        assert r.condition_name == "Strabismus, Unspecified"
        assert r.urgency_tier   == URGENCY_MONITOR   # safety unchanged

    def test_test02_scenario_suppressed(self):
        """Reproduce test02 PDF: temporal+MILD+asymmetry=0.093+disagreement → was Exotropia, now Unspecified."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_TEMPORAL,
            severity=SEVERITY_MILD,
            asymmetry_score=0.093,
            upstream_flags=["pupil_disagreement_left"],
        )
        assert r.condition_name == "Strabismus, Unspecified"
        assert r.urgency_tier   == URGENCY_MONITOR   # safety unchanged

    def test_referral_recommendation_preserved(self):
        """Referral recommendation must survive direction suppression unchanged."""
        r = classify_strabismus(
            dominant_direction=DIRECTION_NASAL,
            severity=SEVERITY_MILD,
            asymmetry_score=0.016,
        )
        assert "3 months" in r.referral_recommendation
        assert r.timeframe == "3 months"
