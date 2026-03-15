"""
BeanHealth CLR Tool — Module 5 Test Suite
==========================================

Pure unit tests for asymmetry score and Hirschberg angle computation.
No real images needed — all tests use synthetic normalised displacement values.

Run:
    cd backend && pytest tests/test_module5.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.module5_asymmetry import (
    AsymmetryResult,
    compute_angle,
    compute_angle_severity,
    compute_asymmetry,
    compute_asymmetry_and_angle,
)
from utils.constants import (
    HIRSCHBERG_CONSTANT,
    IRIS_RADIUS_MM,
    SEVERITY_MILD,
    SEVERITY_MODERATE,
    SEVERITY_NORMAL,
    SEVERITY_SEVERE,
)


# ─────────────────────────────────────────────────────────────
# compute_asymmetry — asymmetry score and dominant eye
# ─────────────────────────────────────────────────────────────

class TestComputeAsymmetry:

    def test_symmetric_eyes_score_zero(self):
        """Identical displacements → asymmetry_score == 0."""
        r = compute_asymmetry(0.2, 0.2)
        assert r["asymmetry_score"] == 0.0

    def test_asymmetric_eyes_correct_score(self):
        """Left=0.5, right=0.2 → score = 0.3."""
        r = compute_asymmetry(0.5, 0.2)
        assert abs(r["asymmetry_score"] - 0.3) < 0.0001

    def test_dominant_eye_left(self):
        """Larger left displacement → dominant_eye = 'left'."""
        r = compute_asymmetry(0.6, 0.1)
        assert r["dominant_eye"] == "left"
        assert abs(r["dominant_norm"] - 0.6) < 0.0001

    def test_dominant_eye_right(self):
        """Larger right displacement → dominant_eye = 'right'."""
        r = compute_asymmetry(0.1, 0.6)
        assert r["dominant_eye"] == "right"
        assert abs(r["dominant_norm"] - 0.6) < 0.0001

    def test_dominant_eye_equal(self):
        """Equal displacements → dominant_eye = 'equal'."""
        r = compute_asymmetry(0.3, 0.3)
        assert r["dominant_eye"] == "equal"

    def test_zero_both_eyes(self):
        """Both eyes at zero → score = 0, dominant = 'equal'."""
        r = compute_asymmetry(0.0, 0.0)
        assert r["asymmetry_score"] == 0.0
        assert r["dominant_eye"] == "equal"
        assert r["dominant_norm"] == 0.0

    def test_dominant_norm_is_larger_value(self):
        """dominant_norm is always the larger of the two values."""
        r = compute_asymmetry(0.3, 0.7)
        assert abs(r["dominant_norm"] - 0.7) < 0.0001

    def test_large_asymmetry(self):
        """One eye at 1.0, other at 0.0 → score = 1.0."""
        r = compute_asymmetry(1.0, 0.0)
        assert abs(r["asymmetry_score"] - 1.0) < 0.0001

    def test_asymmetry_is_absolute(self):
        """Asymmetry score must be non-negative regardless of order."""
        r1 = compute_asymmetry(0.2, 0.5)
        r2 = compute_asymmetry(0.5, 0.2)
        assert r1["asymmetry_score"] == r2["asymmetry_score"]


# ─────────────────────────────────────────────────────────────
# compute_angle — Hirschberg formula
# ─────────────────────────────────────────────────────────────

class TestComputeAngle:

    def test_zero_displacement_zero_degrees(self):
        """No displacement → 0 degrees."""
        r = compute_angle(0.0)
        assert r["deviation_degrees"] == 0.0
        assert r["deviation_mm"] == 0.0

    def test_hirschberg_1mm_is_7_degrees(self):
        """1mm displacement → 7°. Use norm = 1/IRIS_RADIUS_MM to get exactly 1mm."""
        norm = 1.0 / IRIS_RADIUS_MM   # exactly 1mm displacement
        r = compute_angle(norm)
        assert abs(r["deviation_degrees"] - 7.0) < 0.01
        assert abs(r["deviation_mm"]      - 1.0) < 0.0001

    def test_formula_is_linear(self):
        """Doubling the normalised displacement doubles the angle."""
        r1 = compute_angle(0.2)
        r2 = compute_angle(0.4)
        assert abs(r2["deviation_degrees"] - 2 * r1["deviation_degrees"]) < 0.0001

    def test_known_value_half_iris(self):
        """Displacement = 0.5 iris radius → 0.5 × 5.75mm × 7°/mm = 20.125°."""
        r = compute_angle(0.5)
        expected = 0.5 * IRIS_RADIUS_MM * HIRSCHBERG_CONSTANT
        assert abs(r["deviation_degrees"] - expected) < 0.001

    def test_deviation_mm_correct(self):
        """Deviation in mm = norm × IRIS_RADIUS_MM."""
        r = compute_angle(0.3)
        assert abs(r["deviation_mm"] - 0.3 * IRIS_RADIUS_MM) < 0.0001

    def test_full_iris_displacement(self):
        """Norm = 1.0 → displacement_mm = 5.75mm → 40.25°."""
        r = compute_angle(1.0)
        assert abs(r["deviation_mm"]      - IRIS_RADIUS_MM)                    < 0.0001
        assert abs(r["deviation_degrees"] - IRIS_RADIUS_MM * HIRSCHBERG_CONSTANT) < 0.001


# ─────────────────────────────────────────────────────────────
# compute_angle_severity — severity tier thresholds
# ─────────────────────────────────────────────────────────────

class TestComputeAngleSeverity:

    @pytest.mark.parametrize("angle,expected_severity", [
        (0.0,  SEVERITY_NORMAL),
        (4.99, SEVERITY_NORMAL),
        (5.0,  SEVERITY_MILD),
        (7.0,  SEVERITY_MILD),
        (14.99, SEVERITY_MILD),
        (15.0,  SEVERITY_MODERATE),
        (20.0,  SEVERITY_MODERATE),
        (29.99, SEVERITY_MODERATE),
        (30.0,  SEVERITY_SEVERE),
        (45.0,  SEVERITY_SEVERE),
    ])
    def test_severity_thresholds(self, angle, expected_severity):
        r = compute_angle_severity(angle)
        assert r["severity"] == expected_severity, (
            f"angle={angle}° → expected {expected_severity}, got {r['severity']}"
        )

    def test_boundary_5_degrees_is_mild(self):
        """Exactly 5° is the first MILD case (< 5 is NORMAL)."""
        assert compute_angle_severity(5.0)["severity"] == SEVERITY_MILD

    def test_boundary_15_degrees_is_moderate(self):
        """Exactly 15° is the first MODERATE case."""
        assert compute_angle_severity(15.0)["severity"] == SEVERITY_MODERATE

    def test_boundary_30_degrees_is_severe(self):
        """Exactly 30° is the first SEVERE case."""
        assert compute_angle_severity(30.0)["severity"] == SEVERITY_SEVERE


# ─────────────────────────────────────────────────────────────
# compute_asymmetry_and_angle — full Module 5 integration
# ─────────────────────────────────────────────────────────────

class TestComputeAsymmetryAndAngle:

    def test_returns_asymmetry_result(self):
        r = compute_asymmetry_and_angle(0.2, 0.1)
        assert isinstance(r, AsymmetryResult)

    def test_symmetric_normal_eyes(self):
        """Both eyes at 0.05 → very low displacement, NORMAL severity."""
        r = compute_asymmetry_and_angle(0.05, 0.05)
        assert r.asymmetry_score == 0.0
        assert r.severity == SEVERITY_NORMAL

    def test_one_squinting_eye_mild(self):
        """Left eye displaced at ~0.12 (≈ 5°), right near 0 → MILD."""
        # 5° / (7 × 5.75) = 5 / 40.25 ≈ 0.1242
        norm = 5.5 / (HIRSCHBERG_CONSTANT * IRIS_RADIUS_MM)
        r = compute_asymmetry_and_angle(norm, 0.0)
        assert r.severity == SEVERITY_MILD
        assert r.dominant_eye == "left"

    def test_one_squinting_eye_moderate(self):
        """Dominant eye at ≈20° → MODERATE."""
        norm = 20.0 / (HIRSCHBERG_CONSTANT * IRIS_RADIUS_MM)
        r = compute_asymmetry_and_angle(norm, 0.0)
        assert r.severity == SEVERITY_MODERATE

    def test_one_squinting_eye_severe(self):
        """Dominant eye at ≈35° → SEVERE."""
        norm = 35.0 / (HIRSCHBERG_CONSTANT * IRIS_RADIUS_MM)
        r = compute_asymmetry_and_angle(norm, 0.0)
        assert r.severity == SEVERITY_SEVERE

    def test_flags_propagated_from_upstream(self):
        """Upstream flags must appear in the result flags."""
        r = compute_asymmetry_and_angle(
            0.2, 0.1,
            upstream_flags=["pupil_disagreement_left"],
        )
        assert "pupil_disagreement_left" in r.flags

    def test_very_low_displacement_flag(self):
        """Both eyes < 0.05 → 'very_low_displacement_both' flag added."""
        r = compute_asymmetry_and_angle(0.02, 0.03)
        assert "very_low_displacement_both" in r.flags

    def test_dominant_eye_in_result(self):
        """dominant_eye field reflects which eye has larger displacement."""
        r = compute_asymmetry_and_angle(0.1, 0.4)
        assert r.dominant_eye == "right"

    def test_all_fields_present(self):
        """AsymmetryResult must have all required fields."""
        r = compute_asymmetry_and_angle(0.3, 0.1)
        for field_name in [
            "asymmetry_score", "dominant_eye", "deviation_degrees",
            "deviation_mm", "severity", "flags",
        ]:
            assert hasattr(r, field_name), f"Missing field: {field_name}"

    def test_deviation_degrees_from_dominant_eye(self):
        """Deviation angle is computed from the dominant (larger) eye."""
        left_norm  = 0.3
        right_norm = 0.1
        r = compute_asymmetry_and_angle(left_norm, right_norm)
        expected_deg = left_norm * IRIS_RADIUS_MM * HIRSCHBERG_CONSTANT
        assert abs(r.deviation_degrees - expected_deg) < 0.001

    def test_equal_eyes_uses_either(self):
        """Equal displacements → result is still computed without error."""
        r = compute_asymmetry_and_angle(0.25, 0.25)
        expected_deg = 0.25 * IRIS_RADIUS_MM * HIRSCHBERG_CONSTANT
        assert abs(r.deviation_degrees - expected_deg) < 0.001
        assert r.dominant_eye == "equal"
