"""
BeanHealth CLR Tool — Module 4 Test Suite
==========================================

Pure unit tests for displacement measurement.
No real images needed — all tests use synthetic coordinates.

Run:
    cd backend && pytest tests/test_module4.py -v
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.module4_displacement import (
    DisplacementResult,
    _angle_to_cardinal,
    _cardinal_to_anatomical,
    compute_displacement,
    measure_displacement,
)
from utils.constants import (
    DIRECTION_INFERIOR,
    DIRECTION_NASAL,
    DIRECTION_SUPERIOR,
    DIRECTION_TEMPORAL,
)
from utils.exceptions import PipelineError


# ─────────────────────────────────────────────────────────────
# measure_displacement — core maths
# ─────────────────────────────────────────────────────────────

class TestMeasureDisplacement:

    def test_zero_displacement(self):
        """CLR exactly at pupil centre → magnitude = 0, normalised = 0."""
        r = measure_displacement(pupil=(40.0, 40.0), clr=(40.0, 40.0), iris_radius=30.0)
        assert r["magnitude"]  == 0.0
        assert r["normalised"] == 0.0

    def test_known_displacement_horizontal(self):
        """CLR 10px right, iris 40px → normalised = 0.25."""
        r = measure_displacement(pupil=(40.0, 40.0), clr=(50.0, 40.0), iris_radius=40.0)
        assert abs(r["normalised"] - 0.25) < 0.001
        assert abs(r["dx"] - 10.0) < 0.001
        assert r["dy"] == 0.0

    def test_known_displacement_diagonal(self):
        """CLR 3px right + 4px down → magnitude = 5px (3-4-5 triangle)."""
        r = measure_displacement(pupil=(0.0, 0.0), clr=(3.0, 4.0), iris_radius=20.0)
        assert abs(r["magnitude"] - 5.0) < 0.001
        assert abs(r["normalised"] - 0.25) < 0.001

    def test_scale_invariance(self):
        """Same physical displacement at different image scales → same normalised value."""
        r_small = measure_displacement(pupil=(40.0, 40.0), clr=(50.0, 40.0), iris_radius=40.0)
        r_large = measure_displacement(pupil=(80.0, 80.0), clr=(100.0, 80.0), iris_radius=80.0)
        assert abs(r_small["normalised"] - r_large["normalised"]) < 0.0001

    def test_negative_dx(self):
        """CLR to the left of pupil → dx is negative."""
        r = measure_displacement(pupil=(50.0, 40.0), clr=(40.0, 40.0), iris_radius=30.0)
        assert r["dx"] < 0

    def test_negative_dy(self):
        """CLR above pupil → dy is negative."""
        r = measure_displacement(pupil=(50.0, 40.0), clr=(50.0, 30.0), iris_radius=30.0)
        assert r["dy"] < 0

    def test_zero_iris_radius_raises(self):
        """iris_radius = 0 → PipelineError (prevents division by zero)."""
        with pytest.raises(PipelineError):
            measure_displacement(pupil=(40.0, 40.0), clr=(50.0, 40.0), iris_radius=0.0)

    def test_negative_iris_radius_raises(self):
        """iris_radius < 0 → PipelineError."""
        with pytest.raises(PipelineError):
            measure_displacement(pupil=(40.0, 40.0), clr=(50.0, 40.0), iris_radius=-5.0)

    def test_large_displacement_exceeds_1(self):
        """CLR displaced by more than iris radius → normalised > 1.0 (allowed, just large)."""
        r = measure_displacement(pupil=(0.0, 0.0), clr=(50.0, 0.0), iris_radius=30.0)
        assert r["normalised"] > 1.0

    def test_angle_is_radians(self):
        """Angle should be in radians: right displacement → angle ≈ 0."""
        r = measure_displacement(pupil=(40.0, 40.0), clr=(50.0, 40.0), iris_radius=30.0)
        assert abs(r["angle_rad"] - 0.0) < 0.001

    def test_angle_upward_is_negative_pi_over_2(self):
        """CLR above pupil (dy negative) → angle ≈ -π/2."""
        r = measure_displacement(pupil=(40.0, 40.0), clr=(40.0, 30.0), iris_radius=30.0)
        assert abs(r["angle_rad"] - (-math.pi / 2)) < 0.001


# ─────────────────────────────────────────────────────────────
# _angle_to_cardinal
# ─────────────────────────────────────────────────────────────

class TestAngleToCardinal:

    def test_right_direction(self):
        """Angle ≈ 0 → right."""
        assert _angle_to_cardinal(0.0) == "right"

    def test_down_direction(self):
        """Angle ≈ π/2 → down."""
        assert _angle_to_cardinal(math.pi / 2) == "down"

    def test_left_direction_positive(self):
        """Angle ≈ π → left."""
        assert _angle_to_cardinal(math.pi) == "left"

    def test_left_direction_negative(self):
        """Angle ≈ -π → left."""
        assert _angle_to_cardinal(-math.pi) == "left"

    def test_up_direction(self):
        """Angle ≈ -π/2 → up."""
        assert _angle_to_cardinal(-math.pi / 2) == "up"

    def test_diagonal_upper_right_is_right(self):
        """Angle = π/6 (30°, upper-right diagonal) → right."""
        assert _angle_to_cardinal(math.pi / 6) == "right"

    def test_diagonal_lower_right_is_down(self):
        """Angle = π/3 (60°) → down."""
        assert _angle_to_cardinal(math.pi / 3) == "down"


# ─────────────────────────────────────────────────────────────
# _cardinal_to_anatomical — direction labelling
# ─────────────────────────────────────────────────────────────

class TestCardinalToAnatomical:
    """
    In a front-camera (mirrored) image:
        Left eye nasal side  = image RIGHT
        Right eye nasal side = image LEFT
    """

    def test_up_always_superior(self):
        assert _cardinal_to_anatomical("up", "left")  == DIRECTION_SUPERIOR
        assert _cardinal_to_anatomical("up", "right") == DIRECTION_SUPERIOR

    def test_down_always_inferior(self):
        assert _cardinal_to_anatomical("down", "left")  == DIRECTION_INFERIOR
        assert _cardinal_to_anatomical("down", "right") == DIRECTION_INFERIOR

    def test_left_eye_right_is_nasal(self):
        """Front-camera: left eye's nasal side is image-right."""
        assert _cardinal_to_anatomical("right", "left") == DIRECTION_NASAL

    def test_left_eye_left_is_temporal(self):
        assert _cardinal_to_anatomical("left", "left") == DIRECTION_TEMPORAL

    def test_right_eye_left_is_nasal(self):
        """Front-camera: right eye's nasal side is image-left."""
        assert _cardinal_to_anatomical("left", "right") == DIRECTION_NASAL

    def test_right_eye_right_is_temporal(self):
        assert _cardinal_to_anatomical("right", "right") == DIRECTION_TEMPORAL


# ─────────────────────────────────────────────────────────────
# Direction integration — known coordinate cases
# ─────────────────────────────────────────────────────────────

class TestDirectionFromCoordinates:
    """
    Test that the full direction pipeline (measure_displacement → direction label)
    returns the expected anatomical label for all four quadrants.
    """

    @pytest.mark.parametrize("clr_offset,eye,expected_dir", [
        # Front-camera (mirrored) — left eye
        ((+10,   0), "left",  DIRECTION_NASAL),     # CLR right → nasal
        ((-10,   0), "left",  DIRECTION_TEMPORAL),  # CLR left  → temporal
        ((  0, -10), "left",  DIRECTION_SUPERIOR),  # CLR up    → superior
        ((  0, +10), "left",  DIRECTION_INFERIOR),  # CLR down  → inferior
        # Right eye
        ((-10,   0), "right", DIRECTION_NASAL),     # CLR left  → nasal
        ((+10,   0), "right", DIRECTION_TEMPORAL),  # CLR right → temporal
        ((  0, -10), "right", DIRECTION_SUPERIOR),
        ((  0, +10), "right", DIRECTION_INFERIOR),
    ])
    def test_direction_all_quadrants(self, clr_offset, eye, expected_dir):
        pupil = (50.0, 40.0)
        clr   = (pupil[0] + clr_offset[0], pupil[1] + clr_offset[1])
        r = measure_displacement(pupil=pupil, clr=clr, iris_radius=30.0, eye=eye, mirror=True)
        assert r["direction"] == expected_dir, (
            f"clr_offset={clr_offset}, eye={eye}: "
            f"expected {expected_dir}, got {r['direction']}"
        )


# ─────────────────────────────────────────────────────────────
# compute_displacement — both eyes together
# ─────────────────────────────────────────────────────────────

class TestComputeDisplacement:

    def test_returns_displacement_result(self):
        r = compute_displacement(
            left_pupil=(40.0, 40.0), right_pupil=(40.0, 40.0),
            left_clr=(50.0, 40.0),   right_clr=(30.0, 40.0),
            left_iris_radius=30.0,   right_iris_radius=30.0,
        )
        assert isinstance(r, DisplacementResult)

    def test_symmetric_displacement_both_normalised_equal(self):
        """Identical displacement in both eyes → left_norm == right_norm."""
        r = compute_displacement(
            left_pupil=(40.0, 40.0), right_pupil=(40.0, 40.0),
            left_clr=(50.0, 40.0),   right_clr=(50.0, 40.0),
            left_iris_radius=40.0,   right_iris_radius=40.0,
        )
        assert abs(r.left_displacement_norm - r.right_displacement_norm) < 0.001

    def test_zero_displacement_both_eyes(self):
        """CLR at pupil for both eyes → all displacements = 0."""
        r = compute_displacement(
            left_pupil=(40.0, 40.0), right_pupil=(60.0, 40.0),
            left_clr=(40.0, 40.0),   right_clr=(60.0, 40.0),
            left_iris_radius=30.0,   right_iris_radius=30.0,
        )
        assert r.left_displacement_px   == 0.0
        assert r.right_displacement_px  == 0.0
        assert r.left_displacement_norm  == 0.0
        assert r.right_displacement_norm == 0.0

    def test_flags_propagated_from_upstream(self):
        """Upstream flags must appear in the result flags."""
        r = compute_displacement(
            left_pupil=(40.0, 40.0), right_pupil=(40.0, 40.0),
            left_clr=(50.0, 40.0),   right_clr=(50.0, 40.0),
            left_iris_radius=40.0,   right_iris_radius=40.0,
            upstream_flags=["pupil_disagreement_left"],
        )
        assert "pupil_disagreement_left" in r.flags

    def test_large_displacement_flag_added(self):
        """Normalised displacement > 1.0 → large_displacement flag added."""
        r = compute_displacement(
            left_pupil=(0.0, 0.0), right_pupil=(0.0, 0.0),
            left_clr=(50.0, 0.0),  right_clr=(0.0, 0.0),
            left_iris_radius=30.0, right_iris_radius=30.0,
        )
        assert "large_displacement_left" in r.flags

    def test_different_iris_radii_normalised_correctly(self):
        """Different iris radii for each eye → independent normalisation."""
        r = compute_displacement(
            left_pupil=(0.0, 0.0),  right_pupil=(0.0, 0.0),
            left_clr=(10.0, 0.0),   right_clr=(10.0, 0.0),
            left_iris_radius=20.0,  right_iris_radius=40.0,
        )
        # Left: 10/20 = 0.5, Right: 10/40 = 0.25
        assert abs(r.left_displacement_norm  - 0.5)  < 0.001
        assert abs(r.right_displacement_norm - 0.25) < 0.001

    def test_all_fields_present(self):
        """DisplacementResult must have all required fields."""
        r = compute_displacement(
            left_pupil=(40.0, 40.0), right_pupil=(40.0, 40.0),
            left_clr=(50.0, 40.0),   right_clr=(50.0, 40.0),
            left_iris_radius=30.0,   right_iris_radius=30.0,
        )
        for field_name in [
            "left_dx", "left_dy", "right_dx", "right_dy",
            "left_displacement_px", "right_displacement_px",
            "left_displacement_norm", "right_displacement_norm",
            "left_direction", "right_direction",
            "left_angle_rad", "right_angle_rad",
            "flags",
        ]:
            assert hasattr(r, field_name), f"Missing field: {field_name}"
