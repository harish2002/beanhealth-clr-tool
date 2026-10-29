"""
BeanHealth CLR Tool — Laterality Regression Tests
==================================================

"left" / "right" everywhere in the pipeline are the PATIENT's eyes (OS / OD)
in an unmirrored image, where the patient's right eye sits on the image left
(utils/constants.py). Before this was fixed, the FaceMesh path named eyes by
image side, which also made the head-roll estimate ~180° on every upright
photo; Module 4 then rotated every displacement by 180°, inverting
nasal/temporal and superior/inferior — an inversion that happened to cancel
Module 6's own inverted reflex → condition mapping.

Run:
    cd backend && pytest tests/test_laterality.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.module4_displacement import compute_displacement
from pipeline.module5_asymmetry import compute_asymmetry_and_angle
from pipeline.module6_classify import classify_strabismus

TEST_IMAGES = Path(__file__).parent / "test_images"
SUCCESS_REFERENCE = TEST_IMAGES / "success_reference" / "clr_both_eyes.jpg"


def _chain(left_clr_offset, right_clr_offset, head_roll_deg=0.0):
    """Modules 4 → 6 on synthetic crops: pupils at (50, 40), iris radius 30 px."""
    pupil = (50.0, 40.0)
    d = compute_displacement(
        left_pupil=pupil, right_pupil=pupil,
        left_clr=(pupil[0] + left_clr_offset[0], pupil[1] + left_clr_offset[1]),
        right_clr=(pupil[0] + right_clr_offset[0], pupil[1] + right_clr_offset[1]),
        left_iris_radius=30.0, right_iris_radius=30.0,
        head_roll_deg=head_roll_deg,
    )
    a = compute_asymmetry_and_angle(
        left_displacement_norm=d.left_displacement_norm,
        right_displacement_norm=d.right_displacement_norm,
        left_dx=d.left_dx, left_dy=d.left_dy, right_dx=d.right_dx, right_dy=d.right_dy,
        left_iris_radius=30.0, right_iris_radius=30.0,
    )
    eye = a.dominant_eye
    direction = d.left_direction if eye == "left" else d.right_direction
    c = classify_strabismus(direction, a.severity, a.asymmetry_score)
    return d, a, c


class TestReflexReading:
    """Image geometry → the patient's eye and the standard Hirschberg reading."""

    def test_right_eye_reflex_toward_image_left_is_right_esotropia(self):
        # OD is image-left; its temporal side is image-left (−x).
        d, a, c = _chain(left_clr_offset=(0, 0), right_clr_offset=(-12, 0))
        assert d.right_direction == "temporal"
        assert a.dominant_eye == "right"
        assert c.condition_name == "Esotropia"

    def test_left_eye_reflex_toward_image_right_is_left_esotropia(self):
        # OS is image-right; its temporal side is image-right (+x).
        d, a, c = _chain(left_clr_offset=(12, 0), right_clr_offset=(0, 0))
        assert d.left_direction == "temporal"
        assert a.dominant_eye == "left"
        assert c.condition_name == "Esotropia"

    def test_reflex_toward_the_nose_is_exotropia(self):
        d, a, c = _chain(left_clr_offset=(-12, 0), right_clr_offset=(0, 0))
        assert d.left_direction == "nasal"
        assert c.condition_name == "Exotropia"

    def test_reflex_above_pupil_is_hypotropia(self):
        d, a, c = _chain(left_clr_offset=(0, -12), right_clr_offset=(0, 0))
        assert d.left_direction == "superior"
        assert c.condition_name == "Hypotropia"

    def test_reflex_below_pupil_is_hypertropia(self):
        d, a, c = _chain(left_clr_offset=(0, 0), right_clr_offset=(0, 12))
        assert d.right_direction == "inferior"
        assert a.dominant_eye == "right"
        assert c.condition_name == "Hypertropia"

    def test_small_head_roll_keeps_the_reading(self):
        _, _, c = _chain(left_clr_offset=(12, 0), right_clr_offset=(0, 0), head_roll_deg=8.0)
        assert c.condition_name == "Esotropia"


@pytest.mark.skipif(not SUCCESS_REFERENCE.exists(),
                    reason="Reference image not present (test images are git-ignored)")
class TestRealPhoto:

    @pytest.fixture(scope="class")
    def detection(self):
        from pipeline.module1_detection import detect_and_crop_eyes
        from utils.image_utils import load_image_from_path
        return detect_and_crop_eyes(load_image_from_path(SUCCESS_REFERENCE))

    def test_patient_left_eye_is_on_the_image_right(self, detection):
        assert detection.left_crop_box[0] > detection.right_crop_box[0]

    def test_upright_face_has_near_zero_roll(self, detection):
        """Was ~180° when eyes were named by image side."""
        assert abs(detection.head_roll_deg) < 15

    def test_canthi_belong_to_their_own_eye(self, detection):
        """Inner canthi face each other across the nose; outer ones face away."""
        (l_inner, l_outer), (r_inner, r_outer) = detection.left_eye_corners, detection.right_eye_corners
        assert r_outer[0] < r_inner[0] < l_inner[0] < l_outer[0]
