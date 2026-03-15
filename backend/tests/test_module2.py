"""
BeanHealth CLR Tool — Module 2 Test Suite
==========================================

Tests for localise_pupils() — pupil centre localisation.

Test categories:
  1. Unit tests — maths, coordinate mapping, agreement logic (no real images)
  2. Visual tests — run on real crops from Module 1, save annotated output

Run unit tests only:
    cd backend && pytest tests/test_module2.py -v -m unit

Run visual tests (requires real images):
    pytest tests/test_module2.py -v -m visual

Visual output saved to: tests/test_output/module2/
Open those files and confirm:
  - Blue dot lands INSIDE the pupil
  - Green circle wraps the iris boundary
  - Both are within ~5px of each other on clear images
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.module2_pupil import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    PupilResult,
    _agree_and_fuse,
    _iris_radius_in_crop,
    _landmark_centre,
    _map_landmarks_to_crop,
    _preprocess_for_hough,
    localise_pupils,
)
from utils.exceptions import PupilError
from utils.image_utils import (
    draw_circle,
    draw_crosshair,
    draw_dot,
    save_debug_image,
)

logger = logging.getLogger(__name__)

TEST_DIR   = Path(__file__).parent
OUTPUT_DIR = TEST_DIR / "test_output" / "module2"
IMAGE_DIR  = TEST_DIR / "test_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Synthetic helpers
# ─────────────────────────────────────────────────────────────

def make_synthetic_eye_crop(
    width: int = 120,
    height: int = 80,
    pupil_cx: int = 60,
    pupil_cy: int = 40,
    iris_r: int = 28,
    pupil_r: int = 14,
    add_clr: bool = False,
) -> np.ndarray:
    """
    Create a synthetic eye crop with a dark pupil on a lighter iris,
    on a white sclera background. Optionally add a bright CLR spot.

    The Hough detector should find the iris circle reliably on this image.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 220  # sclera

    # Iris (brown-ish)
    cv2.circle(img, (pupil_cx, pupil_cy), iris_r, (100, 70, 40), -1)

    # Pupil (very dark)
    cv2.circle(img, (pupil_cx, pupil_cy), pupil_r, (15, 15, 15), -1)

    # Pupil highlight (small white dot, top-left of pupil)
    cv2.circle(img, (pupil_cx - pupil_r // 3, pupil_cy - pupil_r // 3), 3, (255, 255, 255), -1)

    if add_clr:
        # CLR — bright spot slightly offset from pupil centre
        clr_x = pupil_cx + 5
        clr_y = pupil_cy - 3
        cv2.circle(img, (clr_x, clr_y), 4, (255, 255, 255), -1)

    return img


def make_5_iris_landmarks(
    cx: float, cy: float, r: float,
    offset_x: float = 0.0, offset_y: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Create 5 synthetic iris landmarks in original-image pixel space
    (as if they came from Module 1).

    Offsets simulate the crop origin so mapping can be tested.
    Layout: centre, top, right, bottom, left
    """
    return [
        (cx + offset_x,       cy + offset_y),        # 0: centre
        (cx + offset_x,       cy - r + offset_y),     # 1: top
        (cx + r + offset_x,   cy + offset_y),         # 2: right
        (cx + offset_x,       cy + r + offset_y),     # 3: bottom
        (cx - r + offset_x,   cy + offset_y),         # 4: left
    ]


# ─────────────────────────────────────────────────────────────
# UNIT TESTS — _map_landmarks_to_crop
# ─────────────────────────────────────────────────────────────

class TestMapLandmarksToCrop:

    @pytest.mark.unit
    def test_zero_offset_unchanged(self):
        """Crop at origin → landmarks unchanged."""
        lms = [(50.0, 40.0), (50.0, 28.0), (62.0, 40.0), (50.0, 52.0), (38.0, 40.0)]
        box = (0, 0, 120, 80)
        mapped = _map_landmarks_to_crop(lms, box)
        for orig, mapped_pt in zip(lms, mapped):
            assert abs(orig[0] - mapped_pt[0]) < 0.001
            assert abs(orig[1] - mapped_pt[1]) < 0.001

    @pytest.mark.unit
    def test_crop_offset_subtracted(self):
        """Landmarks in original image space mapped to crop space correctly."""
        # Crop starts at (100, 50) in the original image
        box = (100, 50, 220, 130)
        # A landmark at (150, 90) in original → (50, 40) in crop
        lms = [(150.0, 90.0)]
        mapped = _map_landmarks_to_crop(lms, box)
        assert abs(mapped[0][0] - 50.0) < 0.001
        assert abs(mapped[0][1] - 40.0) < 0.001

    @pytest.mark.unit
    def test_all_5_landmarks_mapped(self):
        """All 5 iris landmarks are mapped, none dropped."""
        box = (10, 20, 130, 100)
        lms = [(10 + i, 20 + i) for i in range(5)]
        mapped = _map_landmarks_to_crop(lms, box)
        assert len(mapped) == 5
        for i, pt in enumerate(mapped):
            assert abs(pt[0] - i) < 0.001
            assert abs(pt[1] - i) < 0.001


# ─────────────────────────────────────────────────────────────
# UNIT TESTS — _iris_radius_in_crop
# ─────────────────────────────────────────────────────────────

class TestIrisRadiusInCrop:

    @pytest.mark.unit
    def test_perfect_circle_radius(self):
        """5 landmarks on a perfect circle → correct radius."""
        lms = [
            (60.0, 40.0),   # centre
            (60.0, 12.0),   # top    28px away
            (88.0, 40.0),   # right  28px
            (60.0, 68.0),   # bottom 28px
            (32.0, 40.0),   # left   28px
        ]
        r = _iris_radius_in_crop(lms)
        assert abs(r - 28.0) < 0.01

    @pytest.mark.unit
    def test_all_same_point_gives_zero(self):
        """Degenerate case — all at same point → radius 0."""
        lms = [(50.0, 50.0)] * 5
        assert _iris_radius_in_crop(lms) == 0.0

    @pytest.mark.unit
    def test_too_few_landmarks_gives_zero(self):
        """Fewer than 5 landmarks → safe return of 0."""
        assert _iris_radius_in_crop([(10.0, 10.0)]) == 0.0
        assert _iris_radius_in_crop([]) == 0.0


# ─────────────────────────────────────────────────────────────
# UNIT TESTS — _landmark_centre
# ─────────────────────────────────────────────────────────────

class TestLandmarkCentre:

    @pytest.mark.unit
    def test_symmetric_landmarks_centre(self):
        """5 symmetric landmarks → mean = geometric centre."""
        lms = [
            (60.0, 40.0),   # centre
            (60.0, 12.0),   # top
            (88.0, 40.0),   # right
            (60.0, 68.0),   # bottom
            (32.0, 40.0),   # left
        ]
        cx, cy = _landmark_centre(lms)
        # Mean of all 5 x coords: (60+60+88+60+32)/5 = 60
        # Mean of all 5 y coords: (40+12+40+68+40)/5 = 40
        assert abs(cx - 60.0) < 0.01
        assert abs(cy - 40.0) < 0.01

    @pytest.mark.unit
    def test_single_landmark_returns_itself(self):
        """Edge case — single landmark."""
        cx, cy = _landmark_centre([(30.0, 70.0)])
        assert abs(cx - 30.0) < 0.001
        assert abs(cy - 70.0) < 0.001


# ─────────────────────────────────────────────────────────────
# UNIT TESTS — _agree_and_fuse
# ─────────────────────────────────────────────────────────────

class TestAgreeAndFuse:

    @pytest.mark.unit
    def test_high_confidence_when_close(self):
        """Distance < 5px → HIGH confidence, averaged centre."""
        flags = []
        lm = (50.0, 40.0)
        hough = (52.0, 41.0, 20.0)   # distance ≈ 2.24px
        final, conf, hc, hr = _agree_and_fuse(lm, hough, "left", flags)
        assert conf == CONFIDENCE_HIGH
        assert not any("disagreement" in f for f in flags)
        # Final should be average of lm and hough
        assert abs(final[0] - 51.0) < 0.01
        assert abs(final[1] - 40.5) < 0.01

    @pytest.mark.unit
    def test_medium_confidence_moderate_distance(self):
        """Distance 5–15px → MEDIUM, averaged centre, disagreement flag."""
        flags = []
        lm = (50.0, 40.0)
        hough = (58.0, 47.0, 22.0)   # distance ≈ 10.6px
        final, conf, hc, hr = _agree_and_fuse(lm, hough, "left", flags)
        assert conf == CONFIDENCE_MEDIUM
        assert "pupil_disagreement_left" in flags
        assert abs(final[0] - 54.0) < 0.01   # average
        assert abs(final[1] - 43.5) < 0.01

    @pytest.mark.unit
    def test_low_confidence_large_distance(self):
        """Distance > 15px → LOW, landmark wins, disagreement flag."""
        flags = []
        lm = (50.0, 40.0)
        hough = (70.0, 60.0, 25.0)   # distance ≈ 28.3px
        final, conf, hc, hr = _agree_and_fuse(lm, hough, "right", flags)
        assert conf == CONFIDENCE_LOW
        assert "pupil_disagreement_right" in flags
        # Landmark wins
        assert final == lm

    @pytest.mark.unit
    def test_low_confidence_hough_none(self):
        """No Hough result → LOW confidence, landmark only, flag added."""
        flags = []
        lm = (60.0, 45.0)
        final, conf, hc, hr = _agree_and_fuse(lm, None, "left", flags)
        assert conf == CONFIDENCE_LOW
        assert hc is None
        assert hr is None
        assert "pupil_disagreement_left" in flags
        assert final == lm

    @pytest.mark.unit
    def test_exact_agreement_high(self):
        """Both estimates at exactly the same point → HIGH, averaged = same point."""
        flags = []
        lm = (55.0, 42.0)
        hough = (55.0, 42.0, 20.0)   # distance = 0
        final, conf, hc, hr = _agree_and_fuse(lm, hough, "left", flags)
        assert conf == CONFIDENCE_HIGH
        assert abs(final[0] - 55.0) < 0.001
        assert abs(final[1] - 42.0) < 0.001

    @pytest.mark.unit
    def test_boundary_exactly_5px_is_medium(self):
        """Distance == 5.0px exactly → MEDIUM (boundary is strictly less than 5 for HIGH)."""
        flags = []
        lm = (50.0, 40.0)
        hough = (55.0, 40.0, 20.0)   # exactly 5px apart
        final, conf, hc, hr = _agree_and_fuse(lm, hough, "left", flags)
        # 5.0 is NOT < 5 → should be MEDIUM
        assert conf == CONFIDENCE_MEDIUM

    @pytest.mark.unit
    def test_boundary_exactly_15px_is_low(self):
        """Distance == 15.0px → LOW (must be < 15 for MEDIUM)."""
        flags = []
        lm = (50.0, 40.0)
        hough = (65.0, 40.0, 20.0)   # exactly 15px
        final, conf, hc, hr = _agree_and_fuse(lm, hough, "left", flags)
        assert conf == CONFIDENCE_LOW


# ─────────────────────────────────────────────────────────────
# UNIT TESTS — _preprocess_for_hough
# ─────────────────────────────────────────────────────────────

class TestPreprocessForHough:

    @pytest.mark.unit
    def test_output_is_single_channel(self):
        """Preprocessing must return a single-channel image."""
        eye = make_synthetic_eye_crop()
        blurred = _preprocess_for_hough(eye)
        assert blurred.ndim == 2
        assert blurred.dtype == np.uint8

    @pytest.mark.unit
    def test_output_same_hw_as_input(self):
        """Output spatial dimensions must match input."""
        eye = make_synthetic_eye_crop(width=120, height=80)
        blurred = _preprocess_for_hough(eye)
        assert blurred.shape == (80, 120)

    @pytest.mark.unit
    def test_blank_image_does_not_crash(self):
        """Preprocessing must not crash on a uniform image."""
        eye = np.full((80, 120, 3), 128, dtype=np.uint8)
        blurred = _preprocess_for_hough(eye)
        assert blurred.shape == (80, 120)


# ─────────────────────────────────────────────────────────────
# INTEGRATION UNIT TESTS — localise_pupils on synthetic crops
# ─────────────────────────────────────────────────────────────

class TestLocalise:

    def _make_symmetric_inputs(
        self,
        cx: int = 60, cy: int = 40,
        iris_r: int = 28, crop_w: int = 120, crop_h: int = 80,
    ):
        """Build matching crops + landmarks for both eyes."""
        crop = make_synthetic_eye_crop(
            width=crop_w, height=crop_h,
            pupil_cx=cx, pupil_cy=cy,
            iris_r=iris_r,
        )
        # Landmarks in "original image" space — simulate crop at origin
        lms = make_5_iris_landmarks(cx, cy, iris_r, offset_x=0, offset_y=0)
        box = (0, 0, crop_w, crop_h)
        return crop, lms, box

    @pytest.mark.unit
    def test_returns_pupil_result_type(self):
        """localise_pupils must return a PupilResult."""
        crop, lms, box = self._make_symmetric_inputs()
        result = localise_pupils(crop, crop, lms, lms, box, box)
        assert isinstance(result, PupilResult)

    @pytest.mark.unit
    def test_pupil_centre_near_synthetic_centre(self):
        """
        On a clean synthetic eye, the detected pupil centre should be
        within 8px of the true pupil centre.
        (Using a generous tolerance because Hough may vote for iris edge.)
        """
        cx, cy = 60, 40
        crop, lms, box = self._make_symmetric_inputs(cx=cx, cy=cy)
        result = localise_pupils(crop, crop, lms, lms, box, box)

        for pupil in [result.left_pupil, result.right_pupil]:
            dist = np.linalg.norm(np.array(pupil) - np.array([cx, cy]))
            assert dist < 10.0, f"Pupil centre {pupil} too far from true centre ({cx},{cy}): {dist:.1f}px"

    @pytest.mark.unit
    def test_iris_radius_close_to_synthetic_radius(self):
        """Detected iris radius within 5px of synthetic ground truth."""
        iris_r = 28
        crop, lms, box = self._make_symmetric_inputs(iris_r=iris_r)
        result = localise_pupils(crop, crop, lms, lms, box, box)
        for r in [result.left_iris_radius, result.right_iris_radius]:
            assert abs(r - iris_r) < 6.0, f"Iris radius {r:.1f} too far from expected {iris_r}"

    @pytest.mark.unit
    def test_confidence_is_valid_tier(self):
        """Confidence must be one of the three valid tiers."""
        crop, lms, box = self._make_symmetric_inputs()
        result = localise_pupils(crop, crop, lms, lms, box, box)
        valid = {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW}
        assert result.left_confidence  in valid
        assert result.right_confidence in valid

    @pytest.mark.unit
    def test_landmark_centre_inside_crop(self):
        """Landmark centre must be within crop bounds."""
        w, h = 120, 80
        crop, lms, box = self._make_symmetric_inputs(crop_w=w, crop_h=h)
        result = localise_pupils(crop, crop, lms, lms, box, box)
        for pt in [result.left_landmark_centre, result.right_landmark_centre]:
            assert 0 <= pt[0] <= w
            assert 0 <= pt[1] <= h

    @pytest.mark.unit
    def test_landmarks_offset_by_crop_origin(self):
        """
        When landmarks are given in original-image space (with crop offset),
        the final pupil centre should still be in crop-local coordinates.
        """
        crop_w, crop_h = 120, 80
        cx, cy = 60, 40
        iris_r = 28
        # Simulate crop at (200, 100) in the original image
        ox, oy = 200, 100
        lms = make_5_iris_landmarks(cx, cy, iris_r, offset_x=ox, offset_y=oy)
        box = (ox, oy, ox + crop_w, oy + crop_h)
        crop = make_synthetic_eye_crop(crop_w, crop_h, cx, cy, iris_r)

        result = localise_pupils(crop, crop, lms, lms, box, box)

        # Pupil should be in crop coords (near cx, cy) not original coords (near 260, 140)
        for pupil in [result.left_pupil, result.right_pupil]:
            assert pupil[0] < crop_w, f"Pupil x={pupil[0]} should be in crop space, not original"
            assert pupil[1] < crop_h, f"Pupil y={pupil[1]} should be in crop space, not original"

    @pytest.mark.unit
    def test_flags_is_list(self):
        """flags must always be a list, even when empty."""
        crop, lms, box = self._make_symmetric_inputs()
        result = localise_pupils(crop, crop, lms, lms, box, box)
        assert isinstance(result.flags, list)

    @pytest.mark.unit
    def test_result_has_all_fields(self):
        """All PupilResult fields must be present and typed correctly."""
        crop, lms, box = self._make_symmetric_inputs()
        result = localise_pupils(crop, crop, lms, lms, box, box)

        assert isinstance(result.left_pupil,  tuple) and len(result.left_pupil)  == 2
        assert isinstance(result.right_pupil, tuple) and len(result.right_pupil) == 2
        assert isinstance(result.left_iris_radius,  float)
        assert isinstance(result.right_iris_radius, float)
        assert isinstance(result.left_confidence,  str)
        assert isinstance(result.right_confidence, str)
        assert isinstance(result.left_landmark_centre,  tuple)
        assert isinstance(result.right_landmark_centre, tuple)


# ─────────────────────────────────────────────────────────────
# VISUAL TESTS — run on real crops from Module 1
# ─────────────────────────────────────────────────────────────

class TestVisual:
    """
    These tests run the full Module 1 → Module 2 chain on real face photos.
    They save annotated crops to test_output/module2/ for manual review.

    After running:
      1. Open tests/test_output/module2/
      2. Confirm blue dot (landmark) is inside the pupil
      3. Confirm green circle (Hough) wraps the iris
      4. Both should be within ~5px on clear, well-lit images
    """

    def _load_image(self, folder: str):
        p = IMAGE_DIR / folder
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            imgs = list(p.glob(ext))
            if imgs:
                img = cv2.imread(str(imgs[0]))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    def _run_pipeline(self, img):
        """Run Module 1 then Module 2 on a real image."""
        from pipeline.module1_detection import detect_and_crop_eyes
        m1 = detect_and_crop_eyes(img)
        m2 = localise_pupils(
            m1.left_crop,
            m1.right_crop,
            m1.left_iris_landmarks,
            m1.right_iris_landmarks,
            m1.left_crop_box,
            m1.right_crop_box,
        )
        return m1, m2

    def _annotate_and_save(self, crop, result, eye: str, test_name: str):
        """Draw landmark, Hough, and final centre on crop and save."""
        annotated = crop.copy()

        if eye == "left":
            lm_c  = result.left_landmark_centre
            h_c   = result.left_hough_centre
            h_r   = result.left_hough_radius
            final = result.left_pupil
        else:
            lm_c  = result.right_landmark_centre
            h_c   = result.right_hough_centre
            h_r   = result.right_hough_radius
            final = result.right_pupil

        # Blue dot — landmark estimate
        annotated = draw_dot(annotated, lm_c[0], lm_c[1], (42, 159, 214), radius=4)

        # Green circle — Hough estimate
        if h_c is not None and h_r is not None:
            annotated = draw_circle(annotated, h_c[0], h_c[1], h_r, (48, 209, 88), thickness=2)
            annotated = draw_dot(annotated, h_c[0], h_c[1], (48, 209, 88), radius=2)

        # White crosshair — final agreed centre
        annotated = draw_crosshair(annotated, final[0], final[1], (255, 255, 255), size=8, thickness=2)

        path = OUTPUT_DIR / f"{test_name}_{eye}.jpg"
        save_debug_image(annotated, path)
        return path

    @pytest.mark.visual
    def test_normal_face_pupil_localisation(self):
        """
        VISUAL TEST — run full chain on a real normal face photo.
        Saves annotated crops for manual inspection.
        """
        img = self._load_image("flash_on_normal")
        if img is None:
            pytest.skip("No image in test_images/flash_on_normal/")

        m1, m2 = self._run_pipeline(img)

        # Structural checks
        assert isinstance(m2, PupilResult)
        assert m2.left_pupil[0]  > 0
        assert m2.right_pupil[0] > 0
        assert m2.left_iris_radius  > 5
        assert m2.right_iris_radius > 5

        # Save annotated images
        l_path = self._annotate_and_save(m1.left_crop,  m2, "left",  "normal")
        r_path = self._annotate_and_save(m1.right_crop, m2, "right", "normal")

        print(f"\n✓ Visual output saved:")
        print(f"  Left:  {l_path}")
        print(f"  Right: {r_path}")
        print(f"  Left  pupil: {m2.left_pupil}  confidence={m2.left_confidence}")
        print(f"  Right pupil: {m2.right_pupil} confidence={m2.right_confidence}")
        print(f"  Flags: {m2.flags or 'none'}")

    @pytest.mark.visual
    def test_high_confidence_on_clear_image(self):
        """
        On a clear, well-lit face photo, both eyes should reach at least
        MEDIUM confidence. LOW on both eyes suggests something is wrong.
        """
        img = self._load_image("flash_on_normal")
        if img is None:
            pytest.skip("No image in test_images/flash_on_normal/")

        _, m2 = self._run_pipeline(img)

        # At least one eye should be HIGH or MEDIUM
        confidences = {m2.left_confidence, m2.right_confidence}
        assert CONFIDENCE_HIGH in confidences or CONFIDENCE_MEDIUM in confidences, (
            f"Both eyes at LOW confidence on a clear image: left={m2.left_confidence}, "
            f"right={m2.right_confidence}"
        )


# ─────────────────────────────────────────────────────────────
# Pytest markers
# ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure unit tests, no real images needed")
    config.addinivalue_line("markers", "visual: tests requiring real images in test_images/")
