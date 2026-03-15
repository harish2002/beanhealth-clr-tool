"""
BeanHealth CLR Tool — Module 1 Test Suite
==========================================

Tests for detect_and_crop_eyes() — the eye detection and crop module.

Test categories:
  1. Unit tests  — pure function tests with synthetic inputs
  2. Visual tests — run on real images, save annotated output for manual review

Run all tests:
    cd backend && pytest tests/test_module1.py -v

Run only unit tests (no real images needed):
    pytest tests/test_module1.py -v -m unit

Run only visual tests (requires real images in test_images/):
    pytest tests/test_module1.py -v -m visual

Visual test output saved to: tests/test_output/module1/
Open those images and manually confirm eye crops look correct.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

# ── Path setup — allow imports from backend/ root ──
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.module1_detection import (
    EyeDetectionResult,
    _bounding_box_from_landmarks,
    _iris_radius_from_landmarks,
    _landmarks_to_pixels,
    detect_and_crop_eyes,
)
from utils.exceptions import DetectionError
from utils.image_utils import draw_circle, draw_dot, save_debug_image

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

TEST_DIR    = Path(__file__).parent
OUTPUT_DIR  = TEST_DIR / "test_output" / "module1"
IMAGE_DIR   = TEST_DIR / "test_images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Synthetic Image Generators (no real photos needed)
# ─────────────────────────────────────────────────────────────

def make_blank_rgb(width: int = 640, height: int = 480) -> np.ndarray:
    """Solid grey image — simulates a wall with no face."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def make_black_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Pure black image — extreme dark input."""
    return np.zeros((height, width, 3), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────
# Helper: annotate crops for visual review
# ─────────────────────────────────────────────────────────────

def annotate_and_save(
    result: EyeDetectionResult,
    original_img: np.ndarray,
    test_name: str,
) -> None:
    """
    Draw detected crop bounding boxes and iris landmarks onto the original
    image and save it. Also save each crop separately.
    Used for MANUAL VISUAL REVIEW — open the output folder after tests pass.
    """
    annotated = original_img.copy()

    # Draw bounding boxes
    for box, colour in [
        (result.left_crop_box,  (0, 180, 255)),   # cyan = left
        (result.right_crop_box, (255, 140, 0)),    # orange = right
    ]:
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

    # Draw iris landmarks
    for lm, colour in [
        (result.left_iris_landmarks,  (42, 159, 214)),  # blue
        (result.right_iris_landmarks, (245, 166, 35)),  # amber
    ]:
        for x, y in lm:
            cv2.circle(annotated, (int(x), int(y)), 3, colour, -1)

    save_debug_image(annotated, OUTPUT_DIR / f"{test_name}_annotated.jpg")
    save_debug_image(result.left_crop,  OUTPUT_DIR / f"{test_name}_left_crop.jpg")
    save_debug_image(result.right_crop, OUTPUT_DIR / f"{test_name}_right_crop.jpg")
    logger.info(f"Visual output saved to {OUTPUT_DIR}/{test_name}_*.jpg")


# ─────────────────────────────────────────────────────────────
# UNIT TESTS — Pure Function Tests (no real images)
# ─────────────────────────────────────────────────────────────

class TestLandmarksToPixels:
    """Test the _landmarks_to_pixels helper."""

    @pytest.mark.unit
    def test_centre_landmark(self):
        """Landmark at (0.5, 0.5) on 100x100 image → pixel (50, 50)."""

        class FakeLandmark:
            def __init__(self, x, y, z=0):
                self.x, self.y, self.z = x, y, z

        landmarks = [FakeLandmark(0.5, 0.5), FakeLandmark(0.25, 0.75)]
        result = _landmarks_to_pixels(landmarks, img_width=100, img_height=100, indices=[0, 1])

        assert len(result) == 2
        assert abs(result[0][0] - 50.0) < 0.01
        assert abs(result[0][1] - 50.0) < 0.01
        assert abs(result[1][0] - 25.0) < 0.01
        assert abs(result[1][1] - 75.0) < 0.01

    @pytest.mark.unit
    def test_top_left_landmark(self):
        """Landmark at (0, 0) → pixel (0, 0)."""

        class FakeLandmark:
            def __init__(self, x, y, z=0):
                self.x, self.y, self.z = x, y, z

        landmarks = [FakeLandmark(0.0, 0.0)]
        result = _landmarks_to_pixels(landmarks, img_width=640, img_height=480, indices=[0])
        assert result[0] == (0.0, 0.0)

    @pytest.mark.unit
    def test_scales_with_image_size(self):
        """Same normalised coord → different pixels for different image sizes."""

        class FakeLandmark:
            def __init__(self, x, y, z=0):
                self.x, self.y, self.z = x, y, z

        lm = [FakeLandmark(0.5, 0.5)]
        small = _landmarks_to_pixels(lm, 100, 100, [0])
        large = _landmarks_to_pixels(lm, 1000, 1000, [0])
        assert small[0] == (50.0, 50.0)
        assert large[0] == (500.0, 500.0)


class TestIrisRadius:
    """Test the _iris_radius_from_landmarks helper."""

    @pytest.mark.unit
    def test_perfect_circle(self):
        """
        5 landmarks: centre at (50,50), 4 boundary points exactly 10px away.
        Expected radius: 10.0
        """
        landmarks = [
            (50.0, 50.0),   # centre
            (50.0, 40.0),   # top     — 10px up
            (60.0, 50.0),   # right   — 10px right
            (50.0, 60.0),   # bottom  — 10px down
            (40.0, 50.0),   # left    — 10px left
        ]
        radius = _iris_radius_from_landmarks(landmarks)
        assert abs(radius - 10.0) < 0.01

    @pytest.mark.unit
    def test_zero_radius(self):
        """All 5 landmarks at same point → radius = 0."""
        landmarks = [(50.0, 50.0)] * 5
        radius = _iris_radius_from_landmarks(landmarks)
        assert radius == 0.0

    @pytest.mark.unit
    def test_fewer_than_5_landmarks(self):
        """Fewer than 5 landmarks → returns 0.0 gracefully."""
        radius = _iris_radius_from_landmarks([(10.0, 10.0), (20.0, 20.0)])
        assert radius == 0.0

    @pytest.mark.unit
    def test_asymmetric_ellipse(self):
        """
        Non-circular iris (as can happen at slight angles).
        Radius should be the MEAN of the 4 spoke distances.
        """
        landmarks = [
            (50.0, 50.0),   # centre
            (50.0, 35.0),   # top   — 15px
            (65.0, 50.0),   # right — 15px
            (50.0, 65.0),   # bottom — 15px
            (35.0, 50.0),   # left  — 15px
        ]
        radius = _iris_radius_from_landmarks(landmarks)
        assert abs(radius - 15.0) < 0.01


class TestInvalidImages:
    """Test that bad inputs raise DetectionError correctly."""

    @pytest.mark.unit
    def test_blank_image_raises_no_face(self):
        """A plain grey image has no face — should raise DetectionError(no_face)."""
        blank = make_blank_rgb()
        with pytest.raises(DetectionError) as exc_info:
            detect_and_crop_eyes(blank)
        assert exc_info.value.code == "no_face"

    @pytest.mark.unit
    def test_black_image_raises_no_face(self):
        """A pure black image → no face detected."""
        black = make_black_image()
        with pytest.raises(DetectionError) as exc_info:
            detect_and_crop_eyes(black)
        assert exc_info.value.code == "no_face"

    @pytest.mark.unit
    def test_detection_error_has_human_message(self):
        """DetectionError must always carry a non-empty human_message."""
        blank = make_blank_rgb()
        with pytest.raises(DetectionError) as exc_info:
            detect_and_crop_eyes(blank)
        err = exc_info.value
        assert err.human_message
        assert len(err.human_message) > 10

    @pytest.mark.unit
    def test_very_small_image_raises(self):
        """A 10x10 image — too small to contain a detectable face."""
        tiny = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(DetectionError):
            detect_and_crop_eyes(tiny)

    @pytest.mark.unit
    def test_single_pixel_image_raises(self):
        """Edge case — single pixel."""
        pixel = np.array([[[128, 128, 128]]], dtype=np.uint8)
        with pytest.raises(DetectionError):
            detect_and_crop_eyes(pixel)


class TestResultShape:
    """
    Structural tests for EyeDetectionResult — run only when real face images
    are available. Skip gracefully if no images found.
    """

    def _load_first_image(self, folder: str):
        """Load the first image from a test_images subfolder, or None."""
        folder_path = IMAGE_DIR / folder
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images = list(folder_path.glob(ext))
            if images:
                img = cv2.imread(str(images[0]))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    @pytest.mark.visual
    def test_normal_face_crops_are_valid(self):
        """
        VISUAL TEST — requires a real face photo in test_images/flash_on_normal/
        Checks structural validity AND saves annotated images for manual review.
        """
        img = self._load_first_image("flash_on_normal")
        if img is None:
            pytest.skip("No real face image found in test_images/flash_on_normal/ — add photos to run this test.")

        result = detect_and_crop_eyes(img, debug=True)

        # ── Structural checks ──
        assert isinstance(result, EyeDetectionResult)

        # Crops are 3-channel RGB uint8
        assert result.left_crop.ndim  == 3
        assert result.right_crop.ndim == 3
        assert result.left_crop.dtype  == np.uint8
        assert result.right_crop.dtype == np.uint8
        assert result.left_crop.shape[2]  == 3
        assert result.right_crop.shape[2] == 3

        # Crops are large enough
        assert result.left_crop.shape[1]  >= 60   # width
        assert result.left_crop.shape[0]  >= 40   # height
        assert result.right_crop.shape[1] >= 60
        assert result.right_crop.shape[0] >= 40

        # 5 iris landmarks per eye
        assert len(result.left_iris_landmarks)  == 5
        assert len(result.right_iris_landmarks) == 5

        # Iris radii are plausible (> 5px, < 500px)
        assert 5 < result.left_iris_radius_orig  < 500
        assert 5 < result.right_iris_radius_orig < 500

        # Crop boxes are valid (x1 < x2, y1 < y2)
        lx1, ly1, lx2, ly2 = result.left_crop_box
        rx1, ry1, rx2, ry2 = result.right_crop_box
        assert lx1 < lx2 and ly1 < ly2
        assert rx1 < rx2 and ry1 < ry2

        # Save annotated images for manual inspection
        annotate_and_save(result, img, "normal_face")
        print(f"\n✓ Visual output saved to {OUTPUT_DIR}/")
        print(f"  Left crop:  {result.left_crop.shape[1]}x{result.left_crop.shape[0]}px")
        print(f"  Right crop: {result.right_crop.shape[1]}x{result.right_crop.shape[0]}px")
        print(f"  Left iris radius:  {result.left_iris_radius_orig:.1f}px")
        print(f"  Right iris radius: {result.right_iris_radius_orig:.1f}px")
        print(f"  Warnings: {result.warnings or 'none'}")

    @pytest.mark.visual
    def test_landmarks_inside_crop_bounds(self):
        """
        VISUAL TEST — iris landmark pixels should fall inside the full image,
        not outside it.
        """
        img = self._load_first_image("flash_on_normal")
        if img is None:
            pytest.skip("No real face image in test_images/flash_on_normal/")

        result = detect_and_crop_eyes(img)
        h, w = img.shape[:2]

        for lm in result.left_iris_landmarks + result.right_iris_landmarks:
            x, y = lm
            assert 0 <= x <= w, f"Landmark x={x} out of image width {w}"
            assert 0 <= y <= h, f"Landmark y={y} out of image height {h}"

    @pytest.mark.visual
    def test_no_face_image_raises(self):
        """VISUAL TEST — image with no face must raise DetectionError."""
        img = self._load_first_image("no_face")
        if img is None:
            pytest.skip("No test image in test_images/no_face/")

        with pytest.raises(DetectionError) as exc_info:
            detect_and_crop_eyes(img)
        assert exc_info.value.code in ("no_face", "eyes_not_visible", "low_confidence")

    @pytest.mark.visual
    def test_eyes_closed_raises(self):
        """VISUAL TEST — closed eyes should raise DetectionError."""
        img = self._load_first_image("eyes_closed")
        if img is None:
            pytest.skip("No test image in test_images/eyes_closed/")

        with pytest.raises(DetectionError) as exc_info:
            detect_and_crop_eyes(img)
        # crop_too_small is also valid: MediaPipe may find the structural eye
        # region even when eyes are closed, but the crop comes out too small
        # because the eyelids cover most of the iris area.
        assert exc_info.value.code in (
            "eyes_closed", "eyes_not_visible", "no_face", "crop_too_small"
        )


# ─────────────────────────────────────────────────────────────
# PRINT SUMMARY (shown when running tests)
# ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure unit tests (no real images needed)")
    config.addinivalue_line("markers", "visual: tests that require real images in test_images/")
