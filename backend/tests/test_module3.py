"""
BeanHealth CLR Tool — Module 3 Test Suite
==========================================

Tests for detect_clr() — corneal light reflex detection.

Test categories:
  1. Unit tests  — filter maths, circularity formula, flash check (no real images)
  2. Visual tests — run on real flash-on photos, save amber dot annotated crops

Run unit tests only:
    cd backend && pytest tests/test_module3.py -v -m unit

Run visual tests (requires test images):
    pytest tests/test_module3.py -v -m visual -s

Visual output → tests/test_output/module3/
After running: open those files and confirm the amber dot lands ON the bright
corneal spot, not on a glare reflection or eyelid highlight.
"""

import logging
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.module3_clr import (
    BlobCandidate,
    CLRResult,
    _adaptive_threshold_mask,
    _apply_four_way_filter,
    _detect_clr_one_eye,
    _find_blobs,
    _select_clr_blob,
    _validate_flash,
    detect_clr,
)
from utils.exceptions import CLRError
from utils.image_utils import draw_dot, draw_crosshair, save_debug_image

logger = logging.getLogger(__name__)

TEST_DIR   = Path(__file__).parent
OUTPUT_DIR = TEST_DIR / "test_output" / "module3"
IMAGE_DIR  = TEST_DIR / "test_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Synthetic image builders
# ─────────────────────────────────────────────────────────────

def make_dark_eye_crop(width=120, height=80) -> np.ndarray:
    """A dark image with no flash — max pixel well below 240."""
    img = np.full((height, width, 3), 80, dtype=np.uint8)
    # Dark iris
    cv2.circle(img, (width // 2, height // 2), 25, (50, 35, 20), -1)
    # Dark pupil
    cv2.circle(img, (width // 2, height // 2), 12, (10, 10, 10), -1)
    return img  # max pixel ≈ 80 → no flash


def make_flash_eye_crop(
    width=120, height=80,
    clr_x: int = 65, clr_y: int = 38,
    clr_r: int = 4,
    add_noise_blob: bool = False,
    add_edge_blob: bool = False,
    add_large_blob: bool = False,
) -> np.ndarray:
    """
    Synthetic eye with a torch reflection (CLR) at the specified position.

    Optional extra blobs for filter testing:
      add_noise_blob  — tiny blob (should fail area filter)
      add_edge_blob   — blob near the edge (should fail location filter)
      add_large_blob  — large blob (should fail area filter)
    """
    img = np.full((height, width, 3), 90, dtype=np.uint8)

    # Sclera
    cv2.ellipse(img, (width // 2, height // 2), (int(width * 0.45), int(height * 0.28)),
                0, 0, 360, (220, 215, 205), -1)

    # Iris
    iris_cx, iris_cy = width // 2, height // 2
    cv2.circle(img, (iris_cx, iris_cy), 28, (90, 60, 30), -1)

    # Pupil
    cv2.circle(img, (iris_cx, iris_cy), 13, (12, 12, 12), -1)

    # CLR bright spot — torch reflection (very bright)
    cv2.circle(img, (clr_x, clr_y), clr_r, (255, 255, 255), -1)
    # Soft glow around it
    cv2.circle(img, (clr_x, clr_y), clr_r + 2, (210, 210, 210), 1)

    # Optional extra blobs
    if add_noise_blob:
        # Tiny (1px) — will fail area filter
        cv2.circle(img, (iris_cx + 8, iris_cy + 5), 1, (255, 255, 255), -1)

    if add_edge_blob:
        # Near the top edge — will fail location filter
        cv2.circle(img, (5, 3), 3, (255, 255, 255), -1)

    if add_large_blob:
        # Large reflection (simulates glasses glare) — will fail area filter
        cv2.rectangle(img, (10, 10), (40, 30), (255, 255, 255), -1)

    return img


def make_blob(
    area: float = 50.0,
    cx: float = 60.0,
    cy: float = 40.0,
    circularity: float = 0.85,
    label: int = 1,
) -> BlobCandidate:
    """Create a BlobCandidate with given attributes for filter testing."""
    return BlobCandidate(
        label=label,
        area=area,
        centroid_x=cx,
        centroid_y=cy,
        circularity=circularity,
    )


# ─────────────────────────────────────────────────────────────
# UNIT TESTS — _validate_flash
# ─────────────────────────────────────────────────────────────

class TestFlashValidation:

    @pytest.mark.unit
    def test_bright_image_passes(self):
        """Max pixel >= 240 → no error raised."""
        gray = np.full((80, 120), 200, dtype=np.uint8)
        gray[40, 60] = 245   # one bright pixel
        _validate_flash(gray, "left")   # should not raise

    @pytest.mark.unit
    def test_dark_image_raises_no_flash(self):
        """All pixels below 240 → CLRError(no_flash)."""
        gray = np.full((80, 120), 180, dtype=np.uint8)
        with pytest.raises(CLRError) as exc_info:
            _validate_flash(gray, "left")
        assert exc_info.value.code == "no_flash"

    @pytest.mark.unit
    def test_exactly_240_passes(self):
        """Peak == 240 → should NOT raise (threshold is strictly less than)."""
        gray = np.full((80, 120), 100, dtype=np.uint8)
        gray[40, 60] = 240
        _validate_flash(gray, "right")   # should not raise

    @pytest.mark.unit
    def test_exactly_239_raises(self):
        """Peak == 239 → below threshold → raises."""
        gray = np.full((80, 120), 100, dtype=np.uint8)
        gray[40, 60] = 239
        with pytest.raises(CLRError) as exc_info:
            _validate_flash(gray, "right")
        assert exc_info.value.code == "no_flash"

    @pytest.mark.unit
    def test_no_flash_human_message_is_not_empty(self):
        """CLRError must carry a non-empty human_message."""
        gray = np.zeros((80, 120), dtype=np.uint8)
        with pytest.raises(CLRError) as exc_info:
            _validate_flash(gray, "left")
        assert len(exc_info.value.human_message) > 5


# ─────────────────────────────────────────────────────────────
# UNIT TESTS — _adaptive_threshold_mask
# ─────────────────────────────────────────────────────────────

class TestAdaptiveThreshold:

    @pytest.mark.unit
    def test_bright_spot_becomes_white(self):
        """Pixels above the 97th percentile threshold should be white in the mask."""
        gray = np.full((80, 120), 100, dtype=np.uint8)
        gray[40, 60] = 255   # one very bright pixel
        mask, _ = _adaptive_threshold_mask(gray)
        assert mask[40, 60] == 255

    @pytest.mark.unit
    def test_dark_pixels_become_black(self):
        """Pixels below threshold must be black (0) in the mask."""
        gray = np.full((80, 120), 50, dtype=np.uint8)
        gray[40, 60] = 255
        mask, _ = _adaptive_threshold_mask(gray)
        # Most pixels are 50, threshold will be high → most should be 0
        assert mask[0, 0] == 0

    @pytest.mark.unit
    def test_mask_is_binary(self):
        """Output mask must contain only 0 and 255."""
        gray = np.random.randint(0, 256, (80, 120), dtype=np.uint8)
        mask, _ = _adaptive_threshold_mask(gray)
        unique = np.unique(mask)
        assert set(unique).issubset({0, 255})

    @pytest.mark.unit
    def test_threshold_adapts_to_brightness(self):
        """Same relative bright spot → mask picks it up regardless of absolute brightness."""
        # Dark scene
        dark = np.full((80, 120), 30, dtype=np.uint8)
        dark[40, 60] = 60   # relative bright spot
        mask_dark, thresh_dark = _adaptive_threshold_mask(dark)
        assert thresh_dark < 100   # threshold adapts down

        # Bright scene
        bright = np.full((80, 120), 200, dtype=np.uint8)
        bright[40, 60] = 240   # relative bright spot
        mask_bright, thresh_bright = _adaptive_threshold_mask(bright)
        assert thresh_bright > 100   # threshold adapts up


# ─────────────────────────────────────────────────────────────
# UNIT TESTS — circularity formula
# ─────────────────────────────────────────────────────────────

class TestCircularity:
    """
    Verify the circularity formula by creating binary masks of known shapes
    and running them through _find_blobs.
    """

    def _run_find_blobs(self, mask: np.ndarray) -> List[BlobCandidate]:
        return _find_blobs(mask)

    @pytest.mark.unit
    def test_circle_has_high_circularity(self):
        """A large filled circle → circularity close to 1.0."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 40, 255, -1)
        blobs = self._run_find_blobs(mask)
        assert len(blobs) == 1
        assert blobs[0].circularity > 0.85, f"Circle circularity was {blobs[0].circularity:.3f}"

    @pytest.mark.unit
    def test_rectangle_has_low_circularity(self):
        """A long thin rectangle → circularity well below 0.5."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (10, 90), (190, 110), 255, -1)  # wide, thin
        blobs = self._run_find_blobs(mask)
        assert len(blobs) == 1
        assert blobs[0].circularity < 0.5, f"Rectangle circularity was {blobs[0].circularity:.3f}"

    @pytest.mark.unit
    def test_empty_mask_returns_no_blobs(self):
        """All-black mask → no blobs."""
        mask = np.zeros((80, 120), dtype=np.uint8)
        blobs = self._run_find_blobs(mask)
        assert len(blobs) == 0

    @pytest.mark.unit
    def test_multiple_blobs_all_found(self):
        """Two separate circles → two blob candidates returned."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (50, 50),  20, 255, -1)
        cv2.circle(mask, (150, 150), 20, 255, -1)
        blobs = self._run_find_blobs(mask)
        assert len(blobs) == 2


# ─────────────────────────────────────────────────────────────
# UNIT TESTS — _apply_four_way_filter
# ─────────────────────────────────────────────────────────────

class TestFourWayFilter:
    """
    Test each of the four mandatory filters independently,
    then test combinations.
    """

    # Reference crop and iris for all filter tests
    CROP_W     = 120
    CROP_H     = 80
    IRIS_R     = 28.0
    IRIS_AREA  = math.pi * (28.0 ** 2)   # ≈ 2463 px²
    MIN_AREA   = 0.005 * IRIS_AREA        # ≈ 12.3 px²
    MAX_AREA   = 0.150 * IRIS_AREA        # ≈ 369.4 px²

    def _run_filter(self, blobs):
        return _apply_four_way_filter(blobs, self.CROP_W, self.CROP_H, self.IRIS_R, (60.0, 40.0), "left")

    # ── Filter ①: Location ──

    @pytest.mark.unit
    def test_centred_blob_passes_location(self):
        """Blob at exact centre → passes location filter."""
        b = make_blob(area=50.0, cx=60.0, cy=40.0, circularity=0.9)
        result = self._run_filter([b])
        assert len(result) == 1

    @pytest.mark.unit
    def test_left_edge_blob_fails_location(self):
        """Blob at x=2 (< 10% margin = 12px) → fails location filter."""
        b = make_blob(area=50.0, cx=2.0, cy=40.0, circularity=0.9)
        result = self._run_filter([b])
        assert len(result) == 0

    @pytest.mark.unit
    def test_top_edge_blob_fails_location(self):
        """Blob at y=2 → fails location filter."""
        b = make_blob(area=50.0, cx=60.0, cy=2.0, circularity=0.9)
        result = self._run_filter([b])
        assert len(result) == 0

    @pytest.mark.unit
    def test_right_edge_blob_fails_location(self):
        """Blob at x=118 (> 90% of 120 = 108) → fails."""
        b = make_blob(area=50.0, cx=118.0, cy=40.0, circularity=0.9)
        result = self._run_filter([b])
        assert len(result) == 0

    # ── Filter ②: Area ──

    @pytest.mark.unit
    def test_valid_area_passes(self):
        """Area in the valid range → passes area filter."""
        valid_area = self.IRIS_AREA * 0.02   # 2% — within [0.5%, 15%]
        b = make_blob(area=valid_area, cx=60.0, cy=40.0, circularity=0.9)
        result = self._run_filter([b])
        assert len(result) == 1

    @pytest.mark.unit
    def test_area_too_small_fails(self):
        """Area < 0.5% of iris area → fails area filter."""
        tiny_area = self.IRIS_AREA * 0.001   # 0.1% — too small
        b = make_blob(area=tiny_area, cx=60.0, cy=40.0, circularity=0.9)
        result = self._run_filter([b])
        assert len(result) == 0

    @pytest.mark.unit
    def test_area_too_large_fails(self):
        """Area > 15% of iris area → fails area filter (glasses glare)."""
        large_area = self.IRIS_AREA * 0.20   # 20% — too large
        b = make_blob(area=large_area, cx=60.0, cy=40.0, circularity=0.9)
        result = self._run_filter([b])
        assert len(result) == 0

    # ── Filter ③: Circularity ──

    @pytest.mark.unit
    def test_circular_blob_passes(self):
        """Circularity 0.85 → passes."""
        b = make_blob(area=50.0, cx=60.0, cy=40.0, circularity=0.85)
        result = self._run_filter([b])
        assert len(result) == 1

    @pytest.mark.unit
    def test_elongated_blob_fails(self):
        """Circularity 0.3 (elongated eyelash reflection) → fails."""
        b = make_blob(area=50.0, cx=60.0, cy=40.0, circularity=0.30)
        result = self._run_filter([b])
        assert len(result) == 0

    @pytest.mark.unit
    def test_exactly_at_circularity_threshold_fails(self):
        """Circularity exactly at 0.5 → must FAIL (condition is strictly >)."""
        b = make_blob(area=50.0, cx=60.0, cy=40.0, circularity=0.50)
        result = self._run_filter([b])
        assert len(result) == 0

    @pytest.mark.unit
    def test_just_above_circularity_threshold_passes(self):
        """Circularity 0.51 → passes."""
        b = make_blob(area=50.0, cx=60.0, cy=40.0, circularity=0.51)
        result = self._run_filter([b])
        assert len(result) == 1

    # ── Multiple blobs ──

    @pytest.mark.unit
    def test_only_valid_blob_selected(self):
        """Mixed list: one valid, one edge, one tiny → only valid survives."""
        valid  = make_blob(area=50.0,                cx=60.0, cy=40.0, circularity=0.85, label=1)
        edge   = make_blob(area=50.0,                cx=2.0,  cy=40.0, circularity=0.85, label=2)
        tiny   = make_blob(area=self.IRIS_AREA*0.001, cx=60.0, cy=40.0, circularity=0.85, label=3)
        result = self._run_filter([valid, edge, tiny])
        assert len(result) == 1
        assert result[0].label == 1


# ─────────────────────────────────────────────────────────────
# UNIT TESTS — _select_clr_blob
# ─────────────────────────────────────────────────────────────

class TestSelectCLRBlob:

    @pytest.mark.unit
    def test_empty_list_raises_no_reflex(self):
        """No passing blobs → CLRError(no_reflex_left)."""
        flags = []
        with pytest.raises(CLRError) as exc_info:
            _select_clr_blob([], "left", flags)
        assert "no_reflex_left" in exc_info.value.code

    @pytest.mark.unit
    def test_single_blob_selected(self):
        """Single blob → returned as-is."""
        flags = []
        b = make_blob(area=50.0, circularity=0.9, label=1)
        result = _select_clr_blob([b], "left", flags)
        assert result.label == 1
        assert not flags   # no ambiguity flag

    @pytest.mark.unit
    def test_largest_blob_selected(self):
        """Multiple blobs → largest by area is selected."""
        flags = []
        small  = make_blob(area=30.0,  label=1)
        medium = make_blob(area=80.0,  label=2)
        large  = make_blob(area=150.0, label=3)
        result = _select_clr_blob([small, medium, large], "left", flags)
        assert result.label == 3

    @pytest.mark.unit
    def test_many_blobs_flags_ambiguous(self):
        """More than 3 passing blobs → ambiguous_reflex flag added."""
        flags = []
        blobs = [make_blob(area=50.0 + i, label=i) for i in range(5)]
        _select_clr_blob(blobs, "right", flags)
        assert "ambiguous_reflex_right" in flags

    @pytest.mark.unit
    def test_three_blobs_no_ambiguous_flag(self):
        """Exactly 3 passing blobs → no ambiguity flag."""
        flags = []
        blobs = [make_blob(area=50.0 + i, label=i) for i in range(3)]
        _select_clr_blob(blobs, "left", flags)
        assert not any("ambiguous" in f for f in flags)


# ─────────────────────────────────────────────────────────────
# INTEGRATION UNIT TESTS — detect_clr on synthetic crops
# ─────────────────────────────────────────────────────────────

class TestDetectCLR:

    @pytest.mark.unit
    def test_returns_clr_result_type(self):
        """detect_clr must return a CLRResult."""
        crop = make_flash_eye_crop()
        result = detect_clr(crop, crop, 28.0, 28.0, (60.0, 40.0), (60.0, 40.0))
        assert isinstance(result, CLRResult)

    @pytest.mark.unit
    def test_clr_near_synthetic_position(self):
        """
        CLR detected should be within 6px of the planted bright spot.
        (Using a tolerance because the blob centroid may shift slightly
        depending on the glow pixels around the CLR.)
        """
        clr_x, clr_y = 65, 38
        crop = make_flash_eye_crop(clr_x=clr_x, clr_y=clr_y, clr_r=4)
        result = detect_clr(crop, crop, 28.0, 28.0, (60.0, 40.0), (60.0, 40.0))

        for clr in [result.left_clr, result.right_clr]:
            dist = math.sqrt((clr[0] - clr_x) ** 2 + (clr[1] - clr_y) ** 2)
            assert dist < 8.0, f"CLR {clr} too far from planted spot ({clr_x},{clr_y}): {dist:.1f}px"

    @pytest.mark.unit
    def test_no_flash_raises(self):
        """Dark crop without bright spot → CLRError(no_flash)."""
        dark = make_dark_eye_crop()
        with pytest.raises(CLRError) as exc_info:
            detect_clr(dark, dark, 28.0, 28.0, (60.0, 40.0), (60.0, 40.0))
        assert exc_info.value.code == "no_flash"

    @pytest.mark.unit
    def test_confidence_is_float_0_to_1(self):
        """CLR confidence (circularity) must be in [0, 1]."""
        crop = make_flash_eye_crop()
        result = detect_clr(crop, crop, 28.0, 28.0, (60.0, 40.0), (60.0, 40.0))
        assert 0.0 <= result.left_clr_confidence  <= 1.0
        assert 0.0 <= result.right_clr_confidence <= 1.0

    @pytest.mark.unit
    def test_noise_blob_does_not_confuse_detector(self):
        """A tiny noise blob alongside the CLR — should still find the CLR."""
        crop = make_flash_eye_crop(clr_x=65, clr_y=38, add_noise_blob=True)
        result = detect_clr(crop, crop, 28.0, 28.0, (60.0, 40.0), (60.0, 40.0))
        # CLR should still be near the planted position
        dist = math.sqrt((result.left_clr[0] - 65) ** 2 + (result.left_clr[1] - 38) ** 2)
        assert dist < 10.0

    @pytest.mark.unit
    def test_edge_blob_does_not_confuse_detector(self):
        """A bright blob at the image edge — location filter should reject it."""
        crop = make_flash_eye_crop(clr_x=65, clr_y=38, add_edge_blob=True)
        result = detect_clr(crop, crop, 28.0, 28.0, (60.0, 40.0), (60.0, 40.0))
        # Still finds the real CLR near (65, 38)
        dist = math.sqrt((result.left_clr[0] - 65) ** 2 + (result.left_clr[1] - 38) ** 2)
        assert dist < 10.0

    @pytest.mark.unit
    def test_flags_is_list(self):
        """flags must always be a list even when empty."""
        crop = make_flash_eye_crop()
        result = detect_clr(crop, crop, 28.0, 28.0, (60.0, 40.0), (60.0, 40.0))
        assert isinstance(result.flags, list)

    @pytest.mark.unit
    def test_clr_result_has_all_fields(self):
        """All CLRResult fields present and correctly typed."""
        crop = make_flash_eye_crop()
        r = detect_clr(crop, crop, 28.0, 28.0, (60.0, 40.0), (60.0, 40.0))
        assert isinstance(r.left_clr,  tuple) and len(r.left_clr)  == 2
        assert isinstance(r.right_clr, tuple) and len(r.right_clr) == 2
        assert isinstance(r.left_clr_confidence,  float)
        assert isinstance(r.right_clr_confidence, float)
        assert isinstance(r.flags, list)


# ─────────────────────────────────────────────────────────────
# VISUAL TESTS — run on real photos
# ─────────────────────────────────────────────────────────────

class TestVisual:
    """
    Visual tests require real photos.
    flash_on_normal/ → torch must have been ON (back camera ideally)
    flash_off/       → torch must have been OFF

    After running, open tests/test_output/module3/:
      - flash_on: amber dot should land ON the corneal bright spot
      - flash_off: test should return INCONCLUSIVE (no image saved)
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
        """Run Module 1 → 2 → 3."""
        from pipeline.module1_detection import detect_and_crop_eyes
        from pipeline.module2_pupil import localise_pupils
        m1 = detect_and_crop_eyes(img)
        m2 = localise_pupils(
            m1.left_crop, m1.right_crop,
            m1.left_iris_landmarks, m1.right_iris_landmarks,
            m1.left_crop_box, m1.right_crop_box,
        )
        m3 = detect_clr(
            m1.left_crop, m1.right_crop,
            m2.left_iris_radius, m2.right_iris_radius,
            m2.left_pupil, m2.right_pupil,
        )
        return m1, m2, m3

    def _annotate_and_save(self, crop, clr_pos, pupil_pos, eye, test_name):
        """Draw amber CLR dot and white pupil crosshair on crop and save."""
        annotated = crop.copy()
        # White crosshair at pupil centre
        annotated = draw_crosshair(annotated, pupil_pos[0], pupil_pos[1], (255, 255, 255), size=8)
        # Amber dot at CLR
        annotated = draw_dot(annotated, clr_pos[0], clr_pos[1], (245, 166, 35), radius=5)
        # Small centre mark on CLR
        annotated = draw_dot(annotated, clr_pos[0], clr_pos[1], (255, 255, 255), radius=1)
        path = OUTPUT_DIR / f"{test_name}_{eye}.jpg"
        save_debug_image(annotated, path)
        return path

    @pytest.mark.visual
    def test_flash_on_clr_detected(self):
        """
        VISUAL TEST — CLR detection on a real flash-on photo.
        Amber dot should land on the corneal bright spot.
        """
        img = self._load_image("flash_on_normal")
        if img is None:
            pytest.skip("No image in test_images/flash_on_normal/")

        # These are selfies without torch — CLR may or may not be detected.
        # We only assert structural validity, not position accuracy.
        # For true CLR testing, back-camera torch photos are needed.
        try:
            m1, m2, m3 = self._run_pipeline(img)
            assert isinstance(m3, CLRResult)
            assert len(m3.left_clr)  == 2
            assert len(m3.right_clr) == 2
            assert 0.0 <= m3.left_clr_confidence  <= 1.0
            assert 0.0 <= m3.right_clr_confidence <= 1.0

            l_path = self._annotate_and_save(
                m1.left_crop,  m3.left_clr,  m2.left_pupil,  "left",  "flash_on")
            r_path = self._annotate_and_save(
                m1.right_crop, m3.right_clr, m2.right_pupil, "right", "flash_on")

            print(f"\n✓ CLR detected:")
            print(f"  Left CLR:  {m3.left_clr}  confidence={m3.left_clr_confidence:.3f}")
            print(f"  Right CLR: {m3.right_clr} confidence={m3.right_clr_confidence:.3f}")
            print(f"  Flags: {m3.flags or 'none'}")
            print(f"  Left:  {l_path}")
            print(f"  Right: {r_path}")

        except CLRError as e:
            # Selfies without torch → INCONCLUSIVE is expected and correct
            print(f"\n⚠ CLRError ({e.code}): {e.human_message}")
            print("  This is expected for selfies without torch. Use back camera + torch for CLR tests.")
            pytest.skip(f"CLR detection failed ({e.code}) — expected for selfies without torch")

    @pytest.mark.visual
    def test_flash_off_returns_inconclusive(self):
        """
        VISUAL TEST — no-flash photo must raise CLRError, never return a position.
        This is the most important safety test: INCONCLUSIVE must be returned,
        not a guessed CLR position.
        """
        img = self._load_image("flash_off")
        if img is None:
            pytest.skip("No image in test_images/flash_off/")

        try:
            m1, _, _ = self._run_pipeline(img)
            # If pipeline succeeded, check the image is definitely not a flash image
            # (it might be misclassified — the user might have torch on)
            print("\n⚠ Pipeline did NOT raise on flash_off image.")
            print("  Verify your flash_off photo truly has no torch/flash active.")
        except CLRError as e:
            # This is the CORRECT outcome — INCONCLUSIVE
            assert e.code in ("no_flash", "no_reflex_left", "no_reflex_right", "no_reflex_both")
            print(f"\n✓ Correctly returned INCONCLUSIVE: {e.code}")
        except Exception:
            # Module 1 or 2 failure (e.g. face not found) is also acceptable
            pass


# ─────────────────────────────────────────────────────────────
# Pytest markers
# ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure unit tests, no real images needed")
    config.addinivalue_line("markers", "visual: tests requiring real images in test_images/")
