"""
BeanHealth CLR Tool — Module 8 Test Suite
==========================================

Pure unit tests for the CLR-free pupil-vs-corner alignment check.
No real images needed — all tests use synthetic pupil and canthus points.

The layout mirrors a frontal face: the image-left eye's inner canthus sits on
its right (toward the nose) and the image-right eye's inner canthus on its
left, so the two inner→outer axes point in opposite directions.

Run:
    cd backend && pytest tests/test_module8.py -v
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.module8_alignment import (
    ALIGN_ALIGNED,
    ALIGN_ASYMMETRIC,
    ALIGN_BORDERLINE,
    compute_corner_alignment,
)

# Crop boxes at the origin, so crop-local pupil coordinates equal image coordinates.
ORIGIN_BOX = (0, 0, 1000, 1000)

# Eye width (inner→outer canthus) is 60 px in every fixture below.
LEFT_CORNERS  = ((200.0, 100.0), (140.0, 100.0))   # image-left eye: (inner, outer)
RIGHT_CORNERS = ((260.0, 100.0), (320.0, 100.0))   # image-right eye: (inner, outer)


def _align(left_pupil, right_pupil, left_corners=LEFT_CORNERS, right_corners=RIGHT_CORNERS):
    return compute_corner_alignment(
        left_pupil=left_pupil,
        right_pupil=right_pupil,
        left_crop_box=ORIGIN_BOX,
        right_crop_box=ORIGIN_BOX,
        left_eye_corners=left_corners,
        right_eye_corners=right_corners,
    )


def _rotate(p, deg, centre=(230.0, 100.0)):
    """Rotate a point about the midpoint between the eyes (simulated head roll)."""
    t = math.radians(deg)
    x, y = p[0] - centre[0], p[1] - centre[1]
    return (centre[0] + x * math.cos(t) - y * math.sin(t),
            centre[1] + x * math.sin(t) + y * math.cos(t))


# ─────────────────────────────────────────────────────────────
# Vertical sign convention (regression: sign used to flip between eyes)
# ─────────────────────────────────────────────────────────────

class TestVerticalSign:

    def test_both_pupils_equally_below_are_aligned(self):
        """
        Pupils sitting the same distance below their canthus lines are
        symmetric and must not produce any vertical asymmetry.

        Before the fix, each eye's v was a cross product against its own
        inner→outer axis; the axes point opposite ways, so the eyes got
        opposite signs and this case read as ~13% vertical asymmetry.
        """
        r = _align(left_pupil=(170.0, 104.0), right_pupil=(290.0, 104.0))
        assert r.left_v_ratio == pytest.approx(r.right_v_ratio, abs=1e-6)
        assert r.v_asymmetry == pytest.approx(0.0, abs=1e-6)
        assert r.verdict == ALIGN_ALIGNED

    def test_positive_v_means_below_in_both_eyes(self):
        r = _align(left_pupil=(170.0, 104.0), right_pupil=(290.0, 104.0))
        assert r.left_v_ratio > 0
        assert r.right_v_ratio > 0

    def test_negative_v_means_above_in_both_eyes(self):
        r = _align(left_pupil=(170.0, 96.0), right_pupil=(290.0, 96.0))
        assert r.left_v_ratio < 0
        assert r.right_v_ratio < 0
        assert r.verdict == ALIGN_ALIGNED

    @pytest.mark.parametrize("roll_deg", [-12.0, -5.0, 5.0, 12.0])
    def test_symmetric_eyes_stay_aligned_under_head_roll(self, roll_deg):
        pts = [(170.0, 104.0), (290.0, 104.0), *LEFT_CORNERS, *RIGHT_CORNERS]
        lp, rp, li, lo, ri, ro = (_rotate(p, roll_deg) for p in pts)
        r = _align(lp, rp, left_corners=(li, lo), right_corners=(ri, ro))
        assert r.v_asymmetry == pytest.approx(0.0, abs=1e-6)
        assert r.verdict == ALIGN_ALIGNED


# ─────────────────────────────────────────────────────────────
# A genuine vertical difference is still caught and named correctly
# ─────────────────────────────────────────────────────────────

class TestVerticalDeviation:

    def test_one_pupil_higher_is_borderline_and_named(self):
        # Left pupil 6 px above its line (-0.10 of eye width), right on its line.
        r = _align(left_pupil=(170.0, 94.0), right_pupil=(290.0, 100.0))
        assert r.left_v_ratio == pytest.approx(-0.10, abs=1e-6)
        assert r.right_v_ratio == pytest.approx(0.0, abs=1e-6)
        assert r.v_asymmetry == pytest.approx(0.10, abs=1e-6)
        assert r.verdict == ALIGN_BORDERLINE
        assert "left pupil sits higher" in r.interpretation

    def test_right_pupil_higher_is_named_right(self):
        r = _align(left_pupil=(170.0, 100.0), right_pupil=(290.0, 93.0))
        assert r.v_asymmetry == pytest.approx(7 / 60, abs=1e-4)   # results are rounded to 4 dp
        assert "right pupil sits higher" in r.interpretation

    def test_large_vertical_difference_is_asymmetric(self):
        # 9 px up vs 3 px down → 12 px = 0.20 of eye width
        r = _align(left_pupil=(170.0, 91.0), right_pupil=(290.0, 103.0))
        assert r.v_asymmetry == pytest.approx(0.20, abs=1e-6)
        assert r.verdict == ALIGN_ASYMMETRIC


# ─────────────────────────────────────────────────────────────
# Horizontal axis is unchanged by the fix
# ─────────────────────────────────────────────────────────────

class TestHorizontal:

    def test_centred_pupils(self):
        r = _align(left_pupil=(170.0, 100.0), right_pupil=(290.0, 100.0))
        assert r.left_h_ratio == pytest.approx(0.5)
        assert r.right_h_ratio == pytest.approx(0.5)
        assert r.verdict == ALIGN_ALIGNED

    def test_horizontal_shift_flagged(self):
        # Left pupil 9 px toward its inner canthus → h 0.35 vs 0.50
        r = _align(left_pupil=(179.0, 100.0), right_pupil=(290.0, 100.0))
        assert r.h_asymmetry == pytest.approx(0.15, abs=1e-6)
        assert r.verdict == ALIGN_ASYMMETRIC


class TestUnavailable:

    def test_missing_corners(self):
        r = compute_corner_alignment(
            left_pupil=(170.0, 100.0), right_pupil=(290.0, 100.0),
            left_crop_box=ORIGIN_BOX, right_crop_box=ORIGIN_BOX,
            left_eye_corners=None, right_eye_corners=RIGHT_CORNERS,
        )
        assert r.available is False
