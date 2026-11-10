"""
Tests for Module 2b (iris-edge centre), Module 5's horizontal-only mode and
Module 3's iris-bound reflex filter.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pipeline.module2_iris_edge import fit_iris_edge, refine_with_iris_edge
from pipeline.module2_pupil import PupilResult
from pipeline.module3_clr import detect_clr
from pipeline.module5_asymmetry import compute_asymmetry_and_angle
from utils.exceptions import CLRError, PupilError


def _eye(centre, r, size=(240, 320), lids=False):
    """Grey-white sclera with a dark iris disk; optional lids over the top and bottom."""
    img = np.full((size[0], size[1], 3), 225, np.uint8)
    cv2.circle(img, (int(round(centre[0] * 8)), int(round(centre[1] * 8))), int(round(r * 8)),
               (45, 35, 30), -1, cv2.LINE_AA, shift=3)
    if lids:
        img[: int(centre[1] - 0.75 * r)] = (120, 90, 80)
        img[int(centre[1] + 0.75 * r):] = (120, 90, 80)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def _pupil_result(lm_left, lm_right, r_left, r_right) -> PupilResult:
    return PupilResult(
        left_pupil=lm_left, right_pupil=lm_right,
        left_iris_radius=r_left, right_iris_radius=r_right,
        left_confidence="HIGH", right_confidence="HIGH",
        left_landmark_centre=lm_left, right_landmark_centre=lm_right,
        left_hough_centre=None, right_hough_centre=None,
        left_hough_radius=None, right_hough_radius=None,
        flags=["pupil_from_darkblob_left"],
    )


class TestFit:

    def test_recovers_centre_and_radius(self):
        gray = cv2.cvtColor(_eye((161.3, 118.6), 40.0), cv2.COLOR_RGB2GRAY)
        fit = fit_iris_edge(gray, (156.0, 122.0), 37.0)      # hint off by ~6 px and 3 px radius
        assert fit is not None
        assert math.hypot(fit.centre[0] - 161.3, fit.centre[1] - 118.6) < 0.5
        assert abs(fit.radius - 40.0) < 1.0

    def test_horizontal_centre_survives_lids(self):
        gray = cv2.cvtColor(_eye((160.0, 120.0), 40.0, lids=True), cv2.COLOR_RGB2GRAY)
        fit = fit_iris_edge(gray, (157.0, 120.0), 40.0)
        assert fit is not None
        assert abs(fit.centre[0] - 160.0) < 0.5

    def test_no_edge_returns_none(self):
        gray = np.full((240, 320), 120, np.uint8)
        assert fit_iris_edge(gray, (160.0, 120.0), 40.0) is None


class TestRefine:

    def _two_eyes(self, r_left, r_right, lc=(560.0, 200.0), rc=(200.0, 200.0)):
        img = np.full((400, 760, 3), 225, np.uint8)
        for (cx, cy), r in ((lc, r_left), (rc, r_right)):
            cv2.circle(img, (int(cx), int(cy)), int(r), (45, 35, 30), -1, cv2.LINE_AA)
        img = cv2.GaussianBlur(img, (0, 0), 1.0)
        boxes = SimpleNamespace(left_crop_box=(460, 100, 660, 300), right_crop_box=(100, 100, 300, 300))
        return img, boxes

    def test_centres_and_shared_radius(self):
        full, det = self._two_eyes(40, 38)
        pr = _pupil_result((97.0, 103.0), (102.0, 98.0), 36.0, 36.0)    # landmark centres a few px off
        out = refine_with_iris_edge(full, 1.0, det, pr)
        assert math.hypot(out.left_pupil[0] - 100, out.left_pupil[1] - 100) < 1.0
        assert math.hypot(out.right_pupil[0] - 100, out.right_pupil[1] - 100) < 1.0
        assert out.left_iris_radius == out.right_iris_radius == pytest.approx(39, abs=1.0)
        assert "centre_from_iris_edge" in out.flags
        assert "pupil_from_darkblob_left" not in out.flags

    def test_scale_maps_back_to_working_frame(self):
        full, det = self._two_eyes(40, 40)
        work_det = SimpleNamespace(left_crop_box=(230, 50, 330, 150), right_crop_box=(50, 50, 150, 150))
        pr = _pupil_result((49.0, 51.0), (51.0, 49.0), 19.0, 19.0)      # working image is half size
        out = refine_with_iris_edge(full, 2.0, work_det, pr)
        assert math.hypot(out.left_pupil[0] - 50, out.left_pupil[1] - 50) < 0.5
        assert out.left_iris_radius == pytest.approx(20, abs=0.6)

    def test_size_mismatch_is_refused(self):
        full, det = self._two_eyes(40, 30)
        pr = _pupil_result((100.0, 100.0), (100.0, 100.0), 36.0, 33.0)
        with pytest.raises(PupilError) as e:
            refine_with_iris_edge(full, 1.0, det, pr)
        assert e.value.code == "iris_size_mismatch"

    def test_missing_edge_is_refused(self):
        full, det = self._two_eyes(40, 40)
        full[100:300, 100:300] = 120                                     # right eye: no iris at all
        pr = _pupil_result((100.0, 100.0), (100.0, 100.0), 40.0, 40.0)
        with pytest.raises(PupilError) as e:
            refine_with_iris_edge(full, 1.0, det, pr)
        assert e.value.code == "iris_edge_not_found_right"

    def test_never_raises_a_lowered_confidence(self):
        full, det = self._two_eyes(40, 40)
        pr = _pupil_result((100.0, 100.0), (100.0, 100.0), 40.0, 40.0)
        pr.left_confidence = "LOW"
        out = refine_with_iris_edge(full, 1.0, det, pr)
        assert out.left_confidence == "LOW"


class TestHorizontalOnly:

    def _asym(self, ldx, ldy, rdx, rdy, r=40.0):
        return compute_asymmetry_and_angle(
            math.hypot(ldx, ldy) / r, math.hypot(rdx, rdy) / r,
            left_dx=ldx, left_dy=ldy, right_dx=rdx, right_dy=rdy,
            left_iris_radius=r, right_iris_radius=r, horizontal_only=True,
        )

    def test_vertical_part_is_not_scored(self):
        a = self._asym(-3.0, -12.0, 3.0, 0.0)          # kappa only, plus a large vertical offset in the left eye
        assert a.asymmetry_degrees == pytest.approx(0.0, abs=1e-6)
        assert "horizontal_only" in a.flags

    def test_left_exo_reads_nasal_reflex_in_the_left_eye(self):
        a = self._asym(-14.0, 0.0, 3.0, 0.0)          # left reflex well nasal (image-left) of its centre
        assert a.dominant_eye == "left"
        assert a.horizontal_direction == "nasal"
        assert a.bav_nasal > 0

    def test_right_eso_reads_temporal_reflex_in_the_right_eye(self):
        a = self._asym(-3.0, 0.0, -12.0, 0.0)          # right reflex temporal (image-left)
        assert a.dominant_eye == "right"
        assert a.horizontal_direction == "temporal"

    def test_two_d_mode_unchanged(self):
        a = compute_asymmetry_and_angle(0.3, 0.075, left_dx=0.0, left_dy=-12.0, right_dx=3.0, right_dy=0.0,
                                        left_iris_radius=40.0, right_iris_radius=40.0)
        assert a.horizontal_direction is None
        assert a.asymmetry_degrees > 10


class TestReflexInsideIris:

    def test_spot_on_sclera_is_rejected_at_one_radius(self):
        crop = np.full((120, 160, 3), 60, np.uint8)
        cv2.circle(crop, (80, 60), 28, (30, 25, 20), -1)
        cv2.circle(crop, (80 + 36, 60), 3, (255, 255, 255), -1)          # bright spot 1.3 r from the centre
        kw = dict(left_pupil=(80.0, 60.0), right_pupil=(80.0, 60.0), min_peak_brightness=180.0,
                  min_area_ratio=0.0003, max_area_ratio=0.25)
        assert detect_clr(crop, crop, 28.0, 28.0, **kw).left_clr is not None          # old 1.5 r bound admits it
        with pytest.raises(CLRError):
            detect_clr(crop, crop, 28.0, 28.0, max_dist_iris_radii=1.0, **kw)
