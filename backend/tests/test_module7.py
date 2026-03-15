"""
BeanHealth CLR Tool — Module 7 Test Suite
==========================================

Tests for report generation — schema completeness, INCONCLUSIVE handling,
annotated image encoding, and field accuracy.

Uses synthetic (mock) module outputs — no real images required.

Run:
    cd backend && pytest tests/test_module7.py -v
"""

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.module1_detection   import EyeDetectionResult
from pipeline.module2_pupil       import PupilResult
from pipeline.module3_clr         import CLRResult
from pipeline.module4_displacement import DisplacementResult
from pipeline.module5_asymmetry   import AsymmetryResult
from pipeline.module6_classify    import ClassificationResult
from pipeline.module7_report import (
    build_error_report,
    build_inconclusive_report,
    build_success_report,
    generate_report,
    _crop_to_full,
    _image_to_base64_jpeg,
)
from utils.constants import (
    URGENCY_URGENT,
    URGENCY_ROUTINE,
    URGENCY_MONITOR,
    URGENCY_NORMAL,
    DIRECTION_NASAL,
    DIRECTION_TEMPORAL,
    SEVERITY_MODERATE,
    SEVERITY_NORMAL,
)
from utils.exceptions import CLRError, DetectionError, PipelineError


# ─────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────

def _make_image(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a synthetic RGB image (solid grey)."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _make_crop(h: int = 120, w: int = 250) -> np.ndarray:
    return np.full((h, w, 3), 100, dtype=np.uint8)


def _make_detection() -> EyeDetectionResult:
    left_crop  = _make_crop(120, 250)
    right_crop = _make_crop(120, 250)
    return EyeDetectionResult(
        left_crop=left_crop,
        right_crop=right_crop,
        left_crop_box=(40,  180, 290, 300),   # x1,y1,x2,y2
        right_crop_box=(350, 180, 600, 300),
        left_iris_landmarks=[(165.0, 240.0), (165.0, 225.0),
                              (175.0, 240.0), (165.0, 255.0), (155.0, 240.0)],
        right_iris_landmarks=[(475.0, 240.0), (475.0, 225.0),
                               (485.0, 240.0), (475.0, 255.0), (465.0, 240.0)],
        left_iris_radius_orig=30.0,
        right_iris_radius_orig=30.0,
        face_confidence=0.98,
        warnings=[],
    )


def _make_pupil_result() -> PupilResult:
    return PupilResult(
        left_pupil=(125.0, 60.0),
        right_pupil=(125.0, 60.0),
        left_iris_radius=30.0,
        right_iris_radius=30.0,
        left_confidence="HIGH",
        right_confidence="HIGH",
        left_landmark_centre=(125.0, 60.0),
        right_landmark_centre=(125.0, 60.0),
        left_hough_centre=(124.0, 60.0),
        right_hough_centre=(125.0, 61.0),
        left_hough_radius=30.0,
        right_hough_radius=30.0,
        flags=[],
    )


def _make_clr_result() -> CLRResult:
    return CLRResult(
        left_clr=(135.0, 60.0),
        right_clr=(115.0, 60.0),
        left_clr_confidence=0.95,
        right_clr_confidence=0.90,
        flags=[],
    )


def _make_displacement() -> DisplacementResult:
    return DisplacementResult(
        left_dx=10.0, left_dy=0.0,
        right_dx=-10.0, right_dy=0.0,
        left_displacement_px=10.0,
        right_displacement_px=10.0,
        left_displacement_norm=0.333,
        right_displacement_norm=0.333,
        left_direction=DIRECTION_NASAL,
        right_direction=DIRECTION_NASAL,
        left_angle_rad=0.0,
        right_angle_rad=3.14159,
        flags=[],
    )


def _make_asymmetry() -> AsymmetryResult:
    return AsymmetryResult(
        asymmetry_score=0.0,
        dominant_eye="equal",
        deviation_degrees=13.4,
        deviation_mm=1.914,
        severity=SEVERITY_MODERATE,
        flags=[],
    )


def _make_classification(urgency: str = URGENCY_ROUTINE) -> ClassificationResult:
    return ClassificationResult(
        condition_name="Esotropia",
        icd10_code="H50.01",
        urgency_tier=urgency,
        referral_recommendation="Refer to ophthalmology within 4 weeks",
        timeframe="4 weeks",
        narrative="Moderate asymmetry detected.",
        flags=[],
    )


def _all_outputs(urgency: str = URGENCY_ROUTINE):
    return dict(
        patient_name="Test Child",
        patient_age=5,
        original_img=_make_image(),
        detection=_make_detection(),
        pupil_result=_make_pupil_result(),
        clr_result=_make_clr_result(),
        displacement=_make_displacement(),
        asymmetry=_make_asymmetry(),
        classification=_make_classification(urgency),
    )


# ─────────────────────────────────────────────────────────────
# _crop_to_full — coordinate mapping
# ─────────────────────────────────────────────────────────────

class TestCropToFull:

    def test_origin_maps_to_box_origin(self):
        assert _crop_to_full((0.0, 0.0), (50, 100, 200, 200)) == (50, 100)

    def test_offset_added_correctly(self):
        result = _crop_to_full((10.0, 20.0), (50, 100, 200, 250))
        assert result == (60, 120)

    def test_float_input_rounded(self):
        result = _crop_to_full((10.6, 20.4), (0, 0, 200, 200))
        assert result == (11, 20)


# ─────────────────────────────────────────────────────────────
# _image_to_base64_jpeg
# ─────────────────────────────────────────────────────────────

class TestImageToBase64Jpeg:

    def test_returns_string(self):
        img = _make_image()
        result = _image_to_base64_jpeg(img)
        assert isinstance(result, str)

    def test_decodes_to_valid_jpeg(self):
        """Decoded bytes must start with JPEG magic bytes (0xFF 0xD8)."""
        img = _make_image()
        b64 = _image_to_base64_jpeg(img)
        decoded = base64.b64decode(b64)
        assert decoded[:2] == b"\xff\xd8", "Not a valid JPEG"

    def test_non_empty(self):
        img = _make_image()
        assert len(_image_to_base64_jpeg(img)) > 100


# ─────────────────────────────────────────────────────────────
# build_success_report — schema and field accuracy
# ─────────────────────────────────────────────────────────────

class TestBuildSuccessReport:

    def setup_method(self):
        self.kwargs = _all_outputs()
        self.report = build_success_report(**self.kwargs)

    def test_status_is_success(self):
        assert self.report["status"] == "SUCCESS"

    def test_patient_fields(self):
        assert self.report["patient"]["name"] == "Test Child"
        assert self.report["patient"]["age"]  == 5

    def test_result_keys_present(self):
        required = [
            "urgency_tier", "condition_name", "icd10_code",
            "deviation_degrees", "asymmetry_score", "severity",
            "referral_recommendation", "timeframe", "narrative",
        ]
        for key in required:
            assert key in self.report["result"], f"Missing result key: {key}"

    def test_technical_keys_present(self):
        required = [
            "left_pupil", "right_pupil", "left_clr", "right_clr",
            "left_displacement_norm", "right_displacement_norm",
            "left_direction", "right_direction",
            "deviation_mm", "dominant_eye", "confidence", "flags",
        ]
        for key in required:
            assert key in self.report["technical"], f"Missing technical key: {key}"

    def test_annotated_image_is_valid_base64_jpeg(self):
        b64 = self.report["annotated_image_b64"]
        decoded = base64.b64decode(b64)
        assert decoded[:2] == b"\xff\xd8"

    def test_timestamp_present_and_non_empty(self):
        assert "timestamp" in self.report
        assert len(self.report["timestamp"]) > 10

    def test_result_values_match_modules(self):
        r = self.report["result"]
        assert r["urgency_tier"]   == URGENCY_ROUTINE
        assert r["condition_name"] == "Esotropia"
        assert r["icd10_code"]     == "H50.01"
        assert r["severity"]       == SEVERITY_MODERATE

    def test_technical_values_match_modules(self):
        t = self.report["technical"]
        assert t["left_displacement_norm"]  == round(0.333, 4)
        assert t["right_displacement_norm"] == round(0.333, 4)
        assert t["left_direction"]          == DIRECTION_NASAL
        assert t["dominant_eye"]            == "equal"
        assert t["confidence"]              == "HIGH"

    def test_flags_is_list(self):
        assert isinstance(self.report["technical"]["flags"], list)

    def test_pupil_coords_are_lists(self):
        t = self.report["technical"]
        assert isinstance(t["left_pupil"],  list)
        assert isinstance(t["right_pupil"], list)
        assert len(t["left_pupil"])  == 2
        assert len(t["right_pupil"]) == 2

    def test_flags_merged_from_all_modules(self):
        """Flags from all modules must appear merged in technical.flags."""
        outputs = _all_outputs()
        outputs["detection"].warnings.append("low_confidence")
        outputs["displacement"].flags.append("large_displacement_left")
        outputs["classification"].flags.append("borderline_asymmetry")
        report = build_success_report(**outputs)
        flags = report["technical"]["flags"]
        assert "low_confidence"         in flags
        assert "large_displacement_left" in flags
        assert "borderline_asymmetry"   in flags

    def test_flags_deduplicated(self):
        """Duplicate flags from multiple modules appear only once."""
        outputs = _all_outputs()
        outputs["displacement"].flags.append("pupil_disagreement_left")
        outputs["asymmetry"].flags.append("pupil_disagreement_left")
        report = build_success_report(**outputs)
        flags = report["technical"]["flags"]
        assert flags.count("pupil_disagreement_left") == 1


# ─────────────────────────────────────────────────────────────
# build_inconclusive_report
# ─────────────────────────────────────────────────────────────

class TestBuildInconclusiveReport:

    def test_status_is_inconclusive(self):
        err = CLRError("no_flash")
        r = build_inconclusive_report(err)
        assert r["status"] == "INCONCLUSIVE"

    def test_reason_code_in_report(self):
        err = CLRError("no_flash")
        r = build_inconclusive_report(err)
        assert r["reason"] == "no_flash"

    def test_reason_human_is_non_empty(self):
        err = CLRError("no_reflex_left")
        r = build_inconclusive_report(err)
        assert isinstance(r["reason_human"], str)
        assert len(r["reason_human"]) > 5

    def test_error_code_in_flags(self):
        err = DetectionError("no_face")
        r = build_inconclusive_report(err)
        assert "no_face" in r["flags"]

    def test_patient_included_when_provided(self):
        err = CLRError("no_flash")
        r = build_inconclusive_report(err, patient_name="Alice", patient_age=4)
        assert r["patient"]["name"] == "Alice"
        assert r["patient"]["age"]  == 4

    def test_patient_omitted_when_not_provided(self):
        err = CLRError("no_flash")
        r = build_inconclusive_report(err)
        assert "patient" not in r

    def test_extra_flags_included(self):
        err = DetectionError("eyes_closed")
        r = build_inconclusive_report(err, extra_flags=["low_confidence"])
        assert "low_confidence" in r["flags"]

    def test_timestamp_present(self):
        err = CLRError("no_flash")
        r = build_inconclusive_report(err)
        assert "timestamp" in r

    def test_no_result_key(self):
        """INCONCLUSIVE report must NOT contain a result block."""
        err = CLRError("no_flash")
        r = build_inconclusive_report(err)
        assert "result" not in r

    def test_no_annotated_image(self):
        """INCONCLUSIVE report must NOT contain an annotated image."""
        err = CLRError("no_flash")
        r = build_inconclusive_report(err)
        assert "annotated_image_b64" not in r


# ─────────────────────────────────────────────────────────────
# build_error_report
# ─────────────────────────────────────────────────────────────

class TestBuildErrorReport:

    def test_status_is_error(self):
        r = build_error_report(RuntimeError("unexpected crash"))
        assert r["status"] == "ERROR"

    def test_message_present(self):
        r = build_error_report(RuntimeError("unexpected crash"))
        assert "message" in r
        assert len(r["message"]) > 0

    def test_no_traceback_exposed(self):
        """Raw traceback must never be in the response."""
        r = build_error_report(RuntimeError("divide by zero"))
        assert "traceback" not in r
        assert "Traceback" not in str(r)

    def test_patient_included_when_provided(self):
        r = build_error_report(RuntimeError("crash"), "Bob", 6)
        assert r["patient"]["name"] == "Bob"

    def test_timestamp_present(self):
        r = build_error_report(RuntimeError("crash"))
        assert "timestamp" in r


# ─────────────────────────────────────────────────────────────
# generate_report — public API routing
# ─────────────────────────────────────────────────────────────

class TestGenerateReport:

    def test_success_path(self):
        r = generate_report(**_all_outputs())
        assert r["status"] == "SUCCESS"

    def test_detection_error_gives_inconclusive(self):
        err = DetectionError("no_face")
        r = generate_report(
            patient_name="Test", patient_age=5,
            original_img=None, error=err,
        )
        assert r["status"] == "INCONCLUSIVE"
        assert r["reason"] == "no_face"

    def test_clr_error_gives_inconclusive(self):
        err = CLRError("no_flash")
        r = generate_report(
            patient_name="Test", patient_age=5,
            original_img=None, error=err,
        )
        assert r["status"] == "INCONCLUSIVE"
        assert r["reason"] == "no_flash"

    def test_generic_exception_gives_error(self):
        r = generate_report(
            patient_name="Test", patient_age=5,
            original_img=None,
            error=RuntimeError("unexpected failure"),
        )
        assert r["status"] == "ERROR"

    def test_missing_modules_without_error_gives_error(self):
        """Calling with all None modules but no error → caught internally → ERROR."""
        r = generate_report(
            patient_name="Test", patient_age=5,
            original_img=None,
            detection=None, pupil_result=None, clr_result=None,
            displacement=None, asymmetry=None, classification=None,
            error=None,
        )
        assert r["status"] == "ERROR"

    def test_always_returns_dict(self):
        """generate_report must never raise — always returns a dict."""
        r = generate_report(
            patient_name="T", patient_age=1,
            original_img=None,
            error=Exception("unknown"),
        )
        assert isinstance(r, dict)
        assert "status" in r

    # ── Urgency border colour verified via classification output ──

    @pytest.mark.parametrize("urgency", [
        URGENCY_URGENT, URGENCY_ROUTINE, URGENCY_MONITOR, URGENCY_NORMAL
    ])
    def test_all_urgency_tiers_produce_success_report(self, urgency):
        r = generate_report(**_all_outputs(urgency=urgency))
        assert r["status"] == "SUCCESS"
        assert r["result"]["urgency_tier"] == urgency
