"""
BeanHealth CLR Tool — Capture Protocol & Frame QC Tests (Phase 3, milestone 4)
==============================================================================

  • The capture plan: positions, head-pose targets (head turns opposite to
    the eyes), tilt roll ranges — all read from the KB.
  • The session format rejects malformed uploads.
  • Frame QC: one test per rejection reason, both eyes required, missing
    observations fail closed, saccade edges, the jitter false-rejection rate,
    per-position sufficiency and session flags.
  • The reference sharpness measure, and the /exam API.

Frame tracks come from exam.sim.frames — synthetic, not physiology.

Run:
    cd backend && pytest tests/test_capture_qc.py -v
"""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from exam.capture import CAPTURE_PROTOCOL_VERSION, GAZE_DIRECTION, CaptureSession, build_plan
from exam.kb import load_kb
from exam.kb.loader import DEFAULT_KB_DIR
from exam.qc import run_qc, sharpness
from exam.sim import FrameNoise, SessionSpec, events, simulate_position, simulate_session

pytestmark = pytest.mark.unit

PROTOCOL_ENTRIES = [
    "capture_distance_min_cm", "capture_distance_max_cm", "capture_distance_tolerance_cm",
    "capture_min_fps", "capture_max_exposure_s", "capture_recommended_frame_height_px",
    "gaze_eccentricity_deg", "head_tilt_min_deg", "head_tilt_max_deg", "hold_duration_s",
]
QC_ENTRIES = [
    "qc_head_pose_tolerance_deg", "qc_blink_openness_ratio", "qc_max_limbus_covered_fraction",
    "qc_sharpness_canonical_iris_radius_px", "qc_sharpness_reference_percentile",
    "qc_min_relative_sharpness", "qc_saccade_velocity_deg_s", "qc_saccade_noise_factor",
]


def kb_with(tmp_path, **values):
    """A KB copy with some capture.yaml values changed (not hash-checked)."""
    dst = tmp_path / "kb"
    shutil.copytree(DEFAULT_KB_DIR, dst)
    path = dst / "capture.yaml"
    data = yaml.safe_load(path.read_text())
    for e in data["entries"]:
        if e["id"] in values:
            e["value"] = values[e["id"]]
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return load_kb(dst, verify_hash=False)


def position_qc(spec=None, name="primary", **kw):
    q = run_qc(simulate_session(spec or SessionSpec(**kw)))
    return next(p for p in q.positions if p.position == name)


def rejected(pqc):
    """{frame index: set of reason codes} for the rejected frames."""
    return {v.index: {r.code for r in v.reasons} for v in pqc.frames if not v.valid}


# ─────────────────────────────────────────────────────────────
# KB
# ─────────────────────────────────────────────────────────────

class TestCaptureKB:
    @pytest.mark.parametrize("entry_id", PROTOCOL_ENTRIES + QC_ENTRIES)
    def test_every_capture_value_is_set_and_sourced(self, entry_id):
        e = load_kb().scalar(entry_id)
        assert e.value is not None
        assert e.source.strip()
        assert e.review_status != "approved"

    @pytest.mark.parametrize("entry_id", PROTOCOL_ENTRIES)
    def test_brief_values_await_clinician_review(self, entry_id):
        e = load_kb().scalar(entry_id)
        assert e.review_status == "needs_clinician_review"
        assert "brief" in e.source

    @pytest.mark.parametrize("entry_id", QC_ENTRIES)
    def test_qc_gates_are_engineering_drafts(self, entry_id):
        e = load_kb().scalar(entry_id)
        assert e.review_status == "draft"
        assert "engineering" in e.source.lower()


# ─────────────────────────────────────────────────────────────
# Protocol plan
# ─────────────────────────────────────────────────────────────

class TestPlan:
    def test_nine_gaze_order_with_tilts(self):
        plan = build_plan("nine_gaze")
        assert [p.position for p in plan.positions] == [
            "primary", "right", "left", "up", "down",
            "up_right", "up_left", "down_right", "down_left", "tilt_right", "tilt_left"]
        assert plan.protocol_version == CAPTURE_PROTOCOL_VERSION

    def test_kids_mode_and_no_tilt(self):
        assert [p.position for p in build_plan("five_gaze", include_tilt=False).positions] == [
            "primary", "right", "left", "up", "down"]

    @pytest.mark.parametrize("position", list(GAZE_DIRECTION))
    def test_head_turns_opposite_to_the_eyes(self, position):
        p = build_plan("nine_gaze").position(position)
        h, v = p.eye_in_orbit_deg
        assert (p.head.yaw_deg, p.head.pitch_deg) == (-h, -v)
        assert p.head.roll_deg is None

    def test_right_gaze_turns_the_head_to_the_patients_left(self):
        p = build_plan().position("right")
        assert p.eye_in_orbit_deg == (25.0, 0.0)
        assert p.head.yaw_deg == -25.0
        assert "patient's left" in p.instruction

    def test_up_gaze_lowers_the_chin(self):
        p = build_plan().position("up")
        assert p.head.pitch_deg == -25.0
        assert "lower the chin" in p.instruction

    def test_tilt_roll_ranges(self):
        plan = build_plan()
        right, left = plan.position("tilt_right"), plan.position("tilt_left")
        assert (right.head.roll_deg.min, right.head.roll_deg.max) == (30, 45)
        assert (left.head.roll_deg.min, left.head.roll_deg.max) == (-45, -30)
        assert right.head.yaw_deg == right.head.pitch_deg == 0

    def test_plan_reads_the_kb(self, tmp_path):
        kb = kb_with(tmp_path, gaze_eccentricity_deg=20, qc_head_pose_tolerance_deg=4, hold_duration_s=3)
        p = build_plan("nine_gaze", kb=kb).position("down_left")
        assert p.eye_in_orbit_deg == (-20, -20)
        assert p.head.tolerance_deg == 4
        assert p.hold_s == 3
        assert p.min_valid_frames == kb.threshold("min_valid_frames_per_position").value

    def test_unset_kb_value_fails_loudly(self, tmp_path):
        with pytest.raises(ValueError, match="gaze_eccentricity_deg"):
            build_plan(kb=kb_with(tmp_path, gaze_eccentricity_deg=None))

    def test_plan_carries_kb_version(self):
        plan = build_plan()
        assert (plan.kb_version, plan.kb_sha256) == (load_kb().version, load_kb().sha256)


# ─────────────────────────────────────────────────────────────
# Session format
# ─────────────────────────────────────────────────────────────

class TestSession:
    def _raw(self):
        return simulate_session(SessionSpec(n_frames=5)).model_dump()

    def test_round_trips_through_json(self):
        s = simulate_session(SessionSpec(n_frames=5))
        assert CaptureSession.model_validate_json(s.model_dump_json()) == s

    def test_unknown_protocol_version_rejected(self):
        raw = self._raw()
        raw["header"]["protocol_version"] = "0.9.0"
        with pytest.raises(ValidationError, match="not supported"):
            CaptureSession.model_validate(raw)

    def test_position_outside_protocol_rejected(self):
        raw = simulate_session(SessionSpec(protocol="five_gaze", n_frames=5)).model_dump()
        raw["captures"][0]["position"] = "up_right"
        with pytest.raises(ValidationError, match="not a position"):
            CaptureSession.model_validate(raw)

    def test_duplicate_position_rejected(self):
        raw = self._raw()
        raw["captures"].append(raw["captures"][0])
        with pytest.raises(ValidationError, match="twice"):
            CaptureSession.model_validate(raw)

    def test_frames_must_be_in_order(self):
        raw = self._raw()
        frames = raw["captures"][0]["frames"]
        frames[1], frames[2] = frames[2], frames[1]
        with pytest.raises(ValidationError, match="increasing"):
            CaptureSession.model_validate(raw)

    def test_unknown_field_rejected(self):
        raw = self._raw()
        raw["captures"][0]["frames"][0]["video_frame_b64"] = "..."
        with pytest.raises(ValidationError):
            CaptureSession.model_validate(raw)


# ─────────────────────────────────────────────────────────────
# Frame QC — one reason at a time
# ─────────────────────────────────────────────────────────────

class TestFrameRejections:
    def test_clean_session_passes_every_frame(self):
        q = run_qc(simulate_session())
        assert all(p.status == "OK" and p.frames_valid == p.frames_total == 60 for p in q.positions)
        assert q.insufficient_positions == []

    @pytest.mark.parametrize("event,code", [
        ("blink", "blink"),
        ("blur", "motion_blur"),
        ("head_off_target", "head_pose_off_target"),
        ("no_reflex_right", "reflex_not_found"),
        ("lid_low", "lid_covers_limbus"),
    ])
    def test_single_event_rejects_only_that_frame(self, event, code):
        pqc = position_qc(events={"primary": events(**{event: [30]})})
        assert rejected(pqc) == {30: {code}}
        assert pqc.reject_counts == {code: 1}

    def test_no_face_is_the_only_reason(self):
        pqc = position_qc(events={"primary": events(no_face=[12])})
        assert rejected(pqc) == {12: {"no_face"}}

    def test_reasons_name_the_eye(self):
        pqc = position_qc(events={"primary": events(no_reflex_right=[8])})
        (reason,) = next(v for v in pqc.frames if v.index == 8).reasons
        assert reason.eye == "R"

    def test_glance_rejects_the_frame_and_both_edges(self):
        pqc = position_qc(events={"primary": events(glance=[20])})
        assert rejected(pqc) == {19: {"saccade"}, 20: {"saccade"}, 21: {"saccade"}}

    def test_head_movement_while_fixating_is_not_a_saccade(self):
        # The head pose changes, the reflex-to-pupil vector does not: no saccade.
        pqc = position_qc(events={"primary": events(head_off_target=list(range(10, 20)))})
        assert set(pqc.reject_counts) == {"head_pose_off_target"}

    def test_multiple_reasons_are_all_logged(self):
        pqc = position_qc(events={"primary": events(blink=[5], blur=[5], head_off_target=[5])})
        assert rejected(pqc)[5] == {"blink", "motion_blur", "head_pose_off_target"}

    def test_tilt_roll_outside_range_is_off_target(self, tmp_path):
        kb = kb_with(tmp_path, head_tilt_min_deg=40, head_tilt_max_deg=45)
        plan_kb = build_plan(kb=kb)
        # Frames generated for the shipped 30–45° plan sit at 37.5°, outside 40–45°.
        cap = simulate_position(build_plan().position("tilt_right"), n_frames=10)
        session = simulate_session(SessionSpec(n_frames=10)).model_copy(
            update={"captures": [cap]})
        q = run_qc(session, kb=kb)
        tilt = next(p for p in q.positions if p.position == "tilt_right")
        assert tilt.frames_valid == 0
        assert plan_kb.position("tilt_right").head.roll_deg.min == 40
        assert "roll" in tilt.frames[0].reasons[0].detail


class TestFailClosed:
    """A check that cannot run rejects the frame — never passes it."""

    def _session_with(self, change):
        raw = simulate_session(SessionSpec(n_frames=12)).model_dump()
        change(raw["captures"][0]["frames"][6])
        return CaptureSession.model_validate(raw)

    def _primary(self, session):
        return next(p for p in run_qc(session).positions if p.position == "primary")

    @pytest.mark.parametrize("field", ["fissure_height_px", "limbus_covered_fraction", "pupil_px"])
    def test_missing_eye_value(self, field):
        pqc = self._primary(self._session_with(lambda f: f["eyes"]["L"].update({field: None})))
        assert "missing_observation" in rejected(pqc)[6]

    def test_missing_sharpness(self):
        pqc = self._primary(self._session_with(lambda f: f.update(sharpness=None)))
        assert rejected(pqc)[6] == {"missing_observation"}

    def test_missing_eye(self):
        pqc = self._primary(self._session_with(lambda f: f["eyes"].pop("R")))
        assert "eye_not_found" in rejected(pqc)[6]

    def test_isolated_frame_saccade_unverifiable(self):
        # Every other frame dropped at 30 fps leaves 67 ms gaps (< 100 ms): fine.
        pqc = position_qc(n_frames=20, events={"primary": events(dropped=range(1, 20, 2))})
        assert pqc.frames_valid == 10
        # Frames whose nearest neighbours are 200 ms away cannot be checked.
        pqc = position_qc(n_frames=20, events={"primary": events(dropped=[*range(1, 6), *range(7, 12)])})
        assert rejected(pqc) == {0: {"saccade_unverifiable"}, 6: {"saccade_unverifiable"}}


# ─────────────────────────────────────────────────────────────
# Relative blur and noise-aware saccades
# ─────────────────────────────────────────────────────────────

class TestAdaptiveGates:
    def test_blur_is_relative_to_the_position(self):
        # A frame at 40% of its neighbours' sharpness is blurred, whatever the absolute level.
        assert rejected(position_qc(events={"primary": events(blur=[7])})) == {7: {"motion_blur"}}

    def test_uniformly_blurred_position_passes(self):
        # Known limitation (capture.yaml): no fixed level works across phones,
        # so a position blurred in every frame is not caught by this check.
        pqc = position_qc(events={"primary": events(blur=range(60))})
        assert "motion_blur" not in pqc.reject_counts

    def test_jitter_alone_rarely_rejects(self):
        total = flagged = 0
        for seed in range(10):
            q = run_qc(simulate_session(SessionSpec(noise=FrameNoise(landmark_px=1.0, seed=seed * 100))))
            total += sum(p.frames_total for p in q.positions)
            flagged += sum(p.reject_counts.get("saccade", 0) for p in q.positions)
        assert flagged / total < 0.02

    def test_glance_still_caught_on_a_noisy_capture(self):
        pqc = position_qc(SessionSpec(noise=FrameNoise(landmark_px=1.0, seed=3),
                                      events={"primary": events(glance=[30], glance_deg=10)}))
        assert {29, 30, 31} <= set(rejected(pqc))


# ─────────────────────────────────────────────────────────────
# Positions and session
# ─────────────────────────────────────────────────────────────

class TestSufficiency:
    def test_too_few_valid_frames(self):
        pqc = position_qc(events={"up": events(blink=range(0, 55))}, name="up")
        assert pqc.status == "INSUFFICIENT_DATA"
        assert pqc.reason == "too_few_valid_frames"
        assert pqc.frames_valid < pqc.min_valid_frames == load_kb().threshold("min_valid_frames_per_position").value

    def test_exactly_the_minimum_is_enough(self):
        need = int(load_kb().threshold("min_valid_frames_per_position").value)
        pqc = position_qc(n_frames=need)
        assert (pqc.status, pqc.frames_valid) == ("OK", need)

    def test_missing_position_is_not_captured(self):
        q = run_qc(simulate_session(SessionSpec(skip=("down_left", "tilt_left"))))
        assert q.insufficient_positions == ["down_left", "tilt_left"]
        dl = next(p for p in q.positions if p.position == "down_left")
        assert (dl.reason, dl.frames_total) == ("not_captured", 0)

    def test_tilts_optional_when_not_expected(self):
        session = simulate_session(SessionSpec(include_tilt=False))
        assert run_qc(session, include_tilt=False).insufficient_positions == []
        assert run_qc(session).insufficient_positions == ["tilt_right", "tilt_left"]

    def test_qc_carries_versions(self):
        q = run_qc(simulate_session(SessionSpec(n_frames=12)))
        assert q.kb_version == load_kb().version and q.qc_version


class TestSessionFlags:
    def flags(self, **kw):
        return {f.code for f in run_qc(simulate_session(SessionSpec(n_frames=12, **kw))).session_flags}

    def test_good_setup_has_no_flags(self):
        assert self.flags() == set()

    @pytest.mark.parametrize("distance,flagged", [(29.0, True), (30.0, False), (43.0, False), (44.0, True)])
    def test_distance_range_with_tolerance(self, distance, flagged):
        assert ("distance_out_of_range" in self.flags(distance_cm=distance)) is flagged

    def test_other_flags(self):
        assert self.flags(distance_cm=None) == {"distance_not_estimated"}
        assert self.flags(fps=24.0) == {"low_fps"}
        assert self.flags(exposure_s=None) == {"exposure_unknown"}
        assert self.flags(exposure_s=1 / 60) == {"exposure_too_long"}
        assert self.flags(width_px=1920, height_px=1080) == {"below_recommended_resolution"}

    def test_portrait_4k_is_not_low_resolution(self):
        assert self.flags(width_px=2160, height_px=3840) == set()


# ─────────────────────────────────────────────────────────────
# Reference sharpness measure
# ─────────────────────────────────────────────────────────────

def _eye_image(scale=1.0):
    """A synthetic eye: dark iris disc with a bright reflex on a mid-grey field."""
    size = int(400 * scale)
    img = np.full((size, size), 150, np.uint8)
    c = (size // 2, size // 2)
    cv2.circle(img, c, int(60 * scale), 70, -1)
    cv2.circle(img, c, int(25 * scale), 20, -1)
    cv2.circle(img, (c[0] + int(8 * scale), c[1] - int(6 * scale)), max(2, int(4 * scale)), 255, -1)
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8), c, 60 * scale


def _motion_blur(img, n):
    k = np.zeros((n, n)); k[n // 2, :] = 1.0 / n
    return cv2.filter2D(img, -1, k)


class TestSharpness:
    def test_blur_lowers_sharpness_monotonically(self):
        img, c, r = _eye_image()
        values = [sharpness(_motion_blur(img, n) if n else img, c, r) for n in (0, 3, 7, 15)]
        assert values == sorted(values, reverse=True)
        assert values[2] < 0.6 * values[0]

    def test_roughly_scale_free(self):
        small, c1, r1 = _eye_image(1.0)
        big, c2, r2 = _eye_image(2.0)
        a, b = sharpness(small, c1, r1), sharpness(big, c2, r2)
        assert abs(a - b) / max(a, b) < 0.35

    def test_needs_grey_image_inside_the_frame(self):
        img, c, r = _eye_image()
        with pytest.raises(ValueError, match="single-channel"):
            sharpness(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), c, r)
        with pytest.raises(ValueError, match="outside"):
            sharpness(img, (-500, -500), r)


# ─────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_protocol_endpoint(self, client):
        r = client.get("/exam/protocol/five_gaze", params={"tilt": "false"})
        assert r.status_code == 200
        assert [p["position"] for p in r.json()["positions"]] == ["primary", "right", "left", "up", "down"]

    def test_unknown_protocol_is_422(self, client):
        assert client.get("/exam/protocol/twelve_gaze").status_code == 422

    def test_qc_endpoint(self, client):
        session = simulate_session(SessionSpec(n_frames=12, skip=("tilt_left",)))
        r = client.post("/exam/qc", content=session.model_dump_json(),
                        headers={"content-type": "application/json"})
        assert r.status_code == 200
        assert r.json()["insufficient_positions"] == ["tilt_left"]

    def test_malformed_session_is_422(self, client):
        assert client.post("/exam/qc", json={"header": {}, "captures": []}).status_code == 422
