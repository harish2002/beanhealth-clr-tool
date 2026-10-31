"""
BeanHealth CLR Tool — Tier 2 Stub Tests (Phase 3, milestone 9)
===============================================================

  • Saccade detector on synthetic 240 fps traces: every injected saccade
    found, with onset, amplitude and direction; nothing on steady noisy
    fixation; tracking gaps never create a saccade; single-sample spikes
    rejected; the threshold rises with the noise.
  • Cover-test session format and its recording rules (fps, resolution,
    cover time, time after uncover).
  • Movement module: the covered eye's movement under cover and on uncover,
    signs in the patient's frame, direction only beyond k × error, and no
    clinical interpretation.
  • POST /exam/tier2/cover-test.

Traces come from exam.tier2.sim — test fixtures, not physiology.

Run:
    cd backend && pytest tests/test_tier2.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from exam.kb import load_kb  # noqa: E402
from exam.tier2 import CoverTrial, analyse_cover_test, check_session, detect_saccades, movements_by  # noqa: E402
from exam.tier2.sim import cover_session, cover_trial, gaze_track, saccade_duration_ms  # noqa: E402

KB = load_kb()
T = np.arange(0.0, 2500.0, 1000.0 / 240)
MOVES = [(300.0, (5.0, 0.0)), (900.0, (-2.0, 1.0)), (1500.0, (0.0, -10.0)), (2100.0, (-8.0, -3.0))]


class TestSaccades:
    def test_finds_every_saccade(self):
        trace = detect_saccades(T, gaze_track(T, (0.0, 0.0), MOVES), KB)
        assert len(trace.saccades) == len(MOVES)
        for s, (onset, (dh, dv)) in zip(trace.saccades, MOVES):
            assert abs(s.onset_ms - onset) <= 2 * 1000 / 240
            assert s.amplitude_deg == pytest.approx(np.hypot(dh, dv), rel=0.1, abs=0.25)
            for got, want in ((s.horizontal_deg, dh), (s.vertical_deg, dv)):
                assert abs(got - want) < 0.3 if want == 0 else np.sign(got) == np.sign(want)
            # The 5-point derivative smears each end by ~2 samples (≈ 8 ms at 240 fps).
            nominal = saccade_duration_ms(np.hypot(dh, dv))
            assert nominal - 5 <= s.duration_ms <= nominal + 25

    @pytest.mark.parametrize("seed", range(5))
    def test_steady_fixation_has_none(self, seed):
        assert detect_saccades(T, gaze_track(T, (2.0, -1.0), [], seed=seed), KB).saccades == []

    def test_gaps_are_not_saccades(self):
        g = gaze_track(T, (0.0, 0.0), [])
        g = [None if 400 <= t <= 480 or 1200 <= t <= 1300 else x for t, x in zip(T, g)]
        assert detect_saccades(T, g, KB).saccades == []

    def test_saccade_next_to_a_gap_is_still_found(self):
        g = gaze_track(T, (0.0, 0.0), [(1000.0, (6.0, 0.0))])
        g = [None if 700 <= t <= 800 else x for t, x in zip(T, g)]
        found = detect_saccades(T, g, KB).saccades
        assert len(found) == 1 and found[0].horizontal_deg == pytest.approx(6.0, abs=0.3)

    def test_single_sample_spike_is_noise(self):
        g = gaze_track(T, (0.0, 0.0), [])
        i = 300
        g[i] = (g[i][0] + 1.5, g[i][1])
        assert detect_saccades(T, g, KB).saccades == []

    def test_threshold_follows_the_noise(self):
        quiet = detect_saccades(T, gaze_track(T, (0, 0), [], noise_deg=0.01), KB).threshold_deg_s
        noisy = detect_saccades(T, gaze_track(T, (0, 0), [], noise_deg=0.2), KB).threshold_deg_s
        assert quiet == KB.scalar("saccade_velocity_floor_deg_s").value and noisy > 5 * quiet

    def test_inputs(self):
        with pytest.raises(ValueError):
            detect_saccades([0.0, 1.0], [(0.0, 0.0)], KB)
        assert detect_saccades([], [], KB).saccades == []


class TestSession:
    def test_protocol_met(self):
        assert check_session(cover_session([cover_trial()]), KB) == []

    def test_recording_rules(self):
        s = cover_session([cover_trial(cover_s=1.0, after_s=0.5, fps=120.0)], fps=120.0, height_px=720)
        assert {p.code for p in check_session(s, KB)} == {"low_fps", "low_resolution", "cover_too_short",
                                                         "too_little_after_uncover"}

    def test_sparse_samples(self):
        t = cover_trial()
        sparse = t.model_copy(update={"samples": t.samples[::3]})
        assert "sparse_samples" in {p.code for p in check_session(cover_session([sparse]), KB)}

    def test_timing_must_be_ordered(self):
        t = cover_trial()
        with pytest.raises(ValidationError):
            CoverTrial(covered_eye="L", cover_on_ms=500, cover_off_ms=400, end_ms=900, samples=t.samples)


class TestMovement:
    def test_latent_deviation(self):
        # Left eye drifts 3° toward the patient's right (nasal for the left eye) under cover, recovers on uncover.
        r = analyse_cover_test(cover_session([cover_trial("L", (3.0, 0.0))]), KB)
        m = movements_by(r)
        assert m["cover:L"].covered and m["cover:L"].horizontal_direction == "nasal"
        assert m["cover:L"].net_horizontal_deg == pytest.approx(3.0, abs=0.1)
        assert m["cover:L"].net_horizontal_pd == pytest.approx(100 * np.tan(np.radians(3.0)), abs=0.2)
        assert m["uncover:L"].horizontal_direction == "temporal" and len(m["uncover:L"].saccades) == 1
        assert m["cover:R"].horizontal_direction is None and m["uncover:R"].horizontal_direction is None

    def test_right_eye_signs(self):
        # Right eye drifting toward the patient's right is temporal for the right eye.
        m = movements_by(analyse_cover_test(cover_session([cover_trial("R", (2.0, -1.5))]), KB))
        assert m["cover:R"].horizontal_direction == "temporal" and m["cover:R"].vertical_direction == "down"

    def test_manifest_moves_the_fellow_eye_on_uncover(self):
        m = movements_by(analyse_cover_test(cover_session([cover_trial("L", (4.0, 0.0), manifest=True)]), KB))
        assert m["uncover:R"].horizontal_direction is not None and len(m["uncover:R"].saccades) == 1

    def test_no_movement_no_direction(self):
        m = movements_by(analyse_cover_test(cover_session([cover_trial("L")]), KB))
        assert all(x.horizontal_direction is None and x.vertical_direction is None and not x.saccades
                   for x in m.values())

    def test_never_interprets(self):
        r = analyse_cover_test(cover_session([cover_trial("L", (3.0, 0.0)), cover_trial("R", seed=4)]), KB)
        assert len(r.trials) == 2
        assert all(t.interpretation is None and "clinician" in t.interpretation_note for t in r.trials)
        assert r.kb_version == KB.version and r.movement_version == "0.1.0"


class TestAPI:
    def test_cover_test_endpoint(self):
        from fastapi.testclient import TestClient
        from main import app
        s = cover_session([cover_trial("L", (3.0, 0.0), after_s=1.0, cover_s=2.0)])
        r = TestClient(app).post("/exam/tier2/cover-test", content=s.model_dump_json(),
                                 headers={"content-type": "application/json"})
        assert r.status_code == 200
        body = r.json()
        assert body["problems"] == [] and body["trials"][0]["interpretation"] is None
