"""
BeanHealth CLR — Tier 2 cover-test session (brief §3, §4.1) — interface
=======================================================================

Tier 2 adds an IR clip to the phone: IR LEDs for illumination and an
IR-pass occluder. The occluder hides the target from the covered eye but
lets the camera keep seeing it, so both eyes' movements are recorded through
the cover and the uncover.

A session is one or more cover trials. Each trial carries its timing marks
and a per-sample gaze track for both eyes (degrees, from an IR pupil–glint
tracker that is not part of Tier 1): horizontal + = the patient's right,
vertical + = up — the same eye-in-orbit convention as the capture protocol.

Recording rules (KB capture.yaml): ≥ cover_test_min_fps at
≥ cover_test_min_frame_height_px; cover held ≥ cover_test_min_cover_s;
≥ cover_test_record_after_uncover_s recorded after the uncover.

Status: interface and validation only. No phone capture or IR tracker
exists yet; exam.tier2.sim produces traces for the tests.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import Field, model_validator

from exam.capture.session import DeviceInfo
from exam.kb import load_kb
from exam.kb.loader import KnowledgeBase
from exam.kb.schema import Eye
from exam.models import Strict

COVER_TEST_PROTOCOL_VERSION = "0.1.0"
MAX_SAMPLES_PER_TRIAL = 240 * 30          # 30 s at 240 fps


class GazeSample(Strict):
    t_ms: float = Field(ge=0)
    # Per-eye gaze (deg; + = patient's right / up); null = not tracked (blink, lost glint).
    eyes: Dict[Eye, Optional[Tuple[float, float]]]


class CoverTrial(Strict):
    covered_eye: Eye
    cover_on_ms: float = Field(ge=0)      # occluder in place
    cover_off_ms: float                   # occluder removed
    end_ms: float
    samples: List[GazeSample] = Field(max_length=MAX_SAMPLES_PER_TRIAL)

    @model_validator(mode="after")
    def _ordered(self) -> "CoverTrial":
        if not self.cover_on_ms < self.cover_off_ms < self.end_ms:
            raise ValueError("need cover_on_ms < cover_off_ms < end_ms")
        for a, b in zip(self.samples, self.samples[1:]):
            if b.t_ms <= a.t_ms:
                raise ValueError("samples must have increasing t_ms")
        return self


class CoverTestSettings(Strict):
    fps: float = Field(gt=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    illumination: Literal["ir"] = "ir"


class CoverTestHeader(Strict):
    session_id: str = Field(min_length=1)
    protocol_version: str = COVER_TEST_PROTOCOL_VERSION
    tier: Literal[2] = 2
    device: DeviceInfo
    settings: CoverTestSettings
    distance_cm: Optional[float] = Field(default=None, gt=0)
    started_at: str


class CoverTestSession(Strict):
    header: CoverTestHeader
    trials: List[CoverTrial] = Field(min_length=1)


class SessionProblem(Strict):
    code: Literal["low_fps", "low_resolution", "cover_too_short", "too_little_after_uncover", "sparse_samples"]
    trial: Optional[int] = None
    detail: str


def check_session(session: CoverTestSession, kb: Optional[KnowledgeBase] = None) -> List[SessionProblem]:
    """Recording-rule violations (empty list = the session meets the Tier 2 protocol)."""
    kb = kb or load_kb()

    def kbv(i: str) -> float:
        v = kb.scalar(i).value
        assert v is not None, i
        return float(v)

    s = session.header.settings
    out: List[SessionProblem] = []
    if s.fps < kbv("cover_test_min_fps"):
        out.append(SessionProblem(code="low_fps", detail=f"{s.fps:g} fps < {kbv('cover_test_min_fps'):g}"))
    if min(s.width_px, s.height_px) < kbv("cover_test_min_frame_height_px"):
        out.append(SessionProblem(code="low_resolution",
                                  detail=f"{min(s.width_px, s.height_px)}p < {kbv('cover_test_min_frame_height_px'):g}p"))
    for i, t in enumerate(session.trials):
        cover_s = (t.cover_off_ms - t.cover_on_ms) / 1000
        if cover_s < kbv("cover_test_min_cover_s"):
            out.append(SessionProblem(code="cover_too_short", trial=i,
                                      detail=f"cover held {cover_s:.2f} s < {kbv('cover_test_min_cover_s'):g} s"))
        after_s = (t.end_ms - t.cover_off_ms) / 1000
        if after_s < kbv("cover_test_record_after_uncover_s"):
            out.append(SessionProblem(code="too_little_after_uncover", trial=i,
                                      detail=f"{after_s:.2f} s after uncover < "
                                             f"{kbv('cover_test_record_after_uncover_s'):g} s"))
        if len(t.samples) >= 2:
            span = (t.samples[-1].t_ms - t.samples[0].t_ms) / 1000
            rate = (len(t.samples) - 1) / span if span > 0 else 0
            if rate < 0.8 * s.fps:
                out.append(SessionProblem(code="sparse_samples", trial=i,
                                          detail=f"{rate:.0f} samples/s against {s.fps:g} fps declared"))
    return out
