"""
BeanHealth CLR — Tier 2 cover-test movement module (brief §3) — skeleton
========================================================================

Measures, per trial, what each eye did in the two windows a cover test is
read from:

  on cover    from cover_on to cover_off: the covered eye (seen through the
              IR-pass occluder) and the fellow eye
  on uncover  from cover_off to the end: both eyes after the occluder goes

Per eye and window: the saccades found (exam.tier2.saccades) and the net
movement (median position over the last 100 ms of the window minus the first
100 ms) with its standard error, in degrees and prism dioptres, and its
direction in the patient's frame (nasal / temporal / up / down for that eye)
— named only when the movement clears k × its error (decision_margin_k).

What it does NOT do: say "tropia", "phoria", "eso" or "exo". That reading
needs clinician-defined criteria (minimum movement, which window counts,
latent vs manifest) that are not in the KB, so `interpretation` is always
null with the reason. Tier 2 is interfaces and stubs in Phase 3 (brief §3).
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np

from exam.kb import load_kb
from exam.kb.loader import KnowledgeBase
from exam.kb.schema import Eye
from exam.models import Strict, degrees_to_pd
from exam.tier2.saccades import SACCADES_VERSION, Saccade, detect_saccades
from exam.tier2.session import COVER_TEST_PROTOCOL_VERSION, CoverTestSession, CoverTrial, SessionProblem, check_session

MOVEMENT_VERSION = "0.1.0"
EDGE_MS = 100.0
NOT_INTERPRETED = ("Cover-test interpretation (tropia / phoria, direction) needs clinician-defined criteria "
                   "that are not in the knowledge base yet; Tier 2 is a stub in Phase 3.")


class EyeMovement(Strict):
    eye: Eye
    window: Literal["cover", "uncover"]
    covered: bool                              # this eye was behind the occluder in this window
    net_horizontal_deg: Optional[float] = None  # + = toward the patient's right
    net_vertical_deg: Optional[float] = None    # + = up
    net_horizontal_pd: Optional[float] = None
    net_vertical_pd: Optional[float] = None
    net_error_deg: Optional[float] = None        # one-sigma, per axis (the larger of the two)
    horizontal_direction: Optional[Literal["nasal", "temporal"]] = None
    vertical_direction: Optional[Literal["up", "down"]] = None
    saccades: List[Saccade] = []


class TrialMovements(Strict):
    trial: int
    covered_eye: Eye
    movements: List[EyeMovement]
    interpretation: None = None
    interpretation_note: str = NOT_INTERPRETED


class CoverTestResult(Strict):
    session_id: str
    problems: List[SessionProblem]
    trials: List[TrialMovements]
    protocol_version: str = COVER_TEST_PROTOCOL_VERSION
    movement_version: str = MOVEMENT_VERSION
    saccades_version: str = SACCADES_VERSION
    kb_version: str
    kb_sha256: str


def _window(trial: CoverTrial, eye: Eye, start: float, end: float, kind: str, kb: KnowledgeBase) -> EyeMovement:
    samples = [s for s in trial.samples if start <= s.t_ms <= end]
    covered = kind == "cover" and eye == trial.covered_eye
    track = [(s.t_ms, s.eyes.get(eye)) for s in samples]
    mv = EyeMovement(eye=eye, window=kind, covered=covered)          # type: ignore[arg-type]
    first = [g for t, g in track if g is not None and t <= start + EDGE_MS]
    last = [g for t, g in track if g is not None and t >= end - EDGE_MS]
    if not first or not last:
        return mv
    a, b = np.array(first), np.array(last)
    h0, v0 = np.median(a, axis=0)
    h1, v1 = np.median(b, axis=0)
    dh, dv = float(h1 - h0), float(v1 - v0)
    # SE of a median ≈ 1.2533·SD/√n; both ends add in quadrature.
    se = 1.2533 * np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b)) \
        if len(a) > 1 and len(b) > 1 else np.array([np.inf, np.inf])
    k = float(kb.threshold("decision_margin_k").value)          # type: ignore[arg-type]
    moved_h, moved_v = abs(dh) > k * se[0], abs(dv) > k * se[1]
    # The right eye's nasal side is the patient's left.
    nasal = (dh < 0) if eye == "R" else (dh > 0)
    trace = detect_saccades([t for t, _ in track], [g for _, g in track], kb)
    return mv.model_copy(update=dict(
        net_horizontal_deg=dh, net_vertical_deg=dv,
        net_horizontal_pd=degrees_to_pd(dh), net_vertical_pd=degrees_to_pd(dv),
        net_error_deg=float(max(se)),
        horizontal_direction=("nasal" if nasal else "temporal") if moved_h else None,
        vertical_direction=("up" if dv > 0 else "down") if moved_v else None,
        saccades=trace.saccades,
    ))


def analyse_cover_test(session: CoverTestSession, kb: Optional[KnowledgeBase] = None) -> CoverTestResult:
    kb = kb or load_kb()
    trials = []
    for i, t in enumerate(session.trials):
        moves = [_window(t, eye, a, b, kind, kb)
                 for kind, a, b in (("cover", t.cover_on_ms, t.cover_off_ms), ("uncover", t.cover_off_ms, t.end_ms))
                 for eye in ("R", "L")]
        trials.append(TrialMovements(trial=i, covered_eye=t.covered_eye, movements=moves))
    return CoverTestResult(session_id=session.header.session_id, problems=check_session(session, kb),
                           trials=trials, **kb.version_info())


def movements_by(result: CoverTestResult, trial: int = 0) -> Dict[str, EyeMovement]:
    """{"cover:R": …, "uncover:L": …} for one trial."""
    return {f"{m.window}:{m.eye}": m for m in result.trials[trial].movements}
