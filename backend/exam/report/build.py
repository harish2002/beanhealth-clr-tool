"""
BeanHealth CLR — Clinician report: the machine-readable report (brief §8)
=========================================================================

    report = build_report(session, result)          # ExamReport (JSON)
    html   = render_html(report)                    # exam.report.html
    pdf    = render_pdf(report)                     # exam.report.pdf

The report reorganises an ExamResult for the reader: red flags first, then
the referral decision, the 9-gaze grid (PD ± error per cell), head tilt,
findings with their evidence, INCONCLUSIVE reasons, eye measures, capture
quality, calibration, the KB entries behind the findings that no clinician
has approved yet, the known limitations and every version.

Numbers are rounded to display precision here (2 decimals), so the JSON, the
HTML and the PDF show the same values and golden-file tests are stable.

Cell colour classes carry no clinical threshold: a cell is `beyond_error`
when either axis clears zero by more than k × error (the engine's own test,
k from the KB), `within_error` otherwise, `insufficient` when unmeasured.
Whether a size matters clinically is the findings' job, not the grid's.

The patient label (name or clinic ID) is optional, printed on the report
only, and never stored by the engine or the database (brief §7).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import Field

from exam.capture.protocol import GAZE_DIRECTION
from exam.capture.session import CaptureSession
from exam.engine.findings import Finding
from exam.examine import ExamResult
from exam.kb import load_kb
from exam.kb.loader import KnowledgeBase
from exam.models import PROTOCOL_POSITIONS, Measured, Strict

REPORT_VERSION = "1.0.0"

CONVENTIONS = ("Right / left are the patient's. Horizontal: + = eso, − = exo. Vertical: + = right hyper, "
               "− = left hyper. Prism dioptres (PD), one-sigma error; degrees = atan(PD / 100). "
               "The grid is drawn as the examiner faces the patient: the patient's right gaze is on the left.")

GAZE_NAME: Dict[str, str] = {
    "primary": "Primary", "right": "Right", "left": "Left", "up": "Up", "down": "Down",
    "up_right": "Up right", "up_left": "Up left", "down_right": "Down right", "down_left": "Down left",
}
EYE_NAME = {"R": "right", "L": "left"}

CellClass = Literal["beyond_error", "within_error", "insufficient"]


class Value(Strict):
    value: float
    error: float
    confidence: str

    @staticmethod
    def of(m: Optional[Measured], digits: int = 2) -> Optional["Value"]:
        if m is None:
            return None
        return Value(value=round(m.value, digits), error=round(m.error, digits), confidence=m.confidence)

    def text(self, unit: str = "", digits: int = 2) -> str:
        return f"{self.value:.{digits}f} ± {self.error:.{digits}f}{(' ' + unit) if unit else ''}"


class ReportPatient(Strict):
    label: Optional[str] = Field(default=None, max_length=120)   # printed only, never stored
    age_years: Optional[int] = Field(default=None, ge=0, le=120)


class ReportCell(Strict):
    gaze: str
    name: str
    status: Literal["OK", "INSUFFICIENT_DATA"]
    cls: CellClass
    horizontal_pd: Optional[Value] = None
    vertical_pd: Optional[Value] = None
    horizontal_deg: Optional[float] = None
    vertical_deg: Optional[float] = None
    horizontal_label: Optional[str] = None     # eso / exo / within error
    vertical_label: Optional[str] = None       # R hyper / L hyper / within error
    confidence: Optional[str] = None
    valid_frames: int = 0
    reason: Optional[str] = None


class ReportTilt(Strict):
    side: Literal["right", "left"]
    vertical_pd: Optional[Value] = None
    label: Optional[str] = None


class ReportFinding(Strict):
    rule_id: str
    label: str
    kind: str
    affected_eye: Optional[str] = None
    status: str
    confidence: str
    urgency: Optional[str] = None
    muscle: Optional[str] = None
    wording: str
    margin_over_threshold: Optional[float] = None
    evidence: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    unapproved_kb: List[str] = Field(default_factory=list)   # "id (status)" behind this finding


class ReportEye(Strict):
    eye: Literal["R", "L"]
    ductions: Dict[str, Value] = Field(default_factory=dict)
    fissure_height_mm: Optional[Value] = None
    fissure_ratio_adduction: Optional[Value] = None
    pupil_diameter_mm: Optional[Value] = None
    scleral_show_mm: Optional[Value] = None


class QualityRow(Strict):
    position: str
    kind: str
    status: str
    frames_valid: int
    frames_total: int
    rejections: List[str] = Field(default_factory=list)   # most frequent first, "code ×n"
    note: Optional[str] = None


class ReportReferral(Strict):
    decision: str
    urgent: bool
    urgent_unconfirmed: bool
    advice: str
    because: List[str]


class ReportCalibration(Strict):
    hirschberg_pd_per_mm: Value
    hirschberg_source: str
    kappa: Dict[str, Optional[Tuple[Value, Value]]]    # (temporal mm, inferior mm)
    flags: List[str] = Field(default_factory=list)


class ReportSession(Strict):
    session_id: str
    started_at: str
    protocol: str
    device: str
    resolution: str
    fps: float
    distance_cm: Optional[Value] = None


class ExamReport(Strict):
    report_version: str = REPORT_VERSION
    generated_at: str
    title: str = "Gaze examination report"
    positioning: str
    conventions: str = CONVENTIONS
    patient: Optional[ReportPatient] = None
    session: ReportSession
    red_flags: List[ReportFinding] = Field(default_factory=list)
    urgent_unconfirmed: bool = False
    referral: ReportReferral
    grid_layout: List[List[Optional[str]]]
    grid: Dict[str, ReportCell]
    tilt: List[ReportTilt] = Field(default_factory=list)
    findings: List[ReportFinding] = Field(default_factory=list)       # REPORTED, abnormal
    inconclusive: List[ReportFinding] = Field(default_factory=list)
    normal: List[ReportFinding] = Field(default_factory=list)          # REPORTED, not abnormal
    not_evaluated: List[str] = Field(default_factory=list)
    eyes: List[ReportEye] = Field(default_factory=list)
    calibration: ReportCalibration
    quality: List[QualityRow] = Field(default_factory=list)
    session_flags: List[str] = Field(default_factory=list)
    measurement_flags: List[str] = Field(default_factory=list)
    unapproved_kb_entries: int = 0
    limitations: List[str] = Field(default_factory=list)
    versions: Dict[str, str]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def grid_layout(protocol: str) -> List[List[Optional[str]]]:
    """Rows top to bottom, columns as the examiner faces the patient (patient's right gaze on the left)."""
    wanted = set(PROTOCOL_POSITIONS[protocol])
    rows = []
    for v in (1, 0, -1):
        row = []
        for h in (1, 0, -1):
            gaze = next(g for g, d in GAZE_DIRECTION.items() if d == (h, v))
            row.append(gaze if gaze in wanted else None)
        rows.append(row)
    return rows


def _beyond(m: Optional[Measured], k: float) -> bool:
    return m is not None and abs(m.value) > k * m.error


def _h_label(m: Measured, k: float) -> str:
    return ("eso" if m.value > 0 else "exo") if _beyond(m, k) else "within error"


def _v_label(m: Measured, k: float) -> str:
    return ("R hyper" if m.value > 0 else "L hyper") if _beyond(m, k) else "within error"


def _finding(f: Finding) -> ReportFinding:
    evidence = []
    for e in f.evidence:
        s = f"{e.quantity} = {e.value:.2f} ± {e.error:.2f} {e.unit}"
        if e.threshold is not None and e.threshold_id:
            s += f" vs {e.threshold:g} ({e.threshold_id})"
        if e.confidence != "HIGH":
            s += f" [{e.confidence}]"
        evidence.append(s)
    return ReportFinding(
        rule_id=f.rule_id, label=f.label, kind=f.kind, affected_eye=f.affected_eye, status=f.status,
        confidence=f.confidence, urgency=f.urgency, muscle=f.muscle, wording=f.wording,
        margin_over_threshold=None if f.margin_over_threshold is None else round(f.margin_over_threshold, 2),
        evidence=list(dict.fromkeys(evidence)), reasons=[r.detail for r in f.reasons],
        unapproved_kb=[f"{r.id} ({r.review_status})" for r in f.kb_entries if r.review_status != "approved"],
    )


# ─────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────

def build_report(session: CaptureSession, result: ExamResult, kb: Optional[KnowledgeBase] = None,
                 patient: Optional[ReportPatient] = None, generated_at: Optional[str] = None) -> ExamReport:
    kb = kb or load_kb()
    k = float(kb.threshold("decision_margin_k").value)        # type: ignore[arg-type]
    engine, grid, cal, qc = result.engine, result.measurement, result.calibration, result.qc
    h = session.header

    cells: Dict[str, ReportCell] = {}
    for gaze, cell in grid.measurements.grid.cells.items():
        pm = grid.position(gaze)
        if cell.status != "OK" or cell.horizontal_pd is None or cell.vertical_pd is None:
            cells[gaze] = ReportCell(gaze=gaze, name=GAZE_NAME[gaze], status="INSUFFICIENT_DATA",
                                     cls="insufficient", valid_frames=cell.valid_frames,
                                     reason=None if pm is None else pm.reason)
            continue
        hm, vm = cell.horizontal_pd, cell.vertical_pd
        cells[gaze] = ReportCell(
            gaze=gaze, name=GAZE_NAME[gaze], status="OK",
            cls="beyond_error" if (_beyond(hm, k) or _beyond(vm, k)) else "within_error",
            horizontal_pd=Value.of(hm), vertical_pd=Value.of(vm),
            horizontal_deg=None if pm is None or pm.horizontal_deg is None else round(pm.horizontal_deg, 1) + 0.0,
            vertical_deg=None if pm is None or pm.vertical_deg is None else round(pm.vertical_deg, 1) + 0.0,
            horizontal_label=_h_label(hm, k), vertical_label=_v_label(vm, k),
            confidence=hm.confidence, valid_frames=cell.valid_frames,
        )

    tilt = []
    t = grid.measurements.tilt
    for side in ("right", "left"):
        m = getattr(t, side) if t is not None else None
        tilt.append(ReportTilt(side=side, vertical_pd=Value.of(m),             # type: ignore[arg-type]
                               label=None if m is None else _v_label(m, k)))

    findings = [f for f in engine.findings if f.status == "REPORTED" and f.abnormal]
    red = [f for f in engine.findings if f.urgency == "URGENT" and f.status == "REPORTED"]
    unapproved = {r.id for f in engine.findings for r in f.kb_entries if r.review_status != "approved"}

    eyes = []
    for eye in ("R", "L"):
        em = grid.measurements.eyes.get(eye)                  # type: ignore[call-overload]
        if em is None:
            continue
        eyes.append(ReportEye(
            eye=eye, ductions={d: Value.of(m) for d, m in em.ductions.items()},   # type: ignore[misc]
            fissure_height_mm=Value.of(em.fissure_height_mm),
            fissure_ratio_adduction=Value.of(em.fissure_ratio_adduction),
            pupil_diameter_mm=Value.of(em.pupil_diameter_mm), scleral_show_mm=Value.of(em.scleral_show_mm),
        ))

    measured_by_name = {p.position: p for p in grid.positions}
    quality = []
    for p in qc.positions:
        top = sorted(p.reject_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        pm = measured_by_name.get(p.position)
        note = None
        if p.reason == "not_captured":
            note = "not captured"
        elif pm is not None and pm.confidence_reasons:
            note = "confidence lowered: " + ", ".join(r.replace("_", " ") for r in pm.confidence_reasons)
        quality.append(QualityRow(position=p.position, kind=p.kind, status=p.status, frames_valid=p.frames_valid,
                                  frames_total=p.frames_total, rejections=[f"{c} ×{n}" for c, n in top], note=note))

    kappa = {e: (None if cal.kappa[e] is None else
                 (Value.of(cal.kappa[e].temporal_mm, 3), Value.of(cal.kappa[e].inferior_mm, 3)))  # type: ignore[union-attr]
             for e in ("R", "L")}
    r = engine.referral
    names = {(f.rule_id, f.affected_eye): f.label + (f" ({EYE_NAME[f.affected_eye]} eye)" if f.affected_eye else "")
             for f in engine.findings}

    def named(tag: str) -> str:
        rule, _, eye = tag.partition(":")
        return names.get((rule, eye or None), tag)

    return ExamReport(
        generated_at=generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        positioning=result.positioning, patient=patient,
        session=ReportSession(session_id=h.session_id, started_at=h.started_at, protocol=h.protocol,
                              device=h.device.model + (f" ({h.device.lens})" if h.device.lens else ""),
                              resolution=f"{h.settings.width_px}×{h.settings.height_px}",
                              fps=round(h.settings.fps, 1), distance_cm=Value.of(h.distance_cm, 1)),
        red_flags=[_finding(f) for f in red], urgent_unconfirmed=r.urgent_unconfirmed,
        referral=ReportReferral(decision=r.decision, urgent=r.urgent, urgent_unconfirmed=r.urgent_unconfirmed,
                                advice=r.advice, because=[named(t) for t in r.because]),
        grid_layout=grid_layout(grid.protocol), grid=cells, tilt=tilt,
        findings=[_finding(f) for f in findings],
        inconclusive=[_finding(f) for f in engine.findings if f.status == "INCONCLUSIVE"],
        normal=[_finding(f) for f in engine.findings if f.status == "REPORTED" and not f.abnormal],
        not_evaluated=[f"{n.rule_id}{(' (' + EYE_NAME[n.affected_eye] + ' eye)') if n.affected_eye else ''}: "
                       + "; ".join(dict.fromkeys(x.detail for x in n.reasons)) for n in engine.not_evaluated],
        eyes=eyes,
        calibration=ReportCalibration(hirschberg_pd_per_mm=Value.of(cal.hirschberg.pd_per_mm),   # type: ignore[arg-type]
                                      hirschberg_source=cal.hirschberg.source, kappa=kappa,
                                      flags=[f.detail for f in cal.flags]),
        quality=quality, session_flags=[f.detail for f in qc.session_flags],
        measurement_flags=list(dict.fromkeys(f.detail for f in grid.flags if f.code != "duction_not_measured")),
        unapproved_kb_entries=len(unapproved), limitations=engine.limitations,
        versions={**result.versions, "report": REPORT_VERSION},
    )
