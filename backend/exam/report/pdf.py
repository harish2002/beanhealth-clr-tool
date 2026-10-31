"""
BeanHealth CLR — Clinician report as PDF (brief §8)
===================================================

The same content as the HTML report, laid out on A4 pages with matplotlib
(already installed with mediapipe, so no extra dependency). Text flows down
the page and onto new pages; the 9-gaze grid is drawn as coloured boxes.
"""

from __future__ import annotations

import io
import textwrap
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from exam.report.build import EYE_NAME, ExamReport, ReportFinding, Value  # noqa: E402
from exam.report.html import _DECISION_TITLE  # noqa: E402

A4 = (8.27, 11.69)
MARGIN = 0.07
INK, MUTED = "#16202c", "#5b6775"
FILL = {"beyond_error": "#fffaeb", "within_error": "#ffffff", "insufficient": "#f2f4f7"}
TONE = {"red": ("#fef3f2", "#b42318"), "amber": ("#fffaeb", "#93370d"), "green": ("#ecfdf3", "#05603a")}


def _v(v: Optional[Value], unit: str = "") -> str:
    return "–" if v is None else v.text(unit)


class _Writer:
    """Top-to-bottom text on A4 pages; starts a new page when the current one is full."""

    def __init__(self, pdf: PdfPages, footer: str):
        self.pdf, self.footer = pdf, footer
        self.fig = None
        self.y = 0.0
        self.page = 0
        self._new_page()

    def _new_page(self) -> None:
        if self.fig is not None:
            self._close_page()
        self.fig = plt.figure(figsize=A4)
        self.page += 1
        self.y = 1 - MARGIN

    def _close_page(self) -> None:
        assert self.fig is not None
        self.fig.text(MARGIN, 0.036, self.footer, fontsize=6, color=MUTED)
        self.fig.text(MARGIN, 0.024, "Decision support — not a diagnosis.", fontsize=6, color=MUTED)
        self.fig.text(1 - MARGIN, 0.024, f"page {self.page}", fontsize=6, color=MUTED, ha="right")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def need(self, height: float) -> None:
        if self.y - height < 0.065:
            self._new_page()

    def text(self, s: str, size: float = 9.0, color: str = INK, bold: bool = False, indent: float = 0.0,
             width: int = 105, gap: float = 0.003, mono: bool = False) -> None:
        wrap = int(width * 9.0 / size) - int(indent * 120)
        lines = textwrap.wrap(s, max(20, wrap)) or [""]
        line_h = size / 72 / A4[1] * 1.35
        for line in lines:
            self.need(line_h)
            self.fig.text(MARGIN + indent, self.y, line, fontsize=size, color=color, va="top",  # type: ignore[union-attr]
                          weight="bold" if bold else "normal", family="monospace" if mono else "sans-serif")
            self.y -= line_h
        self.y -= gap

    def heading(self, s: str) -> None:
        self.need(0.05)
        self.y -= 0.012
        self.text(s.upper(), size=8.5, color=MUTED, bold=True, gap=0.006)

    def box(self, title: str, lines: List[str], tone: str) -> None:
        bg, fg = TONE[tone]
        wrapped = [w for ln in lines for w in (textwrap.wrap(ln, 100) or [""])]
        h = 0.025 + 0.0155 * len(wrapped)
        self.need(h + 0.01)
        self.fig.add_artist(Rectangle((MARGIN, self.y - h), 1 - 2 * MARGIN, h, transform=self.fig.transFigure,  # type: ignore[union-attr]
                                      facecolor=bg, edgecolor=fg, linewidth=0.6))
        self.y -= 0.008
        self.text(title, size=11, color=fg, bold=True, indent=0.012)
        for ln in wrapped:
            self.text(ln, size=8.5, color=fg, indent=0.012, gap=0)
        self.y -= 0.012

    def grid(self, r: ExamReport) -> None:
        size, gap, cell_h = 0.19, 0.008, 0.078
        rows = len(r.grid_layout)
        h = rows * cell_h + (rows - 1) * gap
        self.need(h + 0.02)
        top = self.y
        for i, row in enumerate(r.grid_layout):
            for j, gaze in enumerate(row):
                if gaze is None:
                    continue
                c = r.grid[gaze]
                x = MARGIN + j * (size + gap)
                y = top - (i + 1) * cell_h - i * gap
                self.fig.add_artist(Rectangle((x, y), size, cell_h, transform=self.fig.transFigure,  # type: ignore[union-attr]
                                              facecolor=FILL[c.cls], edgecolor="#d9dee5", linewidth=0.8,
                                              linestyle="--" if c.cls == "insufficient" else "-"))
                tx = x + 0.008
                self.fig.text(tx, y + cell_h - 0.008, c.name + ("" if c.confidence in (None, "HIGH") else f"  [{c.confidence}]"),  # type: ignore[union-attr]
                              fontsize=7, color=MUTED, va="top")
                if c.status != "OK":
                    self.fig.text(tx, y + cell_h / 2, "not measured", fontsize=8, color=MUTED, va="center")  # type: ignore[union-attr]
                    continue
                lines = [f"H {_v(c.horizontal_pd)}", f"  {c.horizontal_label} · {c.horizontal_deg}°",
                         f"V {_v(c.vertical_pd)}", f"  {c.vertical_label}"]
                for k, ln in enumerate(lines):
                    self.fig.text(tx, y + cell_h - 0.026 - k * 0.0135, ln, fontsize=7.2, va="top",  # type: ignore[union-attr]
                                  family="monospace" if k % 2 == 0 else "sans-serif", color=INK)
        self.y = top - h - 0.008
        self.text("Amber: beyond measurement error · white: within error · dashed: not measured. "
                  "Drawn as the examiner faces the patient.", size=7.5, color=MUTED)

    def close(self) -> None:
        self._close_page()


def _findings(w: _Writer, findings: List[ReportFinding], reasons: bool = False) -> None:
    for f in findings:
        eye = f" — {EYE_NAME[f.affected_eye]} eye" if f.affected_eye else ""
        tag = "inconclusive" if f.status == "INCONCLUSIVE" else f"{f.confidence.lower()} confidence"
        w.text(f"{f.label}{eye}  ({tag}{', URGENT' if f.urgency else ''})", size=9, bold=True, gap=0.001)
        w.text(f.wording, size=8.5, indent=0.015, gap=0.001)
        if reasons and f.reasons:
            w.text("Why: " + "; ".join(f.reasons), size=7.8, color=MUTED, indent=0.015, gap=0.001)
        for e in f.evidence[:4]:
            w.text(e, size=7.2, color=MUTED, indent=0.015, gap=0, mono=True)
        w.y -= 0.006


def render_pdf(r: ExamReport) -> bytes:
    buf = io.BytesIO()
    footer = " · ".join(f"{k} {v if k != 'kb_sha256' else v[:12]}" for k, v in r.versions.items())
    with PdfPages(buf, metadata={"Title": f"{r.title} — {r.session.session_id}", "Creator": "BeanHealth CLR"}) as pdf:
        w = _Writer(pdf, footer)
        w.text(r.title, size=16, bold=True)
        who = ""
        if r.patient and (r.patient.label or r.patient.age_years is not None):
            who = " · ".join(x for x in (r.patient.label, f"{r.patient.age_years} y"
                                         if r.patient.age_years is not None else None) if x) + " · "
        w.text(f"{who}Session {r.session.session_id} · captured {r.session.started_at} · report {r.generated_at}",
               size=8, color=MUTED, gap=0.008)
        w.text(r.positioning, size=8.5, bold=True, gap=0.008)
        if r.red_flags or r.urgent_unconfirmed:
            lines = [f"{f.label}{(' — ' + EYE_NAME[f.affected_eye] + ' eye') if f.affected_eye else ''}: {f.wording}"
                     for f in r.red_flags]
            if r.urgent_unconfirmed:
                lines.append("A possible URGENT red flag was seen at LOW confidence or within error — repeat without delay.")
            w.box("Red flags", lines, "red")
        tone = {"REFER": "red" if r.referral.urgent else "amber", "REPEAT_TEST": "amber", "NO_REFERRAL": "green"}
        w.box(_DECISION_TITLE[r.referral.decision],
              [r.referral.advice] + ([f"Based on: {', '.join(r.referral.because)}"] if r.referral.because else []),
              tone[r.referral.decision])

        w.heading("Deviation by gaze (PD ± 1 SE)")
        w.grid(r)
        w.text(r.conventions, size=7.5, color=MUTED)
        w.heading("Head tilt")
        for t in r.tilt:
            w.text(f"Toward the {t.side} shoulder: {_v(t.vertical_pd)} PD  {t.label or 'not measured'}", size=8.5)

        w.heading("Findings")
        if r.findings:
            _findings(w, r.findings)
        else:
            w.text("No deviation or pattern beyond measurement error.", size=9)
        if r.inconclusive:
            w.heading("Inconclusive")
            _findings(w, r.inconclusive, reasons=True)
        if r.not_evaluated:
            w.heading("Not evaluated")
            for x in r.not_evaluated:
                w.text("• " + x, size=7.8, color=MUTED, gap=0)

        w.heading("Eye measures")
        for e in r.eyes:
            d = e.ductions
            w.text(f"{EYE_NAME[e.eye].capitalize()} eye — ductions abd {_v(d.get('abduction'))}, add {_v(d.get('adduction'))}, "
                   f"elev {_v(d.get('elevation'))}, dep {_v(d.get('depression'))}; fissure {_v(e.fissure_height_mm, 'mm')} "
                   f"(adduction ratio {_v(e.fissure_ratio_adduction)}); scleral show {_v(e.scleral_show_mm, 'mm')}; "
                   f"pupil {_v(e.pupil_diameter_mm, 'mm')}", size=8)

        w.heading("Capture quality")
        for x in [*r.session_flags, *r.measurement_flags]:
            w.text("⚠ " + x, size=7.8, color="#93370d", gap=0)
        for q in r.quality:
            w.text(f"{q.position:<16} {q.frames_valid:>4}/{q.frames_total:<4} {'OK ' if q.status == 'OK' else 'INS'} "
                   f"{', '.join(q.rejections)}{('  — ' + q.note) if q.note else ''}", size=7.2, mono=True, gap=0)

        c = r.calibration
        w.heading("Calibration")
        w.text(f"Hirschberg ratio {_v(c.hirschberg_pd_per_mm, 'PD/mm')} "
               f"({'fitted for this patient' if c.hirschberg_source == 'fitted' else 'KB default'})", size=8.5)
        for eye, k in c.kappa.items():
            w.text(f"{EYE_NAME[eye].capitalize()} eye kappa: " +
                   ("not measured" if k is None else f"{k[0].text('mm', 3)} temporal, {k[1].text('mm', 3)} inferior"), size=8.5)
        for x in c.flags:
            w.text("⚠ " + x, size=7.8, color="#93370d", gap=0)

        w.heading("Known limitations")
        for x in r.limitations:
            w.text("• " + x, size=7.8, color=MUTED, gap=0)
        if r.unapproved_kb_entries:
            w.text(f"{r.unapproved_kb_entries} knowledge-base values behind these findings are draft or awaiting "
                   "clinician review.", size=7.8, color=MUTED)
        w.close()
    return buf.getvalue()
