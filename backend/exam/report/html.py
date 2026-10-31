"""
BeanHealth CLR — Clinician report as a self-contained HTML page (brief §8)
=========================================================================

One file, no external assets, so it can be saved, emailed or printed to PDF
from any browser. Order follows the brief: red flags at the top, the
decision-support disclaimer, the grid, tilt, findings with evidence,
INCONCLUSIVE reasons, eye measures, capture quality, calibration, KB review
status, limitations and versions.
"""

from __future__ import annotations

from html import escape
from typing import Iterable, List, Optional

from exam.report.build import EYE_NAME, ExamReport, ReportCell, ReportFinding, Value

_DECISION_TITLE = {
    "REFER": "Refer to ophthalmology",
    "REPEAT_TEST": "Repeat the examination",
    "NO_REFERRAL": "No deviation or pattern beyond measurement error",
}

_CSS = """
:root{--ink:#16202c;--muted:#5b6775;--line:#d9dee5;--soft:#f5f7f9;--red:#b42318;--red-bg:#fef3f2;
--amber:#93370d;--amber-bg:#fffaeb;--green:#05603a;--green-bg:#ecfdf3;--blue:#1d4ed8}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0}h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);
margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.meta{color:var(--muted);font-size:12.5px;margin-top:4px}.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.box{border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:12px 0}
.disclaimer{background:var(--soft);font-size:13px}
.red{background:var(--red-bg);border-color:#fecdca;color:var(--red)}
.amber{background:var(--amber-bg);border-color:#fedf89;color:var(--amber)}
.green{background:var(--green-bg);border-color:#abefc6;color:var(--green)}
.box h3{margin:0 0 4px;font-size:17px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;max-width:560px}
.cell{border:1px solid var(--line);border-radius:6px;padding:8px;min-height:84px;font-size:12px}
.cell .n{color:var(--muted);display:flex;justify-content:space-between}
.cell .v{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px}
.cell.beyond_error{background:var(--amber-bg);border-color:#fedf89}
.cell.within_error{background:#fff}
.cell.insufficient{background:var(--soft);border-style:dashed;color:var(--muted)}
.badge{display:inline-block;font-size:10.5px;border-radius:4px;padding:0 5px;border:1px solid currentColor}
.legend{font-size:12px;color:var(--muted);margin-top:6px}
.legend span{display:inline-block;width:10px;height:10px;border:1px solid var(--line);vertical-align:-1px;margin:0 4px 0 10px}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{text-align:left;padding:4px 6px;border-top:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;border-top:0}
ul.f{padding-left:0;list-style:none;margin:0}ul.f>li{border-left:3px solid var(--line);padding:4px 0 6px 10px;margin:8px 0}
ul.f>li.abn{border-color:#f79009}ul.f>li.urg{border-color:var(--red)}
.ev{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;color:var(--muted);margin:2px 0 0;padding-left:16px}
.small{font-size:12px;color:var(--muted)}
footer{margin-top:28px;font-size:11.5px;color:var(--muted);border-top:1px solid var(--line);padding-top:8px}
@media print{main{padding:0}.noprint{display:none}h2{break-after:avoid}.box,.cell,ul.f>li{break-inside:avoid}}
"""


def _e(s: object) -> str:
    return escape(str(s))


def _v(v: Optional[Value], unit: str = "") -> str:
    return "–" if v is None else _e(v.text(unit))


def _finding_items(findings: Iterable[ReportFinding], show_reasons: bool = False) -> str:
    out = []
    for f in findings:
        cls = "urg" if f.urgency else ("abn" if f.status == "REPORTED" else "")
        eye = f" — {EYE_NAME[f.affected_eye]} eye" if f.affected_eye else ""
        tag = "inconclusive" if f.status == "INCONCLUSIVE" else f"{f.confidence.lower()} confidence"
        parts = [f'<li class="{cls}"><strong>{_e(f.label)}{_e(eye)}</strong> '
                 f'<span class="badge">{_e(tag)}</span>'
                 + (' <span class="badge" style="color:var(--red)">URGENT</span>' if f.urgency else "")
                 + f"<div>{_e(f.wording)}</div>"]
        if show_reasons and f.reasons:
            parts.append('<div class="small">Why: ' + _e("; ".join(f.reasons)) + "</div>")
        if f.evidence:
            parts.append('<ul class="ev">' + "".join(f"<li>{_e(x)}</li>" for x in f.evidence[:6]) + "</ul>")
        if f.unapproved_kb:
            parts.append('<div class="small">KB values not yet clinician-approved: '
                         + _e(", ".join(f.unapproved_kb)) + "</div>")
        out.append("".join(parts) + "</li>")
    return '<ul class="f">' + "".join(out) + "</ul>"


def _cell(c: ReportCell) -> str:
    conf = f'<span class="badge">{_e(c.confidence)}</span>' if c.confidence and c.confidence != "HIGH" else ""
    head = f'<div class="n"><span>{_e(c.name)}</span>{conf}</div>'
    if c.status != "OK":
        why = {"not_captured": "not captured", "kappa_not_measured": "no kappa",
               "too_few_valid_frames": "too few valid frames"}.get(c.reason or "", "not measured")
        return f'<div class="cell insufficient">{head}<div>{_e(why)}</div></div>'
    return (f'<div class="cell {c.cls}">{head}'
            f'<div class="v">H {_v(c.horizontal_pd)}</div><div>{_e(c.horizontal_label)} · {c.horizontal_deg}°</div>'
            f'<div class="v">V {_v(c.vertical_pd)}</div><div>{_e(c.vertical_label)}</div></div>')


def render_html(r: ExamReport) -> str:
    s: List[str] = []
    who = ""
    if r.patient and (r.patient.label or r.patient.age_years is not None):
        bits = [x for x in (r.patient.label, f"{r.patient.age_years} y" if r.patient.age_years is not None else None) if x]
        who = " · ".join(_e(b) for b in bits) + " · "
    s.append(f"<header><h1>{_e(r.title)}</h1><div class='meta'>{who}Session {_e(r.session.session_id)} · "
             f"captured {_e(r.session.started_at)} · report {_e(r.generated_at)}</div></header>")
    s.append(f'<div class="box disclaimer"><strong>{_e(r.positioning)}</strong></div>')

    if r.red_flags or r.urgent_unconfirmed:
        inner = _finding_items(r.red_flags) if r.red_flags else ""
        if r.urgent_unconfirmed:
            inner += "<div>A possible URGENT red flag was seen at LOW confidence or within error — repeat without delay.</div>"
        s.append(f'<div class="box red"><h3>Red flags</h3>{inner}</div>')

    tone = {"REFER": "red" if r.referral.urgent else "amber", "REPEAT_TEST": "amber", "NO_REFERRAL": "green"}
    because = (f'<div class="small">Based on: {_e(", ".join(r.referral.because))}</div>'
               if r.referral.because else "")
    s.append(f'<div class="box {tone[r.referral.decision]}"><h3>{_e(_DECISION_TITLE[r.referral.decision])}</h3>'
             f"<div>{_e(r.referral.advice)}</div>{because}</div>")

    s.append("<h2>Deviation by gaze (PD ± 1 SE) — as you face the patient</h2>")
    cells = []
    for row in r.grid_layout:
        for g in row:
            cells.append(_cell(r.grid[g]) if g else "<div></div>")
    s.append('<div class="grid">' + "".join(cells) + "</div>")
    s.append('<div class="legend"><span style="background:var(--amber-bg)"></span>beyond measurement error'
             '<span></span>within error<span style="background:var(--soft);border-style:dashed"></span>not measured</div>')
    s.append(f'<p class="small">{_e(r.conventions)}</p>')

    s.append("<h2>Head tilt (Bielschowsky)</h2><table><tr><th>Tilt</th><th>Vertical (PD)</th><th></th></tr>")
    for t in r.tilt:
        s.append(f"<tr><td>toward the {t.side} shoulder</td><td class='mono'>{_v(t.vertical_pd)}</td>"
                 f"<td>{_e(t.label or 'not measured')}</td></tr>")
    s.append("</table>")

    s.append("<h2>Findings</h2>")
    s.append(_finding_items(r.findings) if r.findings
             else "<p>No deviation or pattern beyond measurement error.</p>")
    if r.normal:
        s.append("<p class='small'>Also measured: " + _e(" ".join(f.wording for f in r.normal)) + "</p>")

    if r.inconclusive:
        s.append("<h2>Inconclusive</h2>" + _finding_items(r.inconclusive, show_reasons=True))
    if r.not_evaluated:
        s.append("<h2>Not evaluated</h2><ul class='small'>"
                 + "".join(f"<li>{_e(x)}</li>" for x in r.not_evaluated) + "</ul>")

    s.append("<h2>Eye measures</h2><table><tr><th></th>"
             + "".join(f"<th>{EYE_NAME[e.eye].capitalize()} eye</th>" for e in r.eyes) + "</tr>")
    rows = [("Abduction (fraction)", lambda e: _v(e.ductions.get("abduction"))),
            ("Adduction (fraction)", lambda e: _v(e.ductions.get("adduction"))),
            ("Elevation (fraction)", lambda e: _v(e.ductions.get("elevation"))),
            ("Depression (fraction)", lambda e: _v(e.ductions.get("depression"))),
            ("Fissure height", lambda e: _v(e.fissure_height_mm, "mm")),
            ("Fissure on adduction ÷ primary", lambda e: _v(e.fissure_ratio_adduction)),
            ("Scleral show", lambda e: _v(e.scleral_show_mm, "mm")),
            ("Pupil diameter", lambda e: _v(e.pupil_diameter_mm, "mm"))]
    for name, fn in rows:
        s.append(f"<tr><td>{_e(name)}</td>" + "".join(f"<td class='mono'>{fn(e)}</td>" for e in r.eyes) + "</tr>")
    s.append("</table><p class='small'>Ductions: excursion from primary over the head rotation asked for "
             "(1 = full). The fixing eye reaches the target by definition.</p>")

    s.append("<h2>Capture quality</h2>")
    if r.session_flags or r.measurement_flags:
        s.append("<ul class='small'>" + "".join(f"<li>⚠ {_e(x)}</li>" for x in [*r.session_flags, *r.measurement_flags])
                 + "</ul>")
    s.append("<table><tr><th>Position</th><th>Valid frames</th><th>Main rejections</th><th>Note</th></tr>")
    for q in r.quality:
        s.append(f"<tr><td>{_e(q.position)}</td><td>{q.frames_valid}/{q.frames_total}"
                 f"{'' if q.status == 'OK' else ' ✗'}</td><td>{_e(', '.join(q.rejections) or '–')}</td>"
                 f"<td>{_e(q.note or '')}</td></tr>")
    s.append("</table>")

    c = r.calibration
    s.append("<h2>Calibration</h2><table>")
    s.append(f"<tr><td>Hirschberg ratio</td><td class='mono'>{_v(c.hirschberg_pd_per_mm, 'PD/mm')}</td>"
             f"<td>{'fitted for this patient' if c.hirschberg_source == 'fitted' else 'KB default'}</td></tr>")
    for eye, k in c.kappa.items():
        val = "not measured" if k is None else f"{k[0].text('mm', 3)} temporal, {k[1].text('mm', 3)} inferior"
        s.append(f"<tr><td>{EYE_NAME[eye].capitalize()} eye kappa</td><td class='mono' colspan='2'>{_e(val)}</td></tr>")
    s.append("</table>")
    if c.flags:
        s.append("<ul class='small'>" + "".join(f"<li>⚠ {_e(x)}</li>" for x in c.flags) + "</ul>")

    s.append("<h2>Known limitations</h2><ul class='small'>"
             + "".join(f"<li>{_e(x)}</li>" for x in r.limitations) + "</ul>")
    if r.unapproved_kb_entries:
        s.append(f"<p class='small'>{r.unapproved_kb_entries} knowledge-base values behind these findings are "
                 "draft or awaiting clinician review.</p>")
    s.append("<footer>" + " · ".join(f"{_e(k)} {_e(v if k != 'kb_sha256' else v[:12])}" for k, v in r.versions.items())
             + f"<br>{_e(r.positioning)}</footer>")
    s.append('<p class="noprint small">Use the browser\'s Print → Save as PDF for a paper copy.</p>')

    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_e(r.title)} — {_e(r.session.session_id)}</title><style>{_CSS}</style></head>"
            f"<body><main>{''.join(s)}</main></body></html>")
