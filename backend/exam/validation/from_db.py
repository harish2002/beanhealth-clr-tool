"""
BeanHealth CLR — Validation from the database (brief §9)
=========================================================

    report = validate(connect("exam.sqlite"))

  • Bland–Altman, device vs orthoptist (reference_measurements), per
    position and axis, plus all positions pooled.
  • Pattern agreement per engine rule against reference_diagnoses: the
    device says present (a REPORTED abnormal finding), inconclusive (an
    INCONCLUSIVE finding) or absent (no such finding).
  • Per-device error breakdown: the pooled Bland–Altman per device model.

Only sessions whose patients consented to research are included.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from pydantic import Field

from exam.models import Strict
from exam.validation.stats import Agreement, BlandAltman, agreement, bland_altman

VALIDATION_VERSION = "1.0.0"


class ValidationReport(Strict):
    sessions: int
    paired_measurements: int
    bland_altman: Dict[str, Dict[str, BlandAltman]] = Field(default_factory=dict)   # position → axis
    pooled: Dict[str, BlandAltman] = Field(default_factory=dict)                    # axis
    per_device: Dict[str, Dict[str, BlandAltman]] = Field(default_factory=dict)     # model → axis
    patterns: Dict[str, Agreement] = Field(default_factory=dict)                    # "rule[:eye]"
    validation_version: str = VALIDATION_VERSION


_PAIRS = """
SELECT s.session_id, d.model, c.position, m.horizontal_pd, m.vertical_pd, r.horizontal_pd, r.vertical_pd
FROM reference_measurements r
JOIN sessions s ON s.session_id = r.session_id
JOIN patients p ON p.pseudo_id = s.pseudo_id
JOIN devices d ON d.id = s.device_id
JOIN captures c ON c.session_id = r.session_id AND c.position = r.position
JOIN measurements m ON m.capture_id = c.id
WHERE m.status = 'OK' AND p.research = 1
"""


def _research_sessions(conn: sqlite3.Connection) -> List[str]:
    return [r[0] for r in conn.execute(
        "SELECT s.session_id FROM sessions s JOIN patients p ON p.pseudo_id = s.pseudo_id "
        "WHERE p.research = 1")]


def validate(conn: sqlite3.Connection) -> ValidationReport:
    rows = conn.execute(_PAIRS).fetchall()
    by_pos: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    by_dev: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    pooled: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    for _sid, model, pos, mh, mv, rh, rv in rows:
        for axis, dev, ref in (("horizontal", mh, rh), ("vertical", mv, rv)):
            if dev is None or ref is None:
                continue
            for bucket in (by_pos[(pos, axis)], by_dev[(model, axis)], pooled[axis]):
                bucket[0].append(dev)
                bucket[1].append(ref)

    ba: Dict[str, Dict[str, BlandAltman]] = defaultdict(dict)
    for (pos, axis), (d, r) in sorted(by_pos.items()):
        ba[pos][axis] = bland_altman(d, r)
    per_device: Dict[str, Dict[str, BlandAltman]] = defaultdict(dict)
    for (model, axis), (d, r) in sorted(by_dev.items()):
        per_device[model][axis] = bland_altman(d, r)

    sessions = set(_research_sessions(conn))
    calls: Dict[str, Tuple[List[Optional[bool]], List[bool]]] = defaultdict(lambda: ([], []))
    for sid, rule, eye, present in conn.execute(
            "SELECT session_id, rule_id, affected_eye, present FROM reference_diagnoses"):
        if sid not in sessions:
            continue
        rows_f = conn.execute("SELECT status, abnormal FROM findings WHERE session_id = ? AND rule_id = ? AND "
                              "COALESCE(affected_eye, '') = COALESCE(?, '')", (sid, rule, eye)).fetchall()
        device: Optional[bool]
        if any(s == "REPORTED" and a for s, a in rows_f):
            device = True
        elif any(s == "INCONCLUSIVE" for s, _ in rows_f):
            device = None
        else:
            device = False
        key = f"{rule}:{eye}" if eye else rule
        calls[key][0].append(device)
        calls[key][1].append(bool(present))

    return ValidationReport(
        sessions=len(sessions), paired_measurements=len(rows),
        bland_altman=dict(ba), pooled={a: bland_altman(*v) for a, v in sorted(pooled.items())},
        per_device=dict(per_device), patterns={k: agreement(*v) for k, v in sorted(calls.items())},
    )


def summary(r: ValidationReport) -> str:
    """Plain-text summary for the console."""
    def ba_line(name: str, b: BlandAltman) -> str:
        if b.sd is None:
            return f"  {name:<24} n={b.n}"
        return (f"  {name:<24} n={b.n:<4} bias {b.bias:+6.2f} PD  SD {b.sd:5.2f}  "
                f"LoA [{b.loa_low:+6.2f}, {b.loa_high:+6.2f}]  MAE {b.mean_abs_error:5.2f}")

    out = [f"Validation {r.validation_version}: {r.sessions} research-consented sessions, "
           f"{r.paired_measurements} paired positions", "", "Bland–Altman, device − orthoptist (PD):"]
    for axis, b in r.pooled.items():
        out.append(ba_line(f"all positions {axis}", b))
    for pos, axes in r.bland_altman.items():
        for axis, b in axes.items():
            out.append(ba_line(f"{pos} {axis}", b))
    out += ["", "Per device:"]
    for model, axes in r.per_device.items():
        for axis, b in axes.items():
            out.append(ba_line(f"{model} {axis}", b))
    out += ["", "Pattern agreement (device vs clinician):"]
    for key, a in r.patterns.items():
        def pct(x: Optional[float]) -> str:
            return "  –  " if x is None else f"{100 * x:5.1f}%"
        kappa = "–" if a.kappa is None else f"{a.kappa:.2f}"
        out.append(f"  {key:<28} n={a.n:<4} sens {pct(a.sensitivity)}  spec {pct(a.specificity)}  "
                   f"κ {kappa}  inconclusive {pct(a.inconclusive_rate)}")
    return "\n".join(out)
