"""
BeanHealth CLR — Gaze examination database (brief §7)
=====================================================

    conn = connect("exam.sqlite")                       # creates the schema
    store_exam(conn, session, examine(session), PatientRecord(pseudo_id=new_pseudo_id(), ...))
    add_reference_measurement(conn, session_id, "primary", 18.0, 0.0)

Pseudonymized by construction: a patient is a pseudo_id (generate one with
new_pseudo_id and keep the mapping to the person in the clinic's own
records), an age band and consent flags. There is no column for a name or
for the report's patient label, and sessions carry landmark tracks only, so
no image or video is ever written (captures.storage_uri stays NULL).

Storing needs consent: PatientRecord.consent.store_measurements must be true.
Development uses SQLite (stdlib); schema.sql is portable to PostgreSQL.
India's DPDP Act: consent is recorded with every patient row, and
delete_patient removes a patient and everything derived from them.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import Field, field_validator

from exam.capture.session import CaptureSession
from exam.examine import ExamResult
from exam.models import Measured, Strict

SCHEMA = Path(__file__).with_name("schema.sql")
DB_VERSION = "1.0.0"

AgeBand = Literal["0-2", "3-5", "6-11", "12-17", "18-39", "40-64", "65+"]
_PSEUDO_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def age_band(age_years: int) -> AgeBand:
    """The age band stored instead of an age or a birth date."""
    for hi, band in ((2, "0-2"), (5, "3-5"), (11, "6-11"), (17, "12-17"), (39, "18-39"), (64, "40-64")):
        if age_years <= hi:
            return band                                    # type: ignore[return-value]
    return "65+"


def new_pseudo_id() -> str:
    return "p_" + uuid.uuid4().hex[:16]


class Consent(Strict):
    store_measurements: bool               # required to store anything
    research: bool = False                 # may be used for validation studies
    raw_video: bool = False                # raw video may be kept (encrypted, outside this database)


class PatientRecord(Strict):
    pseudo_id: str
    age_band: Optional[AgeBand] = None
    consent: Consent

    @field_validator("pseudo_id")
    @classmethod
    def _pseudonymous(cls, v: str) -> str:
        if not _PSEUDO_ID.match(v) or " " in v:
            raise ValueError("pseudo_id must be 8–64 letters, digits, '_' or '-' (use new_pseudo_id())")
        return v


class StoredSession(Strict):
    session_id: str
    pseudo_id: str
    captures: int
    findings: int
    db_version: str = DB_VERSION


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _m(m: Optional[Measured]) -> Optional[dict]:
    return None if m is None else {"value": m.value, "error": m.error, "confidence": m.confidence}


def connect(path: Union[str, Path] = ":memory:") -> sqlite3.Connection:
    """Open (and create) the database with foreign keys enforced."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA.read_text())
    return conn


def _id(conn: sqlite3.Connection, select: str, insert: str, args: tuple) -> int:
    row = conn.execute(select, args).fetchone()
    if row:
        return int(row[0])
    return int(conn.execute(insert, args).lastrowid)       # type: ignore[arg-type]


def store_exam(conn: sqlite3.Connection, session: CaptureSession, result: ExamResult,
               patient: PatientRecord) -> StoredSession:
    """Write one examined session and everything derived from it, in one transaction."""
    if not patient.consent.store_measurements:
        raise PermissionError("the patient has not consented to storing measurements")
    h, v = session.header, result.versions
    now = _now()
    with conn:
        if conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (h.session_id,)).fetchone():
            raise ValueError(f"session {h.session_id!r} is already stored")
        conn.execute("INSERT INTO patients (pseudo_id, age_band, consent_json, research, created_at) "
                     "VALUES (?, ?, ?, ?, ?) ON CONFLICT (pseudo_id) DO UPDATE SET age_band = excluded.age_band, "
                     "consent_json = excluded.consent_json, research = excluded.research",
                     (patient.pseudo_id, patient.age_band, patient.consent.model_dump_json(),
                      int(patient.consent.research), now))
        device_id = _id(conn, "SELECT id FROM devices WHERE model = ? AND lens = ? AND calibration_profile = ?",
                        "INSERT INTO devices (model, lens, calibration_profile) VALUES (?, ?, ?)",
                        (h.device.model, h.device.lens or "", h.device.calibration_profile_id or ""))
        conn.execute("INSERT OR IGNORE INTO kb_versions (kb_sha256, kb_version, first_seen) VALUES (?, ?, ?)",
                     (v["kb_sha256"], v["kb"], now))
        ev = (v["engine"], v["measurement"], v["calibration"], v["qc"], v["capture_protocol"])
        engine_id = _id(conn, "SELECT id FROM engine_versions WHERE engine_version = ? AND measurement_version = ? "
                              "AND calibration_version = ? AND qc_version = ? AND capture_protocol_version = ?",
                        "INSERT INTO engine_versions (engine_version, measurement_version, calibration_version, "
                        "qc_version, capture_protocol_version) VALUES (?, ?, ?, ?, ?)", ev)
        r = result.engine.referral
        hb = result.calibration.hirschberg
        conn.execute(
            "INSERT INTO sessions (session_id, pseudo_id, device_id, protocol, protocol_version, tier, started_at, "
            "stored_at, distance_cm, distance_error_cm, hirschberg_pd_per_mm, hirschberg_source, referral_decision, "
            "referral_urgent, kb_sha256, engine_versions_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (h.session_id, patient.pseudo_id, device_id, h.protocol, h.protocol_version, h.tier, h.started_at, now,
             None if h.distance_cm is None else h.distance_cm.value,
             None if h.distance_cm is None else h.distance_cm.error,
             hb.pd_per_mm.value, hb.source, r.decision, int(r.urgent), v["kb_sha256"], engine_id))

        measured = {p.position: p for p in result.measurement.positions}
        n_captures = 0
        for pqc in result.qc.positions:
            cap = session.capture(pqc.position)
            if cap is None:
                continue
            n_captures += 1
            cid = int(conn.execute(
                "INSERT INTO captures (session_id, position, kind, tilt, fps, width_px, height_px, frames_total, "
                "frames_valid, status, storage_uri) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (h.session_id, pqc.position, pqc.kind, int(pqc.kind == "tilt"), h.settings.fps, h.settings.width_px,
                 h.settings.height_px, pqc.frames_total, pqc.frames_valid, pqc.status)).lastrowid)  # type: ignore[arg-type]
            conn.executemany(
                "INSERT INTO frame_qc (capture_id, frame_idx, valid, reject_reason) VALUES (?, ?, ?, ?)",
                [(cid, f.index, int(f.valid), ",".join(dict.fromkeys(x.code for x in f.reasons)) or None)
                 for f in pqc.frames])
            pm = measured.get(pqc.position)
            if pm is not None:
                conn.execute(
                    "INSERT INTO measurements (capture_id, status, horizontal_pd, horizontal_error, vertical_pd, "
                    "vertical_error, confidence, frames_used, eyes_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (cid, pm.status, None if pm.horizontal_pd is None else pm.horizontal_pd.value,
                     None if pm.horizontal_pd is None else pm.horizontal_pd.error,
                     None if pm.vertical_pd is None else pm.vertical_pd.value,
                     None if pm.vertical_pd is None else pm.vertical_pd.error,
                     None if pm.horizontal_pd is None else pm.horizontal_pd.confidence, pm.frames_used,
                     json.dumps({e: r.model_dump() for e, r in pm.eyes.items()})))

        for eye, em in result.measurement.measurements.eyes.items():
            conn.execute(
                "INSERT INTO eye_measures (session_id, eye, ductions_json, fissure_height_mm, fissure_ratio_adduction, "
                "pupil_diameter_mm, scleral_show_mm, errors_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (h.session_id, eye, json.dumps({d: _m(m) for d, m in em.ductions.items()}),
                 *(None if x is None else x.value for x in (em.fissure_height_mm, em.fissure_ratio_adduction,
                                                            em.pupil_diameter_mm, em.scleral_show_mm)),
                 json.dumps({k: _m(getattr(em, k)) for k in ("fissure_height_mm", "fissure_ratio_adduction",
                                                             "pupil_diameter_mm", "scleral_show_mm")})))

        for f in result.engine.findings:
            conn.execute(
                "INSERT INTO findings (session_id, rule_id, affected_eye, status, abnormal, confidence, urgency, muscle, "
                "margin_over_threshold, evidence_json, reasons_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (h.session_id, f.rule_id, f.affected_eye, f.status, int(f.abnormal), f.confidence, f.urgency, f.muscle,
                 f.margin_over_threshold, json.dumps([e.model_dump() for e in f.evidence]),
                 json.dumps([x.model_dump() for x in f.reasons])))
    return StoredSession(session_id=h.session_id, pseudo_id=patient.pseudo_id, captures=n_captures,
                         findings=len(result.engine.findings))


def add_reference_measurement(conn: sqlite3.Connection, session_id: str, position: str,
                              horizontal_pd: Optional[float], vertical_pd: Optional[float], method: str = "PBCT",
                              examiner: Optional[str] = None, measured_at: Optional[str] = None) -> None:
    """An orthoptist's value for one position of a stored session (same sign convention)."""
    with conn:
        conn.execute("INSERT INTO reference_measurements (session_id, position, horizontal_pd, vertical_pd, method, "
                     "examiner, measured_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (session_id, position, horizontal_pd, vertical_pd, method, examiner, measured_at))


def add_reference_diagnosis(conn: sqlite3.Connection, session_id: str, rule_id: str, present: bool,
                            affected_eye: Optional[str] = None, examiner: Optional[str] = None) -> None:
    """A clinician's yes / no for one engine rule on a stored session."""
    with conn:
        conn.execute("INSERT INTO reference_diagnoses (session_id, rule_id, affected_eye, present, examiner) "
                     "VALUES (?, ?, ?, ?, ?)", (session_id, rule_id, affected_eye, int(present), examiner))


def delete_patient(conn: sqlite3.Connection, pseudo_id: str) -> int:
    """Erase a patient and everything derived from them (DPDP right to erasure). Returns sessions removed."""
    with conn:
        sessions = [r[0] for r in conn.execute("SELECT session_id FROM sessions WHERE pseudo_id = ?", (pseudo_id,))]
        for sid in sessions:
            caps = [r[0] for r in conn.execute("SELECT id FROM captures WHERE session_id = ?", (sid,))]
            for table in ("frame_qc", "measurements"):
                conn.executemany(f"DELETE FROM {table} WHERE capture_id = ?", [(c,) for c in caps])
            for table in ("captures", "eye_measures", "findings", "reference_measurements", "reference_diagnoses"):
                conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (sid,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
        conn.execute("DELETE FROM patients WHERE pseudo_id = ?", (pseudo_id,))
    return len(sessions)
