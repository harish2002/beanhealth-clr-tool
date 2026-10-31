-- BeanHealth CLR — gaze examination database (brief §7)
--
-- Pseudonymized: no names, no images. A patient is a pseudo_id the clinic
-- maps to a person outside this database; the report's patient label is
-- never written here. Raw video is not stored (captures.storage_uri stays
-- NULL unless the patient consented to raw video and it lives encrypted
-- elsewhere).
--
-- Portable DDL: runs on SQLite (development) and PostgreSQL (production;
-- use SERIAL / IDENTITY for the INTEGER PRIMARY KEY columns and JSONB for
-- the *_json columns). JSON is stored as text.

CREATE TABLE IF NOT EXISTS patients (
    pseudo_id      TEXT PRIMARY KEY,
    age_band       TEXT,
    consent_json   TEXT NOT NULL,          -- {"store_measurements": true, "research": false, "raw_video": false}
    research       INTEGER NOT NULL,       -- consented to validation studies (copied out of consent_json)
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS devices (
    id                   INTEGER PRIMARY KEY,
    model                TEXT NOT NULL,
    lens                 TEXT NOT NULL DEFAULT '',
    calibration_profile  TEXT NOT NULL DEFAULT '',
    UNIQUE (model, lens, calibration_profile)
);

CREATE TABLE IF NOT EXISTS kb_versions (
    kb_sha256   TEXT PRIMARY KEY,
    kb_version  TEXT NOT NULL,
    first_seen  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS engine_versions (
    id                        INTEGER PRIMARY KEY,
    engine_version            TEXT NOT NULL,
    measurement_version       TEXT NOT NULL,
    calibration_version       TEXT NOT NULL,
    qc_version                TEXT NOT NULL,
    capture_protocol_version  TEXT NOT NULL,
    UNIQUE (engine_version, measurement_version, calibration_version, qc_version, capture_protocol_version)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id            TEXT PRIMARY KEY,
    pseudo_id             TEXT NOT NULL REFERENCES patients (pseudo_id),
    device_id             INTEGER NOT NULL REFERENCES devices (id),
    protocol              TEXT NOT NULL,
    protocol_version      TEXT NOT NULL,
    tier                  INTEGER NOT NULL,
    started_at            TEXT NOT NULL,
    stored_at             TEXT NOT NULL,
    distance_cm           REAL,
    distance_error_cm     REAL,
    hirschberg_pd_per_mm  REAL,
    hirschberg_source     TEXT,
    referral_decision     TEXT NOT NULL,
    referral_urgent       INTEGER NOT NULL,
    kb_sha256             TEXT NOT NULL REFERENCES kb_versions (kb_sha256),
    engine_versions_id    INTEGER NOT NULL REFERENCES engine_versions (id)
);

CREATE TABLE IF NOT EXISTS captures (
    id            INTEGER PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions (session_id),
    position      TEXT NOT NULL,
    kind          TEXT NOT NULL,           -- calibration / gaze / tilt
    tilt          INTEGER NOT NULL,
    fps           REAL NOT NULL,
    width_px      INTEGER NOT NULL,
    height_px     INTEGER NOT NULL,
    frames_total  INTEGER NOT NULL,
    frames_valid  INTEGER NOT NULL,
    status        TEXT NOT NULL,           -- OK / INSUFFICIENT_DATA
    storage_uri   TEXT,                    -- NULL: landmark tracks only, no video
    UNIQUE (session_id, position)
);

CREATE TABLE IF NOT EXISTS frame_qc (
    capture_id     INTEGER NOT NULL REFERENCES captures (id),
    frame_idx      INTEGER NOT NULL,
    valid          INTEGER NOT NULL,
    reject_reason  TEXT,                   -- comma-separated codes; NULL when valid
    PRIMARY KEY (capture_id, frame_idx)
);

CREATE TABLE IF NOT EXISTS measurements (
    capture_id        INTEGER PRIMARY KEY REFERENCES captures (id),
    status            TEXT NOT NULL,
    horizontal_pd     REAL,                -- + = eso
    horizontal_error  REAL,
    vertical_pd       REAL,                -- + = right hyper
    vertical_error    REAL,
    confidence        TEXT,
    frames_used       INTEGER NOT NULL,
    eyes_json         TEXT                 -- per-eye rotation and in-orbit angle
);

CREATE TABLE IF NOT EXISTS eye_measures (
    session_id               TEXT NOT NULL REFERENCES sessions (session_id),
    eye                      TEXT NOT NULL,            -- R / L (the patient's)
    ductions_json            TEXT NOT NULL,
    fissure_height_mm        REAL,
    fissure_ratio_adduction  REAL,
    pupil_diameter_mm        REAL,
    scleral_show_mm          REAL,
    errors_json              TEXT NOT NULL,
    PRIMARY KEY (session_id, eye)
);

CREATE TABLE IF NOT EXISTS findings (
    id                     INTEGER PRIMARY KEY,
    session_id             TEXT NOT NULL REFERENCES sessions (session_id),
    rule_id                TEXT NOT NULL,
    affected_eye           TEXT,
    status                 TEXT NOT NULL,              -- REPORTED / INCONCLUSIVE
    abnormal               INTEGER NOT NULL,
    confidence             TEXT NOT NULL,
    urgency                TEXT,
    muscle                 TEXT,
    margin_over_threshold  REAL,
    evidence_json          TEXT NOT NULL,
    reasons_json           TEXT NOT NULL
);

-- Validation (brief §7, §9): what an orthoptist measured on the same visit.
CREATE TABLE IF NOT EXISTS reference_measurements (
    id             INTEGER PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions (session_id),
    position       TEXT NOT NULL,                      -- gaze or tilt position
    horizontal_pd  REAL,                               -- same signs as measurements
    vertical_pd    REAL,
    method         TEXT NOT NULL DEFAULT 'PBCT',
    examiner       TEXT,                               -- a role or code, not a name
    measured_at    TEXT,
    UNIQUE (session_id, position, method)
);

CREATE TABLE IF NOT EXISTS reference_diagnoses (
    id            INTEGER PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions (session_id),
    rule_id       TEXT NOT NULL,                       -- the engine rule it is compared with
    affected_eye  TEXT,
    present       INTEGER NOT NULL,
    examiner      TEXT,
    UNIQUE (session_id, rule_id, affected_eye)
);

CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions (pseudo_id);
CREATE INDEX IF NOT EXISTS idx_findings_session ON findings (session_id);
CREATE INDEX IF NOT EXISTS idx_reference_session ON reference_measurements (session_id);
