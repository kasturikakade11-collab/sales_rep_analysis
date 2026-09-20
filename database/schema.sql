CREATE TABLE IF NOT EXISTS calls (
    call_id TEXT PRIMARY KEY,
    rep_id TEXT,
    call_date TEXT,
    duration_sec REAL,
    overall_score INTEGER,
    risk_score INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS turns (
    call_id TEXT,
    turn_idx INTEGER,
    speaker TEXT,
    start_time REAL,
    end_time REAL,
    text TEXT,
    word_count INTEGER,
    PRIMARY KEY (call_id, turn_idx),
    FOREIGN KEY (call_id) REFERENCES calls(call_id)
);

CREATE TABLE IF NOT EXISTS moments (
    moment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT,
    turn_idx INTEGER,
    moment_type TEXT,
    category TEXT,
    text_snippet TEXT,
    extra_json TEXT,
    FOREIGN KEY (call_id) REFERENCES calls(call_id)
);

CREATE TABLE IF NOT EXISTS risk_flags (
    call_id TEXT PRIMARY KEY,
    budget_addressed BOOLEAN,
    authority_addressed BOOLEAN,
    need_addressed BOOLEAN,
    timeline_addressed BOOLEAN,
    reasons_json TEXT,
    FOREIGN KEY (call_id) REFERENCES calls(call_id)
);

CREATE TABLE IF NOT EXISTS coaching_reports (
    call_id TEXT PRIMARY KEY,
    strengths_json TEXT,
    improvements_json TEXT,
    generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (call_id) REFERENCES calls(call_id)
);

CREATE TABLE IF NOT EXISTS talk_metrics (
    call_id TEXT PRIMARY KEY,
    talk_ratio_rep REAL,
    talk_ratio_prospect REAL,
    pace_rep_wpm REAL,
    pace_prospect_wpm REAL,
    filler_word_rate REAL,
    FOREIGN KEY (call_id) REFERENCES calls(call_id)
);
