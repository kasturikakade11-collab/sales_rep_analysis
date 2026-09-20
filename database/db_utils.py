import sqlite3
import pandas as pd
import json
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(THIS_DIR, "..", "..", "data", "calls.db")
SCHEMA_PATH = os.path.join(THIS_DIR, "schema.sql")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    print(f"Database initialized at {os.path.abspath(DB_PATH)}")


def insert_call(call_id, rep_id, call_date, duration_sec, overall_score=None, risk_score=None):
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO calls
           (call_id, rep_id, call_date, duration_sec, overall_score, risk_score)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (call_id, rep_id, call_date, duration_sec, overall_score, risk_score)
    )
    conn.commit()
    conn.close()


def insert_turns(df_turns: pd.DataFrame):
    conn = get_connection()
    df_turns.to_sql("turns", conn, if_exists="append", index=False)
    conn.close()


def insert_moments(call_id, moments_list):
    conn = get_connection()
    for m in moments_list:
        conn.execute(
            """INSERT INTO moments (call_id, turn_idx, moment_type, category, text_snippet, extra_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (call_id, m["turn_idx"], m["moment_type"], m.get("category"),
             m["text"], json.dumps(m.get("extra", {})))
        )
    conn.commit()
    conn.close()


def insert_risk_flags(call_id, flags: dict, reasons: list):
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO risk_flags
           (call_id, budget_addressed, authority_addressed, need_addressed, timeline_addressed, reasons_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (call_id, flags.get("budget"), flags.get("authority"),
         flags.get("need"), flags.get("timeline"), json.dumps(reasons))
    )
    conn.commit()
    conn.close()


def insert_coaching_report(call_id, strengths: list, improvements: list):
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO coaching_reports (call_id, strengths_json, improvements_json)
           VALUES (?, ?, ?)""",
        (call_id, json.dumps(strengths), json.dumps(improvements))
    )
    conn.commit()
    conn.close()


def insert_talk_metrics(call_id, metrics: dict):
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO talk_metrics
           (call_id, talk_ratio_rep, talk_ratio_prospect, pace_rep_wpm, pace_prospect_wpm, filler_word_rate)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (call_id, metrics.get("talk_ratio_rep"), metrics.get("talk_ratio_prospect"),
         metrics.get("pace_rep_wpm"), metrics.get("pace_prospect_wpm"), metrics.get("filler_word_rate"))
    )
    conn.commit()
    conn.close()


def get_calls_for_rep(rep_id):
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM calls WHERE rep_id = ?", conn, params=(rep_id,))
    conn.close()
    return df


def get_moments_for_call(call_id):
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM moments WHERE call_id = ?", conn, params=(call_id,))
    conn.close()
    return df


def get_all_calls():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM calls", conn)
    conn.close()
    return df
