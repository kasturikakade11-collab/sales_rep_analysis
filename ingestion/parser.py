"""
Ingestion module: reads a raw transcript .txt file and converts it
into a structured pandas DataFrame — one row per speaker turn.

NOTE ON TIMESTAMPS: our synthetic transcripts don't include real
timing data (they weren't generated from timed audio). To still
support pace calculations, this parser ESTIMATES a timestamp for
each turn based on an average speaking rate (130 words/minute is a
reasonable conversational average). If you later get transcripts
with real timestamps (e.g. from Whisper output, which does provide
them), swap the estimate_timestamps() step for parsing the real
timestamps instead — the rest of the pipeline doesn't need to change.
"""

import pandas as pd
import re

AVERAGE_WORDS_PER_MINUTE = 130  # used only for estimated timestamps


def parse_transcript(file_path: str) -> pd.DataFrame:
    """
    Reads a transcript file formatted as:
        Rep: some text here
        Customer: some text here
    Returns a DataFrame with columns:
        turn_number, speaker, text, word_count,
        est_start_time_sec, est_end_time_sec
    """
    rows = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    turn_number = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^(Rep|Customer):\s*(.*)$", line)
        if not match:
            continue  # skip malformed lines rather than crashing

        speaker, text = match.groups()
        word_count = len(text.split())

        rows.append({
            "turn_number": turn_number,
            "speaker": speaker,
            "text": text,
            "word_count": word_count,
        })
        turn_number += 1

    df = pd.DataFrame(rows)
    df = _estimate_timestamps(df)
    return df


def _estimate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds est_start_time_sec and est_end_time_sec columns based on
    cumulative word count and an average speaking pace. This is a
    stand-in for real timing data — see module docstring.
    """
    seconds_per_word = 60 / AVERAGE_WORDS_PER_MINUTE
    current_time = 0.0
    starts, ends = [], []

    for _, row in df.iterrows():
        duration = row["word_count"] * seconds_per_word
        starts.append(round(current_time, 1))
        current_time += duration
        ends.append(round(current_time, 1))

    df["est_start_time_sec"] = starts
    df["est_end_time_sec"] = ends
    return df


if __name__ == "__main__":
    # Quick manual test
    df = parse_transcript("data/transcripts/saas_001.txt")
    print(df)
    print(f"\nTotal turns: {len(df)}")
    print(f"Total estimated call duration: {df['est_end_time_sec'].iloc[-1]} sec")
