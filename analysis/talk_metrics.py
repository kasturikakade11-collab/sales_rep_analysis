"""
Analysis module: computes talk-ratio, pace, and filler-word metrics
from the DataFrame produced by ingestion/parser.py.

These metrics feed directly into Agent 3 (coaching feedback) later —
e.g. "you talked 78% of the call" or "your pace spiked during the
pricing discussion" are the kinds of things Agent 3 will reference.
"""

import pandas as pd
import re

# Common filler words to flag. Expand this list based on what you
# actually see in real transcripts.
FILLER_WORDS = {"um", "uh", "like", "basically", "actually", "you know", "sort of", "kind of"}


def calculate_talk_ratio(df: pd.DataFrame) -> dict:
    """
    Returns rep vs. customer talk-time share, measured by word count
    (a reasonable proxy for talk time in the absence of real audio
    duration).
    """
    totals = df.groupby("speaker")["word_count"].sum()
    total_words = totals.sum()

    rep_words = totals.get("Rep", 0)
    customer_words = totals.get("Customer", 0)

    return {
        "rep_word_count": int(rep_words),
        "customer_word_count": int(customer_words),
        "rep_talk_ratio_pct": round((rep_words / total_words) * 100, 1) if total_words else 0,
        "customer_talk_ratio_pct": round((customer_words / total_words) * 100, 1) if total_words else 0,
    }


def calculate_pace(df: pd.DataFrame) -> dict:
    """
    Returns overall and rep-only speaking pace in words per minute,
    using the estimated timestamps from the parser.
    NOTE: since these are estimated timestamps (see parser.py), pace
    here will be close to the AVERAGE_WORDS_PER_MINUTE constant by
    construction. This becomes meaningful once real timestamps are
    used — keep the calculation in place so the pipeline doesn't
    need to change later.
    """
    total_duration_sec = df["est_end_time_sec"].iloc[-1] if len(df) else 0
    total_words = df["word_count"].sum()

    rep_df = df[df["speaker"] == "Rep"]
    rep_words = rep_df["word_count"].sum()
    rep_duration_sec = (
        rep_df["est_end_time_sec"].max() - rep_df["est_start_time_sec"].min()
        if len(rep_df) else 0
    )

    overall_wpm = (total_words / total_duration_sec) * 60 if total_duration_sec else 0
    rep_wpm = (rep_words / rep_duration_sec) * 60 if rep_duration_sec else 0

    return {
        "overall_pace_wpm": round(overall_wpm, 1),
        "rep_pace_wpm": round(rep_wpm, 1),
    }


def count_filler_words(df: pd.DataFrame) -> dict:
    """
    Counts filler-word usage, broken down by speaker.
    """
    counts = {"Rep": 0, "Customer": 0}
    for _, row in df.iterrows():
        text_lower = row["text"].lower()
        for filler in FILLER_WORDS:
            counts[row["speaker"]] += len(re.findall(rf"\b{re.escape(filler)}\b", text_lower))
    return {
        "rep_filler_count": counts["Rep"],
        "customer_filler_count": counts["Customer"],
    }


def get_full_metrics(df: pd.DataFrame) -> dict:
    """
    Convenience function — runs all three analyses and returns one
    combined dict. This is what gets passed into Agent 3 later.
    """
    metrics = {}
    metrics.update(calculate_talk_ratio(df))
    metrics.update(calculate_pace(df))
    metrics.update(count_filler_words(df))
    return metrics


if __name__ == "__main__":
    from parser import parse_transcript
    import json

    df = parse_transcript("data/transcripts/saas_001.txt")
    metrics = get_full_metrics(df)
    print(json.dumps(metrics, indent=2))
