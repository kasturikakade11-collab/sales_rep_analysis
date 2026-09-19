"""
Analysis module for sales-call metrics.

Calculates:
1. Rep vs customer talk ratio
2. Speaking pace
3. Filler-word usage

These metrics are later passed to the coaching agent.
"""

import pandas as pd
import re


# Common filler words/phrases.
FILLER_WORDS = {
    "um",
    "uh",
    "like",
    "basically",
    "actually",
    "you know",
    "sort of",
    "kind of",
}


def calculate_talk_ratio(df: pd.DataFrame) -> dict:
    """
    Calculates the percentage of words spoken by the Rep
    and Customer.

    Word count is used as a proxy for talk time because
    transcript files do not contain actual audio duration.
    """

    if df.empty:
        return {
            "rep_word_count": 0,
            "customer_word_count": 0,
            "rep_talk_ratio_pct": 0,
            "customer_talk_ratio_pct": 0,
        }

    totals = df.groupby("speaker")["word_count"].sum()

    rep_words = int(totals.get("Rep", 0))
    customer_words = int(totals.get("Customer", 0))

    total_words = rep_words + customer_words

    return {
        "rep_word_count": rep_words,
        "customer_word_count": customer_words,
        "rep_talk_ratio_pct": round(
            (rep_words / total_words) * 100, 1
        ) if total_words else 0,

        "customer_talk_ratio_pct": round(
            (customer_words / total_words) * 100, 1
        ) if total_words else 0,
    }


def calculate_pace(df: pd.DataFrame) -> dict:
    """
    Calculates approximate speaking pace using transcript timestamps.

    Since transcript timestamps represent the boundaries between
    speaker turns, the calculated pace is an approximate observed
    pace rather than true audio-level speaking speed.

    Returns:
        overall_pace_wpm
        rep_pace_wpm
    """

    if df.empty:
        return {
            "overall_pace_wpm": 0,
            "rep_pace_wpm": 0,
        }

    # Overall call duration
    call_start = df["start_time_sec"].min()
    call_end = df["end_time_sec"].max()

    total_duration_sec = call_end - call_start
    total_words = df["word_count"].sum()

    overall_wpm = (
        (total_words / total_duration_sec) * 60
        if total_duration_sec > 0
        else 0
    )

    # Rep-only turns
    rep_df = df[df["speaker"] == "Rep"].copy()

    if len(rep_df):
        rep_duration_sec = (
            rep_df["end_time_sec"] - rep_df["start_time_sec"]
        ).sum()

        rep_words = rep_df["word_count"].sum()

        rep_wpm = (
            (rep_words / rep_duration_sec) * 60
            if rep_duration_sec > 0
            else 0
        )
    else:
        rep_wpm = 0

    return {
        "overall_pace_wpm": round(overall_wpm, 1),
        "rep_pace_wpm": round(rep_wpm, 1),
    }


def count_filler_words(df: pd.DataFrame) -> dict:
    """
    Counts common filler words/phrases separately for
    the Rep and Customer.
    """

    counts = {
        "Rep": 0,
        "Customer": 0,
    }

    if df.empty:
        return {
            "rep_filler_count": 0,
            "customer_filler_count": 0,
        }

    for _, row in df.iterrows():

        speaker = row["speaker"]

        if speaker not in counts:
            continue

        text_lower = str(row["text"]).lower()

        for filler in FILLER_WORDS:
            pattern = rf"\b{re.escape(filler)}\b"
            counts[speaker] += len(
                re.findall(pattern, text_lower)
            )

    return {
        "rep_filler_count": counts["Rep"],
        "customer_filler_count": counts["Customer"],
    }


def get_full_metrics(df: pd.DataFrame) -> dict:
    """
    Runs all available quantitative analyses and combines
    the results into a single dictionary.
    """

    metrics = {}

    metrics.update(calculate_talk_ratio(df))
    metrics.update(calculate_pace(df))
    metrics.update(count_filler_words(df))

    return metrics


if __name__ == "__main__":

    import json
    from ingestion.parser import parse_transcript

    df = parse_transcript(
        "data/transcripts/saas_002.txt"
    )

    metrics = get_full_metrics(df)

    print(json.dumps(metrics, indent=2))