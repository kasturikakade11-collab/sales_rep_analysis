"""
Ingestion module: reads a raw sales-call transcript .txt file and
converts it into a structured pandas DataFrame — one row per speaker turn.

Expected transcript format:

    [00:01] Rep: Hello, thanks for joining.
    [00:07] Customer: Thanks, happy to be here.
    [00:17] Rep: Let's discuss your requirements.

The timestamps from the transcript are preserved as real timestamps.
"""

import pandas as pd
import re


# Matches:
# [00:01] Rep: Hello...
# [01:14] Customer: Yes...
TIMESTAMP_PATTERN = re.compile(
    r"^\[(\d{2}):(\d{2})\]\s*(Rep|Customer):\s*(.*)$"
)


def parse_transcript(file_path: str) -> pd.DataFrame:
    """
    Reads a transcript and returns a DataFrame with columns:

        turn_number
        speaker
        text
        word_count
        start_time_sec
        end_time_sec
    """

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    turn_number = 0

    for line in lines:
        line = line.strip()

        if not line:
            continue

        match = TIMESTAMP_PATTERN.match(line)

        if not match:
            # Skip malformed/unrecognized lines
            continue

        minutes, seconds, speaker, text = match.groups()

        start_time_sec = int(minutes) * 60 + int(seconds)

        word_count = len(text.split())

        rows.append({
            "turn_number": turn_number,
            "speaker": speaker,
            "text": text,
            "word_count": word_count,
            "start_time_sec": start_time_sec,
        })

        turn_number += 1

    df = pd.DataFrame(rows)

    if df.empty:
        # Return a DataFrame with the expected columns
        # instead of creating a partially structured DataFrame.
        return pd.DataFrame(columns=[
            "turn_number",
            "speaker",
            "text",
            "word_count",
            "start_time_sec",
            "end_time_sec"
        ])

    # Calculate end time of each speaker turn.
    #
    # For every turn except the last:
    # current turn ends when the next turn begins.
    #
    # For the final turn:
    # estimate its duration from word count using 130 WPM.
    df["end_time_sec"] = df["start_time_sec"].shift(-1)

    average_words_per_minute = 130
    seconds_per_word = 60 / average_words_per_minute

    last_index = df.index[-1]

    final_duration = (
        df.loc[last_index, "word_count"] * seconds_per_word
    )

    df.loc[last_index, "end_time_sec"] = (
        df.loc[last_index, "start_time_sec"] + final_duration
    )

    df["start_time_sec"] = df["start_time_sec"].astype(float)
    df["end_time_sec"] = df["end_time_sec"].astype(float)

    return df


if __name__ == "__main__":
    # Manual test using one of the actual transcripts
    df = parse_transcript("data/transcripts/saas_002.txt")

    print(df.to_string(index=False))

    print(f"\nTotal turns: {len(df)}")

    if len(df):
        print(
            f"Estimated final call time: "
            f"{df['end_time_sec'].iloc[-1]:.1f} seconds"
        )