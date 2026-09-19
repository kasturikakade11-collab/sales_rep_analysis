"""
Runs the parser + talk_metrics pipeline across every transcript in
data/transcripts/ and saves one summary CSV with all results.

This is Member 1's first deliverable: a single file showing talk-ratio,
pace, and filler-word metrics for every call in the dataset.
"""

import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "ingestion"))
sys.path.insert(0, os.path.join(BASE_DIR, "analysis"))

from parser import parse_transcript
from talk_metrics import get_full_metrics

TRANSCRIPT_DIR = "data/transcripts"
OUTPUT_PATH = "outputs/talk_metrics_summary.csv"


def run_all():
    results = []

    for fname in sorted(os.listdir(TRANSCRIPT_DIR)):
        if not fname.endswith(".txt"):
            continue

        call_id = fname.replace(".txt", "")
        file_path = os.path.join(TRANSCRIPT_DIR, fname)

        df = parse_transcript(file_path)
        metrics = get_full_metrics(df)
        metrics["call_id"] = call_id

        results.append(metrics)
        print(f"Processed {call_id}: rep talked {metrics['rep_talk_ratio_pct']}%, "
              f"pace {metrics['overall_pace_wpm']} wpm")

    summary_df = pd.DataFrame(results)
    # put call_id first for readability
    cols = ["call_id"] + [c for c in summary_df.columns if c != "call_id"]
    summary_df = summary_df[cols]

    os.makedirs("outputs", exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved summary for {len(summary_df)} calls to {OUTPUT_PATH}")

    return summary_df


if __name__ == "__main__":
    run_all()