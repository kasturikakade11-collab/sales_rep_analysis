from src.ingestion.parser import parse_transcript
from src.analysis.talk_metrics import calculate_metrics

from src.agents.extraction_agent import (
    extract_moments
)

from src.agents.risk_agent import (
    analyze_risk
)

from src.agents.coaching_agent import (
    generate_coaching
)


def run_pipeline(
    transcript_path: str
):

    print("\n1. Loading transcript...")

    df = parse_transcript(
        transcript_path
    )

    transcript = "\n".join(
        (
            f"[{row.timestamp}] "
            f"{row.speaker}: "
            f"{row.text}"
        )
        for row in df.itertuples()
    )

    print("Transcript loaded.")


    print("\n2. Calculating metrics...")

    metrics = calculate_metrics(df)

    print(metrics)


    print("\n3. Running Agent 1...")

    extracted = extract_moments(
        transcript
    )

    print("Agent 1 completed.")


    print("\n4. Running Agent 2...")

    risk = analyze_risk(
        extracted,
        metrics
    )

    print("Agent 2 completed.")


    print("\n5. Running Agent 3...")

    coaching = generate_coaching(
        extracted,
        risk,
        metrics
    )

    print("Agent 3 completed.")


    return {
        "metrics": metrics,
        "extracted_moments":
            extracted.model_dump(),
        "risk_analysis":
            risk.model_dump(),
        "coaching_report":
            coaching.model_dump()
    }


if __name__ == "__main__":

    result = run_pipeline(
        "data/transcripts/saas_001.txt"
    )

    print("\n")
    print("=" * 50)
    print("FINAL SALES CALL REPORT")
    print("=" * 50)

    print("\nMETRICS")
    print(result["metrics"])

    print("\nEXTRACTED MOMENTS")
    print(
        result["extracted_moments"]
    )

    print("\nRISK ANALYSIS")
    print(
        result["risk_analysis"]
    )

    print("\nCOACHING REPORT")
    print(
        result["coaching_report"]
    )