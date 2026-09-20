from ingestion.parser import parse_transcript
from analysis.talk_metrics import get_full_metrics

from agents.extraction_agent import (extract_moments)

from agents.risk_agent import (analyze_risk)

from agents.coaching_agent import (generate_coaching)

from validation.extraction_validator import validate_extraction

from validation.coaching_validator import validate_coaching

from agents.followup_email_agent import generate_followup_email

def run_pipeline(
    transcript_path: str
):
    print("\n1. Loading transcript...")

    df = parse_transcript(
        transcript_path
    )


    def format_timestamp(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


    transcript = "\n".join(
        (
            f"[{format_timestamp(row.start_time_sec)}] "
            f"{row.speaker}: "
            f"{row.text}"
        )
        for row in df.itertuples()
    )

    print("Transcript loaded.")


    print("\n2. Calculating metrics...")

    metrics = get_full_metrics(df)

    print(metrics)


    print("\n3. Running Agent 1...")

    extracted = extract_moments(
        transcript
    )

    print("Agent 1 completed.")

    print("\n3.5 Validating Agent 1 output...")

    extracted, validation_errors = validate_extraction(
        extracted,
        transcript
    )

    print(
        f"Validation completed. "
        f"Removed {len(validation_errors)} invalid moments."
    )

    if validation_errors:
        print("\nValidation issues:")

        for error in validation_errors:
            print(error)

        print("\n3.7 Generating follow-up email...")
    followup_email = generate_followup_email(extracted)
    print("Follow-up email generated.")


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

    print("\n5.5 Validating Agent 3 output...")

    coaching, coaching_errors = validate_coaching(
        coaching,
        extracted
    )

    print(
        f"Coaching validation completed. "
        f"Removed {len(coaching_errors)} invalid coaching points."
    )

    if coaching_errors:
        print("\nCoaching validation issues:")

        for error in coaching_errors:
            print(error)




    return {
        "metrics": metrics,
        "extracted_moments":
            extracted.model_dump(),
        "risk_analysis":
            risk.model_dump(),
        "coaching_report":
            coaching.model_dump(),
        "followup_email": followup_email
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

    print("\nFOLLOW-UP EMAIL DRAFT")
    print(result["followup_email"])