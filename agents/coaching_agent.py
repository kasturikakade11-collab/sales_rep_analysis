from typing import List

from pydantic import BaseModel

from .llm_client import client, MODEL_NAME


class CoachingPoint(BaseModel):

    timestamp: str

    topic: str

    what_happened: str

    why_it_matters: str

    coaching: str

    suggested_phrase: str


class CoachingReport(BaseModel):

    strengths: List[str]

    coaching_points: List[CoachingPoint]

    summary: str


COACHING_PROMPT = """
You are Agent 3, the Sales Coaching Agent.

Your job is to generate specific,
actionable coaching.

Use:

1. Extracted moments
2. Risk analysis
3. Call metrics

Every coaching point MUST have a
real timestamp.

Each coaching point must contain:

- timestamp
- topic
- what happened
- why it matters
- coaching
- suggested phrase/action

Rules:

- Never give generic advice.
- Ground recommendations in evidence.
- Never invent events.
- Focus on observable sales behavior.
- Mention strengths when supported.
- Prioritize important improvements.
"""


def generate_coaching(
    extracted_moments,
    risk_analysis,
    metrics
) -> CoachingReport:

    prompt = f"""
{COACHING_PROMPT}

EXTRACTED MOMENTS:

{extracted_moments.model_dump_json(
    indent=2
)}

RISK ANALYSIS:

{risk_analysis.model_dump_json(
    indent=2
)}

CALL METRICS:

{metrics}

Generate the coaching report.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": CoachingReport
        }
    )

    return CoachingReport.model_validate_json(
        response.text
    )