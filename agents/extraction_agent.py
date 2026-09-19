from typing import List, Literal

from pydantic import BaseModel, Field

from .llm_client import client, MODEL_NAME


class Moment(BaseModel):

    timestamp: str = Field(
        description="Timestamp of the moment."
    )

    speaker: Literal[
        "REP",
        "CUSTOMER"
    ]

    category: str = Field(
        description=(
            "Category such as pain_point, "
            "objection, pricing, competitor, "
            "buying_signal, hesitation, "
            "budget, authority, need, timeline, "
            "next_step, unanswered_question."
        )
    )

    evidence: str = Field(
        description=(
            "Evidence directly supported by "
            "the transcript."
        )
    )

    status: Literal[
        "resolved",
        "unresolved",
        "neutral"
    ]


class ExtractionResult(BaseModel):

    moments: List[Moment]


EXTRACTION_PROMPT = """
You are Agent 1 in a Sales Call Intelligence system.

Your job is to extract important sales-call moments.

Extract:

1. Customer pain points
2. Objections
3. Pricing discussions
4. Competitor mentions
5. Buying signals
6. Hesitation signals
7. Budget information
8. Decision-maker / authority information
9. Customer needs
10. Timeline information
11. Next-step commitments
12. Unanswered questions

Rules:

- Use ONLY information explicitly present in the transcript.
- Never invent facts.
- Never assume a competitor.
- Never infer budget.
- Never give coaching advice.
- Never judge the salesperson.
- Include timestamps.
- Evidence must be supported by the transcript.
"""


def extract_moments(
    transcript: str
) -> ExtractionResult:

    prompt = f"""
{EXTRACTION_PROMPT}

TRANSCRIPT:

--- START ---

{transcript}

--- END ---

Extract the important moments.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ExtractionResult
        }
    )

    return ExtractionResult.model_validate_json(
        response.text
    )