from typing import List, Literal

from pydantic import BaseModel, Field

from .llm_client import client, MODEL_NAME
from .extraction_agent import ExtractionResult

class BANT(BaseModel):

    budget: Literal[
        "strong",
        "partial",
        "unknown"
    ]

    authority: Literal[
        "strong",
        "partial",
        "unknown"
    ]

    need: Literal[
        "strong",
        "partial",
        "unknown"
    ]

    timeline: Literal[
        "strong",
        "partial",
        "unknown"
    ]


class RiskReason(BaseModel):

    timestamp: str

    reason: str


class DealRisk(BaseModel):

    level: Literal[
        "LOW",
        "MEDIUM",
        "HIGH"
    ]

    reasons: List[RiskReason]


class CallQuality(BaseModel):

    score: float = Field(
        ge=0,
        le=10
    )

    reasoning: str


class RiskAnalysis(BaseModel):

    call_quality: CallQuality

    bant: BANT

    deal_risk: DealRisk


RISK_PROMPT = """
You are Agent 2 in a Sales Call Intelligence system.

Analyze the extracted sales-call evidence and call metrics.

Analyze:

- Budget
- Authority
- Need
- Timeline
- Unresolved objections
- Competitor pressure
- Hesitation
- Buying signals
- Next-step quality
- Talk ratio
- Speaking pace
- Filler words

BANT values:

strong
partial
unknown

Never assume missing information.

Deal risk:

LOW:
Few meaningful unresolved risk signals.

MEDIUM:
Some meaningful unresolved signals.

HIGH:
Multiple important unresolved signals
or a major unresolved blocker.

Every risk reason must be supported
by evidence and timestamp.

Do not provide coaching advice.
Do not invent facts.
"""


def analyze_risk(
    extracted_moments: ExtractionResult,
    metrics: dict
) -> RiskAnalysis:

    prompt = f"""
{RISK_PROMPT}

EXTRACTED MOMENTS:

{extracted_moments.model_dump_json(
    indent=2
)}

CALL METRICS:

{metrics}

Return the structured risk analysis.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": RiskAnalysis
        }
    )

    return RiskAnalysis.model_validate_json(
        response.text
    )