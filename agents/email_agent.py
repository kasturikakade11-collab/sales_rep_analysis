from pydantic import BaseModel

from .llm_client import client, MODEL_NAME


class FollowUpEmail(BaseModel):

    subject: str

    body: str


EMAIL_PROMPT = """
You are Agent 4, the Follow-Up Email Agent.

Your job is to draft a short, professional
recap email from the sales rep to the prospect,
based only on what was actually discussed in
the call.

Use:

1. Extracted moments (next steps, pricing
   mentions, competitor mentions, objections)
2. The original transcript for tone and context

Rules:

- Only reference next steps, pricing, or
  competitors that were actually extracted.
- Never invent commitments that were not made.
- Keep the tone professional and warm, not pushy.
- Keep the email concise — no more than 150 words.
- End with a clear, specific next action matching
  the extracted next steps.
- Use [Customer Name] and [Your Name] as
  placeholders since real names are not provided.
"""


def generate_followup_email(
    transcript,
    extracted_moments
) -> FollowUpEmail:

    prompt = f"""
{EMAIL_PROMPT}

TRANSCRIPT:

{transcript}

EXTRACTED MOMENTS:

{extracted_moments.model_dump_json(
    indent=2
)}

Generate the follow-up email.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": FollowUpEmail
        }
    )

    return FollowUpEmail.model_validate_json(
        response.text
    )