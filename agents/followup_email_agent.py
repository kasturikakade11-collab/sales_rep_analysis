from typing import List


def generate_followup_email(extracted_result) -> str:
    """
    Generate a follow-up email draft using validated next-step moments.
    No additional LLM/API call is required.
    """

    next_steps = []

    for moment in extracted_result.moments:
        if "next_step" in moment.categories:
            next_steps.append(moment)

    if not next_steps:
        return (
            "Subject: Follow-up on our conversation\n\n"
            "Hi,\n\n"
            "Thank you for taking the time to speak with me today.\n\n"
            "I'll follow up with the relevant information we discussed "
            "and look forward to continuing the conversation.\n\n"
            "Best regards,\n"
            "Sales Representative"
        )

    # Extract useful information from the already validated moments
    commitments = []

    for moment in next_steps:
        commitments.append(moment.evidence.strip())

    # Build a concise recap email
    email = "Subject: Follow-up on our conversation\n\n"
    email += "Hi,\n\n"
    email += (
        "Thank you for taking the time to speak with me today. "
        "It was great discussing your requirements and next steps.\n\n"
    )

    email += "As discussed, here are the next steps:\n"

    for commitment in commitments:
        email += f"- {commitment}\n"

    email += (
        "\nI'll make sure the discussed follow-up items are taken care of. "
        "Please feel free to let me know if there is anything else you "
        "would like me to include.\n\n"
    )

    email += "Looking forward to speaking with you.\n\n"
    email += "Best regards,\n"
    email += "Sales Representative"

    return email