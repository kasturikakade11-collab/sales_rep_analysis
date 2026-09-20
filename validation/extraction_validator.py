import re
from typing import List

from agents.extraction_agent import ExtractionResult


# ============================================================
# ALLOWED CATEGORIES
# ============================================================

ALLOWED_CATEGORIES = {
    "pain_point",
    "objection",
    "pricing",
    "competitor",
    "buying_signal",
    "hesitation",
    "budget",
    "authority",
    "need",
    "timeline",
    "next_step",
    "unanswered_question",
}


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_moment_structure(moment):
    """
    Validate the basic structure of one extracted moment.
    """

    errors = []

    # Timestamp
    if not moment.timestamp:
        errors.append("Missing timestamp")

    # Speaker
    if moment.speaker not in {"REP", "CUSTOMER"}:
        errors.append(f"Invalid speaker: {moment.speaker}")

    # Categories
    if not moment.categories:
        errors.append("No categories")

    for category in moment.categories:
        if category not in ALLOWED_CATEGORIES:
            errors.append(f"Invalid category: {category}")

    # Evidence
    if not moment.evidence or not moment.evidence.strip():
        errors.append("Missing evidence")

    # Status
    if moment.status not in {"resolved", "unresolved", "neutral"}:
        errors.append(f"Invalid status: {moment.status}")

    return errors


# ============================================================
# TIMESTAMP VALIDATION
# ============================================================

def is_valid_timestamp(timestamp: str) -> bool:
    """
    Check whether timestamp follows MM:SS format.
    """

    pattern = r"^\d{2}:\d{2}$"

    if not re.match(pattern, timestamp):
        return False

    minutes, seconds = timestamp.split(":")

    seconds = int(seconds)

    return 0 <= seconds < 60


# ============================================================
# EVIDENCE OVERLAP
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text while preserving currency symbols and
    useful pricing characters.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9₹$€£/\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def evidence_has_overlap(
    evidence: str,
    transcript: str,
    minimum_words: int = 3
) -> bool:
    """
    Check whether extracted evidence shares meaningful words
    with the original transcript.
    """

    evidence_words = set(
        normalize_text(evidence).split()
    )

    transcript_words = set(
        normalize_text(transcript).split()
    )

    overlap = evidence_words.intersection(
        transcript_words
    )

    return len(overlap) >= minimum_words


# ============================================================
# CATEGORY-SPECIFIC VALIDATION
# ============================================================

def validate_category(
    moment,
    context
) -> List[str]:
    """
    Validate whether the assigned categories are supported
    by the extracted evidence and nearby conversation context.
    """

    errors = []

    evidence = normalize_text(moment.evidence)
    context_text = normalize_text(context)

    categories = set(moment.categories)

    # ========================================================
    # COMPETITOR
    # ========================================================

    if "competitor" in categories:

        competitor_indicators = [
            "hubspot",
            "salesforce",
            "zoho",
            "pipedrive",
            "microsoft",
            "oracle",
            "sap",
            "competitor",
            "alternative",
            "vendor",
            "provider",
        ]

        if not any(
            indicator in context_text
            for indicator in competitor_indicators
        ):
            errors.append(
                "Competitor category is not supported "
                "by evidence or nearby context"
            )

    # ========================================================
    # PRICING
    # ========================================================

    if "pricing" in categories:

        pricing_indicators = [
            "$",
            "₹",
            "€",
            "£",
            "usd",
            "inr",
            "eur",
            "gbp",
            "dollar",
            "dollars",
            "rupee",
            "rupees",
            "pricing",
            "price",
            "cost",
            "fee",
            "per month",
            "per year",
            "monthly",
            "annually",
            "/month",
            "/year",
            "per seat",
            "per user",
        ]

        if not any(
            indicator in evidence
            for indicator in pricing_indicators
        ):
            errors.append(
                "Pricing category is not supported "
                "by the evidence"
            )

    # ========================================================
    # BUDGET
    # ========================================================

    if "budget" in categories:

        budget_indicators = [
            "budget",
            "approved",
            "approval",
            "spend",
            "spending",
            "budget limit",
            "budget ceiling",
            "afford",
            "financial",
            "allocated",
            "$",
            "₹",
            "€",
            "£",
        ]

        if not any(
            indicator in evidence
            for indicator in budget_indicators
        ):
            errors.append(
                "Budget category is not supported "
                "by the evidence"
            )

    # ========================================================
    # AUTHORITY
    # ========================================================

    if "authority" in categories:

        authority_indicators = [
            "cfo",
            "ceo",
            "cto",
            "manager",
            "director",
            "vp",
            "vice president",
            "decision maker",
            "decision-maker",
            "approver",
            "approval",
            "sign-off",
            "signoff",
            "approve",
        ]

        if not any(
            indicator in evidence
            for indicator in authority_indicators
        ):
            errors.append(
                "Authority category is not supported "
                "by the evidence"
            )

    # ========================================================
    # TIMELINE
    # ========================================================

    if "timeline" in categories:

        timeline_indicators = [
            "today",
            "tomorrow",
            "this week",
            "next week",
            "this month",
            "next month",
            "day",
            "days",
            "week",
            "weeks",
            "month",
            "months",
            "deadline",
            "schedule",
            "by ",
            "within ",
            "before ",
            "after ",
        ]

        has_timeline = any(
            indicator in evidence
            for indicator in timeline_indicators
        )

        if not has_timeline:
            errors.append(
                "Timeline category is not supported "
                "by the evidence"
            )

    # ========================================================
    # OBJECTION
    # ========================================================

    if "objection" in categories:

        objection_indicators = [
            "concern",
            "worried",
            "worry",
            "problem",
            "issue",
            "difficulty",
            "difficult",
            "challenge",
            "can't",
            "cannot",
            "unable",
            "don't want",
            "not sure",
            "hesitant",
            "hesitation",
            "but",
            "however",
        ]

        if not any(
            indicator in evidence
            for indicator in objection_indicators
        ):
            errors.append(
                "Objection category is not supported "
                "by the evidence"
            )

    # ========================================================
    # NEED
    # ========================================================

    if "need" in categories:

        need_indicators = [
            "need",
            "needs",
            "require",
            "requires",
            "required",
            "requirement",
            "must have",
            "must-have",
            "want",
            "wants",
            "looking for",
            "we're looking for",
            "we are looking for",
            "have to",
        ]

        if not any(
            indicator in evidence
            for indicator in need_indicators
        ):
            errors.append(
                "Need category is not explicitly supported "
                "by the evidence"
            )

    # ========================================================
    # BUYING SIGNAL
    # ========================================================

    if "buying_signal" in categories:

        buying_signal_indicators = [
            "ready to move forward",
            "ready to proceed",
            "move forward",
            "proceed",
            "get started",
            "let's get started",
            "sign",
            "purchase",
            "buy",
            "go ahead",
            "we'll take it",
            "this looks good",
            "good fit",
            "ready",
        ]

        if not any(
            indicator in evidence
            for indicator in buying_signal_indicators
        ):
            errors.append(
                "Buying signal is not explicitly supported "
                "by the evidence"
            )

    # ========================================================
    # NEXT STEP VALIDATION
    # ========================================================

    if "next_step" in categories:

        # ----------------------------------------------------
        # Explicit action indicators
        # ----------------------------------------------------

        action_indicators = [
            "schedule",
            "scheduled",
            "book",
            "call",
            "meet",
            "meeting",
            "send",
            "share",
            "review",
            "discuss",
            "follow up",
            "follow-up",
            "followup",
            "demo",
            "proposal",
            "contract",
            "invite",
            "start",
            "begin",
            "implement",
            "implementation",
            "sign",
            "approve",
            "confirm",
            "provide",
            "deliver",
            "submit",
            "email",
            "lock in",
        ]

        # ----------------------------------------------------
        # Explicit acceptance indicators
        #
        # IMPORTANT:
        # normalize_text() removes apostrophes.
        # Therefore "let's do that" becomes "let s do that".
        # ----------------------------------------------------

        acceptance_indicators = [
            "yes",
            "that works",
            "works for me",
            "that works for me",
            "sounds good",
            "sounds great",
            "let s do that",
            "that is fine",
            "that s fine",
            "perfect",
            "great",
            "sure",
            "absolutely",
            "please do",
            "that would be great",
            "that would work",
        ]

        # ----------------------------------------------------
        # Normalize evidence once.
        # ----------------------------------------------------

        evidence_normalized = normalize_text(
            moment.evidence
        )

        # ----------------------------------------------------
        # 1. Direct concrete action in the evidence
        #
        # Examples:
        #
        # "I'll send the proposal tomorrow."
        # "Let's schedule the demo."
        # "I'll email the pricing sheet."
        # ----------------------------------------------------

        has_direct_action = any(
            indicator in evidence_normalized
            for indicator in action_indicators
        )

        has_valid_next_step = has_direct_action

        # ----------------------------------------------------
        # 2. Explicit acceptance in the evidence
        # ----------------------------------------------------

        has_acceptance = any(
            indicator in evidence_normalized
            for indicator in acceptance_indicators
        )

        # ----------------------------------------------------
        # 3. Check nearby context for a concrete REP action
        # ----------------------------------------------------

        context_has_action = has_contextual_next_step_action(
            context,
            moment.evidence
        )

        # ----------------------------------------------------
        # 4. Acceptance + contextual REP action
        #
        # Example:
        #
        # REP:
        # "Want me to send a one-pager and lock in a
        # follow-up for next Thursday?"
        #
        # CUSTOMER:
        # "Yes, let's do that. Thursday at 2pm works."
        #
        # This is a valid next step.
        # ----------------------------------------------------

        if has_acceptance and context_has_action:
            has_valid_next_step = True

        # ----------------------------------------------------
        # 5. Accepted proposal language + contextual action
        # ----------------------------------------------------

        accepted_proposal_patterns = [
            "yes please",
            "sure",
            "absolutely",
            "go ahead",
            "please do",
            "that would be great",
            "that would work",
        ]

        has_proposal_acceptance = any(
            phrase in evidence_normalized
            for phrase in accepted_proposal_patterns
        )

        if has_proposal_acceptance and context_has_action:
            has_valid_next_step = True

        # ----------------------------------------------------
        # 6. Standalone time/date availability
        #
        # Examples:
        #
        # "Thursday at 2pm works."
        # "Friday works for me."
        #
        # These are NOT next steps by themselves.
        #
        # They are valid only when nearby REP context
        # establishes a concrete action.
        # ----------------------------------------------------

        standalone_time_acceptance = any(
            phrase in evidence_normalized
            for phrase in [
                "thursday works",
                "friday works",
                "monday works",
                "tuesday works",
                "wednesday works",
                "saturday works",
                "sunday works",
                "works for me",
            ]
        )

        if standalone_time_acceptance:
            has_valid_next_step = context_has_action

        # ----------------------------------------------------
        # No sufficient evidence of a concrete next step
        # ----------------------------------------------------

        if not has_valid_next_step:
            errors.append(
                "Next step is not supported by a concrete action "
                "or accepted proposal"
            )

    # ========================================================
    # HESITATION
    # ========================================================

    if "hesitation" in categories:

        hesitation_indicators = [
            "not sure",
            "unsure",
            "hesitant",
            "hesitation",
            "maybe",
            "perhaps",
            "might",
            "uncertain",
            "let me think",
            "need to think",
            "need some time",
        ]

        if not any(
            indicator in evidence
            for indicator in hesitation_indicators
        ):
            errors.append(
                "Hesitation category is not supported "
                "by the evidence"
            )

    # ========================================================
    # PAIN POINT
    # ========================================================

    if "pain_point" in categories:

        pain_indicators = [
            "problem",
            "issue",
            "challenge",
            "pain",
            "struggle",
            "difficult",
            "difficulty",
            "frustrating",
            "frustration",
            "slow",
            "manual",
            "lack",
            "missing",
            "costly",
            "inefficient",
        ]

        if not any(
            indicator in evidence
            for indicator in pain_indicators
        ):
            errors.append(
                "Pain point category is not supported "
                "by the evidence"
            )

    # ========================================================
    # UNANSWERED QUESTION
    # ========================================================

    if "unanswered_question" in categories:

        if "?" not in moment.evidence:
            errors.append(
                "Unanswered question category requires "
                "question evidence"
            )

    return errors


# ============================================================
# CONTEXT
# ============================================================

def get_context(
    transcript: str,
    timestamp: str,
    context_turns: int = 2
) -> str:
    """
    Return nearby transcript lines around the extracted
    moment's timestamp.
    """

    lines = transcript.splitlines()

    target_index = None

    for i, line in enumerate(lines):

        if line.startswith(
            f"[{timestamp}]"
        ):
            target_index = i
            break

    if target_index is None:
        return transcript

    start = max(
        0,
        target_index - context_turns
    )

    end = min(
        len(lines),
        target_index + context_turns + 1
    )

    return "\n".join(
        lines[start:end]
    )


# ============================================================
# CONTEXT-AWARE NEXT STEP VALIDATION
# ============================================================

def has_contextual_next_step_action(
    context: str,
    evidence: str
) -> bool:
    """
    Check whether the nearby conversation contains a concrete
    action proposed by the REP that the current evidence accepts.

    Example:

    REP:
    "Want me to send a one-pager you can forward,
    and we lock in a follow-up for next Thursday?"

    CUSTOMER:
    "Yes, let's do that. Thursday at 2pm works."

    This should be recognized as a valid next step.

    A standalone statement such as:

    "Thursday at 2pm works."

    is only considered a next step when the nearby REP turn
    establishes a concrete action.
    """

    lines = context.splitlines()

    evidence_normalized = normalize_text(evidence)

    # --------------------------------------------------------
    # Acceptance indicators
    #
    # These are normalized because normalize_text()
    # removes punctuation/apostrophes.
    # --------------------------------------------------------

    acceptance_indicators = [
        "yes",
        "yes please",
        "that works",
        "works for me",
        "that works for me",
        "sounds good",
        "sounds great",
        "let s do that",
        "that is fine",
        "that s fine",
        "perfect",
        "great",
        "sure",
        "absolutely",
        "go ahead",
        "please do",
        "that would be great",
        "that would work",
    ]

    # --------------------------------------------------------
    # Check whether current evidence actually accepts something
    # --------------------------------------------------------

    has_acceptance = any(
        indicator in evidence_normalized
        for indicator in acceptance_indicators
    )

    if not has_acceptance:
        return False

    # --------------------------------------------------------
    # Concrete REP action indicators
    # --------------------------------------------------------

    action_indicators = [
        "schedule",
        "scheduled",
        "book",
        "call",
        "meet",
        "meeting",
        "send",
        "share",
        "review",
        "discuss",
        "follow up",
        "follow-up",
        "followup",
        "demo",
        "proposal",
        "contract",
        "invite",
        "start",
        "begin",
        "implement",
        "implementation",
        "sign",
        "approve",
        "confirm",
        "provide",
        "deliver",
        "submit",
        "email",
        "lock in",
    ]

    # --------------------------------------------------------
    # Look for a concrete action in a nearby REP turn.
    #
    # Actual transcript format:
    #
    # [01:02] Rep: Want me to send a one-pager...
    #
    # --------------------------------------------------------

    for line in lines:

        # Match actual speaker label robustly.
        if not re.search(
            r"\]\s*rep\s*:",
            line,
            re.IGNORECASE
        ):
            continue

        line_normalized = normalize_text(line)

        # Check whether the REP actually proposed
        # a concrete action.
        if any(
            indicator in line_normalized
            for indicator in action_indicators
        ):
            return True

    return False


# ============================================================
# FULL EXTRACTION VALIDATOR
# ============================================================

def validate_extraction(
    result: ExtractionResult,
    transcript: str
):
    """
    Validate complete Agent 1 output.

    Returns:
        valid_result
        errors
    """

    valid_moments = []
    errors = []

    for index, moment in enumerate(result.moments):

        moment_errors = []

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        moment_errors.extend(
            validate_moment_structure(moment)
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        if moment.timestamp:

            if not is_valid_timestamp(
                moment.timestamp
            ):
                moment_errors.append(
                    "Invalid timestamp format"
                )

        # ----------------------------------------------------
        # Evidence overlap
        # ----------------------------------------------------

        if moment.evidence:

            if not evidence_has_overlap(
                moment.evidence,
                transcript
            ):
                moment_errors.append(
                    "Evidence has insufficient transcript overlap"
                )

        # ----------------------------------------------------
        # Category-specific validation
        # ----------------------------------------------------

        context = get_context(
            transcript,
            moment.timestamp
        )

        moment_errors.extend(
            validate_category(
                moment,
                context
            )
        )

        # ----------------------------------------------------
        # Keep valid moments
        # ----------------------------------------------------

        if not moment_errors:

            valid_moments.append(moment)

        else:

            errors.append({
                "moment_index": index,
                "timestamp": moment.timestamp,
                "evidence": moment.evidence,
                "errors": moment_errors,
            })

    # ========================================================
    # BUILD VALIDATED RESULT
    # ========================================================

    validated_result = ExtractionResult(
        moments=valid_moments
    )

    return validated_result, errors