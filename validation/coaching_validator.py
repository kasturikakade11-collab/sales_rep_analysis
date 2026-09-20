import re
from typing import List, Tuple

from agents.coaching_agent import (
    CoachingReport,
    CoachingPoint
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for lightweight evidence comparison.

    This is intentionally conservative.
    It does not attempt semantic rewriting.
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


# ============================================================
# TIMESTAMP VALIDATION
# ============================================================

def timestamp_exists(
    timestamp: str,
    extracted_moments
) -> bool:
    """
    Check whether the coaching timestamp exists
    in the extracted moments.
    """

    return any(
        moment.timestamp == timestamp
        for moment in extracted_moments.moments
    )


# ============================================================
# GET EXTRACTED MOMENT
# ============================================================

def get_moment_by_timestamp(
    timestamp: str,
    extracted_moments
):
    """
    Return the extracted moment corresponding to a timestamp.
    """

    for moment in extracted_moments.moments:
        if moment.timestamp == timestamp:
            return moment

    return None


# ============================================================
# UNSUPPORTED CLAIM DETECTION
# ============================================================

FORBIDDEN_CLAIMS = [
    "win probability",
    "winning probability",
    "closing probability",
    "close probability",
    "increase the chance of winning",
    "increase win probability",
    "increase closing probability",
    "guarantee the deal",
    "guaranteed deal",
    "will close",
    "will definitely close",
    "will lose the deal",
    "will definitely lose",
]


def contains_forbidden_claim(text: str) -> bool:
    """
    Detect unsupported outcome/prediction language.
    """

    normalized = normalize_text(text)

    return any(
        phrase in normalized
        for phrase in FORBIDDEN_CLAIMS
    )


# ============================================================
# UNSUPPORTED COMPETITOR CLAIM DETECTION
# ============================================================

COMPETITOR_CLAIM_PATTERNS = [
    r"\b[a-z0-9]+ is cheaper\b",
    r"\b[a-z0-9]+ is more expensive\b",
    r"\b[a-z0-9]+ has better\b",
    r"\b[a-z0-9]+ has worse\b",
    r"\b[a-z0-9]+ offers better\b",
    r"\b[a-z0-9]+ offers worse\b",
    r"\b[a-z0-9]+ is better\b",
    r"\b[a-z0-9]+ is worse\b",
    r"\b[a-z0-9]+ has more\b",
    r"\b[a-z0-9]+ has less\b",
]


def contains_unsupported_competitor_claim(
    text: str,
    extracted_moments
) -> bool:
    """
    Detect broad competitor claims that are not clearly supported
    by extracted evidence.

    This is intentionally conservative.
    """

    normalized = normalize_text(text)

    competitor_names = set()

    for moment in extracted_moments.moments:

        if "competitor" not in moment.categories:
            continue

        evidence = moment.evidence

        words = re.findall(
            r"\b[A-Z][A-Za-z0-9_-]*\b",
            evidence
        )

        for word in words:
            competitor_names.add(
                word.lower()
            )

    if not competitor_names:
        return False

    for competitor in competitor_names:

        for pattern in COMPETITOR_CLAIM_PATTERNS:

            candidate = pattern.replace(
                r"\b[a-z0-9]+",
                re.escape(competitor),
                1
            )

            if re.search(
                candidate,
                normalized
            ):
                return True

    return False


# ============================================================
# UNSUPPORTED PRODUCT CLAIM DETECTION
# ============================================================

PRODUCT_CLAIM_PATTERNS = [
    r"\bour product has\b",
    r"\bour product offers\b",
    r"\bour platform has\b",
    r"\bour platform offers\b",
    r"\bour software has\b",
    r"\bour software offers\b",
    r"\bour solution has\b",
    r"\bour solution offers\b",
    r"\bour product provides\b",
    r"\bour platform provides\b",
    r"\bour solution provides\b",
    r"\bour product includes\b",
    r"\bour platform includes\b",
    r"\bour solution includes\b",
]


def contains_unsupported_product_claim(
    text: str,
    extracted_moments
) -> bool:
    """
    Detect product capability claims that are not directly
    represented in extracted evidence.
    """

    normalized = normalize_text(text)

    for pattern in PRODUCT_CLAIM_PATTERNS:

        if re.search(
            pattern,
            normalized
        ):
            return True

    return False


# ============================================================
# UNSUPPORTED ROI CLAIM DETECTION
# ============================================================

ROI_TERMS = [
    "roi",
    "return on investment",
    "financial return",
    "cost savings",
    "saves money",
    "save money",
    "increase revenue",
    "reduce costs",
]


def contains_unsupported_roi_claim(
    text: str,
    extracted_moments
) -> bool:
    """
    Detect ROI/financial outcome claims unless the same concept
    is explicitly present in extracted evidence.
    """

    normalized = normalize_text(text)

    evidence_text = " ".join(
        moment.evidence.lower()
        for moment in extracted_moments.moments
    )

    for term in ROI_TERMS:

        if term in normalized and term not in evidence_text:
            return True

    return False


# ============================================================
# UNSUPPORTED IMPLEMENTATION CLAIM DETECTION
# ============================================================

# ============================================================
# IMPLEMENTATION CLAIM VALIDATION
# ============================================================

IMPLEMENTATION_CLAIM_PATTERNS = [
    "implementation milestone",
    "implementation milestones",
    "implementation phase",
    "implementation phases",
    "project phase",
    "project phases",
    "deployment phase",
    "deployment phases",
    "implementation steps",
    "implementation plan",
]


def contains_unsupported_implementation_claim(
    text: str,
    extracted_moments
) -> bool:
    """
    Detect specific implementation claims that are not supported
    by any extracted evidence.

    Generic use of the word 'implementation' is NOT automatically
    considered unsupported.

    Example:

    Evidence:
    "Our implementation team can get you live in 2 weeks."

    Allowed:
    "Confirm that the two-week implementation timeline meets
    the customer's three-week requirement."

    Not supported:
    "Walk the customer through the implementation milestones."

    unless milestones actually appear in the extracted evidence.
    """

    normalized = normalize_text(text)

    evidence_text = " ".join(
        normalize_text(moment.evidence)
        for moment in extracted_moments.moments
    )

    for claim in IMPLEMENTATION_CLAIM_PATTERNS:

        if claim in normalized and claim not in evidence_text:
            return True

    return False

# ============================================================
# WHAT HAPPENED VALIDATION
# ============================================================

def validate_what_happened(
    coaching_point: CoachingPoint,
    extracted_moments
) -> List[str]:

    errors = []

    moment = get_moment_by_timestamp(
        coaching_point.timestamp,
        extracted_moments
    )

    if moment is None:
        errors.append(
            "Timestamp does not match an extracted moment."
        )
        return errors

    what_happened = normalize_text(
        coaching_point.what_happened
    )

    evidence = normalize_text(
        moment.evidence
    )

    # Exact evidence should normally appear inside the
    # what_happened explanation, unless the model is simply
    # describing the moment more concisely.

    evidence_words = set(
        evidence.split()
    )

    happened_words = set(
        what_happened.split()
    )

    if evidence_words:

        overlap = len(
            evidence_words.intersection(
                happened_words
            )
        ) / len(evidence_words)

        if overlap < 0.20:
            errors.append(
                "what_happened has very little lexical overlap "
                "with the extracted evidence."
            )

    return errors


# ============================================================
# GENERAL COACHING POINT VALIDATION
# ============================================================

def validate_coaching_point(
    coaching_point: CoachingPoint,
    extracted_moments
) -> List[str]:

    errors = []

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if not timestamp_exists(
        coaching_point.timestamp,
        extracted_moments
    ):
        errors.append(
            "Timestamp does not exist in extracted moments."
        )

        return errors

    # --------------------------------------------------------
    # What happened
    # --------------------------------------------------------

    errors.extend(
        validate_what_happened(
            coaching_point,
            extracted_moments
        )
    )

    # --------------------------------------------------------
    # Combined text
    # --------------------------------------------------------

    combined_text = " ".join([
        coaching_point.topic,
        coaching_point.what_happened,
        coaching_point.why_it_matters,
        coaching_point.coaching,
        coaching_point.suggested_phrase
    ])

    # --------------------------------------------------------
    # Forbidden outcome claims
    # --------------------------------------------------------

    if contains_forbidden_claim(
        combined_text
    ):
        errors.append(
            "Contains unsupported deal outcome or "
            "probability claim."
        )

    # --------------------------------------------------------
    # Competitor claims
    # --------------------------------------------------------

    if contains_unsupported_competitor_claim(
        combined_text,
        extracted_moments
    ):
        errors.append(
            "Contains an unsupported objective claim "
            "about a competitor."
        )

    # --------------------------------------------------------
    # Product claims
    # --------------------------------------------------------

    if contains_unsupported_product_claim(
        combined_text,
        extracted_moments
    ):
        errors.append(
            "Contains an unsupported product capability claim."
        )

    # --------------------------------------------------------
    # ROI claims
    # --------------------------------------------------------

    if contains_unsupported_roi_claim(
        combined_text,
        extracted_moments
    ):
        errors.append(
            "Contains an unsupported ROI or financial outcome claim."
        )

    # --------------------------------------------------------
    # Implementation claims
    # --------------------------------------------------------

    if contains_unsupported_implementation_claim(
        combined_text,
        extracted_moments
    ):
        errors.append(
            "Contains unsupported implementation details."
        )

    return errors


# ============================================================
# FULL COACHING VALIDATION
# ============================================================

def validate_coaching(
    coaching_report: CoachingReport,
    extracted_moments
) -> Tuple[CoachingReport, List[dict]]:
    """
    Validate Agent 3 output.

    Invalid coaching points are removed.

    The original coaching report is never rewritten by this
    validator.
    """

    valid_points = []

    errors = []

    for index, coaching_point in enumerate(
        coaching_report.coaching_points
    ):

        point_errors = validate_coaching_point(
            coaching_point,
            extracted_moments
        )

        if point_errors:

            errors.append({
                "point_index": index,
                "timestamp": coaching_point.timestamp,
                "topic": coaching_point.topic,
                "errors": point_errors
            })

        else:

            valid_points.append(
                coaching_point
            )

    validated_report = CoachingReport(
        strengths=coaching_report.strengths,
        coaching_points=valid_points,
        summary=coaching_report.summary
    )

    return validated_report, errors