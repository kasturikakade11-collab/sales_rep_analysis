"""
Agent 1 - Extraction Evaluation

Evaluates Gemini Agent 1 extraction against data/ground_truth.json.

Categories:
    - objections
    - pricing_mentions
    - competitor_mentions
    - next_steps

Important:
    Objection evaluation uses controlled candidate detection.

    Agent 1 may express an objection using categories such as:
        objection
        hesitation
        pain_point
        pricing
        competitor
        budget
        authority
        timeline

    We do NOT automatically treat every moment in those categories
    as an objection. Controlled signals are required.

Matching:
    - Objections: controlled semantic concept matching + lexical similarity
    - Pricing: numerical + lexical matching
    - Competitors: entity/token + lexical matching
    - Next steps: action + temporal + lexical matching
    - One-to-one matching is enforced.

Outputs:
    outputs/extraction_evaluation.json
    outputs/extraction_evaluation_summary.csv
"""

from __future__ import annotations

import sys
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ============================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Add project root so Python can find:
# agents/
# data/
# outputs/
# etc.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# PATHS
# ============================================================

GROUND_TRUTH_PATH = BASE_DIR / "data" / "ground_truth.json"

OUTPUT_DIR = BASE_DIR / "outputs"

JSON_OUTPUT_PATH = (
    OUTPUT_DIR / "extraction_evaluation.json"
)

CSV_OUTPUT_PATH = (
    OUTPUT_DIR / "extraction_evaluation_summary.csv"
)


# ============================================================
# IMPORT AGENT 1
# ============================================================

from agents.extraction_agent import extract_moments


# ============================================================
# PATHS
# ============================================================



GROUND_TRUTH_PATH = BASE_DIR / "data" / "ground_truth.json"

OUTPUT_DIR = BASE_DIR / "outputs"

JSON_OUTPUT_PATH = OUTPUT_DIR / "extraction_evaluation.json"

CSV_OUTPUT_PATH = OUTPUT_DIR / "extraction_evaluation_summary.csv"


# ============================================================
# IMPORT AGENT 1
# ============================================================

from agents.extraction_agent import extract_moments


# ============================================================
# CATEGORY ALIASES
# ============================================================

CATEGORY_ALIASES = {
    "objections": {
        "objection",
        "objections",
    },
    "pricing_mentions": {
        "pricing",
        "price",
        "pricing_mention",
        "pricing_mentions",
    },
    "competitor_mentions": {
        "competitor",
        "competitors",
        "competitor_mention",
        "competitor_mentions",
    },
    "next_steps": {
        "next_step",
        "next_steps",
    },
}


# ============================================================
# CONTROLLED OBJECTION CANDIDATE CATEGORIES
# ============================================================

# These are the Agent 1 categories that MAY contain objection evidence.
#
# IMPORTANT:
# We do not simply classify every moment from these categories as
# an objection. The text must also contain objection-related signals.
#
# This is necessary because, for example:
#
#   pricing -> "Our plan is $499/month."
#
# is pricing information, not necessarily an objection.
#
# Whereas:
#
#   pricing -> "The price feels too high for our size."
#
# is an objection.

OBJECTION_SOURCE_CATEGORIES = {
    "objection",
    "objections",
    "hesitation",
    "pain_point",
    "pricing",
    "price",
    "competitor",
    "competitors",
    "budget",
    "authority",
    "timeline",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "them",
    "there",
    "this",
    "to",
    "us",
    "was",
    "we",
    "what",
    "when",
    "which",
    "with",
    "you",
    "your",
}


def normalize_text(text: Any) -> str:
    """
    Normalize text for lexical comparison.
    """
    if text is None:
        return ""

    text = str(text).lower()

    # Normalize common currency symbols.
    text = text.replace("₹", " inr ")
    text = text.replace("$", " usd ")
    text = text.replace("€", " eur ")
    text = text.replace("£", " gbp ")

    # Normalize apostrophes.
    text = text.replace("’", "'")

    # Remove punctuation while preserving numbers.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: Any) -> List[str]:
    """
    Tokenize normalized text and remove stopwords.
    """
    normalized = normalize_text(text)

    tokens = [
        token
        for token in normalized.split()
        if token not in STOPWORDS
    ]

    return tokens


def token_set(text: Any) -> set:
    return set(tokenize(text))


def lexical_similarity(text_a: Any, text_b: Any) -> float:
    """
    SequenceMatcher similarity.
    """
    a = normalize_text(text_a)
    b = normalize_text(text_b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def token_overlap(text_a: Any, text_b: Any) -> float:
    """
    Jaccard-style token overlap.
    """
    a = token_set(text_a)
    b = token_set(text_b)

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


# ============================================================
# NUMERIC EXTRACTION
# ============================================================

def extract_numbers(text: Any) -> List[str]:
    """
    Extract useful numeric expressions.

    Examples:
        $499
        499/month
        50 users
        85 lakh
        10%
        3 weeks
        120k
    """
    if text is None:
        return []

    text = str(text).lower()

    patterns = [
        r"\₹\s?\d+(?:,\d{3})*(?:\.\d+)?",
        r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?",
        r"\€\s?\d+(?:,\d{3})*(?:\.\d+)?",
        r"\£\s?\d+(?:,\d{3})*(?:\.\d+)?",
        r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:k|m|lakh|crore)?",
        r"\d+(?:\.\d+)?\s*%",
        r"\d+(?:\.\d+)?\s*(?:weeks?|months?|days?|years?|users?|seats?)",
    ]

    found = []

    for pattern in patterns:
        matches = re.findall(pattern, text)
        found.extend(matches)

    # Normalize numeric strings.
    cleaned = []

    for value in found:
        value = value.strip()
        value = re.sub(r"\s+", "", value)

        if value and value not in cleaned:
            cleaned.append(value)

    return cleaned


def numeric_overlap(text_a: Any, text_b: Any) -> float:
    """
    Measure overlap between numeric expressions.
    """
    a = set(extract_numbers(text_a))
    b = set(extract_numbers(text_b))

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


# ============================================================
# CATEGORY NORMALIZATION
# ============================================================

def normalize_category(category: Any) -> str:
    """
    Convert Agent 1's category into a normalized category.
    """
    if category is None:
        return ""

    category = str(category).strip().lower()

    for canonical, aliases in CATEGORY_ALIASES.items():
        if category in aliases:
            return canonical

    return category


# ============================================================
# EXTRACT MOMENT FIELDS
# ============================================================

def moment_to_dict(moment: Any) -> Dict[str, Any]:
    """
    Convert Pydantic Moment object or dictionary into a dictionary.
    """

    if isinstance(moment, dict):
        return {
            "timestamp": moment.get("timestamp", ""),
            "speaker": moment.get("speaker", ""),
            "category": moment.get("category", ""),
            "evidence": moment.get("evidence", ""),
            "status": moment.get("status", ""),
        }

    return {
        "timestamp": getattr(moment, "timestamp", ""),
        "speaker": getattr(moment, "speaker", ""),
        "category": getattr(moment, "category", ""),
        "evidence": getattr(moment, "evidence", ""),
        "status": getattr(moment, "status", ""),
    }


# ============================================================
# CONTROLLED OBJECTION DETECTION
# ============================================================

OBJECTION_SIGNAL_PHRASES = [
    # General objection / concern
    "concern",
    "concerns",
    "worried",
    "worry",
    "problem",
    "issue",
    "not sure",
    "unsure",
    "hesitant",
    "hesitation",
    "i need to think",
    "need to think",
    "think about it",
    "not ready",
    "not ready to decide",
    "not comfortable",
    "uncomfortable",
    "doesn't work",
    "does not work",
    "won't work",
    "cannot work",
    "can't work",
    "not convinced",
    "hard to justify",
    "difficult to justify",

    # Price objections
    "too expensive",
    "expensive",
    "too high",
    "higher than",
    "more than",
    "more expensive",
    "lower than",
    "too much",
    "costly",
    "price is",
    "price feels",
    "pricing feels",
    "pricing is",
    "price per",
    "cost per",
    "budget",
    "over budget",
    "above budget",
    "within budget",
    "budget constraint",
    "budget constraints",
    "money is tight",
    "can't afford",
    "cannot afford",

    # Comparison objections
    "comparing",
    "compare",
    "comparison",
    "competitor",
    "competitors",
    "another provider",
    "another company",
    "another vendor",
    "another dealership",
    "another option",
    "other provider",
    "other company",
    "other vendor",
    "quoted lower",
    "quoted higher",
    "better price",
    "lower price",

    # Timeline / implementation objections
    "timeline",
    "time to implement",
    "implementation time",
    "implementation timeline",
    "onboarding time",
    "onboarding timeline",
    "too long",
    "takes too long",
    "take too long",
    "before next quarter",
    "by next quarter",
    "limited time",
    "not enough time",
    "don't have time",
    "do not have time",
    "time to evaluate",

    # Authority / approval objections
    "not the one who signs",
    "not the decision maker",
    "not the decision-maker",
    "need approval",
    "needs approval",
    "approval process",
    "decision maker",
    "decision-maker",
    "sign off",
    "sign-off",
    "procurement",
    "director",
    "cfo",
    "need to involve",
    "need to loop in",

    # Need / product objections
    "not a gap",
    "feature gap",
    "missing feature",
    "deprecated",
    "dropped usage",
    "not useful",
    "doesn't meet",
    "does not meet",
    "not meeting",
    "doesn't support",
    "does not support",
    "not available",
]


def contains_objection_signal(text: str) -> bool:
    """
    Determine whether evidence contains a controlled objection signal.
    """
    normalized = normalize_text(text)

    if not normalized:
        return False

    for phrase in OBJECTION_SIGNAL_PHRASES:
        if normalize_text(phrase) in normalized:
            return True

    return False


def has_comparison_signal(text: str) -> bool:
    """
    Strong signals that a competitor/comparison mention is being
    used as an objection rather than merely being informational.
    """
    normalized = normalize_text(text)

    comparison_phrases = [
        "comparing",
        "compare",
        "comparison",
        "higher than",
        "lower than",
        "more than",
        "less than",
        "quoted lower",
        "quoted higher",
        "better price",
        "another provider",
        "another company",
        "another dealership",
        "competitor",
        "competitors",
    ]

    return any(
        normalize_text(phrase) in normalized
        for phrase in comparison_phrases
    )


def has_concern_signal(text: str) -> bool:
    """
    Signals that the speaker is expressing concern or resistance.
    """
    normalized = normalize_text(text)

    concern_phrases = [
        "concern",
        "worried",
        "worry",
        "problem",
        "issue",
        "not sure",
        "unsure",
        "hesitant",
        "hesitation",
        "need to think",
        "think about it",
        "not ready",
        "not comfortable",
        "too expensive",
        "too high",
        "too much",
        "over budget",
        "above budget",
        "budget constraint",
        "budget constraints",
        "money is tight",
        "limited time",
        "not enough time",
        "don't have time",
        "do not have time",
        "takes too long",
        "take too long",
        "too long",
        "not the one who signs",
        "approval process",
        "not the decision maker",
        "not the decision-maker",
    ]

    return any(
        normalize_text(phrase) in normalized
        for phrase in concern_phrases
    )


def is_objection_candidate(moment: Dict[str, Any]) -> bool:
    """
    Controlled objection candidate detection.

    Agent 1 can classify objection evidence under:
        objection
        hesitation
        pain_point
        pricing
        competitor
        budget
        authority
        timeline

    We use category-specific rules so that ordinary information does
    not automatically become an objection.
    """

    category = str(moment.get("category", "")).strip().lower()
    evidence = str(moment.get("evidence", "")).strip()

    if not evidence:
        return False

    # --------------------------------------------------------
    # Explicit objection
    # --------------------------------------------------------

    if category in {"objection", "objections"}:
        return True

    # --------------------------------------------------------
    # Hesitation
    # --------------------------------------------------------

    if category == "hesitation":
        return True

    # --------------------------------------------------------
    # Pain point
    #
    # A pain point is treated as an objection candidate because
    # the project ground truth may represent unresolved customer
    # problems/complaints as objections.
    #
    # But resolved/neutral pain points are not automatically
    # treated as objections unless the text contains a signal.
    # --------------------------------------------------------

    if category == "pain_point":
        status = str(moment.get("status", "")).lower()

        if status == "unresolved":
            return True

        return contains_objection_signal(evidence)

    # --------------------------------------------------------
    # Pricing
    #
    # Ordinary pricing information is NOT an objection.
    #
    # Example:
    #   "Our starter plan is $499/month."
    #
    # is not an objection.
    #
    # But:
    #   "The price per seat once we scale past 50 users..."
    #
    # is an objection candidate.
    # --------------------------------------------------------

    if category in {"pricing", "price"}:
        return (
            contains_objection_signal(evidence)
            or has_concern_signal(evidence)
            or has_comparison_signal(evidence)
        )

    # --------------------------------------------------------
    # Competitor
    #
    # A competitor mention becomes an objection candidate when
    # the evidence indicates active comparison or resistance.
    # --------------------------------------------------------

    if category in {"competitor", "competitors"}:
        return (
            has_comparison_signal(evidence)
            or contains_objection_signal(evidence)
        )

    # --------------------------------------------------------
    # Budget
    #
    # Budget information alone is not necessarily an objection.
    #
    # Unresolved budget concerns or explicit budget language are.
    # --------------------------------------------------------

    if category == "budget":
        status = str(moment.get("status", "")).lower()

        if status == "unresolved":
            return True

        return (
            "budget" in normalize_text(evidence)
            or "money" in normalize_text(evidence)
            or "afford" in normalize_text(evidence)
        ) and has_concern_signal(evidence)

    # --------------------------------------------------------
    # Authority
    #
    # An unresolved approval/decision-maker problem is an
    # objection candidate.
    # --------------------------------------------------------

    if category == "authority":
        status = str(moment.get("status", "")).lower()

        if status == "unresolved":
            return True

        return (
            "not the one who signs" in normalize_text(evidence)
            or "approval" in normalize_text(evidence)
            or "decision maker" in normalize_text(evidence)
            or "decision-maker" in normalize_text(evidence)
            or "procurement" in normalize_text(evidence)
        )

    # --------------------------------------------------------
    # Timeline
    #
    # Timeline information itself is not an objection.
    #
    # Example:
    #   "We want to be live before next quarter."
    #
    # is simply timeline information.
    #
    # But a concern about implementation/evaluation time is an
    # objection candidate.
    # --------------------------------------------------------

    if category == "timeline":
        return (
            has_concern_signal(evidence)
            or contains_objection_signal(evidence)
        )

    return False


# ============================================================
# GET CATEGORY PREDICTIONS
# ============================================================

def get_predictions_for_category(
    moments: List[Dict[str, Any]],
    category: str,
) -> List[Dict[str, Any]]:
    """
    Get prediction evidence for a canonical category.
    """

    results = []

    for moment in moments:

        normalized_category = normalize_category(
            moment.get("category", "")
        )

        if category == "objections":

            if is_objection_candidate(moment):
                results.append(moment)

        elif normalized_category == category:

            results.append(moment)

    return results


# ============================================================
# OBJECTION MATCHING
# ============================================================

def objection_concept_score(
    ground_truth: str,
    prediction: str,
) -> float:
    """
    Controlled semantic/lexical score for objections.

    This combines:
        - lexical similarity
        - token overlap
        - controlled concept signals

    It does NOT use an external embedding model.
    """

    gt = normalize_text(ground_truth)
    pred = normalize_text(prediction)

    if not gt or not pred:
        return 0.0

    sequence_score = lexical_similarity(gt, pred)
    overlap_score = token_overlap(gt, pred)

    gt_tokens = token_set(gt)
    pred_tokens = token_set(pred)

    concept_score = 0.0

    # --------------------------------------------------------
    # Price / cost concepts
    # --------------------------------------------------------

    price_terms = {
        "price",
        "pricing",
        "cost",
        "expensive",
        "higher",
        "lower",
        "budget",
        "money",
        "premium",
        "compensation",
        "salary",
        "quoted",
    }

    if gt_tokens & price_terms and pred_tokens & price_terms:
        concept_score = max(concept_score, 0.75)

    # --------------------------------------------------------
    # Timeline concepts
    # --------------------------------------------------------

    timeline_terms = {
        "timeline",
        "time",
        "timing",
        "weeks",
        "week",
        "month",
        "months",
        "quarter",
        "onboarding",
        "implementation",
        "evaluate",
        "evaluation",
    }

    if gt_tokens & timeline_terms and pred_tokens & timeline_terms:
        concept_score = max(concept_score, 0.75)

    # --------------------------------------------------------
    # Authority concepts
    # --------------------------------------------------------

    authority_terms = {
        "authority",
        "approval",
        "approve",
        "procurement",
        "director",
        "cfo",
        "sign",
        "signoff",
        "decision",
        "stakeholder",
    }

    if gt_tokens & authority_terms and pred_tokens & authority_terms:
        concept_score = max(concept_score, 0.75)

    # --------------------------------------------------------
    # Hesitation concepts
    # --------------------------------------------------------

    hesitation_terms = {
        "hesitation",
        "hesitant",
        "think",
        "decide",
        "decision",
        "ready",
        "uncertain",
        "unsure",
        "not",
    }

    if gt_tokens & hesitation_terms and pred_tokens & hesitation_terms:
        concept_score = max(concept_score, 0.75)

    # --------------------------------------------------------
    # Competitor/comparison concepts
    # --------------------------------------------------------

    comparison_terms = {
        "competitor",
        "competitors",
        "comparison",
        "comparing",
        "compare",
        "provider",
        "company",
        "vendor",
        "dealership",
        "salesforce",
        "hubspot",
        "looker",
        "icici",
    }

    if gt_tokens & comparison_terms and pred_tokens & comparison_terms:
        concept_score = max(concept_score, 0.75)

    # --------------------------------------------------------
    # Product/feature/pain concepts
    # --------------------------------------------------------

    pain_terms = {
        "problem",
        "issue",
        "pain",
        "feature",
        "deprecated",
        "usage",
        "gap",
        "missing",
        "slow",
        "reporting",
        "export",
    }

    if gt_tokens & pain_terms and pred_tokens & pain_terms:
        concept_score = max(concept_score, 0.75)

    # --------------------------------------------------------
    # Combine scores
    # --------------------------------------------------------

    combined = max(
        sequence_score,
        overlap_score,
        concept_score,
    )

    # A small bonus when both lexical and concept evidence agree.
    if concept_score >= 0.75 and (
        sequence_score >= 0.20
        or overlap_score >= 0.15
    ):
        combined = min(1.0, combined + 0.10)

    return combined


def match_objections(
    ground_truth: List[str],
    predictions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int, int]:

    """
    One-to-one objection matching.

    A prediction can match only one ground-truth objection.
    A ground-truth objection can match only one prediction.
    """

    if not ground_truth and not predictions:
        return [], 0, 0, 0

    if not ground_truth:
        return [], 0, len(predictions), 0

    if not predictions:
        return [], 0, 0, len(ground_truth)

    candidate_pairs = []

    for gt_index, gt_text in enumerate(ground_truth):

        for pred_index, prediction in enumerate(predictions):

            pred_text = prediction.get("evidence", "")

            score = objection_concept_score(
                gt_text,
                pred_text,
            )

            candidate_pairs.append(
                (
                    score,
                    gt_index,
                    pred_index,
                )
            )

    # Strongest matches first.
    candidate_pairs.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    matched_gt = set()
    matched_predictions = set()

    matches = []

    # Conservative threshold.
    MATCH_THRESHOLD = 0.50

    for score, gt_index, pred_index in candidate_pairs:

        if score < MATCH_THRESHOLD:
            continue

        if gt_index in matched_gt:
            continue

        if pred_index in matched_predictions:
            continue

        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)

        matches.append(
            {
                "ground_truth_index": gt_index,
                "prediction_index": pred_index,
                "ground_truth": ground_truth[gt_index],
                "prediction": predictions[pred_index].get(
                    "evidence",
                    "",
                ),
                "prediction_category": predictions[pred_index].get(
                    "category",
                    "",
                ),
                "score": round(score, 4),
            }
        )

    true_positives = len(matches)

    false_positives = (
        len(predictions) - true_positives
    )

    false_negatives = (
        len(ground_truth) - true_positives
    )

    return (
        matches,
        true_positives,
        false_positives,
        false_negatives,
    )


# ============================================================
# PRICING MATCHING
# ============================================================

def pricing_match_score(
    ground_truth: str,
    prediction: str,
) -> float:
    """
    Pricing matcher.

    Uses numerical agreement + lexical overlap.
    """

    number_score = numeric_overlap(
        ground_truth,
        prediction,
    )

    lexical_score = lexical_similarity(
        ground_truth,
        prediction,
    )

    overlap_score = token_overlap(
        ground_truth,
        prediction,
    )

    # Strong numerical agreement.
    if number_score > 0:
        return max(
            number_score,
            lexical_score,
            overlap_score,
        )

    return max(
        lexical_score,
        overlap_score,
    )


def match_pricing(
    ground_truth: List[str],
    predictions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int, int]:

    if not ground_truth and not predictions:
        return [], 0, 0, 0

    if not ground_truth:
        return [], 0, len(predictions), 0

    if not predictions:
        return [], 0, 0, len(ground_truth)

    candidate_pairs = []

    for gt_index, gt_text in enumerate(ground_truth):

        for pred_index, prediction in enumerate(predictions):

            pred_text = prediction.get(
                "evidence",
                "",
            )

            score = pricing_match_score(
                gt_text,
                pred_text,
            )

            candidate_pairs.append(
                (
                    score,
                    gt_index,
                    pred_index,
                )
            )

    candidate_pairs.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    matched_gt = set()
    matched_predictions = set()

    matches = []

    MATCH_THRESHOLD = 0.50

    for score, gt_index, pred_index in candidate_pairs:

        if score < MATCH_THRESHOLD:
            continue

        if gt_index in matched_gt:
            continue

        if pred_index in matched_predictions:
            continue

        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)

        matches.append(
            {
                "ground_truth_index": gt_index,
                "prediction_index": pred_index,
                "ground_truth": ground_truth[gt_index],
                "prediction": predictions[pred_index].get(
                    "evidence",
                    "",
                ),
                "score": round(score, 4),
            }
        )

    true_positives = len(matches)

    false_positives = (
        len(predictions) - true_positives
    )

    false_negatives = (
        len(ground_truth) - true_positives
    )

    return (
        matches,
        true_positives,
        false_positives,
        false_negatives,
    )


# ============================================================
# COMPETITOR MATCHING
# ============================================================

def competitor_match_score(
    ground_truth: str,
    prediction: str,
) -> float:

    gt = normalize_text(ground_truth)
    pred = normalize_text(prediction)

    if not gt or not pred:
        return 0.0

    # Exact containment.
    if gt in pred or pred in gt:
        return 1.0

    gt_tokens = token_set(gt)
    pred_tokens = token_set(pred)

    if gt_tokens and gt_tokens.issubset(pred_tokens):
        return 1.0

    overlap = token_overlap(gt, pred)

    lexical = lexical_similarity(
        gt,
        pred,
    )

    return max(
        overlap,
        lexical,
    )


def match_competitors(
    ground_truth: List[str],
    predictions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int, int]:

    if not ground_truth and not predictions:
        return [], 0, 0, 0

    if not ground_truth:
        return [], 0, len(predictions), 0

    if not predictions:
        return [], 0, 0, len(ground_truth)

    candidate_pairs = []

    for gt_index, gt_text in enumerate(ground_truth):

        for pred_index, prediction in enumerate(predictions):

            score = competitor_match_score(
                gt_text,
                prediction.get("evidence", ""),
            )

            candidate_pairs.append(
                (
                    score,
                    gt_index,
                    pred_index,
                )
            )

    candidate_pairs.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    matched_gt = set()
    matched_predictions = set()

    matches = []

    MATCH_THRESHOLD = 0.50

    for score, gt_index, pred_index in candidate_pairs:

        if score < MATCH_THRESHOLD:
            continue

        if gt_index in matched_gt:
            continue

        if pred_index in matched_predictions:
            continue

        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)

        matches.append(
            {
                "ground_truth_index": gt_index,
                "prediction_index": pred_index,
                "ground_truth": ground_truth[gt_index],
                "prediction": predictions[pred_index].get(
                    "evidence",
                    "",
                ),
                "score": round(score, 4),
            }
        )

    true_positives = len(matches)

    false_positives = (
        len(predictions) - true_positives
    )

    false_negatives = (
        len(ground_truth) - true_positives
    )

    return (
        matches,
        true_positives,
        false_positives,
        false_negatives,
    )


# ============================================================
# NEXT-STEP MATCHING
# ============================================================

ACTION_WORDS = {
    "send",
    "share",
    "call",
    "follow",
    "followup",
    "follow-up",
    "schedule",
    "arrange",
    "provide",
    "submit",
    "update",
    "email",
    "demo",
    "meet",
    "meeting",
    "review",
    "try",
    "identify",
    "sign",
    "hold",
    "upgrade",
    "checkout",
    "check",
    "contact",
}


TEMPORAL_WORDS = {
    "today",
    "tomorrow",
    "tonight",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "next",
    "week",
    "weeks",
    "month",
    "months",
    "quarter",
    "hour",
    "hours",
    "day",
    "days",
}


def action_overlap(
    text_a: str,
    text_b: str,
) -> float:

    a = token_set(text_a)
    b = token_set(text_b)

    a_actions = a & ACTION_WORDS
    b_actions = b & ACTION_WORDS

    if not a_actions or not b_actions:
        return 0.0

    return len(a_actions & b_actions) / len(
        a_actions | b_actions
    )


def temporal_overlap(
    text_a: str,
    text_b: str,
) -> float:

    a = token_set(text_a)
    b = token_set(text_b)

    a_time = a & TEMPORAL_WORDS
    b_time = b & TEMPORAL_WORDS

    if not a_time or not b_time:
        return 0.0

    return len(a_time & b_time) / len(
        a_time | b_time
    )


def next_step_match_score(
    ground_truth: str,
    prediction: str,
) -> float:

    lexical = lexical_similarity(
        ground_truth,
        prediction,
    )

    overlap = token_overlap(
        ground_truth,
        prediction,
    )

    action = action_overlap(
        ground_truth,
        prediction,
    )

    temporal = temporal_overlap(
        ground_truth,
        prediction,
    )

    # Strong action + temporal agreement.
    if action > 0 and temporal > 0:
        return max(
            lexical,
            overlap,
            min(
                1.0,
                0.50 + action * 0.25 + temporal * 0.25,
            ),
        )

    return max(
        lexical,
        overlap,
        action,
    )


def match_next_steps(
    ground_truth: List[str],
    predictions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int, int]:

    if not ground_truth and not predictions:
        return [], 0, 0, 0

    if not ground_truth:
        return [], 0, len(predictions), 0

    if not predictions:
        return [], 0, 0, len(ground_truth)

    candidate_pairs = []

    for gt_index, gt_text in enumerate(ground_truth):

        for pred_index, prediction in enumerate(predictions):

            score = next_step_match_score(
                gt_text,
                prediction.get(
                    "evidence",
                    "",
                ),
            )

            candidate_pairs.append(
                (
                    score,
                    gt_index,
                    pred_index,
                )
            )

    candidate_pairs.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    matched_gt = set()
    matched_predictions = set()

    matches = []

    MATCH_THRESHOLD = 0.50

    for score, gt_index, pred_index in candidate_pairs:

        if score < MATCH_THRESHOLD:
            continue

        if gt_index in matched_gt:
            continue

        if pred_index in matched_predictions:
            continue

        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)

        matches.append(
            {
                "ground_truth_index": gt_index,
                "prediction_index": pred_index,
                "ground_truth": ground_truth[gt_index],
                "prediction": predictions[pred_index].get(
                    "evidence",
                    "",
                ),
                "score": round(score, 4),
            }
        )

    true_positives = len(matches)

    false_positives = (
        len(predictions) - true_positives
    )

    false_negatives = (
        len(ground_truth) - true_positives
    )

    return (
        matches,
        true_positives,
        false_positives,
        false_negatives,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> Dict[str, Any]:

    precision_denominator = (
        true_positives + false_positives
    )

    recall_denominator = (
        true_positives + false_negatives
    )

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = (
            true_positives
            / precision_denominator
        )

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = (
            true_positives
            / recall_denominator
        )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth() -> Dict[str, Any]:

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Ground truth file not found:\n"
            f"{GROUND_TRUTH_PATH}"
        )

    with open(
        GROUND_TRUTH_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# TRANSCRIPT LOADING
# ============================================================

def load_transcript(call_id: str) -> str:

    transcript_dir = (
        BASE_DIR
        / "data"
        / "transcripts"
    )

    possible_files = [
        transcript_dir / f"{call_id}.txt",
        transcript_dir / f"{call_id}.md",
        transcript_dir / f"{call_id}.csv",
    ]

    for path in possible_files:

        if path.exists():

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as file:

                return file.read()

    raise FileNotFoundError(
        f"Transcript not found for {call_id} "
        f"in {transcript_dir}"
    )


# ============================================================
# EVALUATE ONE CALL
# ============================================================

def evaluate_call(
    call_id: str,
    ground_truth: Dict[str, Any],
    moments: List[Dict[str, Any]],
) -> Dict[str, Any]:

    category_results = {}

    categories = [
        "objections",
        "pricing_mentions",
        "competitor_mentions",
        "next_steps",
    ]

    for category in categories:

        gt_items = ground_truth.get(
            category,
            [],
        )

        predictions = get_predictions_for_category(
            moments,
            category,
        )

        if category == "objections":

            (
                matches,
                tp,
                fp,
                fn,
            ) = match_objections(
                gt_items,
                predictions,
            )

        elif category == "pricing_mentions":

            (
                matches,
                tp,
                fp,
                fn,
            ) = match_pricing(
                gt_items,
                predictions,
            )

        elif category == "competitor_mentions":

            (
                matches,
                tp,
                fp,
                fn,
            ) = match_competitors(
                gt_items,
                predictions,
            )

        elif category == "next_steps":

            (
                matches,
                tp,
                fp,
                fn,
            ) = match_next_steps(
                gt_items,
                predictions,
            )

        else:
            matches = []
            tp = fp = fn = 0

        metrics = calculate_metrics(
            tp,
            fp,
            fn,
        )

        category_results[category] = {
            "ground_truth": gt_items,
            "predictions": [
                prediction.get(
                    "evidence",
                    "",
                )
                for prediction in predictions
            ],
            "prediction_details": predictions,
            "matches": matches,
            **metrics,
        }

    return {
        "call_id": call_id,
        "predictions": moments,
        "categories": category_results,
    }


# ============================================================
# AGGREGATE RESULTS
# ============================================================

def aggregate_results(
    per_call_results: List[Dict[str, Any]],
) -> Dict[str, Any]:

    categories = [
        "objections",
        "pricing_mentions",
        "competitor_mentions",
        "next_steps",
    ]

    category_totals = {}

    for category in categories:

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for call_result in per_call_results:

            result = call_result[
                "categories"
            ][category]

            total_tp += result[
                "true_positives"
            ]

            total_fp += result[
                "false_positives"
            ]

            total_fn += result[
                "false_negatives"
            ]

        category_totals[category] = (
            calculate_metrics(
                total_tp,
                total_fp,
                total_fn,
            )
        )

    overall_tp = sum(
        category_totals[category][
            "true_positives"
        ]
        for category in categories
    )

    overall_fp = sum(
        category_totals[category][
            "false_positives"
        ]
        for category in categories
    )

    overall_fn = sum(
        category_totals[category][
            "false_negatives"
        ]
        for category in categories
    )

    overall = calculate_metrics(
        overall_tp,
        overall_fp,
        overall_fn,
    )

    return {
        "number_of_calls": len(
            per_call_results
        ),
        "categories": categories,
        "matching_method": {
            "objections": (
                "controlled semantic concept "
                "matching + lexical similarity "
                "with multi-category objection "
                "candidate detection"
            ),
            "pricing_mentions": (
                "numerical + lexical matching"
            ),
            "competitor_mentions": (
                "entity/token + lexical matching"
            ),
            "next_steps": (
                "action + temporal + lexical matching"
            ),
            "one_to_one_matching": True,
        },
        "category_performance": category_totals,
        "overall": overall,
    }


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    summary: Dict[str, Any],
    per_call_results: List[Dict[str, Any]],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "evaluation_summary": summary,
        "per_call_results": per_call_results,
    }

    with open(
        JSON_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    per_call_results: List[Dict[str, Any]],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    categories = [
        "objections",
        "pricing_mentions",
        "competitor_mentions",
        "next_steps",
    ]

    for call_result in per_call_results:

        call_id = call_result[
            "call_id"
        ]

        for category in categories:

            result = call_result[
                "categories"
            ][category]

            rows.append(
                {
                    "call_id": call_id,
                    "category": category,
                    "precision": result[
                        "precision"
                    ],
                    "recall": result[
                        "recall"
                    ],
                    "f1_score": result[
                        "f1"
                    ],
                    "true_positives": result[
                        "true_positives"
                    ],
                    "false_positives": result[
                        "false_positives"
                    ],
                    "false_negatives": result[
                        "false_negatives"
                    ],
                }
            )

    fieldnames = [
        "call_id",
        "category",
        "precision",
        "recall",
        "f1_score",
        "true_positives",
        "false_positives",
        "false_negatives",
    ]

    with open(
        CSV_OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    summary: Dict[str, Any],
    per_call_results: List[Dict[str, Any]],
) -> None:

    print()
    print("=" * 75)
    print(
        "              AGENT 1 EXTRACTION EVALUATION"
    )
    print("=" * 75)

    print()
    print("CATEGORY PERFORMANCE")
    print("-" * 75)

    print(
        f"{'Category':<25}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
        f"{'F1 Score':<15}"
    )

    print("-" * 75)

    for category in summary["categories"]:

        result = summary[
            "category_performance"
        ][category]

        print(
            f"{category:<25}"
            f"{result['precision']:<15.3f}"
            f"{result['recall']:<15.3f}"
            f"{result['f1']:<15.3f}"
        )

    print("-" * 75)

    overall = summary["overall"]

    print(
        f"{'OVERALL':<25}"
        f"{overall['precision']:<15.3f}"
        f"{overall['recall']:<15.3f}"
        f"{overall['f1']:<15.3f}"
    )

    print("=" * 75)

    print()
    print("PER-CALL RESULTS")
    print("-" * 75)

    for call_result in per_call_results:

        print()
        print(call_result["call_id"])

        for category in summary["categories"]:

            result = call_result[
                "categories"
            ][category]

            print(
                f"  {category:<20}"
                f"P={result['precision']:.3f} "
                f"R={result['recall']:.3f} "
                f"F1={result['f1']:.3f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "Starting Agent 1 extraction evaluation..."
    )
    print("=" * 75)

    ground_truth = load_ground_truth()

    print()
    print(
        f"Loaded ground truth for "
        f"{len(ground_truth)} calls."
    )

    per_call_results = []

    for call_id, call_ground_truth in ground_truth.items():

        print()
        print(
            f"Processing {call_id}..."
        )

        try:

            transcript = load_transcript(
                call_id
            )

            extraction_result = extract_moments(
                transcript
            )

            raw_moments = getattr(
                extraction_result,
                "moments",
                [],
            )

            moments = [
                moment_to_dict(moment)
                for moment in raw_moments
            ]

            result = evaluate_call(
                call_id,
                call_ground_truth,
                moments,
            )

            per_call_results.append(
                result
            )

        except Exception as error:

            print(
                f"ERROR processing "
                f"{call_id}: {error}"
            )

            # Keep the evaluation running for
            # the remaining calls.
            continue

    summary = aggregate_results(
        per_call_results
    )

    save_json(
        summary,
        per_call_results,
    )

    save_csv(
        per_call_results
    )

    print_results(
        summary,
        per_call_results,
    )

    print()
    print("=" * 75)
    print(
        "Detailed evaluation saved to:"
    )
    print(
        JSON_OUTPUT_PATH
    )

    print()
    print(
        "CSV summary saved to:"
    )
    print(
        CSV_OUTPUT_PATH
    )

    print("=" * 75)


if __name__ == "__main__":
    main()