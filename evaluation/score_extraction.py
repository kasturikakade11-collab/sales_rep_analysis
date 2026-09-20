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

UPDATED: Agent 1 now returns a "categories" LIST per moment (a single
moment can belong to more than one category, e.g. a sentence with both
a price and a future commitment gets both "pricing" and "next_step").
All matching logic below checks membership in that list/set instead of
comparing a single "category" string.

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

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


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
    "objections": {"objection", "objections"},
    "pricing_mentions": {"pricing", "price", "pricing_mention", "pricing_mentions"},
    "competitor_mentions": {
        "competitor", "competitors", "competitor_mention", "competitor_mentions",
    },
    "next_steps": {"next_step", "next_steps"},
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "how", "i", "if", "in", "is", "it", "its", "me", "my",
    "of", "on", "or", "our", "that", "the", "their", "them", "there",
    "this", "to", "us", "was", "we", "what", "when", "which", "with",
    "you", "your",
}


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = text.replace("₹", " inr ")
    text = text.replace("$", " usd ")
    text = text.replace("€", " eur ")
    text = text.replace("£", " gbp ")
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: Any) -> List[str]:
    normalized = normalize_text(text)
    return [t for t in normalized.split() if t not in STOPWORDS]


def token_set(text: Any) -> set:
    return set(tokenize(text))


def lexical_similarity(text_a: Any, text_b: Any) -> float:
    a = normalize_text(text_a)
    b = normalize_text(text_b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def token_overlap(text_a: Any, text_b: Any) -> float:
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
    UPDATED: also adds bare-digit versions (e.g. "78" from "78 lakh" or
    "78lakh") so formatting differences between ground truth and
    predictions ("78lakh" vs "around 78") still match numerically.
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
        found.extend(re.findall(pattern, text))

    cleaned = []
    for value in found:
        value = value.strip()
        value = re.sub(r"\s+", "", value)
        if value and value not in cleaned:
            cleaned.append(value)

    # NEW: bare digit fallback, strips any attached unit/word so
    # "78lakh" and "78" both produce a comparable "78" token.
    bare_digits = re.findall(r"\d+(?:\.\d+)?", text)
    for value in bare_digits:
        if value not in cleaned:
            cleaned.append(value)

    return cleaned


def numeric_overlap(text_a: Any, text_b: Any) -> float:
    a = set(extract_numbers(text_a))
    b = set(extract_numbers(text_b))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ============================================================
# CATEGORY NORMALIZATION
# ============================================================

def normalize_category(category: Any) -> str:
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
    UPDATED: reads "categories" (a list) instead of "category" (a
    single string), for both dict and Pydantic-object inputs.
    """
    if isinstance(moment, dict):
        return {
            "timestamp": moment.get("timestamp", ""),
            "speaker": moment.get("speaker", ""),
            "categories": moment.get("categories", []),
            "evidence": moment.get("evidence", ""),
            "status": moment.get("status", ""),
        }

    return {
        "timestamp": getattr(moment, "timestamp", ""),
        "speaker": getattr(moment, "speaker", ""),
        "categories": getattr(moment, "categories", []),
        "evidence": getattr(moment, "evidence", ""),
        "status": getattr(moment, "status", ""),
    }


# ============================================================
# CONTROLLED OBJECTION DETECTION
# ============================================================

OBJECTION_SIGNAL_PHRASES = [
    "concern", "concerns", "worried", "worry", "problem", "issue",
    "not sure", "unsure", "hesitant", "hesitation", "i need to think",
    "need to think", "think about it", "not ready", "not ready to decide",
    "not comfortable", "uncomfortable", "doesn't work", "does not work",
    "won't work", "cannot work", "can't work", "not convinced",
    "hard to justify", "difficult to justify",
    "too expensive", "expensive", "too high", "higher than", "more than",
    "more expensive", "lower than", "too much", "costly", "price is",
    "price feels", "pricing feels", "pricing is", "price per", "cost per",
    "budget", "over budget", "above budget", "within budget",
    "budget constraint", "budget constraints", "money is tight",
    "can't afford", "cannot afford",
    "comparing", "compare", "comparison", "competitor", "competitors",
    "another provider", "another company", "another vendor",
    "another dealership", "another option", "other provider",
    "other company", "other vendor", "quoted lower", "quoted higher",
    "better price", "lower price",
    "timeline", "time to implement", "implementation time",
    "implementation timeline", "onboarding time", "onboarding timeline",
    "too long", "takes too long", "take too long", "before next quarter",
    "by next quarter", "limited time", "not enough time",
    "don't have time", "do not have time", "time to evaluate",
    "not the one who signs", "not the decision maker",
    "not the decision-maker", "need approval", "needs approval",
    "approval process", "decision maker", "decision-maker", "sign off",
    "sign-off", "procurement", "director", "cfo", "need to involve",
    "need to loop in",
    "not a gap", "feature gap", "missing feature", "deprecated",
    "dropped usage", "not useful", "doesn't meet", "does not meet",
    "not meeting", "doesn't support", "does not support", "not available",
]


def contains_objection_signal(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(normalize_text(p) in normalized for p in OBJECTION_SIGNAL_PHRASES)


def has_comparison_signal(text: str) -> bool:
    normalized = normalize_text(text)
    comparison_phrases = [
        "comparing", "compare", "comparison", "higher than", "lower than",
        "more than", "less than", "quoted lower", "quoted higher",
        "better price", "another provider", "another company",
        "another dealership", "competitor", "competitors",
    ]
    return any(normalize_text(p) in normalized for p in comparison_phrases)


def has_concern_signal(text: str) -> bool:
    normalized = normalize_text(text)
    concern_phrases = [
        "concern", "worried", "worry", "problem", "issue", "not sure",
        "unsure", "hesitant", "hesitation", "need to think",
        "think about it", "not ready", "not comfortable", "too expensive",
        "too high", "too much", "over budget", "above budget",
        "budget constraint", "budget constraints", "money is tight",
        "limited time", "not enough time", "don't have time",
        "do not have time", "takes too long", "take too long", "too long",
        "not the one who signs", "approval process",
        "not the decision maker", "not the decision-maker",
    ]
    return any(normalize_text(p) in normalized for p in concern_phrases)


def is_objection_candidate(moment: Dict[str, Any]) -> bool:
    """
    Controlled objection candidate detection.
    UPDATED: checks membership against a SET of categories instead of
    a single category string, since Agent 1 now returns "categories"
    as a list.
    """
    raw_categories = moment.get("categories", [])
    categories_lower = {str(c).strip().lower() for c in raw_categories}
    evidence = str(moment.get("evidence", "")).strip()

    if not evidence:
        return False

    # Explicit objection or hesitation
    if categories_lower & {"objection", "objections"}:
        return True
    if "hesitation" in categories_lower:
        return True

    # Pain point
    if "pain_point" in categories_lower:
        status = str(moment.get("status", "")).lower()
        if status == "unresolved":
            return True
        return contains_objection_signal(evidence)

    # Pricing
    if categories_lower & {"pricing", "price"}:
        return (
            contains_objection_signal(evidence)
            or has_concern_signal(evidence)
            or has_comparison_signal(evidence)
        )

    # Competitor
    if categories_lower & {"competitor", "competitors"}:
        return has_comparison_signal(evidence) or contains_objection_signal(evidence)

    # Budget
    if "budget" in categories_lower:
        status = str(moment.get("status", "")).lower()
        if status == "unresolved":
            return True
        return (
            "budget" in normalize_text(evidence)
            or "money" in normalize_text(evidence)
            or "afford" in normalize_text(evidence)
        ) and has_concern_signal(evidence)

    # Authority
    if "authority" in categories_lower:
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

    # Timeline
    if "timeline" in categories_lower:
        return has_concern_signal(evidence) or contains_objection_signal(evidence)

    return False


# ============================================================
# GET CATEGORY PREDICTIONS
# ============================================================

def get_predictions_for_category(
    moments: List[Dict[str, Any]],
    category: str,
) -> List[Dict[str, Any]]:
    """
    UPDATED: checks whether the canonical category is present ANYWHERE
    in the moment's categories list, not equal to a single string.
    """
    results = []
    for moment in moments:
        raw_categories = moment.get("categories", [])
        normalized_categories = {normalize_category(c) for c in raw_categories}

        if category == "objections":
            if is_objection_candidate(moment):
                results.append(moment)
        elif category in normalized_categories:
            results.append(moment)

    return results


# ============================================================
# OBJECTION MATCHING
# ============================================================

def objection_concept_score(ground_truth: str, prediction: str) -> float:
    gt = normalize_text(ground_truth)
    pred = normalize_text(prediction)
    if not gt or not pred:
        return 0.0

    sequence_score = lexical_similarity(gt, pred)
    overlap_score = token_overlap(gt, pred)
    gt_tokens = token_set(gt)
    pred_tokens = token_set(pred)
    concept_score = 0.0

    price_terms = {"price", "pricing", "cost", "expensive", "higher", "lower",
                   "budget", "money", "premium", "compensation", "salary", "quoted"}
    if gt_tokens & price_terms and pred_tokens & price_terms:
        concept_score = max(concept_score, 0.75)

    timeline_terms = {"timeline", "time", "timing", "weeks", "week", "month",
                       "months", "quarter", "onboarding", "implementation",
                       "evaluate", "evaluation"}
    if gt_tokens & timeline_terms and pred_tokens & timeline_terms:
        concept_score = max(concept_score, 0.75)

    authority_terms = {"authority", "approval", "approve", "procurement",
                        "director", "cfo", "sign", "signoff", "decision", "stakeholder"}
    if gt_tokens & authority_terms and pred_tokens & authority_terms:
        concept_score = max(concept_score, 0.75)

    hesitation_terms = {"hesitation", "hesitant", "think", "decide", "decision",
                         "ready", "uncertain", "unsure", "not"}
    if gt_tokens & hesitation_terms and pred_tokens & hesitation_terms:
        concept_score = max(concept_score, 0.75)

    comparison_terms = {"competitor", "competitors", "comparison", "comparing",
                         "compare", "provider", "company", "vendor", "dealership",
                         "salesforce", "hubspot", "looker", "icici"}
    if gt_tokens & comparison_terms and pred_tokens & comparison_terms:
        concept_score = max(concept_score, 0.75)

    pain_terms = {"problem", "issue", "pain", "feature", "deprecated", "usage",
                  "gap", "missing", "slow", "reporting", "export"}
    if gt_tokens & pain_terms and pred_tokens & pain_terms:
        concept_score = max(concept_score, 0.75)

    combined = max(sequence_score, overlap_score, concept_score)

    if concept_score >= 0.75 and (sequence_score >= 0.20 or overlap_score >= 0.15):
        combined = min(1.0, combined + 0.10)

    return combined


def match_objections(
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
            pred_text = prediction.get("evidence", "")
            score = objection_concept_score(gt_text, pred_text)
            candidate_pairs.append((score, gt_index, pred_index))

    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_predictions = set()
    matches = []
    MATCH_THRESHOLD = 0.50

    for score, gt_index, pred_index in candidate_pairs:
        if score < MATCH_THRESHOLD:
            continue
        if gt_index in matched_gt or pred_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)
        matches.append({
            "ground_truth_index": gt_index,
            "prediction_index": pred_index,
            "ground_truth": ground_truth[gt_index],
            "prediction": predictions[pred_index].get("evidence", ""),
            "prediction_categories": predictions[pred_index].get("categories", []),
            "score": round(score, 4),
        })

    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives

    return matches, true_positives, false_positives, false_negatives


# ============================================================
# PRICING MATCHING
# ============================================================

def pricing_match_score(ground_truth: str, prediction: str) -> float:
    number_score = numeric_overlap(ground_truth, prediction)
    lexical_score = lexical_similarity(ground_truth, prediction)
    overlap_score = token_overlap(ground_truth, prediction)
    gt_numbers = extract_numbers(ground_truth)
    pred_numbers = extract_numbers(prediction)

    # If the ground truth names a concrete amount, require the same amount.
    # This prevents a vague question such as "any discount codes?" from
    # matching a specific price such as "$64".
    if gt_numbers:
        if number_score > 0:
            return max(number_score, lexical_score, overlap_score)
        return 0.0

    return max(lexical_score, overlap_score)


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
            pred_text = prediction.get("evidence", "")
            score = pricing_match_score(gt_text, pred_text)
            candidate_pairs.append((score, gt_index, pred_index))

    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_predictions = set()
    matches = []
    MATCH_THRESHOLD = 0.50

    for score, gt_index, pred_index in candidate_pairs:
        if score < MATCH_THRESHOLD:
            continue
        if gt_index in matched_gt or pred_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)
        matches.append({
            "ground_truth_index": gt_index,
            "prediction_index": pred_index,
            "ground_truth": ground_truth[gt_index],
            "prediction": predictions[pred_index].get("evidence", ""),
            "score": round(score, 4),
        })

    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives

    return matches, true_positives, false_positives, false_negatives


# ============================================================
# COMPETITOR MATCHING
# ============================================================

def competitor_match_score(ground_truth: str, prediction: str) -> float:
    gt = normalize_text(ground_truth)
    pred = normalize_text(prediction)
    if not gt or not pred:
        return 0.0

    if gt in pred or pred in gt:
        return 1.0

    gt_tokens = token_set(gt)
    pred_tokens = token_set(pred)
    if gt_tokens and gt_tokens.issubset(pred_tokens):
        return 1.0

    # Possessive/plural normalization, e.g. "another agent's listing"
    # vs "another agent listing".
    gt_stem = re.sub(r"\b(agents|agent|providers|provider|companies|company|vendors|vendor)s?\b", "competitor", gt)
    pred_stem = re.sub(r"\b(agents|agent|providers|provider|companies|company|vendors|vendor)s?\b", "competitor", pred)
    if token_overlap(gt_stem, pred_stem) >= 0.45:
        return 0.75

    overlap = token_overlap(gt, pred)
    lexical = lexical_similarity(gt, pred)
    return max(overlap, lexical)


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
            score = competitor_match_score(gt_text, prediction.get("evidence", ""))
            candidate_pairs.append((score, gt_index, pred_index))

    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_predictions = set()
    matches = []
    MATCH_THRESHOLD = 0.50

    for score, gt_index, pred_index in candidate_pairs:
        if score < MATCH_THRESHOLD:
            continue
        if gt_index in matched_gt or pred_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)
        matches.append({
            "ground_truth_index": gt_index,
            "prediction_index": pred_index,
            "ground_truth": ground_truth[gt_index],
            "prediction": predictions[pred_index].get("evidence", ""),
            "score": round(score, 4),
        })

    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives

    return matches, true_positives, false_positives, false_negatives


# ============================================================
# NEXT-STEP MATCHING
# ============================================================

ACTION_WORDS = {
    "send", "share", "call", "follow", "followup", "follow-up", "schedule",
    "arrange", "provide", "submit", "update", "email", "demo", "meet",
    "meeting", "review", "try", "identify", "sign", "hold", "upgrade",
    "checkout", "check", "contact",
}

TEMPORAL_WORDS = {
    "today", "tomorrow", "tonight", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "next", "week", "weeks",
    "month", "months", "quarter", "hour", "hours", "day", "days",
}


def action_overlap(text_a: str, text_b: str) -> float:
    a_actions = token_set(text_a) & ACTION_WORDS
    b_actions = token_set(text_b) & ACTION_WORDS
    if not a_actions or not b_actions:
        return 0.0
    return len(a_actions & b_actions) / len(a_actions | b_actions)


def temporal_overlap(text_a: str, text_b: str) -> float:
    a_time = token_set(text_a) & TEMPORAL_WORDS
    b_time = token_set(text_b) & TEMPORAL_WORDS
    if not a_time or not b_time:
        return 0.0
    return len(a_time & b_time) / len(a_time | b_time)


def next_step_action_group(text: str) -> set:
    """Map different phrasings to normalized next-step intents."""
    tokens = set(tokenize(text))
    normalized = normalize_text(text)

    groups = {
        "send_material": {"send", "forward", "share", "provide", "email", "mail", "deliver"},
        "follow_up": {"follow", "followup", "call", "callback", "checkin", "check", "touch", "reconnect", "contact"},
        "schedule": {"schedule", "scheduled", "arrange", "book", "setup", "set", "meeting", "session", "calendar"},
        "purchase": {"buy", "purchase", "order", "checkout", "proceed", "signup", "sign", "upgrade"},
        "submit_offer": {"submit", "submission", "offer", "proposal", "application", "bid"},
        "internal_review": {"review", "evaluate", "assess", "consider", "numbers", "internally", "stakeholders", "team"},
        "discuss": {"discuss", "discussion", "talk", "speak", "meet"},
        "confirm": {"confirm", "confirmation", "approve", "approval", "signoff", "signature"},
        "contract": {"contract", "agreement", "sign"},
        "implementation": {"implement", "implementation", "activate", "activation", "launch", "start", "deploy", "onboard"},
        "hold_quote": {"hold", "reserve", "lock"},
        "identify_stakeholder": {"identify", "stakeholder", "decision", "director", "cfo", "procurement"},
    }

    matched = {name for name, words in groups.items() if tokens.intersection(words)}

    # Strong phrase-level intent signals.
    phrase_groups = {
        "internal_review": ["go through the numbers", "review internally", "review the proposal", "review with the team", "discuss internally", "take a look internally"],
        "follow_up": ["get back to you", "get back to me", "touch base", "check in", "follow up", "call back", "call again"],
        "send_material": ["send it over", "send across", "send the", "share the", "forward the", "provide the"],
        "schedule": ["set up", "set a meeting", "calendar invite", "follow-up session", "book a"],
        "purchase": ["place the order", "proceed with", "move forward", "go ahead", "proceed", "checkout"],
        "submit_offer": ["submit the offer", "send the offer", "send an offer", "submit the proposal", "place the bid"],
        "contract": ["send the contract", "get the contract", "sign the contract", "contract over"],
        "hold_quote": ["hold the quote", "keep the quote", "lock the quote"],
        "implementation": ["go live", "start implementation", "activate", "activation", "launch"],
        "identify_stakeholder": ["identify the stakeholder", "find the decision maker", "loop in", "get approval", "bring in the"],
    }
    for group, phrases in phrase_groups.items():
        if any(normalize_text(phrase) in normalized for phrase in phrases):
            matched.add(group)

    return matched


def contains_acceptance_signal(text: str) -> bool:
    normalized = normalize_text(text)
    phrases = [
        "that works", "works for me", "works well", "sounds good", "sounds great",
        "yes lets do that", "yes lets", "lets do that", "that is fine", "thats fine",
        "perfect", "that will work", "that should work", "okay lets do it",
        "okay lets", "sure lets do it", "sure why not", "why not", "go ahead",
        "i am fine with that", "im fine with that", "fine with that",
    ]
    return any(normalize_text(p) in normalized for p in phrases)


def contains_future_reference(text: str) -> bool:
    n = normalize_text(text)
    return bool(set(tokenize(n)) & TEMPORAL_WORDS)


def _split_next_step_evidence(evidence: str) -> List[str]:
    """Split one extraction moment containing multiple concrete actions."""
    text = str(evidence or "").strip()
    if not text:
        return []
    # Preserve short acceptance statements as a single item.
    if contains_acceptance_signal(text) and len(text.split()) <= 10:
        return [text]
    parts = re.split(r"\s+(?:and then|and|;|, then)\s+", text, flags=re.IGNORECASE)
    parts = [p.strip(" .") for p in parts if p.strip(" .")]
    return parts or [text]


def next_step_match_score(ground_truth: str, prediction: str) -> float:
    """Robust semantic matching for next-step intent."""
    gt = str(ground_truth or "")
    pred = str(prediction or "")
    lexical = lexical_similarity(gt, pred)
    overlap = token_overlap(gt, pred)
    action = action_overlap(gt, pred)
    temporal = temporal_overlap(gt, pred)

    gt_groups = next_step_action_group(gt)
    pred_groups = next_step_action_group(pred)
    group_overlap = gt_groups & pred_groups

    # Exact/near-exact evidence is always strong.
    if lexical >= 0.82 or overlap >= 0.70:
        return max(lexical, overlap)

    # Shared normalized intent + supporting context.
    if group_overlap:
        score = 0.64
        if temporal > 0:
            score += 0.12
        if lexical >= 0.25 or overlap >= 0.20:
            score += 0.08
        # Shared object words make the semantic match safer.
        object_words = {"contract", "proposal", "pricing", "sheet", "case", "studies", "offer", "resume", "calendar", "demo", "quote", "order", "checkout", "numbers", "stakeholder", "implementation", "renewal", "coverage", "application"}
        if token_set(gt) & token_set(pred) & object_words:
            score += 0.08
        return min(1.0, max(score, lexical, overlap))

    # Customer acceptance can correspond to a previously proposed concrete action.
    if contains_acceptance_signal(pred) and gt_groups:
        if temporal > 0 or any(g in gt_groups for g in {"follow_up", "schedule", "purchase", "implementation", "hold_quote", "contract"}):
            return 0.76

    # Similar action + timing.
    if action > 0 and temporal > 0:
        return max(lexical, overlap, min(1.0, 0.52 + action * 0.24 + temporal * 0.24))

    return max(lexical, overlap, action)


def _expand_next_step_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded = []
    for prediction in predictions:
        evidence = prediction.get("evidence", "")
        parts = _split_next_step_evidence(evidence)
        if len(parts) <= 1:
            expanded.append(prediction)
            continue
        for part in parts:
            clone = dict(prediction)
            clone["evidence"] = part
            clone["_expanded_from"] = evidence
            expanded.append(clone)
    return expanded


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

    predictions = _expand_next_step_predictions(predictions)
    candidate_pairs = []
    for gt_index, gt_text in enumerate(ground_truth):
        for pred_index, prediction in enumerate(predictions):
            score = next_step_match_score(gt_text, prediction.get("evidence", ""))
            candidate_pairs.append((score, gt_index, pred_index))

    candidate_pairs.sort(key=lambda x: x[0], reverse=True)
    matched_gt = set()
    matched_predictions = set()
    matches = []
    MATCH_THRESHOLD = 0.55

    for score, gt_index, pred_index in candidate_pairs:
        if score < MATCH_THRESHOLD:
            continue
        if gt_index in matched_gt or pred_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)
        matches.append({
            "ground_truth_index": gt_index,
            "prediction_index": pred_index,
            "ground_truth": ground_truth[gt_index],
            "prediction": predictions[pred_index].get("evidence", ""),
            "score": round(score, 4),
        })

    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    return matches, true_positives, false_positives, false_negatives


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> Dict[str, Any]:
    """
    UPDATED: when there is no ground truth AND nothing was predicted,
    that is a correct empty match (not a failure) — score it as
    perfect (1.0) instead of 0.0, so correctly-empty categories don't
    drag down the aggregate averages.
    """
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    if precision_denominator == 0 and recall_denominator == 0:
        return {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

    precision = (
        true_positives / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

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
        raise FileNotFoundError(f"Ground truth file not found:\n{GROUND_TRUTH_PATH}")
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# TRANSCRIPT LOADING
# ============================================================

def load_transcript(call_id: str) -> str:
    transcript_dir = BASE_DIR / "data" / "transcripts"
    possible_files = [
        transcript_dir / f"{call_id}.txt",
        transcript_dir / f"{call_id}.md",
        transcript_dir / f"{call_id}.csv",
    ]
    for path in possible_files:
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                return file.read()
    raise FileNotFoundError(f"Transcript not found for {call_id} in {transcript_dir}")


# ============================================================
# EVALUATE ONE CALL
# ============================================================

def evaluate_call(
    call_id: str,
    ground_truth: Dict[str, Any],
    moments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    category_results = {}
    categories = ["objections", "pricing_mentions", "competitor_mentions", "next_steps"]

    for category in categories:
        gt_items = ground_truth.get(category, [])
        predictions = get_predictions_for_category(moments, category)

        if category == "objections":
            matches, tp, fp, fn = match_objections(gt_items, predictions)
        elif category == "pricing_mentions":
            matches, tp, fp, fn = match_pricing(gt_items, predictions)
        elif category == "competitor_mentions":
            matches, tp, fp, fn = match_competitors(gt_items, predictions)
        elif category == "next_steps":
            matches, tp, fp, fn = match_next_steps(gt_items, predictions)
        else:
            matches, tp, fp, fn = [], 0, 0, 0

        metrics = calculate_metrics(tp, fp, fn)

        category_results[category] = {
            "ground_truth": gt_items,
            "predictions": [p.get("evidence", "") for p in predictions],
            "prediction_details": predictions,
            "matches": matches,
            **metrics,
        }

    return {"call_id": call_id, "predictions": moments, "categories": category_results}


# ============================================================
# AGGREGATE RESULTS
# ============================================================

def aggregate_results(per_call_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    categories = ["objections", "pricing_mentions", "competitor_mentions", "next_steps"]
    category_totals = {}

    for category in categories:
        total_tp = total_fp = total_fn = 0
        for call_result in per_call_results:
            result = call_result["categories"][category]
            total_tp += result["true_positives"]
            total_fp += result["false_positives"]
            total_fn += result["false_negatives"]
        category_totals[category] = calculate_metrics(total_tp, total_fp, total_fn)

    overall_tp = sum(category_totals[c]["true_positives"] for c in categories)
    overall_fp = sum(category_totals[c]["false_positives"] for c in categories)
    overall_fn = sum(category_totals[c]["false_negatives"] for c in categories)
    overall = calculate_metrics(overall_tp, overall_fp, overall_fn)

    return {
        "number_of_calls": len(per_call_results),
        "categories": categories,
        "matching_method": {
            "objections": (
                "controlled semantic concept matching + lexical similarity "
                "with multi-category objection candidate detection"
            ),
            "pricing_mentions": "numerical + lexical matching",
            "competitor_mentions": "entity/token + lexical matching",
            "next_steps": "semantic action-group + acceptance + action/temporal/lexical matching",
            "one_to_one_matching": True,
        },
        "category_performance": category_totals,
        "overall": overall,
    }


# ============================================================
# SAVE JSON / CSV
# ============================================================

def save_json(summary: Dict[str, Any], per_call_results: List[Dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {"evaluation_summary": summary, "per_call_results": per_call_results}
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)


def save_csv(per_call_results: List[Dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    categories = ["objections", "pricing_mentions", "competitor_mentions", "next_steps"]

    for call_result in per_call_results:
        call_id = call_result["call_id"]
        for category in categories:
            result = call_result["categories"][category]
            rows.append({
                "call_id": call_id,
                "category": category,
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1"],
                "true_positives": result["true_positives"],
                "false_positives": result["false_positives"],
                "false_negatives": result["false_negatives"],
            })

    fieldnames = [
        "call_id", "category", "precision", "recall", "f1_score",
        "true_positives", "false_positives", "false_negatives",
    ]
    with open(CSV_OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(summary: Dict[str, Any], per_call_results: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 75)
    print("              AGENT 1 EXTRACTION EVALUATION")
    print("=" * 75)
    print()
    print("CATEGORY PERFORMANCE")
    print("-" * 75)
    print(f"{'Category':<25}{'Precision':<15}{'Recall':<15}{'F1 Score':<15}")
    print("-" * 75)

    for category in summary["categories"]:
        result = summary["category_performance"][category]
        print(f"{category:<25}{result['precision']:<15.3f}{result['recall']:<15.3f}{result['f1']:<15.3f}")

    print("-" * 75)
    overall = summary["overall"]
    print(f"{'OVERALL':<25}{overall['precision']:<15.3f}{overall['recall']:<15.3f}{overall['f1']:<15.3f}")
    print("=" * 75)

    print()
    print("PER-CALL RESULTS")
    print("-" * 75)

    for call_result in per_call_results:
        print()
        print(call_result["call_id"])
        for category in summary["categories"]:
            result = call_result["categories"][category]
            print(f"  {category:<20}P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1']:.3f}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 75)
    print("Starting Agent 1 extraction evaluation...")
    print("=" * 75)

    ground_truth = load_ground_truth()
    print()
    print(f"Loaded ground truth for {len(ground_truth)} calls.")

    per_call_results = []

    for call_id, call_ground_truth in ground_truth.items():
        print()
        print(f"Processing {call_id}...")

        try:
            transcript = load_transcript(call_id)
            extraction_result = extract_moments(transcript)
            raw_moments = getattr(extraction_result, "moments", [])
            moments = [moment_to_dict(m) for m in raw_moments]
            result = evaluate_call(call_id, call_ground_truth, moments)
            per_call_results.append(result)
        except Exception as error:
            print(f"ERROR processing {call_id}: {error}")
            continue

    summary = aggregate_results(per_call_results)
    save_json(summary, per_call_results)
    save_csv(per_call_results)
    print_results(summary, per_call_results)

    print()
    print("=" * 75)
    print("Detailed evaluation saved to:")
    print(JSON_OUTPUT_PATH)
    print()
    print("CSV summary saved to:")
    print(CSV_OUTPUT_PATH)
    print("=" * 75)


if __name__ == "__main__":
    main()
