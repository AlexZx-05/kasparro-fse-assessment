"""Lossless error categorization for evaluation reports."""

from __future__ import annotations

from collections import Counter

from app.evaluation.scorer import ScoringResult


def analyse_errors(scored: ScoringResult) -> dict[str, object]:
    """Return report-ready error records without modifying any predictions."""
    missing = [_gold_record(claim) for claim in scored.missing_gold]
    extra = [_prediction_record(prediction) for prediction in scored.extra_predictions]
    wrong_labels = [
        {
            "claim_id": match.gold.claim_id,
            "answer_id": match.gold.answer_id,
            "start_char": match.gold.start_char,
            "end_char": match.gold.end_char,
            "quote": match.gold.quote,
            "expected_label": match.gold.label,
            "predicted_label": match.prediction.label,
            "evidence_fact_ids": list(match.prediction.fact_ids),
            "difficulty_tags": list(match.gold.difficulty_tags),
        }
        for match in scored.wrong_label_matches
    ]
    failures = [
        {
            "answer_id": outcome.answer_id,
            "error_type": outcome.error_type,
            "error_message": outcome.error_message,
            "duration_ms": outcome.duration_ms,
        }
        for outcome in scored.error_outcomes
    ]

    recurring = Counter[str]()
    recurring["missing_claim"] += len(missing)
    recurring["extra_claim"] += len(extra)
    recurring["wrong_label"] += len(wrong_labels)
    recurring["answer_error"] += len(failures)
    for error in wrong_labels:
        recurring[
            f"wrong_label:{error['expected_label']}->{error['predicted_label']}"
        ] += 1
    for claim in missing:
        for tag in claim["difficulty_tags"]:
            recurring[f"missing_claim_tag:{tag}"] += 1

    patterns = [
        {"category": category, "count": count}
        for category, count in sorted(recurring.items(), key=lambda item: (-item[1], item[0]))
        if count
    ]
    return {
        "missing_claims": missing,
        "extra_claims": extra,
        "wrong_label_claims": wrong_labels,
        "answer_level_failures": failures,
        "recurring_patterns": patterns,
        "top_measured_pattern": patterns[0] if patterns else None,
    }


def _gold_record(claim: object) -> dict[str, object]:
    return {
        "claim_id": getattr(claim, "claim_id"),
        "answer_id": getattr(claim, "answer_id"),
        "start_char": getattr(claim, "start_char"),
        "end_char": getattr(claim, "end_char"),
        "quote": getattr(claim, "quote"),
        "expected_label": getattr(claim, "label"),
        "evidence_fact_ids": list(getattr(claim, "evidence_fact_ids")),
        "difficulty_tags": list(getattr(claim, "difficulty_tags")),
    }


def _prediction_record(prediction: object) -> dict[str, object]:
    return {
        "claim_id": getattr(prediction, "claim_id"),
        "answer_id": getattr(prediction, "answer_id"),
        "start_char": getattr(prediction, "start_char"),
        "end_char": getattr(prediction, "end_char"),
        "quote": getattr(prediction, "quote"),
        "predicted_label": getattr(prediction, "label"),
        "evidence_fact_ids": list(getattr(prediction, "fact_ids")),
    }
