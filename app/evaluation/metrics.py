"""Deterministic metric calculations for exact-span evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas import ClaimLabel
from app.evaluation.scorer import ScoringResult


LABELS: tuple[ClaimLabel, ...] = ("supported", "contradicted", "unknown")
MISSING_OR_ERROR = "missing_or_error"
NO_GOLD = "no_gold"


@dataclass(frozen=True, slots=True)
class PerLabelMetrics:
    """One-vs-rest exact-span metrics for a classification label."""

    label: ClaimLabel
    true_positives: int
    predicted_positives: int
    actual_positives: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, object]:
        return {
            "true_positives": self.true_positives,
            "predicted_positives": self.predicted_positives,
            "actual_positives": self.actual_positives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """All Phase 4 metrics, counts, and the required confusion matrix."""

    gold_claim_count: int
    prediction_count: int
    exact_match_count: int
    correct_joint_count: int
    extraction_precision: float
    extraction_recall: float
    extraction_f1: float
    joint_precision: float
    joint_recall: float
    joint_f1: float
    macro_f1: float
    per_label: dict[ClaimLabel, PerLabelMetrics]
    coverage_answer_count: int
    attempted_answer_count: int
    error_answer_count: int
    coverage_rate: float
    missing_claim_count: int
    extra_claim_count: int
    wrong_label_claim_count: int
    confusion_matrix: dict[str, dict[str, int]]

    def as_dict(self) -> dict[str, object]:
        return {
            "counts": {
                "gold_claims": self.gold_claim_count,
                "accepted_predictions": self.prediction_count,
                "exact_span_matches": self.exact_match_count,
                "joint_correct_matches": self.correct_joint_count,
                "missing_claims": self.missing_claim_count,
                "extra_claims": self.extra_claim_count,
                "wrong_label_claims": self.wrong_label_claim_count,
            },
            "extraction": {
                "precision": self.extraction_precision,
                "recall": self.extraction_recall,
                "f1": self.extraction_f1,
            },
            "joint": {
                "precision": self.joint_precision,
                "recall": self.joint_recall,
                "f1": self.joint_f1,
            },
            "macro_f1": self.macro_f1,
            "per_label": {
                label: metric.as_dict() for label, metric in self.per_label.items()
            },
            "coverage": {
                "answers_attempted": self.attempted_answer_count,
                "answers_successfully_processed": self.coverage_answer_count,
                "answer_level_errors": self.error_answer_count,
                "rate": self.coverage_rate,
            },
            "confusion_matrix": self.confusion_matrix,
            "zero_division_policy": "Metrics with a zero denominator are 0.0.",
        }


def calculate_metrics(scored: ScoringResult) -> EvaluationMetrics:
    """Calculate metrics without rounding intermediate values."""
    gold_count = len(scored.gold_claims)
    prediction_count = len(scored.predictions)
    exact_match_count = len(scored.matches)
    correct_joint_count = sum(match.label_correct for match in scored.matches)

    extraction_precision = _ratio(exact_match_count, prediction_count)
    extraction_recall = _ratio(exact_match_count, gold_count)
    joint_precision = _ratio(correct_joint_count, prediction_count)
    joint_recall = _ratio(correct_joint_count, gold_count)

    per_label: dict[ClaimLabel, PerLabelMetrics] = {}
    for label in LABELS:
        true_positives = sum(
            match.label_correct and match.gold.label == label for match in scored.matches
        )
        predicted_positives = sum(
            prediction.label == label for prediction in scored.predictions
        )
        actual_positives = sum(gold.label == label for gold in scored.gold_claims)
        precision = _ratio(true_positives, predicted_positives)
        recall = _ratio(true_positives, actual_positives)
        per_label[label] = PerLabelMetrics(
            label=label,
            true_positives=true_positives,
            predicted_positives=predicted_positives,
            actual_positives=actual_positives,
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
        )

    attempted = len(scored.scheduled_answer_ids)
    successful = sum(outcome.status == "success" for outcome in scored.outcomes)
    matrix = _confusion_matrix(scored)
    return EvaluationMetrics(
        gold_claim_count=gold_count,
        prediction_count=prediction_count,
        exact_match_count=exact_match_count,
        correct_joint_count=correct_joint_count,
        extraction_precision=extraction_precision,
        extraction_recall=extraction_recall,
        extraction_f1=_f1(extraction_precision, extraction_recall),
        joint_precision=joint_precision,
        joint_recall=joint_recall,
        joint_f1=_f1(joint_precision, joint_recall),
        macro_f1=sum(metric.f1 for metric in per_label.values()) / len(LABELS),
        per_label=per_label,
        coverage_answer_count=successful,
        attempted_answer_count=attempted,
        error_answer_count=attempted - successful,
        coverage_rate=_ratio(successful, attempted),
        missing_claim_count=len(scored.missing_gold),
        extra_claim_count=len(scored.extra_predictions),
        wrong_label_claim_count=len(scored.wrong_label_matches),
        confusion_matrix=matrix,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _confusion_matrix(scored: ScoringResult) -> dict[str, dict[str, int]]:
    columns = (*LABELS, MISSING_OR_ERROR)
    matrix: dict[str, dict[str, int]] = {
        row: {column: 0 for column in columns} for row in (*LABELS, NO_GOLD)
    }
    matched_by_gold_id = {match.gold.claim_id: match for match in scored.matches}
    for gold in scored.gold_claims:
        match = matched_by_gold_id.get(gold.claim_id)
        if match is None:
            matrix[gold.label][MISSING_OR_ERROR] += 1
        else:
            # scorer validation ensures this is one of the three labels.
            matrix[gold.label][str(match.prediction.label)] += 1
    for prediction in scored.extra_predictions:
        matrix[NO_GOLD][str(prediction.label)] += 1
    return matrix
