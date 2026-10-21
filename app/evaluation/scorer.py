"""Exact-span validation and matching for development evaluation.

This module is the only Phase 4 component that turns development-gold rows
into evaluation objects.  Inference modules must never import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from app.data_loader import (
    DEFAULT_DATA_DIR,
    DatasetError,
    VALID_LABELS,
    load_development_gold,
)
from app.schemas import Answer, ClassifiedClaim, ClaimLabel


OutcomeStatus = Literal["success", "error"]
_NON_NEGATIVE_INTEGER = re.compile(r"^(?:0|[1-9][0-9]*)$")


class GoldValidationError(ValueError):
    """Raised when development gold cannot safely be used for scoring."""


class PredictionValidationError(ValueError):
    """Raised when a purported accepted prediction violates the agent contract."""


@dataclass(frozen=True, slots=True)
class GoldClaim:
    """A validated development-gold claim with parsed source positions."""

    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: ClaimLabel
    evidence_fact_ids: tuple[str, ...]
    explanation: str
    difficulty_tags: tuple[str, ...]

    @property
    def identity(self) -> tuple[str, int, int]:
        return (self.answer_id, self.start_char, self.end_char)


@dataclass(frozen=True, slots=True)
class Prediction:
    """A saved, accepted claim prediction used as scorer input."""

    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: str | None
    fact_ids: tuple[str, ...] = ()

    @property
    def identity(self) -> tuple[str, int, int]:
        return (self.answer_id, self.start_char, self.end_char)

    @classmethod
    def from_classified_claim(cls, claim: ClassifiedClaim) -> "Prediction":
        return cls(
            answer_id=claim.answer_id,
            claim_id=claim.claim_id,
            start_char=claim.start_char,
            end_char=claim.end_char,
            quote=claim.claim_text,
            label=claim.label,
            fact_ids=claim.fact_ids,
        )


@dataclass(frozen=True, slots=True)
class AnswerOutcome:
    """One terminal inference result for one scheduled answer."""

    answer_id: str
    status: OutcomeStatus
    predictions: tuple[Prediction, ...]
    duration_ms: float
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class GoldValidationSummary:
    """Dataset validation details written into evaluation reports."""

    development_answer_count: int
    gold_claim_count: int
    missing_value_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "development_answer_count": self.development_answer_count,
            "gold_claim_count": self.gold_claim_count,
            "missing_value_counts": dict(sorted(self.missing_value_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class MatchedClaim:
    """One exact extraction match, independently of label correctness."""

    gold: GoldClaim
    prediction: Prediction

    @property
    def label_correct(self) -> bool:
        return self.gold.label == self.prediction.label


@dataclass(frozen=True, slots=True)
class ScoringResult:
    """Complete exact-span matching result used by metrics and error analysis."""

    scheduled_answer_ids: tuple[str, ...]
    outcomes: tuple[AnswerOutcome, ...]
    gold_claims: tuple[GoldClaim, ...]
    predictions: tuple[Prediction, ...]
    matches: tuple[MatchedClaim, ...]
    missing_gold: tuple[GoldClaim, ...]
    extra_predictions: tuple[Prediction, ...]
    wrong_label_matches: tuple[MatchedClaim, ...]

    @property
    def error_outcomes(self) -> tuple[AnswerOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.status == "error")


def validate_development_gold(
    answers: tuple[Answer, ...] | list[Answer],
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[tuple[GoldClaim, ...], GoldValidationSummary]:
    """Load and fully validate gold against the unmodified development answers.

    Validation deliberately rejects errors rather than repairing a row.  It is
    run by the evaluation harness only and never by the inference workflow.
    """
    development_answers = tuple(
        sorted((answer for answer in answers if answer.split == "development"), key=lambda a: a.answer_id)
    )
    answers_by_id = {answer.answer_id: answer for answer in development_answers}
    if len(answers_by_id) != len(development_answers):
        raise GoldValidationError("Development answers contain duplicate answer IDs")

    try:
        raw_claims = load_development_gold(data_dir)
    except DatasetError as error:
        raise GoldValidationError(str(error)) from error

    missing_value_counts = {
        field: sum(not getattr(claim, field) for claim in raw_claims)
        for field in (
            "answer_id",
            "claim_id",
            "start_char",
            "end_char",
            "quote",
            "expected_label",
            "evidence_fact_ids",
            "explanation",
            "difficulty_tags",
        )
    }
    errors: list[str] = []
    seen_claim_ids: set[str] = set()
    seen_spans: set[tuple[str, int, int]] = set()
    parsed_claims: list[GoldClaim] = []

    for raw_claim in raw_claims:
        if raw_claim.answer_id not in answers_by_id:
            errors.append(
                f"Gold claim {raw_claim.claim_id!r} references non-development "
                f"answer {raw_claim.answer_id!r}"
            )
            continue
        if raw_claim.claim_id in seen_claim_ids:
            errors.append(f"Duplicate gold claim_id {raw_claim.claim_id!r}")
            continue
        seen_claim_ids.add(raw_claim.claim_id)
        try:
            start_char = _parse_offset(raw_claim.start_char, "start_char", raw_claim.claim_id)
            end_char = _parse_offset(raw_claim.end_char, "end_char", raw_claim.claim_id)
        except GoldValidationError as error:
            errors.append(str(error))
            continue

        answer = answers_by_id[raw_claim.answer_id]
        canonical_claim_id = f"{raw_claim.answer_id}:{start_char}:{end_char}"
        if raw_claim.claim_id != canonical_claim_id:
            errors.append(
                f"Gold claim {raw_claim.claim_id!r} must use canonical ID "
                f"{canonical_claim_id!r}"
            )
        if start_char >= end_char or end_char > len(answer.ai_answer):
            errors.append(
                f"Gold claim {raw_claim.claim_id!r} has invalid character span"
            )
            continue
        if raw_claim.quote != answer.ai_answer[start_char:end_char]:
            errors.append(
                f"Gold claim {raw_claim.claim_id!r} quote does not match answers.csv"
            )
        if raw_claim.quote != raw_claim.quote.strip():
            errors.append(
                f"Gold claim {raw_claim.claim_id!r} quote has leading or trailing whitespace"
            )
        if raw_claim.expected_label not in VALID_LABELS:
            errors.append(
                f"Gold claim {raw_claim.claim_id!r} has invalid label "
                f"{raw_claim.expected_label!r}"
            )
            continue
        identity = (raw_claim.answer_id, start_char, end_char)
        if identity in seen_spans:
            errors.append(
                f"Duplicate or conflicting gold span {raw_claim.answer_id}:{start_char}:{end_char}"
            )
            continue
        seen_spans.add(identity)
        parsed_claims.append(
            GoldClaim(
                answer_id=raw_claim.answer_id,
                claim_id=raw_claim.claim_id,
                start_char=start_char,
                end_char=end_char,
                quote=raw_claim.quote,
                label=raw_claim.expected_label,
                evidence_fact_ids=tuple(
                    fact_id for fact_id in raw_claim.evidence_fact_ids.split("|") if fact_id
                ),
                explanation=raw_claim.explanation,
                difficulty_tags=tuple(
                    tag for tag in raw_claim.difficulty_tags.split("|") if tag
                ),
            )
        )

    represented_answer_ids = {claim.answer_id for claim in parsed_claims}
    missing_answers = sorted(set(answers_by_id) - represented_answer_ids)
    if missing_answers:
        errors.append(
            "Development answers missing gold claims: " + ", ".join(missing_answers)
        )
    errors.extend(_overlap_errors(parsed_claims))
    if errors:
        raise GoldValidationError("; ".join(errors))

    return (
        tuple(sorted(parsed_claims, key=lambda claim: claim.identity)),
        GoldValidationSummary(
            development_answer_count=len(development_answers),
            gold_claim_count=len(parsed_claims),
            missing_value_counts=missing_value_counts,
        ),
    )


def score_exact_spans(
    answers: tuple[Answer, ...] | list[Answer],
    gold_claims: tuple[GoldClaim, ...] | list[GoldClaim],
    outcomes: tuple[AnswerOutcome, ...] | list[AnswerOutcome],
) -> ScoringResult:
    """Score accepted predictions by exact answer-ID and character-span identity."""
    scheduled_answers = tuple(sorted(answers, key=lambda answer: answer.answer_id))
    answers_by_id = {answer.answer_id: answer for answer in scheduled_answers}
    if len(answers_by_id) != len(scheduled_answers):
        raise PredictionValidationError("Scheduled answers contain duplicate answer IDs")
    ordered_outcomes = _validate_outcomes(outcomes, answers_by_id)
    ordered_gold = tuple(sorted(gold_claims, key=lambda claim: claim.identity))
    _validate_gold_for_scoring(ordered_gold, answers_by_id)

    accepted_predictions: list[Prediction] = []
    for outcome in ordered_outcomes:
        if outcome.status == "error":
            if outcome.predictions:
                raise PredictionValidationError(
                    f"Error outcome {outcome.answer_id!r} cannot contain accepted predictions"
                )
            continue
        _validate_predictions_for_answer(
            outcome.answer_id,
            outcome.predictions,
            answers_by_id[outcome.answer_id].ai_answer,
        )
        accepted_predictions.extend(outcome.predictions)

    predictions = tuple(sorted(accepted_predictions, key=lambda prediction: prediction.identity))
    gold_by_identity = {claim.identity: claim for claim in ordered_gold}
    predictions_by_identity = {prediction.identity: prediction for prediction in predictions}
    matches = tuple(
        MatchedClaim(gold=gold, prediction=predictions_by_identity[gold.identity])
        for gold in ordered_gold
        if gold.identity in predictions_by_identity
    )
    missing_gold = tuple(
        gold for gold in ordered_gold if gold.identity not in predictions_by_identity
    )
    extra_predictions = tuple(
        prediction
        for prediction in predictions
        if prediction.identity not in gold_by_identity
    )
    wrong_labels = tuple(match for match in matches if not match.label_correct)
    return ScoringResult(
        scheduled_answer_ids=tuple(answer.answer_id for answer in scheduled_answers),
        outcomes=ordered_outcomes,
        gold_claims=ordered_gold,
        predictions=predictions,
        matches=matches,
        missing_gold=missing_gold,
        extra_predictions=extra_predictions,
        wrong_label_matches=wrong_labels,
    )


def _parse_offset(value: str, field: str, claim_id: str) -> int:
    if not _NON_NEGATIVE_INTEGER.fullmatch(value):
        raise GoldValidationError(
            f"Gold claim {claim_id!r} has invalid {field} value {value!r}"
        )
    return int(value)


def _overlap_errors(claims: list[GoldClaim]) -> list[str]:
    errors: list[str] = []
    by_answer: dict[str, list[GoldClaim]] = {}
    for claim in claims:
        by_answer.setdefault(claim.answer_id, []).append(claim)
    for answer_id, answer_claims in by_answer.items():
        previous: GoldClaim | None = None
        for claim in sorted(answer_claims, key=lambda item: (item.start_char, item.end_char)):
            if previous is not None and claim.start_char < previous.end_char:
                errors.append(
                    f"Gold claims {previous.claim_id!r} and {claim.claim_id!r} overlap "
                    f"in answer {answer_id!r}"
                )
            if previous is None or claim.end_char > previous.end_char:
                previous = claim
    return errors


def _validate_outcomes(
    outcomes: tuple[AnswerOutcome, ...] | list[AnswerOutcome],
    answers_by_id: dict[str, Answer],
) -> tuple[AnswerOutcome, ...]:
    by_answer_id: dict[str, AnswerOutcome] = {}
    for outcome in outcomes:
        if outcome.answer_id not in answers_by_id:
            raise PredictionValidationError(
                f"Outcome references unscheduled answer {outcome.answer_id!r}"
            )
        if outcome.answer_id in by_answer_id:
            raise PredictionValidationError(
                f"Duplicate answer outcome {outcome.answer_id!r}"
            )
        if outcome.status not in {"success", "error"}:
            raise PredictionValidationError(
                f"Outcome {outcome.answer_id!r} has invalid status {outcome.status!r}"
            )
        by_answer_id[outcome.answer_id] = outcome
    missing = sorted(set(answers_by_id) - set(by_answer_id))
    if missing:
        raise PredictionValidationError("Missing answer outcomes: " + ", ".join(missing))
    return tuple(by_answer_id[answer_id] for answer_id in sorted(answers_by_id))


def _validate_gold_for_scoring(
    gold_claims: tuple[GoldClaim, ...], answers_by_id: dict[str, Answer]
) -> None:
    seen: set[tuple[str, int, int]] = set()
    for gold in gold_claims:
        if gold.answer_id not in answers_by_id:
            raise GoldValidationError(
                f"Gold claim {gold.claim_id!r} references an unscheduled answer"
            )
        if gold.identity in seen:
            raise GoldValidationError(f"Duplicate gold span {gold.claim_id!r}")
        seen.add(gold.identity)


def _validate_predictions_for_answer(
    answer_id: str,
    predictions: tuple[Prediction, ...],
    answer_text: str,
) -> None:
    seen_ids: set[str] = set()
    seen_spans: set[tuple[int, int]] = set()
    ordered = sorted(predictions, key=lambda prediction: (prediction.start_char, prediction.end_char))
    previous: Prediction | None = None
    for prediction in ordered:
        if prediction.answer_id != answer_id:
            raise PredictionValidationError(
                f"Prediction {prediction.claim_id!r} has the wrong answer_id"
            )
        if prediction.claim_id in seen_ids:
            raise PredictionValidationError(
                f"Duplicate prediction claim_id {prediction.claim_id!r}"
            )
        seen_ids.add(prediction.claim_id)
        if prediction.label not in VALID_LABELS:
            raise PredictionValidationError(
                f"Prediction {prediction.claim_id!r} has missing or invalid label"
            )
        canonical_claim_id = (
            f"{prediction.answer_id}:{prediction.start_char}:{prediction.end_char}"
        )
        if prediction.claim_id != canonical_claim_id:
            raise PredictionValidationError(
                f"Prediction {prediction.claim_id!r} has a non-canonical claim_id"
            )
        if (
            prediction.start_char < 0
            or prediction.end_char <= prediction.start_char
            or prediction.end_char > len(answer_text)
        ):
            raise PredictionValidationError(
                f"Prediction {prediction.claim_id!r} has an invalid character span"
            )
        if prediction.quote != answer_text[prediction.start_char : prediction.end_char]:
            raise PredictionValidationError(
                f"Prediction {prediction.claim_id!r} quote does not match its source span"
            )
        if prediction.quote != prediction.quote.strip():
            raise PredictionValidationError(
                f"Prediction {prediction.claim_id!r} quote has leading or trailing whitespace"
            )
        span = (prediction.start_char, prediction.end_char)
        if span in seen_spans:
            raise PredictionValidationError(
                f"Duplicate prediction span {prediction.claim_id!r}"
            )
        seen_spans.add(span)
        if previous is not None and prediction.start_char < previous.end_char:
            raise PredictionValidationError(
                f"Predictions {previous.claim_id!r} and {prediction.claim_id!r} overlap"
            )
        previous = prediction
