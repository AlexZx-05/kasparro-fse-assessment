import csv
from pathlib import Path

import pytest

from app.data_loader import load_inference_data
from app.evaluation.metrics import calculate_metrics
from app.evaluation.runner import run_development_evaluation
from app.evaluation.versions import load_frozen_version
from app.evaluation.scorer import (
    AnswerOutcome,
    GoldClaim,
    GoldValidationError,
    Prediction,
    PredictionValidationError,
    score_exact_spans,
    validate_development_gold,
)
from app.schemas import Answer, ClaimClassificationResult, ClassifiedClaim


TEXT = "Alpha is supported. Beta is contradicted. Gamma is unknown."


def _answer(answer_id: str = "A01", text: str = TEXT) -> Answer:
    return Answer(
        answer_id=answer_id,
        split="development",
        brand_id="brand",
        question="Question",
        ai_answer=text,
    )


def _gold(answer: Answer, quote: str, label: str) -> GoldClaim:
    start = answer.ai_answer.index(quote)
    end = start + len(quote)
    return GoldClaim(
        answer_id=answer.answer_id,
        claim_id=f"{answer.answer_id}:{start}:{end}",
        start_char=start,
        end_char=end,
        quote=quote,
        label=label,  # type: ignore[arg-type]
        evidence_fact_ids=(),
        explanation="",
        difficulty_tags=(),
    )


def _prediction(answer: Answer, quote: str, label: str | None) -> Prediction:
    start = answer.ai_answer.index(quote)
    end = start + len(quote)
    return Prediction(
        answer_id=answer.answer_id,
        claim_id=f"{answer.answer_id}:{start}:{end}",
        start_char=start,
        end_char=end,
        quote=quote,
        label=label,
    )


def _outcome(answer: Answer, *predictions: Prediction) -> AnswerOutcome:
    return AnswerOutcome(
        answer_id=answer.answer_id,
        status="success",
        predictions=tuple(predictions),
        duration_ms=1.0,
    )


def test_perfect_extraction_classification_and_all_labels() -> None:
    answer = _answer()
    gold = (
        _gold(answer, "Alpha is supported.", "supported"),
        _gold(answer, "Beta is contradicted.", "contradicted"),
        _gold(answer, "Gamma is unknown.", "unknown"),
    )
    scored = score_exact_spans(
        [answer], gold, [_outcome(answer, *(_prediction(answer, item.quote, item.label) for item in gold))]
    )
    metrics = calculate_metrics(scored)

    assert (metrics.extraction_precision, metrics.extraction_recall, metrics.extraction_f1) == (1.0, 1.0, 1.0)
    assert (metrics.joint_precision, metrics.joint_recall, metrics.joint_f1) == (1.0, 1.0, 1.0)
    assert metrics.macro_f1 == 1.0
    assert metrics.confusion_matrix["supported"]["supported"] == 1
    assert metrics.confusion_matrix["contradicted"]["contradicted"] == 1
    assert metrics.confusion_matrix["unknown"]["unknown"] == 1


def test_correct_span_with_wrong_label_reduces_joint_and_populates_confusion() -> None:
    answer = _answer()
    gold = (_gold(answer, "Alpha is supported.", "supported"),)
    scored = score_exact_spans(
        [answer], gold, [_outcome(answer, _prediction(answer, gold[0].quote, "contradicted"))]
    )
    metrics = calculate_metrics(scored)

    assert len(scored.matches) == 1
    assert len(scored.wrong_label_matches) == 1
    assert metrics.extraction_f1 == 1.0
    assert metrics.joint_f1 == 0.0
    assert metrics.confusion_matrix["supported"]["contradicted"] == 1


def test_missing_extra_and_wrong_span_claims_are_counted_without_fuzzy_matching() -> None:
    answer = _answer()
    gold = (
        _gold(answer, "Alpha is supported.", "supported"),
        _gold(answer, "Beta is contradicted.", "contradicted"),
    )
    extra = _prediction(answer, "Gamma is unknown.", "unknown")
    scored = score_exact_spans([answer], gold, [_outcome(answer, extra)])
    metrics = calculate_metrics(scored)

    assert len(scored.matches) == 0
    assert len(scored.missing_gold) == 2
    assert len(scored.extra_predictions) == 1
    assert metrics.extraction_precision == 0.0
    assert metrics.extraction_recall == 0.0
    assert metrics.confusion_matrix["supported"]["missing_or_error"] == 1
    assert metrics.confusion_matrix["no_gold"]["unknown"] == 1


def test_correct_text_at_incorrect_offsets_is_not_an_exact_match() -> None:
    answer = _answer(text="Same fact. Same fact.")
    first = _gold(answer, "Same fact.", "supported")
    second_start = answer.ai_answer.rindex("Same fact.")
    second = Prediction(
        answer_id=answer.answer_id,
        claim_id=f"{answer.answer_id}:{second_start}:{second_start + len('Same fact.')}",
        start_char=second_start,
        end_char=second_start + len("Same fact."),
        quote="Same fact.",
        label="supported",
    )

    scored = score_exact_spans([answer], [first], [_outcome(answer, second)])

    assert not scored.matches
    assert scored.missing_gold == (first,)
    assert scored.extra_predictions == (second,)


def test_duplicate_predictions_and_missing_labels_are_rejected() -> None:
    answer = _answer()
    gold = (_gold(answer, "Alpha is supported.", "supported"),)
    prediction = _prediction(answer, gold[0].quote, "supported")

    with pytest.raises(PredictionValidationError, match="Duplicate prediction"):
        score_exact_spans([answer], gold, [_outcome(answer, prediction, prediction)])
    with pytest.raises(PredictionValidationError, match="missing or invalid label"):
        score_exact_spans(
            [answer], gold, [_outcome(answer, _prediction(answer, gold[0].quote, None))]
        )


def test_empty_predictions_and_empty_gold_use_documented_zero_division_policy() -> None:
    answer = _answer()
    scored = score_exact_spans([answer], [], [_outcome(answer)])
    metrics = calculate_metrics(scored)

    assert metrics.extraction_precision == 0.0
    assert metrics.extraction_recall == 0.0
    assert metrics.joint_f1 == 0.0
    assert metrics.macro_f1 == 0.0
    assert metrics.coverage_rate == 1.0


def test_g10_p10_m8_c7_uses_overall_joint_denominator() -> None:
    answer = _answer(text="a b c d e f g h i j k l")
    positions = [index * 2 for index in range(12)]
    gold = tuple(
        GoldClaim(
            answer_id="A01",
            claim_id=f"A01:{positions[index]}:{positions[index] + 1}",
            start_char=positions[index],
            end_char=positions[index] + 1,
            quote=chr(ord("a") + index),
            label="supported",
            evidence_fact_ids=(),
            explanation="",
            difficulty_tags=(),
        )
        for index in range(10)
    )
    predictions = [
        Prediction(
            answer_id="A01",
            claim_id=f"A01:{positions[index]}:{positions[index] + 1}",
            start_char=positions[index],
            end_char=positions[index] + 1,
            quote=chr(ord("a") + index),
            label="supported" if index < 7 else "contradicted",
        )
        for index in range(8)
    ]
    predictions.extend(
        Prediction(
            answer_id="A01",
            claim_id=f"A01:{positions[index]}:{positions[index] + 1}",
            start_char=positions[index],
            end_char=positions[index] + 1,
            quote=chr(ord("a") + index),
            label="supported",
        )
        for index in (10, 11)
    )

    metrics = calculate_metrics(score_exact_spans([answer], gold, [_outcome(answer, *predictions)]))

    assert metrics.gold_claim_count == 10
    assert metrics.prediction_count == 10
    assert metrics.exact_match_count == 8
    assert metrics.correct_joint_count == 7
    assert metrics.extraction_precision == pytest.approx(0.8)
    assert metrics.extraction_recall == pytest.approx(0.8)
    assert metrics.extraction_f1 == pytest.approx(0.8)
    assert metrics.joint_precision == pytest.approx(0.7)
    assert metrics.joint_recall == pytest.approx(0.7)
    assert metrics.joint_f1 == pytest.approx(0.7)


def test_coverage_keeps_failed_answers_and_their_gold_claims() -> None:
    first = _answer("A01", "First claim.")
    second = _answer("A02", "Second claim.")
    gold = (
        _gold(first, "First claim.", "supported"),
        _gold(second, "Second claim.", "unknown"),
    )
    failed = AnswerOutcome(
        answer_id="A02",
        status="error",
        predictions=(),
        duration_ms=2.0,
        error_type="GroqTimeoutError",
        error_message="Groq request timed out",
    )
    scored = score_exact_spans(
        [first, second],
        gold,
        [_outcome(first, _prediction(first, "First claim.", "supported")), failed],
    )
    metrics = calculate_metrics(scored)

    assert metrics.coverage_answer_count == 1
    assert metrics.attempted_answer_count == 2
    assert metrics.error_answer_count == 1
    assert metrics.coverage_rate == 0.5
    assert metrics.missing_claim_count == 1
    assert metrics.joint_recall == 0.5


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _evaluation_data(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _write_csv(
        tmp_path / "answers.csv",
        ("answer_id", "split", "brand_id", "question", "ai_answer"),
        [{"answer_id": "A01", "split": "development", "brand_id": "brand", "question": "Q", "ai_answer": "A fact."}],
    )
    _write_csv(
        tmp_path / "brand_facts.csv",
        ("fact_id", "brand_id", "model_id", "field", "value", "value_type", "unit", "condition", "valid_from", "valid_to"),
        [{"fact_id": "brand.fact", "brand_id": "brand", "model_id": "m", "field": "f", "value": "true", "value_type": "boolean", "unit": "", "condition": "", "valid_from": "", "valid_to": ""}],
    )
    _write_csv(
        tmp_path / "development_gold.csv",
        ("answer_id", "claim_id", "start_char", "end_char", "quote", "expected_label", "evidence_fact_ids", "explanation", "difficulty_tags"),
        [{"answer_id": "A01", "claim_id": "A01:0:7", "start_char": "0", "end_char": "7", "quote": "A fact.", "expected_label": "supported", "evidence_fact_ids": "brand.fact", "explanation": "", "difficulty_tags": ""}],
    )
    return tmp_path


def test_gold_validation_reports_bad_offsets_and_inference_loader_never_reads_gold(
    tmp_path: Path,
) -> None:
    data_dir = _evaluation_data(tmp_path)
    (data_dir / "development_gold.csv").write_text("not,a,valid,gold,file\n", encoding="utf-8")

    # The inference-only view succeeds even with malformed gold data.
    inference_data = load_inference_data(data_dir)
    assert inference_data.answers[0].ai_answer == "A fact."
    with pytest.raises(GoldValidationError, match="missing required column"):
        validate_development_gold(inference_data.answers, data_dir=data_dir)


def test_runner_saves_actual_fake_outcomes_without_passing_gold_to_processor(
    tmp_path: Path,
) -> None:
    data_dir = _evaluation_data(tmp_path / "data")
    seen_answers: list[str] = []

    def processor(answer: Answer) -> ClaimClassificationResult:
        seen_answers.append(answer.answer_id)
        return ClaimClassificationResult(
            answer_id=answer.answer_id,
            claims=(
                ClassifiedClaim(
                    answer_id=answer.answer_id,
                    claim_id="A01:0:7",
                    start_char=0,
                    end_char=7,
                    claim_text="A fact.",
                    label="supported",
                    fact_ids=("brand.fact",),
                ),
            ),
        )

    scored, metrics, paths = run_development_evaluation(
        version="v1",
        data_dir=data_dir,
        reports_dir=tmp_path / "reports",
        answer_processor=processor,
    )

    assert seen_answers == ["A01"]
    assert len(scored.matches) == 1
    assert metrics.joint_f1 == 1.0
    assert all(path.is_file() for path in paths.values())


def test_frozen_version_prompts_are_available() -> None:
    v1 = load_frozen_version("v1")
    v2 = load_frozen_version("v2")

    assert v1.version == "v1"
    assert v2.version == "v2"
    assert v1.extraction_prompt.strip()
    assert v1.classification_prompt.strip()
    assert v2.extraction_prompt.strip()
    assert v2.classification_prompt.strip()
