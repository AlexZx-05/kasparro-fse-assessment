"""Command-line runner for saved, live development evaluations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
from typing import Callable, Literal

from app.agent.claim_classifier import create_default_classifier
from app.agent.claim_extractor import create_default_extractor
from app.config import get_settings
from app.data_loader import DEFAULT_DATA_DIR, PROJECT_ROOT, load_answers
from app.evaluation.error_analysis import analyse_errors
from app.evaluation.metrics import EvaluationMetrics, calculate_metrics
from app.evaluation.report import (
    ensure_run_destination,
    write_development_reports,
    write_v1_v2_comparison,
)
from app.evaluation.scorer import (
    AnswerOutcome,
    Prediction,
    ScoringResult,
    score_exact_spans,
    validate_development_gold,
)
from app.evaluation.versions import AgentVersion, FrozenVersion, load_frozen_version
from app.schemas import Answer, ClaimClassificationResult


DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
AnswerProcessor = Callable[[Answer], ClaimClassificationResult]


def run_development_evaluation(
    *,
    version: AgentVersion,
    data_dir: Path = DEFAULT_DATA_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    overwrite: bool = False,
    answer_processor: AnswerProcessor | None = None,
) -> tuple[ScoringResult, EvaluationMetrics, dict[str, Path]]:
    """Run one frozen version over every development answer and save evidence.

    The optional processor is intended only for unit tests.  In normal use the
    default invokes the live extractor and classifier.  Gold is loaded here in
    the evaluator, never passed into the processor or the agent.
    """
    frozen_version = load_frozen_version(version)
    ensure_run_destination(reports_dir, version, "development", overwrite=overwrite)
    if version == "v2" and not (reports_dir / "v1" / "development_metrics.json").is_file():
        raise FileNotFoundError(
            "Version 2 requires saved Version 1 development metrics for comparison"
        )

    answers = tuple(
        sorted(
            (answer for answer in load_answers(data_dir) if answer.split == "development"),
            key=lambda answer: answer.answer_id,
        )
    )
    gold_claims, gold_validation = validate_development_gold(answers, data_dir=data_dir)
    processor = answer_processor or _build_live_processor(frozen_version, data_dir)

    outcomes = tuple(_run_one_answer(answer, processor) for answer in answers)
    scored = score_exact_spans(answers, gold_claims, outcomes)
    metrics = calculate_metrics(scored)
    analysis = analyse_errors(scored)
    run_metadata = _run_metadata(frozen_version, data_dir)
    artifact_paths = write_development_reports(
        reports_root=reports_dir,
        version=version,
        run_metadata=run_metadata,
        outcomes=[_outcome_as_dict(outcome) for outcome in outcomes],
        scheduled_answer_ids=list(scored.scheduled_answer_ids),
        metrics=metrics,
        gold_validation=gold_validation.as_dict(),
        error_analysis=analysis,
    )
    if version == "v2":
        artifact_paths.update(
            {
                f"comparison_{name}": path
                for name, path in write_v1_v2_comparison(reports_dir).items()
            }
        )
    return scored, metrics, artifact_paths


def _build_live_processor(
    frozen_version: FrozenVersion,
    data_dir: Path,
) -> AnswerProcessor:
    """Create inference objects whose inputs exclude development gold entirely."""
    extractor = create_default_extractor(system_prompt=frozen_version.extraction_prompt)
    classifier = create_default_classifier(
        system_prompt=frozen_version.classification_prompt,
        data_dir=data_dir,
    )

    def process(answer: Answer) -> ClaimClassificationResult:
        extraction = extractor.extract(answer.answer_id, answer.ai_answer)
        return classifier.classify(answer.answer_id, answer.ai_answer, extraction)

    return process


def _run_one_answer(answer: Answer, processor: AnswerProcessor) -> AnswerOutcome:
    started = time.perf_counter()
    try:
        result = processor(answer)
        predictions = tuple(
            Prediction.from_classified_claim(claim) for claim in result.claims
        )
        return AnswerOutcome(
            answer_id=answer.answer_id,
            status="success",
            predictions=predictions,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as error:
        return AnswerOutcome(
            answer_id=answer.answer_id,
            status="error",
            predictions=(),
            duration_ms=(time.perf_counter() - started) * 1000,
            error_type=type(error).__name__,
            error_message=str(error),
        )


def _run_metadata(frozen_version: FrozenVersion, data_dir: Path) -> dict[str, object]:
    settings = get_settings()
    configuration = frozen_version.as_dict()
    configuration["prompt_snapshots"] = {
        "extraction": frozen_version.extraction_prompt,
        "classification": frozen_version.classification_prompt,
    }
    configuration["max_model_turns"] = 3
    configuration["provider_timeout_seconds"] = settings.groq_timeout_seconds
    return {
        "version": frozen_version.version,
        "split": "development",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": settings.model_provider,
        "model": settings.model_name,
        "configuration": configuration,
        "targeted_change": frozen_version.targeted_change,
        "dataset_files": {
            name: _sha256_file(data_dir / name)
            for name in ("answers.csv", "brand_facts.csv", "development_gold.csv")
        },
        "usage": None,
        "usage_note": "The current provider adapter does not expose per-answer usage.",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _outcome_as_dict(outcome: AnswerOutcome) -> dict[str, object]:
    return {
        "answer_id": outcome.answer_id,
        "status": outcome.status,
        "duration_ms": outcome.duration_ms,
        "error_type": outcome.error_type,
        "error_message": outcome.error_message,
        "predictions": [
            {
                "answer_id": prediction.answer_id,
                "claim_id": prediction.claim_id,
                "start_char": prediction.start_char,
                "end_char": prediction.end_char,
                "quote": prediction.quote,
                "label": prediction.label,
                "fact_ids": list(prediction.fact_ids),
            }
            for prediction in outcome.predictions
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and save a frozen live evaluation on A01-A40 only."
    )
    parser.add_argument("--version", choices=("v1", "v2"), required=True)
    parser.add_argument(
        "--split",
        choices=("development",),
        default="development",
        help="Only the development split is eligible for Phase 4 tuning.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace existing artifacts for this version and split.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    _, metrics, paths = run_development_evaluation(
        version=args.version,
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        overwrite=args.overwrite,
    )
    print(
        f"{args.version} development evaluation complete: "
        f"joint_f1={metrics.joint_f1:.6f}, "
        f"coverage={metrics.coverage_rate:.6f}, "
        f"artifacts={', '.join(str(path) for path in paths.values())}"
    )


if __name__ == "__main__":
    main()
