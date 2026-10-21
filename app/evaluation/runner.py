"""Command-line runner for saved, live development evaluations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import time
from typing import Callable, Literal
from uuid import uuid4

from app.agent.claim_classifier import create_default_classifier
from app.agent.claim_extractor import create_default_extractor
from app.config import get_settings
from app.data_loader import DEFAULT_DATA_DIR, PROJECT_ROOT, load_answers
from app.langfuse import tracked_trace
from app.evaluation.error_analysis import analyse_errors
from app.evaluation.metrics import EvaluationMetrics, calculate_metrics
from app.evaluation.report import (
    ensure_run_destination,
    ensure_phase7_report_destination,
    export_phase7_development_report,
    write_phase7_report,
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
ProgressCallback = Callable[[int, int, AnswerOutcome], None]
logger = logging.getLogger(__name__)


def run_development_evaluation(
    *,
    version: AgentVersion,
    data_dir: Path = DEFAULT_DATA_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    baseline_reports_dir: Path | None = None,
    overwrite: bool = False,
    answer_processor: AnswerProcessor | None = None,
    progress_callback: ProgressCallback | None = None,
    run_id: str | int | None = None,
    phase7_report_path: Path | None = None,
) -> tuple[ScoringResult, EvaluationMetrics, dict[str, Path]]:
    """Run one frozen version over every development answer and save evidence.

    The optional processor is intended only for unit tests.  In normal use the
    default invokes the live extractor and classifier.  Gold is loaded here in
    the evaluator, never passed into the processor or the agent.
    """
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    frozen_version = load_frozen_version(version)
    ensure_run_destination(reports_dir, version, "development", overwrite=overwrite)
    summary_path = phase7_report_path or reports_dir / f"{version}_development.json"
    ensure_phase7_report_destination(summary_path, overwrite=overwrite)
    v1_reports_dir = baseline_reports_dir or reports_dir
    if version == "v2" and not (
        v1_reports_dir / "v1" / "development_metrics.json"
    ).is_file():
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
    evaluation_run_id = run_id or str(uuid4())
    processor = answer_processor or _build_live_processor(
        frozen_version,
        data_dir,
        run_id=evaluation_run_id,
    )

    outcomes_list: list[AnswerOutcome] = []
    for completed_count, answer in enumerate(answers, start=1):
        outcome = _run_one_answer(answer, processor)
        outcomes_list.append(outcome)
        if progress_callback is not None:
            progress_callback(completed_count, len(answers), outcome)
    outcomes = tuple(outcomes_list)
    scored = score_exact_spans(answers, gold_claims, outcomes)
    metrics = calculate_metrics(scored)
    analysis = analyse_errors(scored)
    settings = get_settings()
    run_metadata = _run_metadata(
        frozen_version,
        data_dir,
        dataset_split="development",
        timestamp=started_at.isoformat(),
    )
    run_metadata["run_id"] = evaluation_run_id
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
                for name, path in write_v1_v2_comparison(
                    reports_dir,
                    v1_reports_root=v1_reports_dir,
                ).items()
            }
        )
    summary_path = write_phase7_report(
        path=summary_path,
        run_id=run_metadata["run_id"],
        version=version,
        dataset_split="development",
        timestamp=started_at.isoformat(),
        model=get_settings().model_name,
        configuration={
            **frozen_version.as_dict(),
            "max_model_turns": 3,
            "provider_timeout_seconds": settings.model_timeout_seconds,
            "dataset_files": run_metadata["dataset_files"],
        },
        outcomes=[_outcome_as_dict(outcome) for outcome in outcomes],
        metrics=metrics.as_dict(),
        runtime_seconds=time.perf_counter() - started_clock,
        overwrite=overwrite,
    )
    artifact_paths["phase7_summary"] = summary_path
    return scored, metrics, artifact_paths


def run_final_check_evaluation(
    *,
    version: AgentVersion,
    data_dir: Path = DEFAULT_DATA_DIR,
    report_path: Path | None = None,
    overwrite: bool = False,
    answer_processor: AnswerProcessor | None = None,
    run_id: str | int | None = None,
) -> tuple[tuple[AnswerOutcome, ...], Path]:
    """Run frozen inference on A41-A50 without reading or deriving final labels."""
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    frozen_version = load_frozen_version(version)
    output_path = report_path or DEFAULT_REPORTS_DIR / f"{version}_final.json"
    ensure_phase7_report_destination(output_path, overwrite=overwrite)
    evaluation_run_id = run_id or str(uuid4())

    answers = tuple(
        sorted(
            (answer for answer in load_answers(data_dir) if answer.split == "check"),
            key=lambda answer: answer.answer_id,
        )
    )
    expected_ids = tuple(f"A{number:02d}" for number in range(41, 51))
    actual_ids = tuple(answer.answer_id for answer in answers)
    if actual_ids != expected_ids:
        raise ValueError(
            "Final-check evaluation requires exactly answers A41-A50; "
            f"found {len(actual_ids)} check answer(s)."
        )

    processor = answer_processor or _build_live_processor(
        frozen_version,
        data_dir,
        run_id=evaluation_run_id,
    )
    outcomes = tuple(_run_one_answer(answer, processor) for answer in answers)
    settings = get_settings()
    metadata = _run_metadata(
        frozen_version,
        data_dir,
        dataset_split="check",
        timestamp=started_at.isoformat(),
    )
    output_path = write_phase7_report(
        path=output_path,
        run_id=evaluation_run_id,
        version=version,
        dataset_split="check",
        timestamp=started_at.isoformat(),
        model=settings.model_name,
        configuration={
            **frozen_version.as_dict(),
            "max_model_turns": 3,
            "provider_timeout_seconds": settings.model_timeout_seconds,
            "dataset_files": metadata["dataset_files"],
        },
        outcomes=[_outcome_as_dict(outcome) for outcome in outcomes],
        metrics=None,
        runtime_seconds=time.perf_counter() - started_clock,
        overwrite=overwrite,
    )
    return outcomes, output_path


def _build_live_processor(
    frozen_version: FrozenVersion,
    data_dir: Path,
    *,
    run_id: str | int | None = None,
) -> AnswerProcessor:
    """Create inference objects whose inputs exclude development gold entirely."""
    extractor = create_default_extractor(system_prompt=frozen_version.extraction_prompt)
    classifier = create_default_classifier(
        system_prompt=frozen_version.classification_prompt,
        data_dir=data_dir,
    )

    def process(answer: Answer) -> ClaimClassificationResult:
        with tracked_trace(
            name="evaluation-answer",
            metadata={
                "answer_id": answer.answer_id,
                "dataset_split": answer.split,
                "agent_version": frozen_version.version,
                "model": get_settings().model_name,
                "run_id": run_id,
            },
        ):
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
        logger.warning(
            "Evaluation inference failed for answer %s (%s)",
            answer.answer_id,
            type(error).__name__,
        )
        return AnswerOutcome(
            answer_id=answer.answer_id,
            status="error",
            predictions=(),
            duration_ms=(time.perf_counter() - started) * 1000,
            error_type=type(error).__name__,
            error_message=str(error),
        )


def _run_metadata(
    frozen_version: FrozenVersion,
    data_dir: Path,
    *,
    dataset_split: str,
    timestamp: str,
) -> dict[str, object]:
    settings = get_settings()
    configuration = frozen_version.as_dict()
    configuration["prompt_snapshots"] = {
        "extraction": frozen_version.extraction_prompt,
        "classification": frozen_version.classification_prompt,
    }
    configuration["max_model_turns"] = 3
    configuration["provider_timeout_seconds"] = settings.model_timeout_seconds
    return {
        "version": frozen_version.version,
        "split": dataset_split,
        "timestamp": timestamp,
        "provider": "openrouter",
        "model": settings.model_name,
        "configuration": configuration,
        "targeted_change": frozen_version.targeted_change,
        "dataset_files": {
            name: _sha256_file(data_dir / name)
            for name in (
                ("answers.csv", "brand_facts.csv", "development_gold.csv")
                if dataset_split == "development"
                else ("answers.csv", "brand_facts.csv")
            )
        },
        "usage": None,
        "usage_note": "The provider adapter does not expose aggregate cost.",
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
        description="Run and save a frozen live evaluation on the selected split."
    )
    parser.add_argument("--version", choices=("v1", "v2"), required=True)
    parser.add_argument(
        "--split",
        choices=("development", "check"),
        default="development",
        help=(
            "Use development for scored evaluation; check runs frozen inference "
            "without gold-based metrics."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--baseline-reports-dir",
        type=Path,
        default=None,
        help="Use a specific frozen V1 development report when comparing V2.",
    )
    parser.add_argument(
        "--export-run-id",
        type=int,
        help="Export an already completed PostgreSQL API development run without rerunning inference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace existing artifacts for this version and split.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.export_run_id is not None:
        from app.repositories import get_run

        run = get_run(args.export_run_id)
        if (
            run is None
            or run.status not in {"completed", "partial"}
            or run.dataset_split != "development"
            or run.completed_at is None
        ):
            raise SystemExit(
                f"Run {args.export_run_id} is missing, incomplete, or not a "
                "development evaluation."
            )
        report_path = export_phase7_development_report(
            reports_root=args.reports_dir,
            run_id=run.id,
            version=run.version,
            timestamp=run.created_at.isoformat(),
            runtime_seconds=(run.completed_at - run.created_at).total_seconds(),
            output_path=args.reports_dir / f"{run.version}_development.json",
            overwrite=args.overwrite,
        )
        print(f"Exported completed run {run.id} report: {report_path}")
        return
    if args.split == "check":
        outcomes, path = run_final_check_evaluation(
            version=args.version,
            data_dir=args.data_dir,
            report_path=args.reports_dir / f"{args.version}_final.json",
            overwrite=args.overwrite,
        )
        successful = sum(outcome.status == "success" for outcome in outcomes)
        print(
            f"{args.version} final-check inference complete: "
            f"answers={len(outcomes)}, successful={successful}, "
            "gold-based metrics=unavailable, "
            f"report={path}"
        )
    else:
        _, metrics, paths = run_development_evaluation(
            version=args.version,
            data_dir=args.data_dir,
            reports_dir=args.reports_dir,
            baseline_reports_dir=args.baseline_reports_dir,
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
