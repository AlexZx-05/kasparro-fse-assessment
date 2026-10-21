"""Repository helpers for storing run, answer, and claim records."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data_loader import VALID_LABELS
from app.db_models import AnswerResult, ClaimResult, Run
from app.database import get_session_factory


def create_run(version: str, dataset_split: str) -> Run:
    session = get_session_factory()()
    try:
        run = Run(version=version, dataset_split=dataset_split, status="running", total_answers=0)
        session.add(run)
        session.commit()
        session.refresh(run)
        return run
    finally:
        session.close()


def update_run(
    run_id: int,
    *,
    status: str | None = None,
    total_answers: int | None = None,
    successful_answers: int | None = None,
    failed_answers: int | None = None,
    metrics_json: dict[str, Any] | None = None,
    error_summary: dict[str, Any] | None = None,
    completed_at: Any | None = None,
) -> Run | None:
    session = get_session_factory()()
    try:
        run = session.get(Run, run_id)
        if run is None:
            return None
        if status is not None:
            run.status = status
        if total_answers is not None:
            run.total_answers = total_answers
        if successful_answers is not None:
            run.successful_answers = successful_answers
        if failed_answers is not None:
            run.failed_answers = failed_answers
        if metrics_json is not None:
            run.metrics_json = metrics_json
        if error_summary is not None:
            run.error_summary = error_summary
        if completed_at is not None:
            run.completed_at = completed_at
        session.commit()
        session.refresh(run)
        return run
    finally:
        session.close()


def save_answer_result(
    run_id: int,
    *,
    answer_id: str,
    status: str,
    predicted_claim_count: int,
    error_message: str | None = None,
) -> AnswerResult:
    session = get_session_factory()()
    try:
        result = AnswerResult(
            run_id=run_id,
            answer_id=answer_id,
            status=status,
            predicted_claim_count=predicted_claim_count,
            error_message=error_message,
        )
        session.add(result)
        session.commit()
        session.refresh(result)
        return result
    finally:
        session.close()


def save_claim_result(
    answer_result_id: int,
    *,
    claim_id: str,
    start_char: int,
    end_char: int,
    claim_text: str,
    label: str,
    fact_ids: list[str],
) -> ClaimResult:
    session = get_session_factory()()
    try:
        result = ClaimResult(
            answer_result_id=answer_result_id,
            claim_id=claim_id,
            start_char=start_char,
            end_char=end_char,
            claim_text=claim_text,
            label=label,
            fact_ids=fact_ids,
        )
        session.add(result)
        session.commit()
        session.refresh(result)
        return result
    finally:
        session.close()


def get_run(run_id: int) -> Run | None:
    session = get_session_factory()()
    try:
        return session.get(Run, run_id)
    finally:
        session.close()


def get_run_with_results(run_id: int) -> tuple[Run | None, list[AnswerResult], list[ClaimResult]]:
    session = get_session_factory()()
    try:
        run = session.get(Run, run_id)
        if run is None:
            return None, [], []
        answer_results = session.execute(
            select(AnswerResult).where(AnswerResult.run_id == run_id).order_by(AnswerResult.answer_id)
        ).scalars().all()
        claim_results = session.execute(
            select(ClaimResult)
            .join(AnswerResult, ClaimResult.answer_result_id == AnswerResult.id)
            .where(AnswerResult.run_id == run_id)
            .order_by(AnswerResult.answer_id, ClaimResult.claim_id)
        ).scalars().all()
        return run, list(answer_results), list(claim_results)
    finally:
        session.close()


def persist_answer_outcome(run_id: int, outcome: Any) -> Run | None:
    """Persist one completed answer and update run counters atomically."""
    session = get_session_factory()()
    try:
        run = session.get(Run, run_id)
        if run is None:
            return None

        answer_result = AnswerResult(
            run_id=run.id,
            answer_id=outcome.answer_id,
            status=outcome.status,
            predicted_claim_count=len(outcome.predictions),
            error_message=(
                None
                if outcome.status == "success"
                else f"{getattr(outcome, 'error_type', 'InferenceError')}: "
                "inference failed; inspect the evaluation report."
            ),
        )
        session.add(answer_result)
        session.flush()
        for prediction in outcome.predictions:
            if prediction.label not in VALID_LABELS:
                raise ValueError(
                    f"Prediction {prediction.claim_id!r} has an invalid label"
                )
            session.add(
                ClaimResult(
                    answer_result_id=answer_result.id,
                    claim_id=prediction.claim_id,
                    start_char=prediction.start_char,
                    end_char=prediction.end_char,
                    claim_text=prediction.quote,
                    label=prediction.label,
                    fact_ids=list(prediction.fact_ids),
                )
            )

        if outcome.status == "success":
            run.successful_answers += 1
        else:
            run.failed_answers += 1
        session.commit()
        session.refresh(run)
        return run
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def finalize_evaluation_run(
    run_id: int,
    *,
    metrics: dict[str, Any],
    error_summary: dict[str, Any],
    completed_at: Any,
) -> Run | None:
    """Set terminal run state after all per-answer outcomes have been saved."""
    session = get_session_factory()()
    try:
        run = session.get(Run, run_id)
        if run is None:
            return None
        run.status = "partial" if run.failed_answers else "completed"
        run.metrics_json = metrics
        run.error_summary = error_summary
        run.completed_at = completed_at
        session.commit()
        session.refresh(run)
        return run
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def persist_run_result_set(
    run_id: int,
    *,
    outcomes: list[Any],
    metrics: dict[str, Any] | None = None,
    error_summary: dict[str, Any] | None = None,
    completed_at: Any | None = None,
) -> Run | None:
    session = get_session_factory()()
    try:
        run = session.get(Run, run_id)
        if run is None:
            return None
        for outcome in outcomes:
            answer_result = AnswerResult(
                run_id=run.id,
                answer_id=outcome.answer_id,
                status=outcome.status,
                predicted_claim_count=len(outcome.predictions),
                error_message=(
                    None
                    if outcome.status == "success"
                    else f"{getattr(outcome, 'error_type', 'InferenceError')}: "
                    "inference failed; inspect the evaluation report."
                ),
            )
            session.add(answer_result)
            session.flush()
            for prediction in outcome.predictions:
                if prediction.label not in VALID_LABELS:
                    raise ValueError(
                        f"Prediction {prediction.claim_id!r} has an invalid label"
                    )
                session.add(
                    ClaimResult(
                        answer_result_id=answer_result.id,
                        claim_id=prediction.claim_id,
                        start_char=prediction.start_char,
                        end_char=prediction.end_char,
                        claim_text=prediction.quote,
                        label=prediction.label,
                        fact_ids=list(prediction.fact_ids),
                    )
                )
        run.total_answers = len(outcomes)
        run.successful_answers = sum(1 for outcome in outcomes if outcome.status == "success")
        run.failed_answers = sum(1 for outcome in outcomes if outcome.status != "success")
        if metrics is not None:
            run.metrics_json = metrics
        if error_summary is not None:
            run.error_summary = error_summary
        if any(outcome.status != "success" for outcome in outcomes) or metrics is None:
            run.status = "partial"
        else:
            run.status = "completed"
        if completed_at is not None:
            run.completed_at = completed_at
        session.commit()
        session.refresh(run)
        return run
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
