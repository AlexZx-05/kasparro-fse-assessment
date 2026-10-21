"""FastAPI application for the Kasparro inference and evaluation backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import AsyncIterator

from fastapi import BackgroundTasks, FastAPI, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.agent.claim_classifier import (
    ClassificationError,
    create_default_classifier,
)
from app.agent.claim_extractor import (
    MalformedModelResponse,
    create_default_extractor,
)
from app.agent.openrouter_client import (
    OpenRouterAuthenticationError,
    OpenRouterClientError,
    OpenRouterRateLimitError,
    OpenRouterTimeoutError,
)
from app.agent.validators import ClaimValidationError
from app.config import get_settings
from app.data_loader import DEFAULT_DATA_DIR, DatasetError, load_answers
from app.database import initialize_database, validate_database_configuration
from app.evaluation.error_analysis import analyse_errors
from app.evaluation.runner import run_development_evaluation
from app.evaluation.scorer import AnswerOutcome
from app.evaluation.versions import AgentVersion
from app.langfuse import tracked_trace
from app.repositories import (
    finalize_evaluation_run,
    create_run,
    get_run_with_results,
    persist_answer_outcome,
    update_run,
)
from app.schemas import (
    AnswerResultResponse,
    ClaimDatabaseResultResponse,
    ClaimResultResponse,
    ClassifyAnswerRequest,
    ClassifyAnswerResponse,
    CreateEvaluationRunRequest,
    EvaluationRunDetailResponse,
    EvaluationRunResponse,
)

logger = logging.getLogger(__name__)


def _startup() -> None:
    try:
        validate_database_configuration()
        initialize_database()
    except RuntimeError as error:
        raise RuntimeError(
            "DATABASE_URL is missing. Set it in the local .env file before starting the API."
        ) from error
    except SQLAlchemyError as error:
        logger.exception("Database startup validation or table initialization failed")
        raise RuntimeError(
            "Could not initialize the PostgreSQL database. Check the configured "
            "database, network access, and database permissions."
        ) from error


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    _startup()
    yield


app = FastAPI(
    title="Kasparro FSE Assessment",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyAnswerResponse)
def classify_answer(request: ClassifyAnswerRequest) -> ClassifyAnswerResponse:
    try:
        answers = load_answers()
    except DatasetError as error:
        logger.exception("Could not load assessment answers")
        raise HTTPException(
            status_code=500,
            detail="Assessment data could not be loaded.",
        ) from error

    matches = [answer for answer in answers if answer.answer_id == request.answer_id]
    if len(matches) != 1:
        raise HTTPException(status_code=404, detail="Unknown answer_id.")
    answer = matches[0]
    if request.brand_id is not None and request.brand_id != answer.brand_id:
        raise HTTPException(
            status_code=400,
            detail="brand_id does not match the assessment record for this answer_id.",
        )
    if request.ai_answer != answer.ai_answer:
        raise HTTPException(
            status_code=400,
            detail="ai_answer must exactly match the assessment record for this answer_id.",
        )

    try:
        with tracked_trace(
            name="classify-answer",
            metadata={
                "answer_id": answer.answer_id,
                "dataset_split": answer.split,
                "brand_id": answer.brand_id,
                "agent_version": "default",
                "model": get_settings().model_name,
            },
        ):
            extraction = create_default_extractor().extract(
                answer.answer_id,
                request.ai_answer,
            )
            classification = create_default_classifier().classify(
                answer.answer_id,
                request.ai_answer,
                extraction,
            )
    except OpenRouterTimeoutError as error:
        logger.warning("Classification timed out for answer %s", answer.answer_id)
        raise HTTPException(
            status_code=504,
            detail="The model provider timed out while classifying this answer.",
        ) from error
    except OpenRouterAuthenticationError as error:
        logger.error(
            "OpenRouter rejected credentials or access for classification of answer %s",
            answer.answer_id,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "OpenRouter rejected the configured credentials or model access. "
                "Verify OPENROUTER_API_KEY and confirm the configured model is "
                "available to this account. Do not share the key."
            ),
        ) from error
    except OpenRouterRateLimitError as error:
        logger.warning(
            "OpenRouter rate limit or quota blocked classification for answer %s",
            answer.answer_id,
        )
        raise HTTPException(
            status_code=429,
            detail=(
                "OpenRouter rate limit or quota was exceeded. Check the account "
                "limits and usage, then retry after the limit resets."
            ),
        ) from error
    except (
        OpenRouterClientError,
        ClassificationError,
        MalformedModelResponse,
        ClaimValidationError,
    ) as error:
        logger.warning(
            "Provider or classification validation failed for answer %s (%s)",
            answer.answer_id,
            type(error).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="The model provider returned an unusable classification response.",
        ) from error
    except Exception as error:
        logger.exception("Classification failed for answer %s", answer.answer_id)
        raise HTTPException(
            status_code=500,
            detail="Classification failed. Check the server logs for details.",
        ) from error

    claims = [
        ClaimResultResponse(
            claim_id=claim.claim_id,
            start_char=claim.start_char,
            end_char=claim.end_char,
            claim_text=claim.claim_text,
            label=claim.label,
            fact_ids=claim.fact_ids,
        )
        for claim in classification.claims
    ]
    return ClassifyAnswerResponse(answer_id=answer.answer_id, claims=claims)


@app.post(
    "/evaluation-runs",
    response_model=EvaluationRunResponse,
    status_code=202,
)
def create_evaluation_run(
    request: CreateEvaluationRunRequest,
    background_tasks: BackgroundTasks,
) -> EvaluationRunResponse:
    try:
        run = create_run(request.version, request.dataset_split)
    except SQLAlchemyError as error:
        logger.exception("Could not create evaluation run record")
        raise HTTPException(
            status_code=503,
            detail="Could not create the evaluation run in the database. "
            "Check server logs for details.",
        ) from error

    background_tasks.add_task(
        _execute_evaluation_run,
        run.id,
        request.version,
        request.dataset_split,
    )
    return EvaluationRunResponse(
        run_id=run.id,
        version=request.version,
        dataset_split=request.dataset_split,
        status="running",
    )


def _execute_evaluation_run(
    run_id: int,
    version: AgentVersion,
    dataset_split: str,
) -> None:
    try:
        if dataset_split != "development":
            raise ValueError("Only the development split is available in this phase.")
        answers = load_answers(DEFAULT_DATA_DIR)
        total_answers = sum(answer.split == dataset_split for answer in answers)
        updated = update_run(
            run_id,
            status="running",
            total_answers=total_answers,
            successful_answers=0,
            failed_answers=0,
        )
        if updated is None:
            raise RuntimeError("Evaluation run record disappeared before execution.")

        def record_progress(
            completed_count: int,
            answer_count: int,
            outcome: AnswerOutcome,
        ) -> None:
            updated_run = persist_answer_outcome(run_id, outcome)
            if updated_run is None:
                raise RuntimeError(
                    f"Evaluation run record disappeared after answer {completed_count}."
                )
            logger.info(
                "Evaluation run %s processed answer %s/%s (%s): %s",
                run_id,
                completed_count,
                answer_count,
                outcome.answer_id,
                outcome.status,
            )

        reports_root = DEFAULT_DATA_DIR.parent / "reports"
        scored, metrics, _ = run_development_evaluation(
            version=version,
            data_dir=DEFAULT_DATA_DIR,
            reports_dir=reports_root / "api_runs" / str(run_id),
            baseline_reports_dir=reports_root,
            overwrite=False,
            progress_callback=record_progress,
            run_id=run_id,
            phase7_report_path=reports_root / f"{version}_development.json",
        )
        persisted = finalize_evaluation_run(
            run_id,
            metrics=metrics.as_dict(),
            error_summary=analyse_errors(scored),
            completed_at=datetime.now(timezone.utc),
        )
        if persisted is None:
            raise RuntimeError("Evaluation run record disappeared before persistence.")
    except Exception as error:
        logger.exception("Evaluation run %s failed", run_id)
        try:
            updated = update_run(
                run_id,
                status="failed",
                error_summary={
                    "error_type": type(error).__name__,
                    "error_message": "Evaluation execution failed; inspect server logs.",
                },
                completed_at=datetime.now(timezone.utc),
            )
            if updated is None:
                logger.error("Could not find evaluation run %s to mark as failed", run_id)
        except SQLAlchemyError:
            logger.exception("Could not persist failed status for run %s", run_id)


@app.get(
    "/evaluation-runs/{run_id}",
    response_model=EvaluationRunDetailResponse,
)
def get_evaluation_run(run_id: int) -> EvaluationRunDetailResponse:
    try:
        run, answer_results, claim_results = get_run_with_results(run_id)
    except SQLAlchemyError as error:
        logger.exception("Could not retrieve evaluation run %s", run_id)
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve the evaluation run from the database.",
        ) from error
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown evaluation run.")

    answer_items = [
        AnswerResultResponse(
            id=result.id,
            answer_id=result.answer_id,
            status=result.status,
            predicted_claim_count=result.predicted_claim_count,
            error_message=result.error_message,
            created_at=result.created_at.isoformat(),
        )
        for result in answer_results
    ]
    claim_items = [
        ClaimDatabaseResultResponse(
            id=claim.id,
            claim_id=claim.claim_id,
            start_char=claim.start_char,
            end_char=claim.end_char,
            claim_text=claim.claim_text,
            label=claim.label,
            fact_ids=claim.fact_ids,
            created_at=claim.created_at.isoformat(),
        )
        for claim in claim_results
    ]
    return EvaluationRunDetailResponse(
        id=run.id,
        version=run.version,
        dataset_split=run.dataset_split,
        status=run.status,
        total_answers=run.total_answers,
        successful_answers=run.successful_answers,
        failed_answers=run.failed_answers,
        metrics_json=run.metrics_json,
        error_summary=run.error_summary,
        created_at=run.created_at.isoformat(),
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        answer_results=answer_items,
        claim_results=claim_items,
    )
