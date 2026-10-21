"""FastAPI application for the Kasparro inference and evaluation backend."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from app.agent.claim_classifier import create_default_classifier
from app.agent.claim_extractor import create_default_extractor
from app.data_loader import DEFAULT_DATA_DIR, load_answers
from app.database import validate_database_configuration
from app.evaluation.error_analysis import analyse_errors
from app.evaluation.runner import run_development_evaluation
from app.repositories import (
    create_run,
    get_run_with_results,
    persist_run_result_set,
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

app = FastAPI(title="Kasparro FSE Assessment", version="0.2.0")


@app.on_event("startup")
def _startup() -> None:
    try:
        validate_database_configuration()
    except RuntimeError as error:
        raise RuntimeError(
            "DATABASE_URL is missing. Set it in the local .env file before starting the API."
        ) from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyAnswerResponse)
def classify_answer(
    request: ClassifyAnswerRequest,
) -> ClassifyAnswerResponse:
    answers = load_answers()
    matches = [answer for answer in answers if answer.answer_id == request.answer_id]
    if len(matches) != 1:
        raise HTTPException(status_code=404, detail=f"Unknown answer_id: {request.answer_id}")
    answer = matches[0]
    if request.brand_id is not None and request.brand_id != answer.brand_id:
        raise HTTPException(
            status_code=400,
            detail=f"brand_id {request.brand_id!r} does not match the assessment record for answer {request.answer_id!r}",
        )
    try:
        extraction = create_default_extractor().extract(answer.answer_id, request.ai_answer)
        classification = create_default_classifier().classify(
            answer.answer_id,
            request.ai_answer,
            extraction,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {type(error).__name__}: {error}",
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


@app.post("/evaluation-runs", response_model=EvaluationRunResponse)
def create_evaluation_run(request: CreateEvaluationRunRequest) -> EvaluationRunResponse:
    run = create_run(request.version, request.dataset_split)
    try:
        scored, metrics, _ = run_development_evaluation(
            version=request.version,
            data_dir=DEFAULT_DATA_DIR,
            reports_dir=DEFAULT_DATA_DIR.parent / "reports",
            overwrite=False,
        )
        analysis = analyse_errors(scored)
        persist_run_result_set(
            run.id,
            outcomes=list(scored.outcomes),
            metrics=metrics.as_dict(),
            error_summary=analysis,
        )
        update_run(
            run.id,
            status="completed",
            total_answers=len(scored.outcomes),
            successful_answers=sum(1 for outcome in scored.outcomes if outcome.status == "success"),
            failed_answers=sum(1 for outcome in scored.outcomes if outcome.status != "success"),
            metrics_json=metrics.as_dict(),
            error_summary=analysis,
            completed_at=datetime.now(timezone.utc),
        )
        return EvaluationRunResponse(
            run_id=run.id,
            version=request.version,
            dataset_split=request.dataset_split,
            status="completed",
        )
    except Exception as error:
        update_run(
            run.id,
            status="failed",
            error_summary={"error_type": type(error).__name__, "error_message": str(error)},
            completed_at=datetime.now(timezone.utc),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation run failed: {type(error).__name__}: {error}",
        ) from error


@app.get("/evaluation-runs/{run_id}", response_model=EvaluationRunDetailResponse)
def get_evaluation_run(run_id: int) -> EvaluationRunDetailResponse:
    run, answer_results, claim_results = get_run_with_results(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown evaluation run: {run_id}")

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
