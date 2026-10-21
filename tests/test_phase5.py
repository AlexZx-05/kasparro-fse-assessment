"""Phase 5 database and API smoke tests."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app import database
from app.db_models import Base
from app.evaluation.scorer import AnswerOutcome
from app.main import app
from app.repositories import (
    create_run,
    finalize_evaluation_run,
    get_run_with_results,
    persist_answer_outcome,
    persist_run_result_set,
    save_answer_result,
    save_claim_result,
)


def _reset_database() -> None:
    database._engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    database._session_factory = sessionmaker(bind=database._engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=database._engine)


def test_psycopg_url_uses_installed_psycopg2_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for configured_url in (
        "postgresql://localhost/db",
        "postgresql+psycopg://localhost/db",
    ):
        monkeypatch.setattr(database, "_engine", None)
        monkeypatch.setattr(
            database,
            "get_database_url",
            lambda: configured_url,
        )

        engine = database.get_engine()
        try:
            assert engine.dialect.name == "postgresql"
            assert engine.dialect.driver == "psycopg2"
        finally:
            engine.dispose()


def test_initialize_database_creates_missing_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    monkeypatch.setattr(database, "_engine", engine)

    database.initialize_database()

    assert set(inspect(engine).get_table_names()).issuperset(
        {"runs", "answer_results", "claim_results"}
    )
    engine.dispose()


def test_persisted_evaluation_with_answer_errors_is_partial() -> None:
    _reset_database()
    run = create_run("v1", "development")

    persisted = persist_run_result_set(
        run.id,
        outcomes=[
            SimpleNamespace(answer_id="A01", status="success", predictions=[]),
            SimpleNamespace(
                answer_id="A02",
                status="error",
                predictions=[],
                error_type="OpenRouterTimeoutError",
            ),
        ],
        metrics={"coverage": {"rate": 0.5}},
    )

    assert persisted is not None
    assert persisted.status == "partial"
    assert persisted.total_answers == 2
    assert persisted.successful_answers == 1
    assert persisted.failed_answers == 1
    assert persisted.metrics_json == {"coverage": {"rate": 0.5}}


def test_incremental_answer_outcomes_are_visible_before_run_finishes() -> None:
    _reset_database()
    run = create_run("v1", "development")

    persisted = persist_answer_outcome(
        run.id,
        SimpleNamespace(
            answer_id="A01",
            status="success",
            predictions=[
                SimpleNamespace(
                    claim_id="A01:0:7",
                    start_char=0,
                    end_char=7,
                    quote="A fact.",
                    label="supported",
                    fact_ids=["brand.fact"],
                )
            ],
        ),
    )
    current, answer_results, claim_results = get_run_with_results(run.id)

    assert persisted is not None
    assert current is not None
    assert current.status == "running"
    assert current.successful_answers == 1
    assert [result.answer_id for result in answer_results] == ["A01"]
    assert [claim.claim_id for claim in claim_results] == ["A01:0:7"]

    completed = finalize_evaluation_run(
        run.id,
        metrics={"coverage": {"rate": 1.0}},
        error_summary={},
        completed_at=datetime.now(timezone.utc),
    )
    assert completed is not None
    assert completed.status == "completed"


def test_repository_can_persist_run_and_claims() -> None:
    _reset_database()
    run = create_run("v1", "development")
    answer_result = save_answer_result(
        run.id,
        answer_id="A01",
        status="success",
        predicted_claim_count=1,
    )
    claim = save_claim_result(
        answer_result.id,
        claim_id="A01:0:7",
        start_char=0,
        end_char=7,
        claim_text="A fact.",
        label="supported",
        fact_ids=["fact-1"],
    )

    assert run.id is not None
    assert answer_result.answer_id == "A01"
    assert claim.claim_id == "A01:0:7"
    assert claim.label == "supported"


def test_get_evaluation_run_exposes_incrementally_persisted_results() -> None:
    _reset_database()
    run = create_run("v1", "development")
    persist_answer_outcome(
        run.id,
        SimpleNamespace(
            answer_id="A01",
            status="success",
            predictions=[
                SimpleNamespace(
                    claim_id="A01:0:7",
                    start_char=0,
                    end_char=7,
                    quote="A fact.",
                    label="supported",
                    fact_ids=["brand.fact"],
                )
            ],
        ),
    )

    response = TestClient(app).get(f"/evaluation-runs/{run.id}")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "running"
    assert body["successful_answers"] == 1
    assert [item["answer_id"] for item in body["answer_results"]] == ["A01"]
    assert [item["claim_id"] for item in body["claim_results"]] == ["A01:0:7"]


def test_classify_route_requires_known_answer_id() -> None:
    _reset_database()
    client = TestClient(app)
    response = client.post(
        "/classify",
        json={"answer_id": "missing", "ai_answer": "Some answer", "brand_id": "brand"},
    )
    assert response.status_code == 404


def test_classify_route_returns_validated_claim_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main_module
    from app.data_loader import load_answers
    from app.schemas import ClaimClassificationResult, ClaimExtractionResult

    _reset_database()
    answer = next(answer for answer in load_answers() if answer.answer_id == "A01")

    class EmptyExtractor:
        def extract(self, answer_id: str, _: str) -> ClaimExtractionResult:
            return ClaimExtractionResult(answer_id=answer_id, claims=())

    class EmptyClassifier:
        def classify(
            self,
            answer_id: str,
            _: str,
            __: ClaimExtractionResult,
        ) -> ClaimClassificationResult:
            return ClaimClassificationResult(answer_id=answer_id, claims=())

    monkeypatch.setattr(main_module, "create_default_extractor", EmptyExtractor)
    monkeypatch.setattr(main_module, "create_default_classifier", EmptyClassifier)
    response = TestClient(app).post(
        "/classify",
        json={"answer_id": answer.answer_id, "ai_answer": answer.ai_answer},
    )

    assert response.status_code == 200
    assert response.json() == {"answer_id": answer.answer_id, "claims": []}


@pytest.mark.parametrize(
    ("provider_error", "expected_status", "expected_detail"),
    [
        (
            "authentication",
            503,
            "OpenRouter rejected the configured credentials or model access.",
        ),
        (
            "rate_limit",
            429,
            "OpenRouter rate limit or quota was exceeded.",
        ),
    ],
)
def test_classify_route_explains_provider_auth_and_rate_limit_errors(
    monkeypatch: pytest.MonkeyPatch,
    provider_error: str,
    expected_status: int,
    expected_detail: str,
) -> None:
    import app.main as main_module
    from app.agent.openrouter_client import (
        OpenRouterAuthenticationError,
        OpenRouterRateLimitError,
    )
    from app.data_loader import load_answers

    _reset_database()
    answer = next(answer for answer in load_answers() if answer.answer_id == "A01")
    error_type = (
        OpenRouterAuthenticationError
        if provider_error == "authentication"
        else OpenRouterRateLimitError
    )

    class FailingExtractor:
        def extract(self, *_: object) -> None:
            raise error_type("provider failure")

    monkeypatch.setattr(
        main_module,
        "create_default_extractor",
        lambda: FailingExtractor(),
    )
    response = TestClient(app).post(
        "/classify",
        json={
            "answer_id": answer.answer_id,
            "brand_id": answer.brand_id,
            "ai_answer": answer.ai_answer,
        },
    )

    assert response.status_code == expected_status
    assert expected_detail in response.json()["detail"]


def test_create_evaluation_run_requires_known_version() -> None:
    _reset_database()
    client = TestClient(app)
    response = client.post(
        "/evaluation-runs",
        json={"version": "v9", "dataset_split": "development"},
    )
    assert response.status_code == 422


def test_evaluation_run_returns_running_and_schedules_background_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main_module

    _reset_database()
    scheduled: list[tuple[int, str, str]] = []

    def _record_task(run_id: int, version: str, dataset_split: str) -> None:
        scheduled.append((run_id, version, dataset_split))

    monkeypatch.setattr(main_module, "_execute_evaluation_run", _record_task)
    response = TestClient(app).post(
        "/evaluation-runs",
        json={"version": "v1", "dataset_split": "development"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "running"
    assert scheduled == [(body["run_id"], "v1", "development")]


def test_api_evaluation_writes_reports_under_run_specific_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main_module

    captured: dict[str, object] = {}
    updates: list[dict[str, object]] = []
    scored = SimpleNamespace(outcomes=[])
    metrics = SimpleNamespace(as_dict=lambda: {})
    persisted_answers: list[AnswerOutcome] = []

    def _fake_evaluation(**kwargs: object) -> tuple[object, object, dict[str, object]]:
        captured.update(kwargs)
        progress_callback = kwargs["progress_callback"]
        progress_callback(
            1,
            1,
            AnswerOutcome(
                answer_id="A01",
                status="success",
                predictions=(),
                duration_ms=0.0,
            ),
        )
        return scored, metrics, {}

    monkeypatch.setattr(main_module, "run_development_evaluation", _fake_evaluation)
    monkeypatch.setattr(main_module, "analyse_errors", lambda _: {})
    monkeypatch.setattr(
        main_module,
        "update_run",
        lambda _run_id, **kwargs: updates.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        main_module,
        "persist_answer_outcome",
        lambda _run_id, outcome: persisted_answers.append(outcome) or object(),
    )
    monkeypatch.setattr(
        main_module,
        "finalize_evaluation_run",
        lambda *_args, **_kwargs: object(),
    )

    main_module._execute_evaluation_run(246, "v1", "development")

    reports_root = main_module.DEFAULT_DATA_DIR.parent / "reports"
    assert captured["reports_dir"] == reports_root / "api_runs" / "246"
    assert captured["baseline_reports_dir"] == reports_root
    assert captured["overwrite"] is False
    assert captured["progress_callback"]
    assert updates[0]["status"] == "running"
    assert updates[0]["total_answers"] > 0
    assert len(persisted_answers) == 1
