"""Phase 5 database and API smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import database
from app.db_models import Base
from app.main import app
from app.repositories import create_run, save_answer_result, save_claim_result


def _reset_database() -> None:
    database._engine = create_engine("sqlite:///:memory:")
    database._session_factory = sessionmaker(bind=database._engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=database._engine)


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


def test_classify_route_requires_known_answer_id() -> None:
    _reset_database()
    client = TestClient(app)
    response = client.post(
        "/classify",
        json={"answer_id": "missing", "ai_answer": "Some answer", "brand_id": "brand"},
    )
    assert response.status_code == 404


def test_create_evaluation_run_requires_known_version() -> None:
    _reset_database()
    client = TestClient(app)
    response = client.post(
        "/evaluation-runs",
        json={"version": "v9", "dataset_split": "development"},
    )
    assert response.status_code == 422
