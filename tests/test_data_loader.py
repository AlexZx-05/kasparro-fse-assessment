import asyncio
import csv
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from app.data_loader import (
    DatasetError,
    load_answers,
    load_brand_facts,
    load_development_gold,
    load_inference_data,
    parse_boolean_value,
    validate_dataset,
)
from app.main import app


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_load_answers_preserves_quoted_commas_and_embedded_newlines(
    tmp_path: Path,
) -> None:
    original_answer = "First sentence, with a comma.\nSecond sentence stays here.  "
    _write_csv(
        tmp_path / "answers.csv",
        ("answer_id", "split", "brand_id", "question", "ai_answer"),
        [
            {
                "answer_id": "A01",
                "split": "development",
                "brand_id": "sample",
                "question": "A question, containing a comma?",
                "ai_answer": original_answer,
            }
        ],
    )

    answer = load_answers(tmp_path)[0]

    assert answer.question == "A question, containing a comma?"
    assert answer.ai_answer == original_answer


@pytest.mark.parametrize(
    ("loader", "filename"),
    [
        (load_answers, "answers.csv"),
        (load_brand_facts, "brand_facts.csv"),
        (load_development_gold, "development_gold.csv"),
    ],
)
def test_load_missing_csv_reports_path(
    tmp_path: Path, loader: Callable[[Path], object], filename: str
) -> None:
    with pytest.raises(DatasetError, match=filename):
        loader(tmp_path)


def test_missing_required_column_is_reported(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "answers.csv",
        ("answer_id", "split", "brand_id", "question"),
        [],
    )

    with pytest.raises(DatasetError, match="ai_answer"):
        load_answers(tmp_path)


def test_duplicate_answer_ids_are_rejected(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "answers.csv",
        ("answer_id", "split", "brand_id", "question", "ai_answer"),
        [
            {
                "answer_id": "A01",
                "split": "development",
                "brand_id": "sample",
                "question": "Question one",
                "ai_answer": "Answer one",
            },
            {
                "answer_id": "A01",
                "split": "development",
                "brand_id": "sample",
                "question": "Question two",
                "ai_answer": "Answer two",
            },
        ],
    )

    with pytest.raises(DatasetError, match="duplicate answer_id"):
        load_answers(tmp_path)


def test_invalid_split_is_rejected(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "answers.csv",
        ("answer_id", "split", "brand_id", "question", "ai_answer"),
        [
            {
                "answer_id": "A01",
                "split": "production",
                "brand_id": "sample",
                "question": "Question",
                "ai_answer": "Answer",
            }
        ],
    )

    with pytest.raises(DatasetError, match="invalid split"):
        load_answers(tmp_path)


def test_malformed_csv_is_reported(tmp_path: Path) -> None:
    (tmp_path / "answers.csv").write_text(
        'answer_id,split,brand_id,question,ai_answer\nA01,development,sample,"open quote',
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="malformed CSV"):
        load_answers(tmp_path)


def test_invalid_fact_value_type_is_rejected(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "brand_facts.csv",
        (
            "fact_id",
            "brand_id",
            "model_id",
            "field",
            "value",
            "value_type",
            "unit",
            "condition",
            "valid_from",
            "valid_to",
        ),
        [
            {
                "fact_id": "speaker.unknown",
                "brand_id": "sample",
                "model_id": "speaker",
                "field": "unknown",
                "value": "yes",
                "value_type": "truthy",
                "unit": "",
                "condition": "",
                "valid_from": "",
                "valid_to": "",
            }
        ],
    )

    with pytest.raises(DatasetError, match="invalid value_type"):
        load_brand_facts(tmp_path)


def test_invalid_development_label_is_rejected(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "development_gold.csv",
        (
            "answer_id",
            "claim_id",
            "start_char",
            "end_char",
            "quote",
            "expected_label",
            "evidence_fact_ids",
            "explanation",
            "difficulty_tags",
        ),
        [
            {
                "answer_id": "A01",
                "claim_id": "A01:0:6",
                "start_char": "0",
                "end_char": "6",
                "quote": "Answer",
                "expected_label": "likely",
                "evidence_fact_ids": "",
                "explanation": "",
                "difficulty_tags": "",
            }
        ],
    )

    with pytest.raises(DatasetError, match="invalid expected_label"):
        load_development_gold(tmp_path)


def test_inference_data_does_not_contain_gold_labels(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "answers.csv",
        ("answer_id", "split", "brand_id", "question", "ai_answer"),
        [
            {
                "answer_id": "A01",
                "split": "development",
                "brand_id": "sample",
                "question": "Question",
                "ai_answer": "Answer",
            }
        ],
    )
    _write_csv(
        tmp_path / "brand_facts.csv",
        (
            "fact_id",
            "brand_id",
            "model_id",
            "field",
            "value",
            "value_type",
            "unit",
            "condition",
            "valid_from",
            "valid_to",
        ),
        [],
    )
    _write_csv(
        tmp_path / "development_gold.csv",
        (
            "answer_id",
            "claim_id",
            "start_char",
            "end_char",
            "quote",
            "expected_label",
            "evidence_fact_ids",
            "explanation",
            "difficulty_tags",
        ),
        [
            {
                "answer_id": "A01",
                "claim_id": "A01:0:6",
                "start_char": "0",
                "end_char": "6",
                "quote": "Answer",
                "expected_label": "supported",
                "evidence_fact_ids": "",
                "explanation": "",
                "difficulty_tags": "",
            }
        ],
    )

    inference_data = load_inference_data(tmp_path)

    assert len(inference_data.answers) == 1
    assert len(inference_data.brand_facts) == 0
    assert not hasattr(inference_data, "development_gold")
    assert load_development_gold(tmp_path)[0].expected_label == "supported"


def test_boolean_catalogue_values_are_interpreted_explicitly(
    tmp_path: Path,
) -> None:
    _write_csv(
        tmp_path / "brand_facts.csv",
        (
            "fact_id",
            "brand_id",
            "model_id",
            "field",
            "value",
            "value_type",
            "unit",
            "condition",
            "valid_from",
            "valid_to",
        ),
        [
            {
                "fact_id": "speaker.immersion",
                "brand_id": "sample",
                "model_id": "speaker",
                "field": "immersion_allowed",
                "value": "false",
                "value_type": "boolean",
                "unit": "",
                "condition": "",
                "valid_from": "",
                "valid_to": "",
            }
        ],
    )

    fact = load_brand_facts(tmp_path)[0]

    assert fact.value == "false"
    assert parse_boolean_value(fact) is False


def test_health_endpoint_returns_ok_http_response() -> None:
    assert any(
        route.path == "/health" and "GET" in route.methods
        for route in app.routes
    )

    async def request_health() -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            messages.append(message)

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/health",
                "raw_path": b"/health",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"host", b"testserver")],
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
            },
            receive,
            send,
        )
        return messages

    messages = asyncio.run(request_health())
    response_start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        message["body"]
        for message in messages
        if message["type"] == "http.response.body"
    )

    assert response_start["status"] == 200
    assert json.loads(response_body) == {"status": "ok"}


def test_supplied_dataset_is_valid() -> None:
    summary = validate_dataset()

    assert summary.answer_count == 50
    assert summary.fact_count == 132
    assert summary.development_gold_count == 376
