import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.claim_classifier import (
    AnswerMappingError,
    ClaimClassifier,
    InvalidClassification,
    ToolCallError,
)
from app.agent.tools import (
    BrandFactsTool,
    InvalidToolArguments,
    execute_registered_tool,
)
from app.agent.validators import build_extraction_result
from app.schemas import ClaimExtractionResult, ExtractedClaim


ANSWER_COLUMNS = ("answer_id", "split", "brand_id", "question", "ai_answer")
FACT_COLUMNS = (
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
)
ANSWER_TEXT = (
    "Bluetooth 5.3 is supported. Bluetooth 4.0 is supported. "
    "It has built-in GPS."
)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def assessment_data(tmp_path: Path) -> Path:
    _write_csv(
        tmp_path / "answers.csv",
        ANSWER_COLUMNS,
        [
            {
                "answer_id": "A_TEST",
                "split": "development",
                "brand_id": "sample_brand",
                "question": "Test question",
                "ai_answer": ANSWER_TEXT,
            }
        ],
    )
    _write_csv(
        tmp_path / "brand_facts.csv",
        FACT_COLUMNS,
        [
            {
                "fact_id": "sample.speaker.bluetooth",
                "brand_id": "sample_brand",
                "model_id": "speaker",
                "field": "bluetooth_version",
                "value": "5.3",
                "value_type": "string",
                "unit": "",
                "condition": "",
                "valid_from": "2026-01-01",
                "valid_to": "2026-12-31",
            },
            {
                "fact_id": "other.speaker.weight",
                "brand_id": "other_brand",
                "model_id": "speaker",
                "field": "weight",
                "value": "2",
                "value_type": "number",
                "unit": "kg",
                "condition": "",
                "valid_from": "",
                "valid_to": "",
            },
        ],
    )
    return tmp_path


def _tool_call(
    *,
    call_id: str = "call_1",
    name: str = "get_brand_facts",
    arguments: str = '{"brand_id":"sample_brand"}',
) -> object:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(
    *,
    content: str | None = None,
    tool_calls: list[object] | None = None,
    finish_reason: str = "stop",
) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls or []),
                finish_reason=finish_reason,
            )
        ]
    )


class RecordingToolCallingClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def chat_completion(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | dict[str, object] | None = None,
    ) -> object:
        self.calls.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        if not self.responses:
            raise AssertionError("Unexpected model turn")
        return self.responses.pop(0)


def _classifications(*items: tuple[str, str, list[str]]) -> str:
    return json.dumps(
        {
            "classifications": [
                {"claim_id": claim_id, "label": label, "fact_ids": fact_ids}
                for claim_id, label, fact_ids in items
            ]
        }
    )


def _extract(assessment_data: Path) -> ClaimExtractionResult:
    return build_extraction_result(
        answer_id="A_TEST",
        ai_answer=ANSWER_TEXT,
        claim_texts=(
            "Bluetooth 5.3 is supported.",
            "Bluetooth 4.0 is supported.",
            "It has built-in GPS.",
        ),
    )


def _classifier(
    assessment_data: Path,
    responses: list[object],
    *,
    max_model_turns: int = 3,
) -> tuple[ClaimClassifier, RecordingToolCallingClient]:
    client = RecordingToolCallingClient(responses)
    return (
        ClaimClassifier(
            client,
            data_dir=assessment_data,
            max_model_turns=max_model_turns,
        ),
        client,
    )


def test_brand_facts_tool_returns_exact_requested_brand_only(
    assessment_data: Path,
) -> None:
    assert not (assessment_data / "development_gold.csv").exists()
    result = BrandFactsTool(assessment_data).get_brand_facts("sample_brand")

    assert result.status == "ok"
    assert [fact.fact_id for fact in result.facts] == ["sample.speaker.bluetooth"]
    assert result.facts[0].value == "5.3"


def test_brand_facts_tool_reports_unknown_brand(assessment_data: Path) -> None:
    result = BrandFactsTool(assessment_data).get_brand_facts("missing_brand")

    assert result.status == "unknown_brand"
    assert result.facts == ()


def test_brand_facts_tool_reports_empty_catalogue(tmp_path: Path) -> None:
    _write_csv(tmp_path / "brand_facts.csv", FACT_COLUMNS, [])

    result = BrandFactsTool(tmp_path).get_brand_facts("sample_brand")

    assert result.status == "empty_catalogue"
    assert result.facts == ()


def test_brand_facts_tool_rejects_invalid_brand_arguments(
    assessment_data: Path,
) -> None:
    with pytest.raises(InvalidToolArguments):
        BrandFactsTool(assessment_data).get_brand_facts("../secrets")


def test_only_registered_tool_can_execute(assessment_data: Path) -> None:
    with pytest.raises(InvalidToolArguments, match="Unregistered tool"):
        execute_registered_tool(
            tool_name="read_file",
            arguments_json='{"path":"private.txt"}',
            brand_facts_tool=BrandFactsTool(assessment_data),
        )


def test_tool_rejects_malformed_or_extra_arguments(assessment_data: Path) -> None:
    with pytest.raises(InvalidToolArguments, match="valid JSON"):
        execute_registered_tool(
            tool_name="get_brand_facts",
            arguments_json="{",
            brand_facts_tool=BrandFactsTool(assessment_data),
        )
    with pytest.raises(InvalidToolArguments, match="exactly"):
        execute_registered_tool(
            tool_name="get_brand_facts",
            arguments_json='{"brand_id":"sample_brand","path":"private.txt"}',
            brand_facts_tool=BrandFactsTool(assessment_data),
        )


def test_real_registered_tool_call_is_executed_and_result_returned_to_model(
    assessment_data: Path,
) -> None:
    final = _classifications(
        ("A_TEST:0:27", "supported", ["sample.speaker.bluetooth"]),
        ("A_TEST:28:55", "contradicted", ["sample.speaker.bluetooth"]),
        ("A_TEST:56:76", "unknown", []),
    )
    classifier, client = _classifier(
        assessment_data,
        [
            _response(tool_calls=[_tool_call()], finish_reason="tool_calls"),
            _response(content=final),
        ],
    )

    result = classifier.classify(
        "A_TEST",
        ANSWER_TEXT,
        _extract(assessment_data),
    )

    assert [claim.label for claim in result.claims] == [
        "supported",
        "contradicted",
        "unknown",
    ]
    assert result.claims[0].fact_ids == ("sample.speaker.bluetooth",)
    assert len(client.calls) == 2
    second_turn_messages = client.calls[1]["messages"]
    tool_message = second_turn_messages[-1]
    assert tool_message["role"] == "tool"
    assert "sample.speaker.bluetooth" in tool_message["content"]
    assert "other.speaker.weight" not in tool_message["content"]
    assert client.calls[0]["tools"][0]["function"]["name"] == "get_brand_facts"
    assert client.calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_brand_facts"},
    }
    assert client.calls[1]["tools"] is None
    assert client.calls[1]["tool_choice"] is None
    initial_user_message = client.calls[0]["messages"][1]["content"]
    assert isinstance(initial_user_message, str)
    assert '"brand_id": "sample_brand"' in initial_user_message
    assert '"question": "Test question"' in initial_user_message


def test_contradicted_claim_cites_actual_catalogue_fact(
    assessment_data: Path,
) -> None:
    final = _classifications(
        ("A_TEST:0:27", "supported", ["sample.speaker.bluetooth"]),
        ("A_TEST:28:55", "contradicted", ["sample.speaker.bluetooth"]),
        ("A_TEST:56:76", "unknown", []),
    )
    classifier, _ = _classifier(
        assessment_data,
        [
            _response(tool_calls=[_tool_call()], finish_reason="tool_calls"),
            _response(content=final),
        ],
    )

    result = classifier.classify(
        "A_TEST",
        ANSWER_TEXT,
        _extract(assessment_data),
    )

    assert result.claims[1].label == "contradicted"
    assert result.claims[1].fact_ids == ("sample.speaker.bluetooth",)


def test_unknown_tool_is_rejected(assessment_data: Path) -> None:
    classifier, _ = _classifier(
        assessment_data,
        [
            _response(
                tool_calls=[_tool_call(name="execute_code", arguments="{}")],
                finish_reason="tool_calls",
            )
        ],
    )

    with pytest.raises(ToolCallError, match="Unregistered tool"):
        classifier.classify(
            "A_TEST",
            ANSWER_TEXT,
            _extract(assessment_data),
        )


def test_tool_call_for_unmapped_brand_is_rejected(assessment_data: Path) -> None:
    classifier, _ = _classifier(
        assessment_data,
        [
            _response(
                tool_calls=[
                    _tool_call(arguments='{"brand_id":"other_brand"}')
                ],
                finish_reason="tool_calls",
            )
        ],
    )

    with pytest.raises(ToolCallError, match="other than the answer"):
        classifier.classify(
            "A_TEST",
            ANSWER_TEXT,
            _extract(assessment_data),
        )


def test_malformed_tool_call_response_is_rejected(assessment_data: Path) -> None:
    malformed_call = SimpleNamespace(id="", function=SimpleNamespace())
    classifier, _ = _classifier(
        assessment_data,
        [_response(tool_calls=[malformed_call], finish_reason="tool_calls")],
    )

    with pytest.raises(ToolCallError, match="malformed tool call"):
        classifier.classify("A_TEST", ANSWER_TEXT, _extract(assessment_data))


def test_repeated_tool_call_is_rejected(assessment_data: Path) -> None:
    call = _tool_call()
    classifier, _ = _classifier(
        assessment_data,
        [
            _response(tool_calls=[call], finish_reason="tool_calls"),
            _response(tool_calls=[call], finish_reason="tool_calls"),
        ],
    )

    with pytest.raises(ToolCallError, match="repeated"):
        classifier.classify(
            "A_TEST",
            ANSWER_TEXT,
            _extract(assessment_data),
        )


def test_model_turn_limit_is_enforced(assessment_data: Path) -> None:
    classifier, client = _classifier(
        assessment_data,
        [
            _response(tool_calls=[_tool_call()], finish_reason="tool_calls"),
            _response(tool_calls=[_tool_call(call_id="call_2")], finish_reason="tool_calls"),
        ],
        max_model_turns=2,
    )

    with pytest.raises(ToolCallError, match="maximum model turns"):
        classifier.classify(
            "A_TEST",
            ANSWER_TEXT,
            _extract(assessment_data),
        )
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "Final response"),
        (
            _classifications(
                ("A_TEST:0:27", "likely", ["sample.speaker.bluetooth"]),
                ("A_TEST:28:55", "contradicted", ["sample.speaker.bluetooth"]),
                ("A_TEST:56:76", "unknown", []),
            ),
            "Final response",
        ),
        (
            _classifications(
                ("A_TEST:0:27", "supported", ["not-a-real-fact"]),
                ("A_TEST:28:55", "contradicted", ["sample.speaker.bluetooth"]),
                ("A_TEST:56:76", "unknown", []),
            ),
            "unavailable fact ID",
        ),
        (
            _classifications(("A_TEST:0:27", "supported", [])),
            "exactly once",
        ),
    ],
)
def test_invalid_final_classification_is_rejected(
    assessment_data: Path, content: str, message: str
) -> None:
    classifier, _ = _classifier(
        assessment_data,
        [
            _response(tool_calls=[_tool_call()], finish_reason="tool_calls"),
            _response(content=content),
        ],
    )

    with pytest.raises(InvalidClassification, match=message):
        classifier.classify(
            "A_TEST",
            ANSWER_TEXT,
            _extract(assessment_data),
        )


def test_final_answer_without_tool_call_is_rejected(assessment_data: Path) -> None:
    classifier, _ = _classifier(
        assessment_data,
        [
            _response(
                content=_classifications(
                    ("A_TEST:0:27", "supported", ["sample.speaker.bluetooth"]),
                    ("A_TEST:28:55", "contradicted", ["sample.speaker.bluetooth"]),
                    ("A_TEST:56:76", "unknown", []),
                )
            )
        ],
    )

    with pytest.raises(InvalidClassification, match="without retrieving"):
        classifier.classify(
            "A_TEST",
            ANSWER_TEXT,
            _extract(assessment_data),
        )


def test_answer_mapping_must_match_original_text(assessment_data: Path) -> None:
    classifier, _ = _classifier(assessment_data, [])

    with pytest.raises(AnswerMappingError, match="does not exactly match"):
        classifier.classify(
            "A_TEST",
            "A rewritten answer.",
            _extract(assessment_data),
        )


def test_invalid_extracted_span_is_rejected_before_model_call(
    assessment_data: Path,
) -> None:
    classifier, client = _classifier(assessment_data, [])
    invalid_extraction = ClaimExtractionResult(
        answer_id="A_TEST",
        claims=(
            ExtractedClaim(
                answer_id="A_TEST",
                claim_id="A_TEST:0:27",
                start_char=0,
                end_char=27,
                claim_text="Bluetooth 4.0 is supported.",
            ),
        ),
    )

    with pytest.raises(ValueError, match="does not exactly match"):
        classifier.classify("A_TEST", ANSWER_TEXT, invalid_extraction)
    assert client.calls == []


def test_classification_preserves_claim_offsets_ids_and_text(
    assessment_data: Path,
) -> None:
    answer_text = ANSWER_TEXT
    extraction = _extract(assessment_data)
    classifier, _ = _classifier(
        assessment_data,
        [
            _response(tool_calls=[_tool_call()], finish_reason="tool_calls"),
            _response(
                content=_classifications(
                    ("A_TEST:0:27", "supported", ["sample.speaker.bluetooth"]),
                    ("A_TEST:28:55", "contradicted", ["sample.speaker.bluetooth"]),
                    ("A_TEST:56:76", "unknown", []),
                )
            ),
        ],
    )

    result = classifier.classify("A_TEST", answer_text, extraction)

    for claim in result.claims:
        assert answer_text[claim.start_char : claim.end_char] == claim.claim_text
        assert claim.claim_id == f"A_TEST:{claim.start_char}:{claim.end_char}"


def test_empty_extraction_returns_empty_without_model_call(
    assessment_data: Path,
) -> None:
    classifier, client = _classifier(assessment_data, [])

    result = classifier.classify(
        "A_TEST",
        ANSWER_TEXT,
        ClaimExtractionResult(answer_id="A_TEST", claims=()),
    )

    assert result.claims == ()
    assert client.calls == []


def test_optional_live_groq_tool_calling() -> None:
    import os

    if os.getenv("RUN_GROQ_TOOL_INTEGRATION") != "1":
        pytest.skip("Set RUN_GROQ_TOOL_INTEGRATION=1 to enable live tool integration")
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is required for live Groq tool integration")

    from app.agent.claim_classifier import create_default_classifier
    from app.data_loader import load_answers
    from app.schemas import ClaimExtractionResult, ExtractedClaim

    answer = next(item for item in load_answers() if item.answer_id == "A01")
    claim_text = (
        "For the current India catalogue, ClearTrail Mini is listed at INR 1,200 "
        "less than ClearTrail Plus, comparing the two tax-inclusive prices before shipping."
    )
    start_char = answer.ai_answer.find(claim_text)
    if start_char < 0:
        pytest.skip("The configured assessment data does not contain the integration sentence")
    end_char = start_char + len(claim_text)
    claim = ExtractedClaim(
        answer_id=answer.answer_id,
        claim_id=f"{answer.answer_id}:{start_char}:{end_char}",
        start_char=start_char,
        end_char=end_char,
        claim_text=claim_text,
    )
    result = create_default_classifier().classify(
        answer.answer_id,
        answer.ai_answer,
        ClaimExtractionResult(answer_id=answer.answer_id, claims=(claim,)),
    )
    assert len(result.claims) == 1
    assert result.claims[0].fact_ids
