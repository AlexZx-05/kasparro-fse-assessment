"""Catalogue-grounded claim classification with a bounded model tool loop."""

import json
import logging
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from app.agent.openrouter_client import OpenRouterClient
from app.langfuse import tracked_tool
from app.agent.prompts import CLASSIFICATION_SYSTEM_PROMPT
from app.agent.tools import (
    GET_BRAND_FACTS_TOOL,
    BrandFactsTool,
    InvalidToolArguments,
    execute_registered_tool,
    serialize_brand_facts_result,
)
from app.agent.validators import ClaimValidationError, validate_claim_span
from app.data_loader import DEFAULT_DATA_DIR, DatasetError, load_answers
from app.schemas import (
    ClaimClassificationResult,
    ClaimExtractionResult,
    ClassifiedClaim,
    ModelClassificationResponse,
)


logger = logging.getLogger(__name__)
MAX_MODEL_TURNS = 3
GET_BRAND_FACTS_REQUIRED_TOOL_CHOICE: dict[str, object] = {
    "type": "function",
    "function": {"name": "get_brand_facts"},
}


class ToolCallingClient(Protocol):
    def chat_completion(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | dict[str, object] | None = None,
    ) -> object: ...


class ClassificationError(RuntimeError):
    """Base class for classification workflow errors."""


class AnswerMappingError(ClassificationError):
    """Raised when answer ID or supplied answer cannot map to one known brand."""


class ToolCallError(ClassificationError):
    """Raised for invalid, unknown, repeated, or excessive tool requests."""


class InvalidClassification(ClassificationError):
    """Raised when model classifications fail schema or evidence validation."""


class ClaimClassifier:
    """Classify extracted claims using catalogue facts retrieved via model tools."""

    def __init__(
        self,
        model_client: ToolCallingClient,
        *,
        data_dir: Path = DEFAULT_DATA_DIR,
        brand_facts_tool: BrandFactsTool | None = None,
        max_model_turns: int = MAX_MODEL_TURNS,
        system_prompt: str = CLASSIFICATION_SYSTEM_PROMPT,
    ) -> None:
        if not 1 <= max_model_turns <= MAX_MODEL_TURNS:
            raise ValueError(f"max_model_turns must be between 1 and {MAX_MODEL_TURNS}")
        self._model_client = model_client
        self._data_dir = data_dir
        self._brand_facts_tool = brand_facts_tool or BrandFactsTool(data_dir)
        self._max_model_turns = max_model_turns
        self._system_prompt = system_prompt

    def classify(
        self,
        answer_id: str,
        ai_answer: str,
        extraction: ClaimExtractionResult,
    ) -> ClaimClassificationResult:
        """Classify each Phase 2 claim against facts for its actual answer brand."""
        if extraction.answer_id != answer_id:
            raise ClaimValidationError(
                "Extraction answer_id does not match classifier answer_id"
            )
        try:
            answers = load_answers(self._data_dir)
        except DatasetError as error:
            raise AnswerMappingError("Assessment answers could not be loaded") from error
        matching_answers = [answer for answer in answers if answer.answer_id == answer_id]
        if len(matching_answers) != 1:
            raise AnswerMappingError(
                f"answer_id {answer_id!r} does not map to exactly one assessment answer"
            )
        answer = matching_answers[0]
        if answer.ai_answer != ai_answer:
            raise AnswerMappingError(
                "Provided ai_answer does not exactly match the assessment answer"
            )

        for claim in extraction.claims:
            if claim.answer_id != answer_id:
                raise ClaimValidationError(
                    "Extracted claim answer_id does not match classifier answer_id"
                )
            validate_claim_span(
                answer_id=answer_id,
                ai_answer=ai_answer,
                start_char=claim.start_char,
                end_char=claim.end_char,
                claim_text=claim.claim_text,
            )
            expected_claim_id = (
                f"{answer_id}:{claim.start_char}:{claim.end_char}"
            )
            if claim.claim_id != expected_claim_id:
                raise ClaimValidationError("Extracted claim_id is inconsistent")
        if not extraction.claims:
            return ClaimClassificationResult(answer_id=answer_id, claims=())

        messages: list[dict[str, object]] = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "answer_id": answer_id,
                        "brand_id": answer.brand_id,
                        "question": answer.question,
                        "answer_text": ai_answer,
                        "claims": [
                            {
                                "claim_id": claim.claim_id,
                                "claim_text": claim.claim_text,
                            }
                            for claim in extraction.claims
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        called_brand_facts = False
        seen_tool_calls: set[tuple[str, str]] = set()
        available_facts: dict[str, object] | None = None

        for turn in range(1, self._max_model_turns + 1):
            # Force the sole registered function on the first turn. Once facts
            # are available, omit tools so the final turn returns classification JSON.
            tools = [GET_BRAND_FACTS_TOOL] if not called_brand_facts else None
            tool_choice = (
                GET_BRAND_FACTS_REQUIRED_TOOL_CHOICE
                if not called_brand_facts
                else None
            )
            response = self._model_client.chat_completion(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            message, finish_reason = _get_assistant_message(response)
            tool_calls = _get_field(message, "tool_calls") or []
            if tool_calls:
                if turn == self._max_model_turns:
                    raise ToolCallError(
                        f"Model requested a tool on turn {turn}; maximum model turns "
                        f"is {self._max_model_turns}"
                    )
                assistant_call_messages: list[dict[str, object]] = []
                tool_result_messages: list[dict[str, object]] = []
                seen_call_ids: set[str] = set()
                for tool_call in tool_calls:
                    call_id, tool_name, arguments_json = _parse_tool_call(tool_call)
                    if call_id in seen_call_ids:
                        raise ToolCallError("Model returned duplicate tool call IDs")
                    seen_call_ids.add(call_id)
                    if tool_name != "get_brand_facts":
                        raise ToolCallError(f"Unregistered tool requested: {tool_name!r}")
                    try:
                        arguments = json.loads(arguments_json)
                    except json.JSONDecodeError as error:
                        raise ToolCallError(
                            "get_brand_facts arguments must be valid JSON"
                        ) from error
                    if not isinstance(arguments, dict):
                        raise ToolCallError("get_brand_facts arguments must be a JSON object")
                    signature = (
                        tool_name,
                        json.dumps(arguments, sort_keys=True, separators=(",", ":")),
                    )
                    if signature in seen_tool_calls:
                        raise ToolCallError("Model repeated a catalogue tool call")
                    seen_tool_calls.add(signature)
                    requested_brand = arguments.get("brand_id")
                    if requested_brand != answer.brand_id:
                        raise ToolCallError(
                            "Model requested facts for a brand other than the answer's "
                            "mapped brand"
                        )
                    try:
                        with tracked_tool(
                            name="get-brand-facts",
                            metadata={
                                "tool_name": tool_name,
                                "brand_id": requested_brand,
                            },
                            input_data={"brand_id": requested_brand},
                        ) as observation:
                            result = execute_registered_tool(
                                tool_name=tool_name,
                                arguments_json=arguments_json,
                                brand_facts_tool=self._brand_facts_tool,
                            )
                            observation.update(
                                output={
                                    "status": result.status,
                                    "fact_count": len(result.facts),
                                }
                            )
                    except InvalidToolArguments as error:
                        raise ToolCallError(str(error)) from error
                    called_brand_facts = True
                    available_facts = result.model_dump(mode="json")
                    assistant_call_messages.append(_assistant_tool_call_message(tool_call))
                    tool_result_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name,
                            "content": serialize_brand_facts_result(result),
                        }
                    )
                messages.append(
                    _join_assistant_tool_calls(message, assistant_call_messages)
                )
                messages.extend(tool_result_messages)
                continue

            if finish_reason == "tool_calls":
                raise ToolCallError("Model marked tool calls but returned none")
            if not called_brand_facts or available_facts is None:
                raise InvalidClassification(
                    "Model returned classifications without retrieving catalogue facts"
                )
            content = _get_field(message, "content")
            if not isinstance(content, str) or not content:
                raise InvalidClassification("Model returned an empty classification response")
            return self._validate_classification_response(
                answer_id=answer_id,
                ai_answer=ai_answer,
                extraction=extraction,
                content=content,
                available_facts=available_facts,
            )

        raise ToolCallError(
            f"Model did not finish within {self._max_model_turns} model turns"
        )

    @staticmethod
    def _validate_classification_response(
        *,
        answer_id: str,
        ai_answer: str,
        extraction: ClaimExtractionResult,
        content: str,
        available_facts: dict[str, object],
    ) -> ClaimClassificationResult:
        try:
            response = ModelClassificationResponse.model_validate_json(content)
        except (ValidationError, ValueError) as error:
            logger.warning("Model returned invalid claim classifications")
            raise InvalidClassification(
                "Final response must be JSON containing valid classifications"
            ) from error

        facts = available_facts.get("facts")
        if not isinstance(facts, list):
            raise InvalidClassification("Catalogue tool returned malformed facts")
        valid_fact_ids = {
            fact.get("fact_id")
            for fact in facts
            if isinstance(fact, dict) and isinstance(fact.get("fact_id"), str)
        }
        expected_claims = {claim.claim_id: claim for claim in extraction.claims}
        if len(expected_claims) != len(extraction.claims):
            raise ClaimValidationError("Extraction contains duplicate claim IDs")
        received_claim_ids = [item.claim_id for item in response.classifications]
        if (
            len(received_claim_ids) != len(set(received_claim_ids))
            or set(received_claim_ids) != set(expected_claims)
        ):
            raise InvalidClassification(
                "Response must classify every extracted claim exactly once"
            )

        classifications_by_id = {
            item.claim_id: item for item in response.classifications
        }
        classified_claims: list[ClassifiedClaim] = []
        for claim in extraction.claims:
            classification = classifications_by_id[claim.claim_id]
            if not classification.fact_ids and classification.label in {
                "supported",
                "contradicted",
            }:
                raise InvalidClassification(
                    f"{classification.label} claims require catalogue fact IDs"
                )
            invalid_fact_ids = sorted(set(classification.fact_ids) - valid_fact_ids)
            if invalid_fact_ids:
                raise InvalidClassification(
                    f"Classification references unavailable fact ID(s): "
                    f"{', '.join(invalid_fact_ids)}"
                )
            if len(classification.fact_ids) != len(set(classification.fact_ids)):
                raise InvalidClassification("Classification contains duplicate fact IDs")
            validated_claim = validate_claim_span(
                answer_id=answer_id,
                ai_answer=ai_answer,
                start_char=claim.start_char,
                end_char=claim.end_char,
                claim_text=claim.claim_text,
            )
            if validated_claim.claim_id != claim.claim_id:
                raise ClaimValidationError("Extracted claim_id is inconsistent")
            classified_claims.append(
                ClassifiedClaim(
                    **claim.model_dump(),
                    label=classification.label,
                    fact_ids=classification.fact_ids,
                )
            )
        return ClaimClassificationResult(
            answer_id=answer_id,
            claims=tuple(classified_claims),
        )


def create_default_classifier(
    *,
    system_prompt: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> ClaimClassifier:
    """Create an OpenRouter-backed classifier with a frozen prompt version."""
    return ClaimClassifier(
        OpenRouterClient(),
        data_dir=data_dir,
        system_prompt=system_prompt or CLASSIFICATION_SYSTEM_PROMPT,
    )


def _get_field(value: object, field: str) -> object | None:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _get_assistant_message(response: object) -> tuple[object, str | None]:
    choices = _get_field(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        raise InvalidClassification("Model returned an unexpected tool-call response")
    choice = choices[0]
    message = _get_field(choice, "message")
    if message is None:
        raise InvalidClassification("Model response did not contain an assistant message")
    finish_reason = _get_field(choice, "finish_reason")
    return message, finish_reason if isinstance(finish_reason, str) else None


def _parse_tool_call(tool_call: object) -> tuple[str, str, str]:
    call_id = _get_field(tool_call, "id")
    function = _get_field(tool_call, "function")
    tool_name = _get_field(function, "name") if function is not None else None
    arguments = _get_field(function, "arguments") if function is not None else None
    if (
        not isinstance(call_id, str)
        or not call_id
        or not isinstance(tool_name, str)
        or not isinstance(arguments, str)
    ):
        raise ToolCallError("Model returned a malformed tool call")
    return call_id, tool_name, arguments


def _assistant_tool_call_message(
    tool_call: object,
) -> dict[str, object]:
    call_id, tool_name, arguments = _parse_tool_call(tool_call)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": tool_name, "arguments": arguments},
    }


def _join_assistant_tool_calls(
    assistant_message: object,
    tool_calls: list[dict[str, object]],
) -> dict[str, object]:
    content = _get_field(assistant_message, "content")
    return {
        "role": "assistant",
        "content": content if isinstance(content, str) else None,
        "tool_calls": tool_calls,
    }
