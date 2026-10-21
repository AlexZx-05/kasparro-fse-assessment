"""Claim extraction workflow independent of the model provider."""

import json
import logging
from typing import Protocol

from pydantic import ValidationError

from app.agent.prompts import build_extraction_user_prompt
from app.agent.validators import (
    ClaimValidationError,
    build_extraction_result,
)
from app.schemas import ClaimExtractionResult, ModelExtractionResponse


logger = logging.getLogger(__name__)


class CompletionClient(Protocol):
    def complete(self, user_prompt: str) -> str: ...


class MalformedModelResponse(ValueError):
    """Raised when the model response is not valid extraction JSON."""


class ClaimExtractor:
    def __init__(self, model_client: CompletionClient) -> None:
        self._model_client = model_client

    def extract(self, answer_id: str, ai_answer: str) -> ClaimExtractionResult:
        """Extract factual sentence claims and validate all offsets locally."""
        if not isinstance(answer_id, str) or not answer_id:
            raise ValueError("answer_id must be a non-empty string")
        if not isinstance(ai_answer, str):
            raise TypeError("ai_answer must be a string")
        if not ai_answer.strip():
            return ClaimExtractionResult(answer_id=answer_id, claims=())

        raw_response = self._model_client.complete(
            build_extraction_user_prompt(answer_id, ai_answer)
        )
        try:
            parsed = ModelExtractionResponse.model_validate_json(raw_response)
        except (ValidationError, ValueError) as error:
            logger.warning("Model returned malformed claim extraction JSON")
            raise MalformedModelResponse(
                "Model response must be JSON with a claims array of exact claim_text values"
            ) from error

        try:
            return build_extraction_result(
                answer_id=answer_id,
                ai_answer=ai_answer,
                claim_texts=tuple(claim.claim_text for claim in parsed.claims),
            )
        except ClaimValidationError:
            logger.warning("Model claims did not validate against the original answer")
            raise


def create_default_extractor(
    *,
    system_prompt: str | None = None,
) -> ClaimExtractor:
    """Create a Groq-backed extractor with an optional frozen prompt version."""
    from app.agent.groq_client import GroqClient
    from app.agent.prompts import EXTRACTION_SYSTEM_PROMPT

    return ClaimExtractor(
        GroqClient(extraction_system_prompt=system_prompt or EXTRACTION_SYSTEM_PROMPT)
    )
