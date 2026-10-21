"""Typed records used by the Phase 1 dataset loader."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ClaimLabel = Literal["supported", "contradicted", "unknown"]
FactValueType = Literal["string", "number", "boolean"]


@dataclass(frozen=True, slots=True)
class Answer:
    answer_id: str
    split: str
    brand_id: str
    question: str
    ai_answer: str


@dataclass(frozen=True, slots=True)
class BrandFact:
    fact_id: str
    brand_id: str
    model_id: str
    field: str
    value: str
    value_type: FactValueType
    unit: str
    condition: str
    valid_from: str
    valid_to: str


@dataclass(frozen=True, slots=True)
class DevelopmentGoldClaim:
    answer_id: str
    claim_id: str
    start_char: str
    end_char: str
    quote: str
    expected_label: ClaimLabel
    evidence_fact_ids: str
    explanation: str
    difficulty_tags: str


@dataclass(frozen=True, slots=True)
class InferenceData:
    """The inference-facing data view intentionally excludes gold labels."""

    answers: tuple[Answer, ...]
    brand_facts: tuple[BrandFact, ...]


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    claim_id: str
    answer_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    claim_text: str = Field(min_length=1)


class ClaimExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    answer_id: str
    claims: tuple[ExtractedClaim, ...]


class ModelClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    claim_text: str = Field(min_length=1)


class ModelExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    claims: tuple[ModelClaim, ...]


class ModelClaimClassification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    claim_id: str
    label: ClaimLabel
    fact_ids: tuple[str, ...]


class ModelClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    classifications: tuple[ModelClaimClassification, ...]


class ClassifiedClaim(ExtractedClaim):
    label: ClaimLabel
    fact_ids: tuple[str, ...]


class ClaimClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    answer_id: str
    claims: tuple[ClassifiedClaim, ...]


class ClassifyAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    answer_id: str = Field(min_length=1)
    ai_answer: str = Field(min_length=1)
    brand_id: str | None = None


class ClaimResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    claim_id: str
    start_char: int
    end_char: int
    claim_text: str
    label: ClaimLabel
    fact_ids: tuple[str, ...]


class ClassifyAnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    answer_id: str
    claims: list[ClaimResultResponse]


class CreateEvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal["v1", "v2"]
    dataset_split: Literal["development", "check"] = "development"


class EvaluationRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_id: int
    version: str
    dataset_split: str
    status: str


class AnswerResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int
    answer_id: str
    status: str
    predicted_claim_count: int
    error_message: str | None = None
    created_at: str


class ClaimDatabaseResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int
    claim_id: str
    start_char: int
    end_char: int
    claim_text: str
    label: ClaimLabel
    fact_ids: list[str]
    created_at: str


class EvaluationRunDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int
    version: str
    dataset_split: str
    status: str
    total_answers: int
    successful_answers: int
    failed_answers: int
    metrics_json: dict[str, object] | None = None
    error_summary: dict[str, object] | None = None
    created_at: str
    completed_at: str | None = None
    answer_results: list[AnswerResultResponse] = []
    claim_results: list[ClaimDatabaseResultResponse] = []
