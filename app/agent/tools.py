"""Explicitly registered, read-only catalogue tools for inference."""

from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.data_loader import DEFAULT_DATA_DIR, load_brand_facts
from app.schemas import BrandFact


_BRAND_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class BrandFactsToolError(ValueError):
    """Base class for safe catalogue tool errors."""


class InvalidToolArguments(BrandFactsToolError):
    """Raised when a model supplies invalid catalogue tool arguments."""


class CatalogueDataError(BrandFactsToolError):
    """Raised when catalogue data cannot be loaded."""


class GetBrandFactsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    brand_id: str = Field(min_length=1, max_length=100)


class BrandFactsResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["ok", "empty_catalogue", "unknown_brand"]
    brand_id: str
    facts: tuple[BrandFact, ...]


GET_BRAND_FACTS_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "get_brand_facts",
        "description": (
            "Return real product catalogue facts for exactly one brand ID. "
            "Call this before classifying claims."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "brand_id": {
                    "type": "string",
                    "description": "Exact brand_id supplied by the application.",
                }
            },
            "required": ["brand_id"],
            "additionalProperties": False,
        },
    },
}


class BrandFactsTool:
    """Load only brand facts; this tool has no development-gold access."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self._data_dir = data_dir

    def get_brand_facts(self, brand_id: str) -> BrandFactsResult:
        """Return catalogue facts for one validated brand ID."""
        try:
            arguments = GetBrandFactsArguments.model_validate({"brand_id": brand_id})
        except ValidationError as error:
            raise InvalidToolArguments("brand_id must be a valid non-empty string") from error
        if not _BRAND_ID_PATTERN.fullmatch(arguments.brand_id):
            raise InvalidToolArguments("brand_id contains unsupported characters")

        try:
            all_facts = load_brand_facts(self._data_dir)
        except (OSError, ValueError) as error:
            raise CatalogueDataError("Catalogue facts could not be loaded") from error

        if not all_facts:
            return BrandFactsResult(
                status="empty_catalogue",
                brand_id=arguments.brand_id,
                facts=(),
            )
        matching_facts = tuple(
            fact for fact in all_facts if fact.brand_id == arguments.brand_id
        )
        if not matching_facts:
            return BrandFactsResult(
                status="unknown_brand",
                brand_id=arguments.brand_id,
                facts=(),
            )
        return BrandFactsResult(
            status="ok",
            brand_id=arguments.brand_id,
            facts=matching_facts,
        )


def execute_registered_tool(
    *,
    tool_name: str,
    arguments_json: str,
    brand_facts_tool: BrandFactsTool,
) -> BrandFactsResult:
    """Execute only the registered tool after strict JSON argument validation."""
    import json

    if tool_name != "get_brand_facts":
        raise InvalidToolArguments(f"Unregistered tool requested: {tool_name!r}")
    try:
        parsed_arguments = json.loads(arguments_json)
    except json.JSONDecodeError as error:
        raise InvalidToolArguments("get_brand_facts arguments must be valid JSON") from error
    try:
        arguments = GetBrandFactsArguments.model_validate(parsed_arguments)
    except ValidationError as error:
        raise InvalidToolArguments(
            "get_brand_facts requires exactly one valid brand_id argument"
        ) from error
    return brand_facts_tool.get_brand_facts(arguments.brand_id)


def serialize_brand_facts_result(result: BrandFactsResult) -> str:
    """Serialize actual catalogue values for a model tool response."""
    return result.model_dump_json()


def get_brand_facts(
    brand_id: str,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> BrandFactsResult:
    """Return catalogue facts for one brand using the assessment CSV."""
    return BrandFactsTool(data_dir).get_brand_facts(brand_id)
