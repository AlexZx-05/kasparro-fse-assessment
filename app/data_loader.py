"""CSV loading and validation for the assessment datasets."""

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, TypeVar

from app.schemas import (
    Answer,
    BrandFact,
    DevelopmentGoldClaim,
    FactValueType,
    InferenceData,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
VALID_SPLITS = frozenset({"development", "check"})
VALID_VALUE_TYPES = frozenset({"string", "number", "boolean"})
VALID_LABELS = frozenset({"supported", "contradicted", "unknown"})

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
GOLD_COLUMNS = (
    "answer_id",
    "claim_id",
    "start_char",
    "end_char",
    "quote",
    "expected_label",
    "evidence_fact_ids",
    "explanation",
    "difficulty_tags",
)


class DatasetError(ValueError):
    """Raised when a dataset file is missing or violates its CSV contract."""


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    answer_count: int
    fact_count: int
    development_gold_count: int


T = TypeVar("T")


def _read_csv(path: Path, required_columns: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise DatasetError(f"Required CSV file not found: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file, strict=True)
            headers = reader.fieldnames
            if headers is None:
                raise DatasetError(f"{path}: CSV file is empty or has no header row")
            if len(headers) != len(set(headers)):
                raise DatasetError(f"{path}: duplicate column names in CSV header")

            missing_columns = [column for column in required_columns if column not in headers]
            if missing_columns:
                missing = ", ".join(missing_columns)
                raise DatasetError(f"{path}: missing required column(s): {missing}")

            rows: list[dict[str, str]] = []
            try:
                for row in reader:
                    if None in row:
                        raise DatasetError(
                            f"{path}: row ending at line {reader.line_num} has more values "
                            "than the header"
                        )
                    missing_values = [key for key, value in row.items() if value is None]
                    if missing_values:
                        raise DatasetError(
                            f"{path}: row ending at line {reader.line_num} has fewer values "
                            "than the header"
                        )
                    rows.append(row)
            except csv.Error as error:
                raise DatasetError(
                    f"{path}: malformed CSV near line {reader.line_num}: {error}"
                ) from error
    except csv.Error as error:
        raise DatasetError(f"{path}: malformed CSV header: {error}") from error
    except UnicodeDecodeError as error:
        raise DatasetError(f"{path}: file is not valid UTF-8: {error}") from error
    except OSError as error:
        raise DatasetError(f"Could not read {path}: {error}") from error

    return rows


def _required_value(row: dict[str, str], column: str, path: Path) -> str:
    value = row[column]
    if value == "":
        raise DatasetError(f"{path}: required value {column!r} is empty")
    return value


def _build_rows(
    path: Path,
    required_columns: tuple[str, ...],
    builder: Callable[[dict[str, str], Path], T],
) -> list[T]:
    return [builder(row, path) for row in _read_csv(path, required_columns)]


def _build_answer(row: dict[str, str], path: Path) -> Answer:
    answer_id = _required_value(row, "answer_id", path)
    split = _required_value(row, "split", path)
    if split not in VALID_SPLITS:
        allowed = ", ".join(sorted(VALID_SPLITS))
        raise DatasetError(
            f"{path}: answer {answer_id!r} has invalid split {split!r}; "
            f"expected one of: {allowed}"
        )
    return Answer(
        answer_id=answer_id,
        split=split,
        brand_id=row["brand_id"],
        question=row["question"],
        ai_answer=row["ai_answer"],
    )


def _build_fact(row: dict[str, str], path: Path) -> BrandFact:
    fact_id = _required_value(row, "fact_id", path)
    value_type = _required_value(row, "value_type", path)
    if value_type not in VALID_VALUE_TYPES:
        allowed = ", ".join(sorted(VALID_VALUE_TYPES))
        raise DatasetError(
            f"{path}: fact {fact_id!r} has invalid value_type {value_type!r}; "
            f"expected one of: {allowed}"
        )

    value = row["value"]
    if value_type == "boolean" and value.strip().casefold() not in {"true", "false"}:
        raise DatasetError(
            f"{path}: boolean fact {fact_id!r} must have value 'true' or 'false'"
        )
    if value_type == "number":
        try:
            number = Decimal(value)
        except InvalidOperation as error:
            raise DatasetError(
                f"{path}: numeric fact {fact_id!r} has invalid value {value!r}"
            ) from error
        if not number.is_finite():
            raise DatasetError(
                f"{path}: numeric fact {fact_id!r} must be a finite number"
            )

    return BrandFact(
        fact_id=fact_id,
        brand_id=row["brand_id"],
        model_id=row["model_id"],
        field=row["field"],
        value=value,
        value_type=value_type,  # type: ignore[arg-type]
        unit=row["unit"],
        condition=row["condition"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
    )


def _build_gold(row: dict[str, str], path: Path) -> DevelopmentGoldClaim:
    answer_id = _required_value(row, "answer_id", path)
    claim_id = _required_value(row, "claim_id", path)
    label = _required_value(row, "expected_label", path)
    if label not in VALID_LABELS:
        allowed = ", ".join(sorted(VALID_LABELS))
        raise DatasetError(
            f"{path}: claim {claim_id!r} has invalid expected_label {label!r}; "
            f"expected one of: {allowed}"
        )
    return DevelopmentGoldClaim(
        answer_id=answer_id,
        claim_id=claim_id,
        start_char=row["start_char"],
        end_char=row["end_char"],
        quote=row["quote"],
        expected_label=label,  # type: ignore[arg-type]
        evidence_fact_ids=row["evidence_fact_ids"],
        explanation=row["explanation"],
        difficulty_tags=row["difficulty_tags"],
    )


def load_answers(data_dir: Path = DEFAULT_DATA_DIR) -> list[Answer]:
    """Load answer text without normalizing its whitespace or punctuation."""
    path = data_dir / "answers.csv"
    answers = _build_rows(path, ANSWER_COLUMNS, _build_answer)
    seen_ids: set[str] = set()
    for answer in answers:
        if answer.answer_id in seen_ids:
            raise DatasetError(f"{path}: duplicate answer_id {answer.answer_id!r}")
        seen_ids.add(answer.answer_id)
    return answers


def load_brand_facts(data_dir: Path = DEFAULT_DATA_DIR) -> list[BrandFact]:
    """Load catalogue facts while preserving each source value as text."""
    return _build_rows(data_dir / "brand_facts.csv", FACT_COLUMNS, _build_fact)


def load_development_gold(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> list[DevelopmentGoldClaim]:
    """Load development labels for validation only, never for inference."""
    return _build_rows(data_dir / "development_gold.csv", GOLD_COLUMNS, _build_gold)


def load_inference_data(data_dir: Path = DEFAULT_DATA_DIR) -> InferenceData:
    """Return only the answer and product-fact inputs available to inference."""
    return InferenceData(
        answers=tuple(load_answers(data_dir)),
        brand_facts=tuple(load_brand_facts(data_dir)),
    )


def validate_dataset(data_dir: Path = DEFAULT_DATA_DIR) -> DatasetSummary:
    """Validate all assessment CSV files without returning their contents."""
    answers = load_answers(data_dir)
    facts = load_brand_facts(data_dir)
    development_gold = load_development_gold(data_dir)
    answer_ids = {answer.answer_id for answer in answers}
    unknown_answer_ids = sorted(
        {claim.answer_id for claim in development_gold} - answer_ids
    )
    if unknown_answer_ids:
        ids = ", ".join(unknown_answer_ids)
        raise DatasetError(
            f"{data_dir / 'development_gold.csv'}: answer_id value(s) "
            f"not found in answers.csv: {ids}"
        )
    return DatasetSummary(
        answer_count=len(answers),
        fact_count=len(facts),
        development_gold_count=len(development_gold),
    )


def parse_boolean_value(fact: BrandFact) -> bool:
    """Interpret a boolean catalogue value explicitly; reject other fact types."""
    if fact.value_type != "boolean":
        raise ValueError(f"Fact {fact.fact_id!r} is not a boolean")
    value = fact.value.strip().casefold()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Fact {fact.fact_id!r} has invalid boolean value {fact.value!r}")


def main() -> None:
    summary = validate_dataset()
    print(
        "Dataset validation passed: "
        f"{summary.answer_count} answers, "
        f"{summary.fact_count} product facts, "
        f"{summary.development_gold_count} development gold claims."
    )


if __name__ == "__main__":
    main()
