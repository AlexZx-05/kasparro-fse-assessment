"""Frozen, file-backed agent configurations for comparable evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal


AgentVersion = Literal["v1", "v2"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"


class VersionConfigurationError(ValueError):
    """Raised when an evaluation version has not been frozen."""


@dataclass(frozen=True, slots=True)
class FrozenVersion:
    """Prompt files and the documented one-change history for an agent version."""

    version: AgentVersion
    extraction_prompt: str
    classification_prompt: str
    extraction_prompt_path: Path
    classification_prompt_path: Path
    targeted_change: dict[str, str]

    @property
    def extraction_prompt_sha256(self) -> str:
        return _sha256_text(self.extraction_prompt)

    @property
    def classification_prompt_sha256(self) -> str:
        return _sha256_text(self.classification_prompt)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "extraction_prompt": {
                "path": str(self.extraction_prompt_path.relative_to(PROJECT_ROOT)),
                "sha256": self.extraction_prompt_sha256,
            },
            "classification_prompt": {
                "path": str(self.classification_prompt_path.relative_to(PROJECT_ROOT)),
                "sha256": self.classification_prompt_sha256,
            },
            "targeted_change": dict(self.targeted_change),
        }


def load_frozen_version(version: AgentVersion) -> FrozenVersion:
    """Load exactly the prompt text recorded for a named, frozen version."""
    filenames = {
        "v1": ("v1_extraction.txt", "v1_classification.txt"),
        "v2": ("v2_extraction.txt", "v2_classification.txt"),
    }
    try:
        extraction_name, classification_name = filenames[version]
    except KeyError as error:
        raise VersionConfigurationError(f"Unknown agent version {version!r}") from error
    extraction_path = PROMPTS_DIR / extraction_name
    classification_path = PROMPTS_DIR / classification_name
    if not extraction_path.is_file() or not classification_path.is_file():
        raise VersionConfigurationError(
            f"Version {version} has not been frozen: expected prompt files "
            f"{extraction_name} and {classification_name}"
        )
    extraction_prompt = extraction_path.read_text(encoding="utf-8")
    classification_prompt = classification_path.read_text(encoding="utf-8")
    if not extraction_prompt.strip() or not classification_prompt.strip():
        raise VersionConfigurationError(f"Version {version} has an empty prompt file")
    return FrozenVersion(
        version=version,
        extraction_prompt=extraction_prompt,
        classification_prompt=classification_prompt,
        extraction_prompt_path=extraction_path,
        classification_prompt_path=classification_path,
        targeted_change=_targeted_change(version),
    )


def _targeted_change(version: AgentVersion) -> dict[str, str]:
    if version == "v1":
        return {
            "status": "baseline",
            "component": "extraction_and_classification",
            "change": "Frozen baseline configuration before a targeted improvement.",
            "rationale": "Version 1 establishes the unmodified prompt and model configuration used for the first measured run.",
            "potential_side_effect": "This baseline may still over-include sentence fragments or weak evidence decisions.",
        }
    return {
        "status": "candidate",
        "component": "extraction_boundary_filter",
        "change": "Tighten extraction to standalone factual sentences and require a direct evidence gate before supported or contradicted labels.",
        "rationale": "The targeted improvement is intended to reduce sentence-fragment extraction errors and false-positive supported/contradicted labels from weak or partial evidence.",
        "potential_side_effect": "The stricter gate may increase unknown classifications when the catalogue evidence is partial but still informative.",
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
