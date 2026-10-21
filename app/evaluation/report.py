"""JSON and Markdown artifact writers for reproducible evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.metrics import EvaluationMetrics


class ReportExistsError(FileExistsError):
    """Raised when a run would overwrite an existing evidence artifact."""


RUN_FILENAMES = (
    "development_predictions.json",
    "development_metrics.json",
    "development_report.md",
    "error_analysis.json",
    "configuration.json",
)


def ensure_run_destination(
    reports_root: Path,
    version: str,
    split: str,
    *,
    overwrite: bool,
) -> Path:
    """Check the version-specific report directory before a billable run starts."""
    directory = reports_root / version
    filenames = tuple(filename.replace("development", split) for filename in RUN_FILENAMES)
    existing = [directory / filename for filename in filenames if (directory / filename).exists()]
    if existing and not overwrite:
        listed = ", ".join(str(path) for path in existing)
        raise ReportExistsError(
            f"Refusing to overwrite existing {version} {split} artifacts: {listed}. "
            "Use --overwrite only for an intentional replacement."
        )
    return directory


def ensure_phase7_report_destination(path: Path, *, overwrite: bool) -> None:
    """Reject an existing Phase 7 summary before starting a billable run."""
    if path.exists() and not overwrite:
        raise ReportExistsError(
            f"Refusing to overwrite existing evaluation report {path}. "
            "Use --overwrite only for an intentional replacement."
        )


def write_phase7_report(
    *,
    path: Path,
    run_id: str | int,
    version: str,
    dataset_split: str,
    timestamp: str,
    model: str,
    configuration: dict[str, object],
    outcomes: list[dict[str, object]],
    metrics: dict[str, object] | None,
    runtime_seconds: float,
    overwrite: bool = False,
) -> Path:
    """Write one actual run summary, using null metrics when gold is unavailable."""
    ensure_phase7_report_destination(path, overwrite=overwrite)
    successful_answers = sum(
        outcome.get("status") == "success" for outcome in outcomes
    )
    payload: dict[str, object] = {
        "run_id": run_id,
        "version": version,
        "dataset_split": dataset_split,
        "timestamp": timestamp,
        "model": model,
        "answers_processed": len(outcomes),
        "successful_answers": successful_answers,
        "operational_failures": len(outcomes) - successful_answers,
        "runtime_seconds": runtime_seconds,
        "actual_cost": None,
        "actual_cost_unavailable_reason": (
            "The provider adapter does not expose billed cost; no estimate was made."
        ),
        "metrics": metrics,
        "metrics_unavailable_reason": (
            "No authorized final-check gold labels are available."
            if metrics is None and dataset_split == "check"
            else None
        ),
        "configuration": configuration,
        "outcomes": outcomes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    return path


def export_phase7_development_report(
    *,
    reports_root: Path,
    run_id: int,
    version: str,
    timestamp: str,
    runtime_seconds: float,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Export a completed API run's actual saved predictions and metrics."""
    run_directory = reports_root / "api_runs" / str(run_id) / version
    metrics_payload = _read_json(run_directory / "development_metrics.json")
    predictions_payload = _read_json(run_directory / "development_predictions.json")
    run_metadata = metrics_payload.get("run")
    outcomes = predictions_payload.get("outcomes")
    metrics = metrics_payload.get("metrics")
    if (
        not isinstance(run_metadata, dict)
        or run_metadata.get("version") != version
        or run_metadata.get("split") != "development"
        or not isinstance(run_metadata.get("model"), str)
        or not isinstance(run_metadata.get("configuration"), dict)
        or not isinstance(outcomes, list)
        or not all(isinstance(outcome, dict) for outcome in outcomes)
        or not isinstance(metrics, dict)
    ):
        raise ValueError(
            f"Run {run_id} does not contain a valid completed development report."
        )
    return write_phase7_report(
        path=output_path,
        run_id=run_id,
        version=version,
        dataset_split="development",
        timestamp=timestamp,
        model=run_metadata["model"],
        configuration=run_metadata["configuration"],
        outcomes=outcomes,
        metrics=metrics,
        runtime_seconds=runtime_seconds,
        overwrite=overwrite,
    )


def write_development_reports(
    *,
    reports_root: Path,
    version: str,
    run_metadata: dict[str, object],
    outcomes: list[dict[str, object]],
    scheduled_answer_ids: list[str],
    metrics: EvaluationMetrics,
    gold_validation: dict[str, object],
    error_analysis: dict[str, object],
) -> dict[str, Path]:
    """Write all required development artifacts from actual completed outcomes."""
    directory = reports_root / version
    directory.mkdir(parents=True, exist_ok=True)
    predictions_path = directory / "development_predictions.json"
    metrics_path = directory / "development_metrics.json"
    analysis_path = directory / "error_analysis.json"
    markdown_path = directory / "development_report.md"
    configuration_path = directory / "configuration.json"

    prediction_payload = {
        "run": run_metadata,
        "scheduled_answer_ids": scheduled_answer_ids,
        "outcomes": outcomes,
    }
    metrics_payload = {
        "run": run_metadata,
        "scheduled_answer_ids": scheduled_answer_ids,
        "gold_validation": gold_validation,
        "metrics": metrics.as_dict(),
    }
    analysis_payload = {
        "run": run_metadata,
        "error_analysis": error_analysis,
    }
    configuration_payload = {
        "version": version,
        "run_configuration": run_metadata["configuration"],
        "targeted_change": run_metadata["targeted_change"],
    }
    _write_json(predictions_path, prediction_payload)
    _write_json(metrics_path, metrics_payload)
    _write_json(analysis_path, analysis_payload)
    _write_json(configuration_path, configuration_payload)
    markdown_path.write_text(
        _development_markdown(metrics_payload, error_analysis), encoding="utf-8"
    )
    return {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "report": markdown_path,
        "analysis": analysis_path,
        "configuration": configuration_path,
    }


def write_v1_v2_comparison(
    reports_root: Path,
    *,
    v1_reports_root: Path | None = None,
) -> dict[str, Path]:
    """Compare saved development reports without rerunning inference."""
    baseline_root = v1_reports_root or reports_root
    v1 = _read_json(baseline_root / "v1" / "development_metrics.json")
    v2 = _read_json(reports_root / "v2" / "development_metrics.json")
    v1_ids = v1.get("scheduled_answer_ids")
    v2_ids = v2.get("scheduled_answer_ids")
    if not isinstance(v1_ids, list) or not isinstance(v2_ids, list) or v1_ids != v2_ids:
        raise ValueError("V1 and V2 reports do not use the same scheduled answers")
    if v1.get("gold_validation", {}).get("gold_claim_count") != v2.get(
        "gold_validation", {}
    ).get("gold_claim_count"):
        raise ValueError("V1 and V2 reports do not use the same gold-claim count")

    comparisons = {}
    paths = {
        "extraction_precision": ("extraction", "precision"),
        "extraction_recall": ("extraction", "recall"),
        "extraction_f1": ("extraction", "f1"),
        "joint_precision": ("joint", "precision"),
        "joint_recall": ("joint", "recall"),
        "joint_f1": ("joint", "f1"),
        "macro_f1": ("macro_f1",),
        "coverage_rate": ("coverage", "rate"),
    }
    for name, path in paths.items():
        before = _nested_number(v1["metrics"], path)
        after = _nested_number(v2["metrics"], path)
        comparisons[name] = {
            "v1": before,
            "v2": after,
            "delta": after - before,
            "change": _change_word(after - before),
        }
    count_names = ("missing_claims", "extra_claims", "wrong_label_claims")
    count_comparisons = {}
    for name in count_names:
        before = _nested_number(v1["metrics"], ("counts", name))
        after = _nested_number(v2["metrics"], ("counts", name))
        count_comparisons[name] = {
            "v1": before,
            "v2": after,
            "delta": after - before,
            "change": _change_word(after - before),
        }
    coverage_counts = {}
    for name in ("answers_successfully_processed", "answer_level_errors"):
        before = _nested_number(v1["metrics"], ("coverage", name))
        after = _nested_number(v2["metrics"], ("coverage", name))
        coverage_counts[name] = {
            "v1": before,
            "v2": after,
            "delta": after - before,
            "change": _change_word(after - before),
        }

    payload = {
        "versions": ["v1", "v2"],
        "split": "development",
        "scheduled_answer_ids": v1_ids,
        "gold_claim_count": v1["gold_validation"]["gold_claim_count"],
        "metrics": comparisons,
        "error_counts": count_comparisons,
        "coverage_counts": coverage_counts,
    }
    directory = reports_root / "comparison"
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "v1_vs_v2.json"
    markdown_path = directory / "v1_vs_v2.md"
    _write_json(json_path, payload)
    markdown_path.write_text(_comparison_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _development_markdown(
    metrics_payload: dict[str, object], error_analysis: dict[str, object]
) -> str:
    run = metrics_payload["run"]
    metrics = metrics_payload["metrics"]
    assert isinstance(run, dict)
    assert isinstance(metrics, dict)
    extraction = metrics["extraction"]
    joint = metrics["joint"]
    coverage = metrics["coverage"]
    counts = metrics["counts"]
    assert isinstance(extraction, dict)
    assert isinstance(joint, dict)
    assert isinstance(coverage, dict)
    assert isinstance(counts, dict)
    targeted = run["targeted_change"]
    assert isinstance(targeted, dict)
    top_pattern = error_analysis.get("top_measured_pattern")
    return "\n".join(
        [
            f"# {run['version']} development evaluation",
            "",
            f"- Timestamp: {run['timestamp']}",
            f"- Model: {run['model']}",
            f"- Dataset split: {run['split']}",
            f"- Answers scheduled: {coverage['answers_attempted']}",
            f"- Gold claims: {counts['gold_claims']}",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Extraction precision | {_format(extraction['precision'])} |",
            f"| Extraction recall | {_format(extraction['recall'])} |",
            f"| Extraction F1 | {_format(extraction['f1'])} |",
            f"| Joint precision | {_format(joint['precision'])} |",
            f"| Joint recall | {_format(joint['recall'])} |",
            f"| Joint F1 | {_format(joint['f1'])} |",
            f"| Macro F1 | {_format(metrics['macro_f1'])} |",
            f"| Coverage | {_format(coverage['rate'])} |",
            "",
            "## Error counts",
            "",
            f"- Missing claims: {counts['missing_claims']}",
            f"- Extra claims: {counts['extra_claims']}",
            f"- Wrong-label claims: {counts['wrong_label_claims']}",
            f"- Answer-level errors: {coverage['answer_level_errors']}",
            "",
            "## Frozen configuration",
            "",
            f"- Change status: {targeted['status']}",
            f"- Component: {targeted['component']}",
            f"- Change: {targeted['change']}",
            f"- Rationale: {targeted['rationale']}",
            f"- Potential side effect: {targeted['potential_side_effect']}",
            "",
            "## Measured error pattern",
            "",
            "No errors were observed." if top_pattern is None else (
                f"Top measured pattern: {top_pattern['category']} "
                f"({top_pattern['count']})."
            ),
            "",
            "Detailed predictions, metrics, and unmodified error records are saved beside this report.",
            "",
        ]
    )


def _comparison_markdown(payload: dict[str, object]) -> str:
    metrics = payload["metrics"]
    errors = payload["error_counts"]
    coverage = payload["coverage_counts"]
    assert isinstance(metrics, dict)
    assert isinstance(errors, dict)
    assert isinstance(coverage, dict)
    rows = [
        "# V1 versus V2 development comparison",
        "",
        f"Both reports scored the same {len(payload['scheduled_answer_ids'])} development answers and {payload['gold_claim_count']} gold claims.",
        "",
        "## Metrics",
        "",
        "| Metric | V1 | V2 | Delta | Change |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, values in metrics.items():
        assert isinstance(values, dict)
        rows.append(
            f"| {name} | {_format(values['v1'])} | {_format(values['v2'])} | "
            f"{_format(values['delta'])} | {values['change']} |"
        )
    rows.extend(["", "## Error counts", "", "| Count | V1 | V2 | Delta | Change |", "| --- | ---: | ---: | ---: | --- |"])
    for name, values in errors.items():
        assert isinstance(values, dict)
        rows.append(
            f"| {name} | {values['v1']} | {values['v2']} | {values['delta']} | {values['change']} |"
        )
    rows.extend(["", "## Answer processing", "", "| Count | V1 | V2 | Delta | Change |", "| --- | ---: | ---: | ---: | --- |"])
    for name, values in coverage.items():
        assert isinstance(values, dict)
        rows.append(
            f"| {name} | {values['v1']} | {values['v2']} | {values['delta']} | {values['change']} |"
        )
    return "\n".join(rows) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required comparison artifact not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Report {path} is not a JSON object")
    return value


def _nested_number(value: object, path: tuple[str, ...]) -> float:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Report is missing metric {'.'.join(path)}")
        current = current[key]
    if not isinstance(current, (int, float)):
        raise ValueError(f"Report metric {'.'.join(path)} is not numeric")
    return float(current)


def _change_word(delta: float) -> str:
    if delta > 0:
        return "improved"
    if delta < 0:
        return "declined"
    return "unchanged"


def _format(value: object) -> str:
    return f"{float(value):.6f}" if isinstance(value, (int, float)) else str(value)
