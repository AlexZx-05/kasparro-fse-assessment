from contextlib import contextmanager
from typing import Any

import pytest

from app.langfuse import tracked_generation, tracked_tool, tracked_trace


class RecordingObservation:
    def __init__(self, name: str, as_type: str) -> None:
        self.name = name
        self.as_type = as_type
        self.updates: list[dict[str, Any]] = []

    def update(self, **values: Any) -> None:
        self.updates.append(values)


class RecordingClient:
    def __init__(self) -> None:
        self.observations: list[RecordingObservation] = []
        self.flush_count = 0

    @contextmanager
    def start_as_current_observation(
        self,
        *,
        as_type: str,
        name: str,
        metadata: dict[str, Any],
        input: Any,
    ):
        observation = RecordingObservation(name, as_type)
        self.observations.append(observation)
        yield observation

    def flush(self) -> None:
        self.flush_count += 1


def test_disabled_langfuse_tracing_does_not_break_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.langfuse._get_client", lambda: None)

    with tracked_trace(name="answer", metadata={"answer_id": "A01"}) as trace:
        trace.update(output={"status": "ok"})


def test_trace_generation_and_tool_use_current_observation_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingClient()
    monkeypatch.setattr("app.langfuse._get_client", lambda: client)

    with tracked_trace(name="answer", metadata={"answer_id": "A01"}):
        with tracked_generation(name="provider-call", model="test-model") as generation:
            generation.update(output={"response_chars": 2})
        with tracked_tool(name="catalogue") as tool:
            tool.update(output={"fact_count": 1})

    assert [item.as_type for item in client.observations] == [
        "span",
        "generation",
        "span",
    ]
    assert client.flush_count == 3
    assert client.observations[1].updates == [{"output": {"response_chars": 2}}]


def test_observation_records_failure_and_propagates_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingClient()
    monkeypatch.setattr("app.langfuse._get_client", lambda: client)

    with pytest.raises(ValueError):
        with tracked_trace(name="answer"):
            raise ValueError("provider failure")

    assert client.observations[0].updates == [
        {"level": "ERROR", "status_message": "ValueError"}
    ]
