import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.agent.openrouter_client import (
    OpenRouterAuthenticationError,
    OpenRouterClient,
    OpenRouterProviderError,
    OpenRouterRateLimitError,
)
from app.config import get_settings


def _settings_with_key():
    return replace(
        get_settings(),
        model_name="test-model",
        openrouter_api_key="test-secret-never-log",
    )


class _TransientStatusError(Exception):
    def __init__(
        self,
        status_code: int,
        *,
        body: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__("transient")
        self.status_code = status_code
        self.body = body
        self.response = SimpleNamespace(headers=headers or {})


def test_provider_retries_transient_errors_once_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.agent.openrouter_client as openrouter_module

    calls: list[int] = []

    def _fake_create(**_: object) -> object:
        calls.append(1)
        if len(calls) == 1:
            raise _TransientStatusError(500)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    monkeypatch.setattr(openrouter_module, "APIStatusError", _TransientStatusError)
    client = OpenRouterClient(
        _settings_with_key(),
        sdk_client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create))
        ),
    )

    assert client.complete("prompt") == "ok"
    assert len(calls) == 2


def test_provider_does_not_retry_auth_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.agent.openrouter_client as openrouter_module

    calls = {"count": 0}

    def _fake_create(**_: object) -> object:
        calls["count"] += 1
        raise _TransientStatusError(401)

    monkeypatch.setattr(openrouter_module, "APIStatusError", _TransientStatusError)
    client = OpenRouterClient(
        _settings_with_key(),
        sdk_client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create))
        ),
    )

    with pytest.raises(OpenRouterAuthenticationError):
        client.complete("prompt")
    assert calls["count"] == 1


def test_provider_retries_twice_then_records_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.agent.openrouter_client as openrouter_module

    calls = {"count": 0}

    def _fake_create(**_: object) -> object:
        calls["count"] += 1
        raise _TransientStatusError(500)

    monkeypatch.setattr(openrouter_module, "APIStatusError", _TransientStatusError)
    client = OpenRouterClient(
        _settings_with_key(),
        sdk_client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create))
        ),
    )

    with pytest.raises(OpenRouterProviderError):
        client.complete("prompt")
    assert calls["count"] == 2


def test_provider_concurrency_limit_never_exceeds_three() -> None:
    active = 0
    maximum = 0
    lock = threading.Lock()

    def _fake_create(**_: object) -> object:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.08)
        with lock:
            active -= 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    client = OpenRouterClient(
        _settings_with_key(),
        sdk_client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create))
        ),
    )
    threads = [
        threading.Thread(target=lambda: client.complete("prompt"))
        for _ in range(12)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert maximum <= 3


def test_provider_rate_limit_is_not_retried() -> None:
    import app.agent.openrouter_client as openrouter_module

    calls = {"count": 0}

    def _fake_create(**_: object) -> object:
        calls["count"] += 1
        raise _TransientStatusError(429)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(openrouter_module, "APIStatusError", _TransientStatusError)
    try:
        client = OpenRouterClient(
            _settings_with_key(),
            sdk_client=SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create))
            ),
        )
        with pytest.raises(OpenRouterRateLimitError):
            client.complete("prompt")
    finally:
        monkeypatch.undo()
    assert calls["count"] == 1


def test_provider_retries_temporary_rate_limit_after_header_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import app.agent.openrouter_client as openrouter_module

    calls = {"count": 0}
    sleeps: list[float] = []
    api_key = "test-secret-never-log"

    def _fake_create(**_: object) -> object:
        calls["count"] += 1
        if calls["count"] == 1:
            raise _TransientStatusError(
                429,
                body={
                    "error": {
                        "message": f"Rate limit reached for key {api_key}.",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
                headers={"Retry-After": "0.25"},
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    monkeypatch.setattr(openrouter_module, "APIStatusError", _TransientStatusError)
    monkeypatch.setattr(openrouter_module.time, "sleep", sleeps.append)
    client = OpenRouterClient(
        _settings_with_key(),
        sdk_client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create))
        ),
    )

    assert client.complete("prompt") == "ok"
    assert calls["count"] == 2
    assert sleeps == [0.25]
    assert "rate_limit_exceeded" in caplog.text
    assert "Retry-After='0.25'" in caplog.text
    assert "Rate limit reached for key [REDACTED]." in caplog.text
    assert api_key not in caplog.text


def test_provider_does_not_retry_exhausted_quota_even_with_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.agent.openrouter_client as openrouter_module

    calls = {"count": 0}

    def _fake_create(**_: object) -> object:
        calls["count"] += 1
        raise _TransientStatusError(
            429,
            body={
                "error": {
                    "message": "Insufficient quota; quota exhausted.",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                }
            },
            headers={"Retry-After": "0"},
        )

    monkeypatch.setattr(openrouter_module, "APIStatusError", _TransientStatusError)
    client = OpenRouterClient(
        _settings_with_key(),
        sdk_client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create))
        ),
    )

    with pytest.raises(OpenRouterRateLimitError):
        client.complete("prompt")
    assert calls["count"] == 1


def test_provider_does_not_retry_after_timeout_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.agent.openrouter_client as openrouter_module

    calls = {"count": 0}

    def _fake_create(**_: object) -> object:
        calls["count"] += 1
        raise _TransientStatusError(
            429,
            body={"error": {"message": "Temporary rate limit"}},
            headers={"Retry-After": "16"},
        )

    monkeypatch.setattr(openrouter_module, "APIStatusError", _TransientStatusError)
    client = OpenRouterClient(
        _settings_with_key(),
        sdk_client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create))
        ),
    )

    with pytest.raises(OpenRouterRateLimitError):
        client.complete("prompt")
    assert calls["count"] == 1


def test_timeout_is_retried_once_then_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.agent.openrouter_client as openrouter_module

    class FakeTimeout(Exception):
        pass

    calls = {"count": 0}

    def _fake_create(**_: object) -> object:
        calls["count"] += 1
        raise FakeTimeout("timeout")

    monkeypatch.setattr(openrouter_module, "APITimeoutError", FakeTimeout)
    client = OpenRouterClient(
        _settings_with_key(),
        sdk_client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create))
        ),
    )

    with pytest.raises(openrouter_module.OpenRouterTimeoutError):
        client.complete("prompt")
    assert calls["count"] == 2


def test_provider_semaphore_is_released_after_failure() -> None:
    calls = {"count": 0}

    def _fake_create(**_: object) -> object:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("failed")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    client = OpenRouterClient(
        _settings_with_key(),
        sdk_client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create))
        ),
    )

    with pytest.raises(OpenRouterProviderError):
        client.complete("first")
    assert client.complete("second") == "ok"


def test_provider_timeout_cannot_exceed_fifteen_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.agent.openrouter_client as openrouter_module

    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(openrouter_module, "OpenAI", FakeOpenAI)
    OpenRouterClient(replace(_settings_with_key(), model_timeout_seconds=30))

    assert captured["timeout"] == 15
    assert captured["max_retries"] == 0
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
