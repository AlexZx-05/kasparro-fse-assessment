"""OpenRouter client adapter using its OpenAI-compatible API."""

import logging
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

from app.agent.prompts import EXTRACTION_SYSTEM_PROMPT
from app.config import Settings, get_settings
from app.langfuse import tracked_generation

logger = logging.getLogger(__name__)

_PROVIDER_SEMAPHORE = threading.BoundedSemaphore(3)


class OpenRouterClientError(RuntimeError):
    """Base class for sanitized OpenRouter configuration and provider errors."""


class OpenRouterConfigurationError(OpenRouterClientError):
    """Raised when OpenRouter settings are absent or invalid."""


class OpenRouterAuthenticationError(OpenRouterClientError):
    """Raised when OpenRouter rejects the configured API key."""


class OpenRouterRateLimitError(OpenRouterClientError):
    """Raised when OpenRouter rate limits a request."""


class OpenRouterModelUnavailableError(OpenRouterClientError):
    """Raised when the configured model is unavailable on OpenRouter."""


class OpenRouterTimeoutError(OpenRouterClientError):
    """Raised when an OpenRouter request exceeds its configured timeout."""


class OpenRouterProviderError(OpenRouterClientError):
    """Raised for other sanitized OpenRouter request or response failures."""


class OpenRouterClient:
    """Small, injectable wrapper around OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sdk_client: OpenAI | None = None,
        extraction_system_prompt: str = EXTRACTION_SYSTEM_PROMPT,
    ) -> None:
        self._settings = settings or get_settings()
        if not self._settings.model_name:
            raise OpenRouterConfigurationError("MODEL_NAME must be configured")
        if sdk_client is None and not self._settings.openrouter_api_key:
            raise OpenRouterConfigurationError(
                "OPENROUTER_API_KEY is required to use OpenRouter"
            )
        self._sdk_client = (
            sdk_client
            if sdk_client is not None
            else OpenAI(
                api_key=self._settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=min(self._settings.model_timeout_seconds, 15.0),
                max_retries=0,
            )
        )
        self._extraction_system_prompt = extraction_system_prompt

    def complete(self, user_prompt: str, *, metadata: dict[str, Any] | None = None) -> str:
        """Call the configured chat model and return its raw response content."""
        with tracked_generation(
            name="openrouter-completion",
            model=self._settings.model_name,
            metadata={"provider": "openrouter", **(metadata or {})},
            input_data={"prompt_chars": len(user_prompt)},
        ) as trace:
            response = self._request(
                messages=[
                    {"role": "system", "content": self._extraction_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                meta=metadata,
                trace=trace,
            )
            try:
                content = response.choices[0].message.content
            except (AttributeError, IndexError, TypeError) as error:
                raise OpenRouterProviderError(
                    "OpenRouter returned an unexpected completion response"
                ) from error
            if not isinstance(content, str) or not content:
                raise OpenRouterProviderError(
                    "OpenRouter returned an empty or unexpected completion response"
                )
            if trace is not None and hasattr(trace, "update"):
                trace.update(output={"response_chars": len(content)})
            return content

    def chat_completion(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> object:
        """Request a chat completion, optionally with registered local tools."""
        request: dict[str, object] = {
            "messages": messages,
            "temperature": 0,
        }
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        with tracked_generation(
            name="openrouter-tool-call",
            model=self._settings.model_name,
            metadata={"provider": "openrouter", **(metadata or {})},
            input_data={
                "message_count": len(messages),
                "tool_names": [
                    tool.get("function", {}).get("name")
                    for tool in tools or []
                    if isinstance(tool.get("function"), dict)
                ],
            },
        ) as trace:
            result = self._request(**request, meta=metadata, trace=trace)
            if trace is not None and hasattr(trace, "update"):
                choices = getattr(result, "choices", ())
                first_choice = choices[0] if choices else None
                message = getattr(first_choice, "message", None)
                tool_calls = getattr(message, "tool_calls", None) or []
                update: dict[str, Any] = {
                    "output": {
                        "finish_reason": getattr(first_choice, "finish_reason", None),
                        "tool_call_count": len(tool_calls),
                    }
                }
                usage = getattr(result, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                    update["usage_details"] = {
                        "input": prompt_tokens,
                        "output": completion_tokens,
                    }
                trace.update(**update)
            return result

    def _request(self, **request: object) -> object:
        request = dict(request)
        request.pop("meta", None)
        trace = request.pop("trace", None)
        timeout_budget = min(self._settings.model_timeout_seconds, 15.0)
        request_started = time.monotonic()
        attempt = 0
        while True:
            remaining_timeout = timeout_budget - (
                time.monotonic() - request_started
            )
            if remaining_timeout <= 0:
                raise OpenRouterTimeoutError("OpenRouter request timed out")
            try:
                with _PROVIDER_SEMAPHORE:
                    response = self._sdk_client.chat.completions.create(
                        model=self._settings.model_name,
                        timeout=remaining_timeout,
                        **request,
                    )
                _record_usage(trace, response)
                return response
            except OpenRouterClientError:
                raise
            except APITimeoutError as error:
                logger.warning("OpenRouter request timed out")
                if attempt < 1:
                    attempt += 1
                    _record_retry(trace, attempt)
                    continue
                raise OpenRouterTimeoutError("OpenRouter request timed out") from error
            except APIStatusError as error:
                status_code = getattr(error, "status_code", None)
                logger.warning(
                    "OpenRouter request failed with HTTP status %s", status_code
                )
                if status_code in {401, 403}:
                    raise OpenRouterAuthenticationError(
                        "OpenRouter rejected the configured API key"
                    ) from None
                if status_code == 429:
                    provider_message, provider_type, provider_code, retry_after = (
                        _rate_limit_details(error, self._settings.openrouter_api_key)
                    )
                    logger.warning(
                        "OpenRouter rate limit response: type=%r code=%r message=%r "
                        "Retry-After=%r",
                        provider_type,
                        provider_code,
                        provider_message,
                        retry_after,
                    )
                    retry_delay = _parse_retry_after(retry_after)
                    remaining_timeout = timeout_budget - (
                        time.monotonic() - request_started
                    )
                    if (
                        attempt < 1
                        and retry_delay is not None
                        and retry_delay < remaining_timeout
                        and not _is_quota_exhaustion(
                            provider_type,
                            provider_code,
                            provider_message,
                        )
                    ):
                        time.sleep(retry_delay)
                        attempt += 1
                        _record_retry(trace, attempt)
                        continue
                    raise OpenRouterRateLimitError(
                        "OpenRouter rate limit exceeded"
                    ) from None
                if status_code in {500, 502, 503, 504}:
                    if attempt < 1:
                        attempt += 1
                        _record_retry(trace, attempt)
                        continue
                    raise OpenRouterProviderError(
                        f"OpenRouter request failed with HTTP status {status_code}"
                    ) from None
                if status_code == 404:
                    raise OpenRouterModelUnavailableError(
                        "Configured OpenRouter model is unavailable (HTTP 404); "
                        "check MODEL_NAME against available OpenRouter models"
                    ) from None
                raise OpenRouterProviderError(
                    f"OpenRouter request failed with HTTP status {status_code}"
                ) from None
            except APIConnectionError as error:
                logger.warning("OpenRouter connection failed")
                if attempt < 1:
                    attempt += 1
                    _record_retry(trace, attempt)
                    continue
                raise OpenRouterProviderError(
                    "Could not connect to OpenRouter"
                ) from error
            except Exception:
                logger.error("Unexpected OpenRouter response or SDK failure")
                raise OpenRouterProviderError(
                    "Unexpected OpenRouter response or SDK failure"
                ) from None


def _rate_limit_details(
    error: APIStatusError,
    api_key: str | None,
) -> tuple[str, str | None, str | None, str | None]:
    payload = getattr(error, "body", None)
    response = getattr(error, "response", None)
    if not isinstance(payload, dict) and response is not None:
        try:
            payload = response.json()
        except (AttributeError, TypeError, ValueError):
            payload = None

    error_payload = payload.get("error", payload) if isinstance(payload, dict) else {}
    message = error_payload.get("message") if isinstance(error_payload, dict) else None
    if not isinstance(message, str):
        message = getattr(error, "message", "OpenRouter returned HTTP 429.")
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = message[:500]

    provider_type = error_payload.get("type") if isinstance(error_payload, dict) else None
    provider_code = error_payload.get("code") if isinstance(error_payload, dict) else None
    provider_type = provider_type if isinstance(provider_type, str) else None
    provider_code = provider_code if isinstance(provider_code, str) else None

    headers = getattr(response, "headers", {})
    retry_after = None
    if hasattr(headers, "get"):
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
    retry_after = str(retry_after)[:100] if retry_after is not None else None
    return message, provider_type, provider_code, retry_after


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return delay if delay >= 0 else None


def _is_quota_exhaustion(
    provider_type: str | None,
    provider_code: str | None,
    message: str,
) -> bool:
    markers = (
        "insufficient_quota",
        "quota_exceeded",
        "quota_exhausted",
        "billing_hard_limit",
        "billing_limit_exceeded",
    )
    structured_reason = " ".join((provider_type or "", provider_code or "")).casefold()
    if any(marker in structured_reason for marker in markers):
        return True

    normalized_message = message.casefold()
    return any(
        phrase in normalized_message
        for phrase in (
            "quota exhausted",
            "quota exceeded",
            "insufficient quota",
            "billing hard limit",
            "billing limit reached",
        )
    )


def _record_retry(trace: object, retry_number: int) -> None:
    update = getattr(trace, "update", None)
    if callable(update):
        try:
            update(metadata={"retry_count": retry_number})
        except Exception:
            logger.debug("Could not record provider retry in Langfuse")


def _record_usage(trace: object, response: object) -> None:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    update = getattr(trace, "update", None)
    if (
        callable(update)
        and isinstance(prompt_tokens, int)
        and isinstance(completion_tokens, int)
    ):
        try:
            update(
                usage_details={
                    "input": prompt_tokens,
                    "output": completion_tokens,
                }
            )
        except Exception:
            logger.debug("Could not record provider token usage in Langfuse")
