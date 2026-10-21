"""Groq SDK adapter for model-provider-specific behavior."""

import logging

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    Groq,
)

from app.config import Settings, get_settings
from app.agent.prompts import EXTRACTION_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


class GroqClientError(RuntimeError):
    """Base class for sanitized Groq configuration and provider errors."""


class GroqConfigurationError(GroqClientError):
    """Raised when Groq settings are absent or invalid."""


class GroqAuthenticationError(GroqClientError):
    """Raised when Groq rejects the configured API key."""


class GroqRateLimitError(GroqClientError):
    """Raised when Groq rate limits a request."""


class GroqModelUnavailableError(GroqClientError):
    """Raised when the configured model is unavailable on Groq."""


class GroqTimeoutError(GroqClientError):
    """Raised when a Groq request exceeds its configured timeout."""


class GroqProviderError(GroqClientError):
    """Raised for other sanitized Groq request or response failures."""


class GroqClient:
    """Small, injectable wrapper around the official Groq Python SDK."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sdk_client: Groq | None = None,
        extraction_system_prompt: str = EXTRACTION_SYSTEM_PROMPT,
    ) -> None:
        self._settings = settings or get_settings()
        if self._settings.model_provider.casefold() != "groq":
            raise GroqConfigurationError(
                "MODEL_PROVIDER must be 'groq'; automatic provider fallback is disabled"
            )
        if not self._settings.model_name:
            raise GroqConfigurationError("MODEL_NAME must be configured for Groq")
        if sdk_client is None and not self._settings.groq_api_key:
            raise GroqConfigurationError("GROQ_API_KEY is required to use Groq")
        self._sdk_client = (
            sdk_client
            if sdk_client is not None
            else Groq(
                api_key=self._settings.groq_api_key,
                timeout=self._settings.groq_timeout_seconds,
            )
        )
        self._extraction_system_prompt = extraction_system_prompt

    def complete(self, user_prompt: str) -> str:
        """Call the configured chat model and return its raw response content."""
        response = self._request(
            messages=[
                {"role": "system", "content": self._extraction_system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as error:
            raise GroqProviderError(
                "Groq returned an unexpected completion response"
            ) from error
        if not isinstance(content, str) or not content:
            raise GroqProviderError(
                "Groq returned an empty or unexpected completion response"
            )
        return content

    def chat_completion(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | dict[str, object] | None = None,
    ) -> object:
        """Request a native Groq chat completion, optionally with local tools.

        Omitting both tool fields after a local tool result prevents a model from
        attempting a second function call while it is producing the final JSON.
        """
        request: dict[str, object] = {
            "messages": messages,
            "temperature": 0,
        }
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        return self._request(**request)

    def _request(self, **request: object) -> object:
        try:
            return self._sdk_client.chat.completions.create(
                model=self._settings.model_name,
                **request,
            )
        except GroqClientError:
            raise
        except APITimeoutError:
            logger.warning("Groq request timed out")
            raise GroqTimeoutError("Groq request timed out") from None
        except APIStatusError as error:
            status_code = error.status_code
            logger.warning("Groq request failed with HTTP status %s", status_code)
            if status_code in {401, 403}:
                raise GroqAuthenticationError(
                    "Groq rejected the configured API key"
                ) from None
            if status_code == 429:
                raise GroqRateLimitError("Groq rate limit exceeded") from None
            if status_code == 404:
                raise GroqModelUnavailableError(
                    "Configured Groq model is unavailable (HTTP 404); "
                    "check MODEL_NAME against the active models in your Groq account"
                ) from None
            raise GroqProviderError(
                f"Groq request failed with HTTP status {status_code}"
            ) from None
        except APIConnectionError:
            logger.warning("Groq connection failed")
            raise GroqProviderError("Could not connect to Groq") from None
        except Exception:
            logger.error("Unexpected Groq provider response or SDK failure")
            raise GroqProviderError(
                "Unexpected Groq provider response or SDK failure"
            ) from None
