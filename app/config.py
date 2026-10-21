"""Environment-backed application settings."""

from dataclasses import dataclass
import math
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    model_name: str
    openrouter_api_key: str | None
    model_timeout_seconds: float
    database_url: str | None
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_base_url: str | None


def get_settings() -> Settings:
    """Read settings without logging or otherwise exposing secret values."""
    timeout_value = os.getenv("MODEL_TIMEOUT_SECONDS", "15")
    try:
        model_timeout_seconds = float(timeout_value)
    except ValueError as error:
        raise ValueError("MODEL_TIMEOUT_SECONDS must be a positive number") from error
    if not math.isfinite(model_timeout_seconds) or model_timeout_seconds <= 0:
        raise ValueError("MODEL_TIMEOUT_SECONDS must be a positive number")
    model_timeout_seconds = min(model_timeout_seconds, 15.0)

    return Settings(
        model_name=os.getenv("MODEL_NAME", "openai/gpt-oss-120b"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        model_timeout_seconds=model_timeout_seconds,
        database_url=os.getenv("DATABASE_URL"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_base_url=os.getenv("LANGFUSE_BASE_URL"),
    )


settings = get_settings()
