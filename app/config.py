"""Environment-backed application settings."""

from dataclasses import dataclass
import math
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    model_provider: str
    model_name: str
    groq_api_key: str | None
    groq_timeout_seconds: float
    openai_api_key: str | None
    gemini_api_key: str | None
    anthropic_api_key: str | None
    database_url: str | None
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_base_url: str | None


def get_settings() -> Settings:
    """Read settings without logging or otherwise exposing secret values."""
    timeout_value = os.getenv("GROQ_TIMEOUT_SECONDS", "30")
    try:
        groq_timeout_seconds = float(timeout_value)
    except ValueError as error:
        raise ValueError("GROQ_TIMEOUT_SECONDS must be a positive number") from error
    if not math.isfinite(groq_timeout_seconds) or groq_timeout_seconds <= 0:
        raise ValueError("GROQ_TIMEOUT_SECONDS must be a positive number")

    return Settings(
        model_provider=os.getenv("MODEL_PROVIDER", "groq"),
        model_name=os.getenv("MODEL_NAME", ""),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_timeout_seconds=groq_timeout_seconds,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        database_url=os.getenv("DATABASE_URL"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_base_url=os.getenv("LANGFUSE_BASE_URL"),
    )


settings = get_settings()
