"""SQLAlchemy configuration for the Phase 5 persistence layer."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_session_factory: sessionmaker[Session] | None = None
_engine: Engine | None = None


def get_database_url() -> str:
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is missing. Set it in the local .env file before using the database."
        )
    return settings.database_url


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        database_url: URL = make_url(get_database_url())
        if database_url.get_backend_name() == "postgresql":
            database_url = database_url.set(drivername="postgresql+psycopg2")
        _engine = create_engine(database_url, pool_pre_ping=True, future=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def validate_database_configuration() -> None:
    """Check that DATABASE_URL is configured without exposing its value."""
    get_database_url()
    with get_engine().connect() as connection:
        connection.execute(text("SELECT 1"))


def initialize_database() -> None:
    """Create missing application tables without modifying existing tables."""
    from app.db_models import Base

    Base.metadata.create_all(bind=get_engine())
