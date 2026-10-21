"""SQLAlchemy models for runs, answer results, and claim results."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'partial')",
            name="ck_runs_status",
        ),
        CheckConstraint(
            "dataset_split = 'development'",
            name="ck_runs_dataset_split",
        ),
        CheckConstraint(
            "total_answers >= 0 AND successful_answers >= 0 AND failed_answers >= 0",
            name="ck_runs_answer_counts_nonnegative",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    version: Mapped[str] = mapped_column(String(16), nullable=False)
    dataset_split: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    total_answers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    successful_answers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_answers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    answer_results: Mapped[list["AnswerResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class AnswerResult(Base):
    __tablename__ = "answer_results"
    __table_args__ = (
        UniqueConstraint("run_id", "answer_id", name="uq_answer_result_run_answer"),
        CheckConstraint(
            "status IN ('success', 'error')",
            name="ck_answer_results_status",
        ),
        CheckConstraint("predicted_claim_count >= 0", name="ck_answer_claim_count"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), nullable=False, index=True)
    answer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    predicted_claim_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    run: Mapped[Run] = relationship(back_populates="answer_results")
    claim_results: Mapped[list["ClaimResult"]] = relationship(
        back_populates="answer_result", cascade="all, delete-orphan"
    )


class ClaimResult(Base):
    __tablename__ = "claim_results"
    __table_args__ = (
        UniqueConstraint("answer_result_id", "claim_id", name="uq_claim_result_answer_claim"),
        CheckConstraint(
            "label IN ('supported', 'contradicted', 'unknown')",
            name="ck_claim_results_label",
        ),
        CheckConstraint("start_char >= 0 AND end_char > start_char", name="ck_claim_span"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    answer_result_id: Mapped[int] = mapped_column(
        ForeignKey("answer_results.id"), nullable=False, index=True
    )
    claim_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    claim_text: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    fact_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    answer_result: Mapped[AnswerResult] = relationship(back_populates="claim_results")
