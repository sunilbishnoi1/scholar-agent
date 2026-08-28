"""SQLAlchemy 2.0 Declarative Models for Scholar Agent.

Defines database schemas for:
- User (users): User authentication, subscription tiers, and monthly budgets
- ResearchProject (research_projects): Core literature review project workspace
- AgentPlan (agent_plans): Multi-agent execution plans and step tracking
- PaperReference (paper_references): Project-associated paper citations and embeddings
- UserUsage (user_usage): Monthly token consumption and cost aggregation
- LLMInteraction (llm_interactions): LLM request audit trail and latency metrics
- PaperCache (paper_cache): Global scientific paper cache with full-text and sections
- ResearchReportModel (research_reports): Synthesized research report documents
- EvidenceMatrixEntry (evidence_matrix_entries): Comparative matrix extraction rows
- ResearchGapModel (research_gaps): Grounded actionable research gaps
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""
    pass


class User(Base):
    """User account entity with subscription tier and usage limits."""
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    institution: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tier: Mapped[str] = mapped_column(String(50), default="free", nullable=False)  # free, pro, enterprise
    monthly_budget_usd: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    research_projects: Mapped[list["ResearchProject"]] = relationship(
        "ResearchProject", back_populates="user", cascade="all, delete-orphan"
    )
    usage_records: Mapped[list["UserUsage"]] = relationship(
        "UserUsage", back_populates="user", cascade="all, delete-orphan"
    )
    llm_interactions: Mapped[list["LLMInteraction"]] = relationship(
        "LLMInteraction", back_populates="user", cascade="all, delete-orphan"
    )


class ResearchProject(Base):
    """Container for scientific literature review workflows."""
    __tablename__ = "research_projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    research_question: Mapped[str] = mapped_column(Text, nullable=False)
    keywords: Mapped[Optional[list[str]]] = mapped_column(JSON, default=list, nullable=True)
    subtopics: Mapped[Optional[list[str]]] = mapped_column(JSON, default=list, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="planning", nullable=False)
    total_papers_found: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_papers: Mapped[Optional[int]] = mapped_column(Integer, default=30, nullable=True)
    report: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)  # Structured output container
    report_status: Mapped[str] = mapped_column(String(50), default="empty", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="research_projects")
    agent_plans: Mapped[list["AgentPlan"]] = relationship(
        "AgentPlan", back_populates="project", cascade="all, delete-orphan"
    )
    paper_references: Mapped[list["PaperReference"]] = relationship(
        "PaperReference", back_populates="project", cascade="all, delete-orphan"
    )
    research_reports: Mapped[list["ResearchReportModel"]] = relationship(
        "ResearchReportModel", back_populates="project", cascade="all, delete-orphan"
    )
    evidence_matrix_entries: Mapped[list["EvidenceMatrixEntry"]] = relationship(
        "EvidenceMatrixEntry", back_populates="project", cascade="all, delete-orphan"
    )
    research_gaps: Mapped[list["ResearchGapModel"]] = relationship(
        "ResearchGapModel", back_populates="project", cascade="all, delete-orphan"
    )


class AgentPlan(Base):
    """Step execution status log for multi-agent workflows."""
    __tablename__ = "agent_plans"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("research_projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agent_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    plan_steps: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    current_step: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    plan_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Relationships
    project: Mapped["ResearchProject"] = relationship("ResearchProject", back_populates="agent_plans")


class PaperReference(Base):
    """Project-associated paper reference metadata and embeddings."""
    __tablename__ = "paper_references"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("research_projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    authors: Mapped[Optional[list[str]]] = mapped_column(JSON, default=list, nullable=True)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    embeddings: Mapped[Optional[list[float]]] = mapped_column(JSON, nullable=True)
    relevance_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    project: Mapped["ResearchProject"] = relationship("ResearchProject", back_populates="paper_references")


class UserUsage(Base):
    """Monthly usage and cost aggregation per user."""
    __tablename__ = "user_usage"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    month: Mapped[date] = mapped_column(Date, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    projects_created: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    papers_analyzed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="usage_records")


class LLMInteraction(Base):
    """Detailed audit log for individual LLM requests and responses."""
    __tablename__ = "llm_interactions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("research_projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    agent_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    task_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    prompt_preview: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    response_preview: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="llm_interactions")


# ============================================================================
# Phase 2 Relational Models
# ============================================================================


class PaperCache(Base):
    """
    Global scientific paper cache across projects and users.
    Supports DOI deduplication, multi-tier OA resolution persistence,
    parsed markdown caching, hierarchical section JSON, and table extraction.
    """
    __tablename__ = "paper_cache"

    doi: Mapped[str] = mapped_column(String(256), primary_key=True)
    arxiv_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    s2_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    authors: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    venue: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parsed_markdown: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sections_json: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    tables_json: Mapped[Optional[list[Any]]] = mapped_column(JSON, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    is_full_text: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ResearchReportModel(Base):
    """
    Synthesized research reports generated for a specific ResearchProject.
    Captures executive summary, methodology distribution, quality scores,
    thematic sections with citation anchors, and conflicting debates.
    """
    __tablename__ = "research_reports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("research_projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    executive_summary: Mapped[str] = mapped_column(Text, nullable=False)
    methodology_overview: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    thematic_sections: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    conflicts_and_debates: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    project: Mapped["ResearchProject"] = relationship("ResearchProject", back_populates="research_reports")


class EvidenceMatrixEntry(Base):
    """
    Comparative matrix rows extracted across acquired papers for a specific project.
    Captures methodology type, benchmark datasets, quantitative metrics, and primary limitations.
    """
    __tablename__ = "evidence_matrix_entries"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("research_projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    paper_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    methodology_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    benchmark_dataset: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    primary_metric: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    primary_limitation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    authors: Mapped[Optional[list[str]]] = mapped_column(JSON, default=list, nullable=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    doi: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    is_full_text: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    project: Mapped["ResearchProject"] = relationship("ResearchProject", back_populates="evidence_matrix_entries")

    __table_args__ = (
        Index("ix_evidence_matrix_project_paper", "project_id", "paper_id"),
    )


class ResearchGapModel(Base):
    """
    Grounded, actionable research gaps identified for a specific project.
    Captures importance priority ('high' | 'medium' | 'low'), recommended methodology,
    and grounding paper IDs.
    """
    __tablename__ = "research_gaps"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("research_projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    gap_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[str] = mapped_column(String(32), default="high", nullable=False)
    recommended_methodology: Mapped[str] = mapped_column(Text, nullable=False)
    grounding_paper_ids: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    project: Mapped["ResearchProject"] = relationship("ResearchProject", back_populates="research_gaps")


__all__ = [
    "Base",
    "User",
    "ResearchProject",
    "AgentPlan",
    "PaperReference",
    "UserUsage",
    "LLMInteraction",
    "PaperCache",
    "ResearchReportModel",
    "EvidenceMatrixEntry",
    "ResearchGapModel",
]
