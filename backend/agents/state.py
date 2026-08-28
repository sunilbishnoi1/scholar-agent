"""
LangGraph State Schema for Scholar Agent Autonomous Scientific Reasoning Pipeline.
Defines AgentState TypedDict, reducer annotations, agent execution metadata, and factory methods.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal, TypedDict

try:
    from agents.schemas import (
        AcademicPaperCandidate,
        BibliographyItem,
        CitationAuditReport,
        ConflictingDebate,
        CriticEvaluation,
        EvidenceMatrixRow,
        MethodologyDistribution,
        ReportMetadata,
        ReportStatus,
        ResearchGapItem,
        ResearchReport,
        SearchQueryPlan,
        ThematicSection,
        ThematicSynthesisDraft,
    )
except ImportError:
    from backend.agents.schemas import (
        AcademicPaperCandidate,
        BibliographyItem,
        CitationAuditReport,
        ConflictingDebate,
        CriticEvaluation,
        EvidenceMatrixRow,
        MethodologyDistribution,
        ReportMetadata,
        ReportStatus,
        ResearchGapItem,
        ResearchReport,
        SearchQueryPlan,
        ThematicSection,
        ThematicSynthesisDraft,
    )


class AgentType(StrEnum):
    """Specialized autonomous reasoning agents in Milestone 4."""

    SUPERVISOR = "supervisor"
    DISCOVERY = "discovery"
    INGESTION = "ingestion"
    MATRIX_BUILDER = "matrix_builder"
    SYNTHESIZER = "synthesizer"
    CRITIC = "critic"
    AUDITOR = "auditor"
    FINALIZER = "finalizer"

    # Backward compatibility aliases
    PLANNER = "planner"
    RETRIEVER = "retriever"
    ANALYZER = "analyzer"
    QUALITY_CHECKER = "quality_checker"


class GoalStatus(StrEnum):
    """Lifecycle status for items on the blackboard goal stack."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GoalItem(TypedDict):
    """Sub-goal or milestone tracked on the working memory blackboard."""

    goal_id: str
    name: str
    description: str
    target_agent: AgentType
    status: GoalStatus
    priority: int
    parent_goal_id: str | None
    created_at: str
    completed_at: str | None
    error_message: str | None


class AgentMessage(TypedDict):
    """Standardized event/message log for LangGraph state history."""

    agent: str
    action: str
    content: Any
    timestamp: str


class TelemetryEvent(TypedDict):
    """Execution telemetry record for agent invocation."""

    agent: str
    action: str
    duration_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    model_name: str
    timestamp: str


class ParsedPaperData(TypedDict):
    """Structured representation of an acquired scientific paper."""

    paper_id: str
    doi: str | None
    arxiv_id: str | None
    s2_id: str | None
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str
    full_text_markdown: str | None
    sections: list[dict[str, Any]]
    tables: list[Any]
    equations: list[str]
    source_url: str | None
    is_full_text: bool
    citation_count: int | None
    relevance_score: float | None


class PaperData(TypedDict):
    """Structure for paper data throughout the pipeline."""

    id: str
    title: str
    abstract: str
    authors: list[str]
    url: str
    source: str
    relevance_score: float | None
    analysis: dict | None


class AgentState(TypedDict):
    """
    The shared state that flows through the LangGraph multi-agent supervisor DAG.
    Supports immutable updates, append-only reducer channels, and full reasoning artifacts.
    """

    # --- Core Identifiers & Inputs ---
    project_id: str
    user_id: str
    title: str
    research_question: str
    academic_level: str
    target_word_count: int
    max_papers: int

    # --- Goal Stack & Control Flow ---
    goal_stack: list[GoalItem]
    current_node: str
    status: str  # "running" | "needs_refinement" | "auditing" | "completed" | "error"
    iteration_count: int  # Current refinement loop index (0, 1, 2)
    max_iterations: int  # Hard cap on refinement loops (default: 2)

    # --- Discovery Agent Artifacts ---
    search_query_plan: dict[str, Any] | SearchQueryPlan | None
    candidate_papers: list[dict[str, Any]]
    total_candidates_found: int

    # --- Ingestion Agent Artifacts ---
    parsed_papers: dict[str, ParsedPaperData] | list[dict[str, Any]]
    papers_analyzed_full_text: int
    papers_analyzed_abstract_only: int
    paper_chunks: list[dict[str, Any]]

    # --- Evidence Matrix Builder Artifacts ---
    evidence_matrix: list[EvidenceMatrixRow] | list[dict[str, Any]]
    evidence_matrix_markdown: str

    # --- Thematic Synthesizer Artifacts ---
    executive_summary: str
    draft_thematic_sections: list[ThematicSection] | list[dict[str, Any]]
    thematic_sections: list[dict[str, Any]]
    conflicting_debates: list[ConflictingDebate] | list[dict[str, Any]]
    debates: list[dict[str, Any]]
    research_gaps: list[ResearchGapItem] | list[dict[str, Any]]
    methodology_overview: MethodologyDistribution | dict[str, Any] | None
    bibliography: list[BibliographyItem] | list[dict[str, Any]]
    synthesis_draft: dict[str, Any] | None

    # --- Adversarial Critic Artifacts ---
    critic_evaluations: Annotated[list[dict[str, Any]], operator.add]
    critic_evaluation: dict[str, Any] | None
    current_critic_score: float
    should_refine: bool
    refinement_guidance: list[str]

    # --- Citation Auditor Artifacts ---
    audit_report: CitationAuditReport | dict[str, Any] | None
    citation_audit_report: dict[str, Any] | None
    audit_precision_score: float
    audit_passed: bool

    # --- Final Output Artifact ---
    final_report: ResearchReport | dict[str, Any] | None

    # --- Reducer-backed Streams (Append-Only) ---
    messages: Annotated[list[AgentMessage], operator.add]
    errors: Annotated[list[str], operator.add]
    telemetry: Annotated[list[TelemetryEvent], operator.add]

    # --- Backward Compatibility Fields for Legacy Endpoints ---
    keywords: list[str]
    subtopics: list[str]
    search_strategy: dict[str, Any]
    papers: list[dict[str, Any]]
    total_papers_found: int
    analyzed_papers: list[dict[str, Any]]
    high_quality_papers: list[dict[str, Any]]
    synthesis: str
    synthesis_sections: list[dict[str, Any]]
    quality_score: float
    quality_feedback: str
    current_agent: AgentType | str
    iteration: int
    relevance_threshold: float


def create_initial_agent_state(
    project_id: str,
    user_id: str = "default_user",
    title: str = "",
    research_question: str = "",
    max_papers: int = 25,
    max_iterations: int = 2,
    relevance_threshold: float = 60.0,
    academic_level: str = "graduate",
    target_word_count: int = 3500,
) -> AgentState:
    """
    Factory creating a fully initialized, type-safe AgentState instance.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    initial_goals: list[GoalItem] = [
        {
            "goal_id": "goal_discovery",
            "name": "Literature Discovery",
            "description": "Formulate search queries and retrieve candidate papers from academic sources",
            "target_agent": AgentType.DISCOVERY,
            "status": GoalStatus.PENDING,
            "priority": 1,
            "parent_goal_id": None,
            "created_at": now_iso,
            "completed_at": None,
            "error_message": None,
        },
        {
            "goal_id": "goal_ingestion",
            "name": "Full-Text Ingestion & Resolution",
            "description": "Resolve open-access PDFs, parse markdown hierarchies, and populate PaperCache",
            "target_agent": AgentType.INGESTION,
            "status": GoalStatus.PENDING,
            "priority": 2,
            "parent_goal_id": None,
            "created_at": now_iso,
            "completed_at": None,
            "error_message": None,
        },
        {
            "goal_id": "goal_matrix",
            "name": "Evidence Matrix Extraction",
            "description": "Extract uniform comparative schemas across acquired papers",
            "target_agent": AgentType.MATRIX_BUILDER,
            "status": GoalStatus.PENDING,
            "priority": 3,
            "parent_goal_id": None,
            "created_at": now_iso,
            "completed_at": None,
            "error_message": None,
        },
        {
            "goal_id": "goal_synthesis",
            "name": "Thematic Synthesis & Gap Formulation",
            "description": "Synthesize thematic sections with [ref_X#secY] anchors and identify research gaps",
            "target_agent": AgentType.SYNTHESIZER,
            "status": GoalStatus.PENDING,
            "priority": 4,
            "parent_goal_id": None,
            "created_at": now_iso,
            "completed_at": None,
            "error_message": None,
        },
        {
            "goal_id": "goal_critic",
            "name": "Adversarial Peer Review",
            "description": "Evaluate synthesis quality (0-100) and identify refinement needs",
            "target_agent": AgentType.CRITIC,
            "status": GoalStatus.PENDING,
            "priority": 5,
            "parent_goal_id": None,
            "created_at": now_iso,
            "completed_at": None,
            "error_message": None,
        },
        {
            "goal_id": "goal_auditor",
            "name": "Citation Grounding & NLI Audit",
            "description": "Verify atomic factual claims against source chunks via structured NLI",
            "target_agent": AgentType.AUDITOR,
            "status": GoalStatus.PENDING,
            "priority": 6,
            "parent_goal_id": None,
            "created_at": now_iso,
            "completed_at": None,
            "error_message": None,
        },
        {
            "goal_id": "goal_finalize",
            "name": "Report Finalization & Persistence",
            "description": "Assemble final ResearchReport and persist into PostgreSQL relational storage",
            "target_agent": AgentType.FINALIZER,
            "status": GoalStatus.PENDING,
            "priority": 7,
            "parent_goal_id": None,
            "created_at": now_iso,
            "completed_at": None,
            "error_message": None,
        },
    ]

    return AgentState(
        project_id=project_id,
        user_id=user_id,
        title=title,
        research_question=research_question,
        academic_level=academic_level,
        target_word_count=target_word_count,
        max_papers=max_papers,
        goal_stack=initial_goals,
        current_node="supervisor",
        status="running",
        iteration_count=0,
        max_iterations=max(1, min(max_iterations, 2)),  # Enforce hard bounded limit [1, 2]
        search_query_plan=None,
        candidate_papers=[],
        total_candidates_found=0,
        parsed_papers={},
        papers_analyzed_full_text=0,
        papers_analyzed_abstract_only=0,
        paper_chunks=[],
        evidence_matrix=[],
        evidence_matrix_markdown="",
        executive_summary="",
        draft_thematic_sections=[],
        thematic_sections=[],
        conflicting_debates=[],
        debates=[],
        research_gaps=[],
        methodology_overview=None,
        bibliography=[],
        synthesis_draft=None,
        critic_evaluations=[],
        critic_evaluation=None,
        current_critic_score=0.0,
        should_refine=False,
        refinement_guidance=[],
        audit_report=None,
        citation_audit_report=None,
        audit_precision_score=0.0,
        audit_passed=False,
        final_report=None,
        messages=[],
        errors=[],
        telemetry=[],
        # Backward compatibility initializers
        keywords=[],
        subtopics=[],
        search_strategy={},
        papers=[],
        total_papers_found=0,
        analyzed_papers=[],
        high_quality_papers=[],
        synthesis="",
        synthesis_sections=[],
        quality_score=0.0,
        quality_feedback="",
        current_agent=AgentType.SUPERVISOR,
        iteration=0,
        relevance_threshold=relevance_threshold,
    )


# Backward compatibility alias
create_initial_state = create_initial_agent_state


@dataclass
class AgentResult:
    """Standardized wrapper for agent execution outcome."""

    success: bool
    data: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self, agent: str, action: str) -> AgentMessage:
        return AgentMessage(
            agent=agent,
            action=action,
            content=self.data if self.success else self.error,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
