# Agent State Definitions for LangGraph
# Defines the shared state that flows through the agent pipeline

import operator
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, TypedDict


class AgentType(str, Enum):
    """Enumeration of agent types in the pipeline."""

    PLANNER = "planner"
    RETRIEVER = "retriever"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"
    QUALITY_CHECKER = "quality_checker"


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


class AgentMessage(TypedDict):
    """Structure for messages passed between agents."""

    agent: str
    action: str
    content: Any
    timestamp: str


class AgentState(TypedDict):
    """
    The shared state that flows through the LangGraph pipeline.

    This state is passed between agents and contains all the information
    needed to complete the research task.
    """

    # Core identifiers
    project_id: str
    user_id: str

    # Research inputs
    title: str
    research_question: str

    # Planner outputs
    keywords: list[str]
    subtopics: list[str]
    search_strategy: dict

    # Retriever outputs
    papers: list[PaperData]
    total_papers_found: int

    # Analyzer outputs
    analyzed_papers: list[dict]
    high_quality_papers: list[dict]  # Papers with relevance_score > threshold

    # Synthesizer outputs
    synthesis: str
    synthesis_sections: list[dict]

    # Quality control
    quality_score: float
    quality_feedback: str

    # Pipeline control
    current_agent: AgentType
    iteration: int
    max_iterations: int

    # Message history for debugging/tracing
    messages: Annotated[Sequence[AgentMessage], operator.add]

    # Error handling
    errors: list[str]
    status: str  # "running", "completed", "error", "needs_refinement"

    # Configuration
    max_papers: int
    relevance_threshold: float
    academic_level: str
    target_word_count: int


def create_initial_state(
    project_id: str,
    user_id: str,
    title: str,
    research_question: str,
    max_papers: int = 50,
    max_iterations: int = 3,
    relevance_threshold: float = 60.0,
    academic_level: str = "graduate",
    target_word_count: int = 500,
) -> AgentState:
    """
    Factory function to create a properly initialized AgentState.

    Args:
        project_id: Database ID of the research project
        user_id: Database ID of the user
        title: Title of the research project
        research_question: The main research question to answer
        max_papers: Maximum number of papers to retrieve
        max_iterations: Maximum iterations for refinement loops
        relevance_threshold: Minimum relevance score (0-100) to include a paper
        academic_level: Academic level for writing style
        target_word_count: Target word count for synthesis

    Returns:
        AgentState: A fully initialized state dictionary
    """
    return AgentState(
        # Core identifiers
        project_id=project_id,
        user_id=user_id,
        # Research inputs
        title=title,
        research_question=research_question,
        # Planner outputs (initialized empty)
        keywords=[],
        subtopics=[],
        search_strategy={},
        # Retriever outputs
        papers=[],
        total_papers_found=0,
        # Analyzer outputs
        analyzed_papers=[],
        high_quality_papers=[],
        # Synthesizer outputs
        synthesis="",
        synthesis_sections=[],
        # Quality control
        quality_score=0.0,
        quality_feedback="",
        # Pipeline control
        current_agent=AgentType.PLANNER,
        iteration=0,
        max_iterations=max_iterations,
        messages=[],
        # Error handling
        errors=[],
        status="running",
        # Configuration
        max_papers=max_papers,
        relevance_threshold=relevance_threshold,
        academic_level=academic_level,
        target_word_count=target_word_count,
    )


@dataclass
class AgentResult:
    """
    Result wrapper for agent execution.

    Provides a consistent interface for agent outputs with
    success/failure tracking and metadata.
    """

    success: bool
    data: Any
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_message(self, agent: str, action: str) -> AgentMessage:
        """Convert result to an AgentMessage for state history."""
        return AgentMessage(
            agent=agent,
            action=action,
            content=self.data if self.success else self.error,
            timestamp=datetime.now(UTC).isoformat(),
        )
