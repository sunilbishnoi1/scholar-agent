# WebSocket Event Types and Broadcasting Utilities
# Standardized event format for real-time agent updates (Scholar Agent v3.2)

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(StrEnum):
    """Types of events that can be broadcast to clients."""

    # Connection events
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"

    # Standard v3.2 Event Types (R1 / Milestone 1)
    DISCOVERY_STARTED = "discovery_started"
    PAPER_DISCOVERED = "paper_discovered"
    PDF_PARSED = "pdf_parsed"
    MATRIX_ROW_ADDED = "matrix_row_added"
    THEMATIC_DRAFT_READY = "thematic_draft_ready"
    CRITIC_VERDICT = "critic_verdict"
    FACT_CHECKED = "fact_checked"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_ERROR = "pipeline_error"
    PIPELINE_STOPPED = "pipeline_stopped"

    # Legacy agent lifecycle events (for backwards compatibility)
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_ERROR = "agent_error"

    # Progress & logging events
    STATUS_UPDATE = "status"
    PROGRESS_UPDATE = "progress"
    LOG_MESSAGE = "log"

    # Legacy paper processing events
    PAPER_FOUND = "paper_found"
    PAPER_ANALYZED = "paper_analyzed"

    # Legacy project completion events
    PROJECT_COMPLETED = "complete"
    PROJECT_ERROR = "error"


@dataclass
class AgentEvent:
    """
    Standardized event format for agent updates.

    Attributes:
        type: The type of event
        agent: Which agent generated this event (discovery, ingestion, matrix_builder, synthesizer, critic, auditor)
        project_id: The project this event relates to
        message: Human-readable message
        progress: Progress percentage (0-100)
        data: Additional event-specific data
        timestamp: When the event occurred (ISO 8601 UTC)
    """

    type: EventType | str
    agent: str | None = None
    project_id: str | None = None
    message: str | None = None
    progress: float | None = None
    data: dict[str, Any] | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "type": self.type.value if isinstance(self.type, EventType) else str(self.type),
            "timestamp": self.timestamp,
        }

        if self.agent:
            result["agent"] = self.agent
        if self.project_id:
            result["project_id"] = self.project_id
        if self.message:
            result["message"] = self.message
        if self.progress is not None:
            result["progress"] = self.progress
        if self.data:
            result["data"] = self.data

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


# =========================================================================
# Event factory functions for Standard v3.2 & Legacy events
# =========================================================================


def create_status_event(
    project_id: str, agent: str, status: str, message: str | None = None
) -> AgentEvent:
    """Create a status update event."""
    return AgentEvent(
        type=EventType.STATUS_UPDATE,
        agent=agent,
        project_id=project_id,
        message=message or f"{agent} is now {status}",
        data={"status": status},
    )


def create_progress_event(
    project_id: str,
    agent: str,
    progress: float,
    message: str | None = None,
    current: int | None = None,
    total: int | None = None,
) -> AgentEvent:
    """Create a progress update event."""
    data: dict[str, Any] = {"progress_percent": progress}
    if current is not None:
        data["current"] = current
    if total is not None:
        data["total"] = total

    return AgentEvent(
        type=EventType.PROGRESS_UPDATE,
        agent=agent,
        project_id=project_id,
        progress=progress,
        message=message,
        data=data,
    )


def create_log_event(project_id: str, agent: str, message: str, level: str = "info") -> AgentEvent:
    """Create a log message event."""
    return AgentEvent(
        type=EventType.LOG_MESSAGE,
        agent=agent,
        project_id=project_id,
        message=message,
        data={"level": level},
    )


def create_paper_event(
    project_id: str, event_type: EventType, paper_title: str, paper_data: dict[str, Any] | None = None
) -> AgentEvent:
    """Create a legacy paper-related event."""
    data: dict[str, Any] = {"paper_title": paper_title}
    if paper_data:
        data.update(paper_data)

    return AgentEvent(
        type=event_type,
        agent="retriever" if event_type == EventType.PAPER_FOUND else "analyzer",
        project_id=project_id,
        message=f"Paper: {paper_title[:50]}...",
        data=data,
    )


def create_completion_event(
    project_id: str,
    success: bool,
    summary: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> AgentEvent:
    """Create a project completion event."""
    if success:
        return AgentEvent(
            type=EventType.PROJECT_COMPLETED,
            project_id=project_id,
            message="Literature review completed successfully",
            progress=100.0,
            data=summary or {},
        )
    else:
        return AgentEvent(
            type=EventType.PROJECT_ERROR,
            project_id=project_id,
            message=error_message or "An error occurred during processing",
            data={"error": error_message},
        )


# --- Standard v3.2 Typed Event Creators ---


def create_discovery_started_event(
    project_id: str,
    queries: list[str] | list[dict[str, Any]] | None = None,
    agent: str = "discovery",
    message: str | None = None,
    progress: float | None = None,
) -> AgentEvent:
    """Create a discovery_started event."""
    queries_list = queries or []
    return AgentEvent(
        type=EventType.DISCOVERY_STARTED,
        agent=agent,
        project_id=project_id,
        message=message or f"Literature discovery started with {len(queries_list)} search queries",
        progress=progress if progress is not None else 5.0,
        data={"queries": queries_list, "agent": agent},
    )


def create_paper_discovered_event(
    project_id: str,
    paper_id: str,
    title: str,
    authors: list[str] | None = None,
    year: int | None = None,
    venue: str | None = None,
    source: str | None = None,
    citation_count: int | None = None,
    relevance_score: float | None = None,
    data: dict[str, Any] | None = None,
    progress: float | None = None,
) -> AgentEvent:
    """Create a paper_discovered event."""
    event_data: dict[str, Any] = {
        "paper_id": paper_id,
        "title": title,
        "authors": authors or [],
        "year": year,
        "venue": venue,
        "source": source,
        "citation_count": citation_count,
        "relevance_score": relevance_score,
    }
    if data:
        event_data.update(data)

    return AgentEvent(
        type=EventType.PAPER_DISCOVERED,
        agent="discovery",
        project_id=project_id,
        message=f"Discovered candidate paper: {title[:60]}...",
        progress=progress,
        data=event_data,
    )


def create_pdf_parsed_event(
    project_id: str,
    paper_id: str,
    title: str,
    is_full_text: bool = False,
    sections_count: int = 0,
    tables_count: int = 0,
    figures_count: int = 0,
    data: dict[str, Any] | None = None,
    progress: float | None = None,
) -> AgentEvent:
    """Create a pdf_parsed event."""
    event_data: dict[str, Any] = {
        "paper_id": paper_id,
        "title": title,
        "is_full_text": is_full_text,
        "sections_count": sections_count,
        "tables_count": tables_count,
        "figures_count": figures_count,
    }
    if data:
        event_data.update(data)

    status_tag = "Full Text" if is_full_text else "Abstract Only"
    return AgentEvent(
        type=EventType.PDF_PARSED,
        agent="ingestion",
        project_id=project_id,
        message=f"Parsed PDF ({status_tag}): {title[:60]}...",
        progress=progress,
        data=event_data,
    )


def create_matrix_row_added_event(
    project_id: str,
    row: Any,
    data: dict[str, Any] | None = None,
    progress: float | None = None,
) -> AgentEvent:
    """Create a matrix_row_added event."""
    if hasattr(row, "model_dump"):
        row_dict = row.model_dump()
    elif hasattr(row, "dict"):
        row_dict = row.dict()
    elif isinstance(row, dict):
        row_dict = row
    elif hasattr(row, "__dict__"):
        row_dict = vars(row)
    else:
        row_dict = {"raw": str(row)}

    event_data: dict[str, Any] = {"row": row_dict}
    if data:
        event_data.update(data)

    paper_title = row_dict.get("title", "Paper")
    return AgentEvent(
        type=EventType.MATRIX_ROW_ADDED,
        agent="matrix_builder",
        project_id=project_id,
        message=f"Matrix row synthesized for: {paper_title[:50]}...",
        progress=progress,
        data=event_data,
    )


def create_thematic_draft_ready_event(
    project_id: str,
    section_count: int = 0,
    debates_count: int = 0,
    gaps_count: int = 0,
    iteration: int = 0,
    sections: list[dict[str, Any]] | None = None,
    data: dict[str, Any] | None = None,
    progress: float | None = None,
) -> AgentEvent:
    """Create a thematic_draft_ready event."""
    event_data: dict[str, Any] = {
        "section_count": section_count,
        "debates_count": debates_count,
        "gaps_count": gaps_count,
        "iteration": iteration,
        "sections": sections or [],
    }
    if data:
        event_data.update(data)

    return AgentEvent(
        type=EventType.THEMATIC_DRAFT_READY,
        agent="synthesizer",
        project_id=project_id,
        message=f"Thematic draft synthesized (Iteration {iteration}): {section_count} sections, {debates_count} debates, {gaps_count} gaps",
        progress=progress,
        data=event_data,
    )


def create_critic_verdict_event(
    project_id: str,
    score: float,
    should_refine: bool,
    iteration: int = 0,
    dimension_scores: dict[str, float] | None = None,
    weaknesses: list[str] | None = None,
    guidance: str | None = None,
    data: dict[str, Any] | None = None,
    progress: float | None = None,
) -> AgentEvent:
    """Create a critic_verdict event."""
    event_data: dict[str, Any] = {
        "score": score,
        "should_refine": should_refine,
        "iteration": iteration,
        "dimension_scores": dimension_scores or {},
        "weaknesses": weaknesses or [],
        "guidance": guidance or "",
    }
    if data:
        event_data.update(data)

    verdict_str = "Needs refinement" if should_refine else "Passed quality threshold"
    return AgentEvent(
        type=EventType.CRITIC_VERDICT,
        agent="critic",
        project_id=project_id,
        message=f"Adversarial Critic verdict: Score {score:.1f}/100 ({verdict_str})",
        progress=progress,
        data=event_data,
    )


def create_fact_checked_event(
    project_id: str,
    precision_score: float,
    passed: bool,
    entailed_count: int = 0,
    neutral_count: int = 0,
    contradiction_count: int = 0,
    total_propositions: int = 0,
    data: dict[str, Any] | None = None,
    progress: float | None = None,
) -> AgentEvent:
    """Create a fact_checked event."""
    event_data: dict[str, Any] = {
        "precision_score": precision_score,
        "passed": passed,
        "entailed_count": entailed_count,
        "neutral_count": neutral_count,
        "contradiction_count": contradiction_count,
        "total_propositions": total_propositions,
    }
    if data:
        event_data.update(data)

    verdict_str = "PASSED" if passed else "FAILED"
    return AgentEvent(
        type=EventType.FACT_CHECKED,
        agent="auditor",
        project_id=project_id,
        message=f"Citation audit completed: Precision {precision_score:.1f}% ({verdict_str})",
        progress=progress,
        data=event_data,
    )


def create_pipeline_completed_event(
    project_id: str,
    report: Any = None,
    summary: dict[str, Any] | None = None,
) -> AgentEvent:
    """Create a pipeline_completed event."""
    report_dict: dict[str, Any] | None = None
    if hasattr(report, "model_dump"):
        report_dict = report.model_dump()
    elif hasattr(report, "dict"):
        report_dict = report.dict()
    elif isinstance(report, dict):
        report_dict = report

    event_data: dict[str, Any] = {"report": report_dict}
    if summary:
        event_data.update(summary)

    return AgentEvent(
        type=EventType.PIPELINE_COMPLETED,
        agent="supervisor",
        project_id=project_id,
        message="Autonomous literature review pipeline completed successfully",
        progress=100.0,
        data=event_data,
    )


def create_pipeline_error_event(
    project_id: str,
    error_message: str,
) -> AgentEvent:
    """Create a pipeline_error event."""
    return AgentEvent(
        type=EventType.PIPELINE_ERROR,
        agent="supervisor",
        project_id=project_id,
        message=f"Pipeline error: {error_message}",
        data={"error": error_message},
    )


def create_pipeline_stopped_event(
    project_id: str,
    message: str = "Research task stopped by user.",
) -> AgentEvent:
    """Create a pipeline_stopped event."""
    return AgentEvent(
        type=EventType.PIPELINE_STOPPED,
        agent="supervisor",
        project_id=project_id,
        message=message,
        data={"status": "stopped"},
    )


# =========================================================================
# Broadcasting utilities (integrates with Redis pub/sub)
# =========================================================================


async def broadcast_agent_update(project_id: str, event: AgentEvent, use_redis: bool = True):
    """
    Broadcast an agent event to all interested clients.

    This function handles both:
    1. Direct WebSocket broadcast (for local connections)
    2. Redis pub/sub broadcast (for distributed deployments)

    Args:
        project_id: The project to broadcast to
        event: The event to broadcast
        use_redis: Whether to also publish to Redis (for distributed)
    """
    from .manager import get_connection_manager

    manager = get_connection_manager()
    event_dict = event.to_dict()

    # Broadcast to local WebSocket connections
    await manager.broadcast_to_project(project_id, event_dict)

    # Also publish to Redis for distributed deployments
    if use_redis:
        try:
            from cache.redis_cache import get_cache

            cache = get_cache()
            if cache and cache.is_connected:
                channel = f"project:{project_id}:updates"
                cache.publish(channel, event_dict)
        except Exception as e:
            logger.warning(f"Redis publish failed: {e}")


def sync_broadcast_agent_update(project_id: str, event: AgentEvent):
    """
    Synchronous version of broadcast for use in background threads & Celery tasks.

    Dispatches directly to local WebSocket clients (in-memory) and also
    publishes to Redis pub/sub if Redis is available.
    """
    event_dict = event.to_dict()

    # 1. Direct in-memory WebSocket broadcast (for local/single-process background threads)
    try:
        from .manager import get_connection_manager
        manager = get_connection_manager()
        manager.sync_broadcast_to_project(project_id, event_dict)
    except Exception as e:
        logger.debug(f"Direct WebSocket broadcast failed: {e}")

    # 2. Redis pub/sub broadcast (for distributed Celery workers)
    try:
        import os
        import sys

        # Ensure project root is in sys.path
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)

        from cache.redis_cache import get_cache

        cache = get_cache()
        if cache and cache.is_connected:
            channel = f"project:{project_id}:updates"
            cache.publish(channel, event_dict)
            logger.debug(f"Broadcast event to Redis channel {channel}: {event.type}")
        else:
            logger.debug(f"Redis cache not connected, skipping Redis pub/sub for {project_id}")
    except Exception as e:
        logger.debug(f"Failed to publish event to Redis: {e}")


# =========================================================================
# Agent Progress Tracker (7-phase multi-agent pipeline helper)
# =========================================================================


class AgentProgressTracker:
    """
    Helper class for tracking and broadcasting agent progress in Celery tasks.
    Configured for the 7-phase Scholar Agent v3.2 architecture:
      - discovery: 15%
      - ingestion: 25%
      - matrix_builder: 20%
      - synthesizer: 20%
      - critic: 10%
      - auditor: 10%
    """

    AGENT_ORDER = [
        "discovery",
        "ingestion",
        "matrix_builder",
        "synthesizer",
        "critic",
        "auditor",
    ]

    AGENT_WEIGHTS = {
        "discovery": 15.0,  # 15%
        "ingestion": 25.0,  # 25%
        "matrix_builder": 20.0,  # 20%
        "synthesizer": 20.0,  # 20%
        "critic": 10.0,  # 10%
        "auditor": 10.0,  # 10%
    }

    AGENT_ALIASES = {
        "planner": "discovery",
        "retriever": "ingestion",
        "analyzer": "matrix_builder",
        "matrix": "matrix_builder",
        "synthesizer": "synthesizer",
        "quality_checker": "critic",
        "critic": "critic",
        "auditor": "auditor",
        "supervisor": "discovery",
    }

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.current_agent: str | None = None
        self.agent_progress: dict[str, float] = {}
        self.completed_agents: list[str] = []

    def _normalize_agent_name(self, agent: str) -> str:
        """Normalize legacy and modern agent names to standard keys."""
        cleaned = agent.lower().replace("_agent", "").strip()
        return self.AGENT_ALIASES.get(cleaned, cleaned)

    def _calculate_total_progress(self) -> float:
        """Calculate overall progress based on 7-phase agent weights."""
        total = 0.0
        normalized_completed = {self._normalize_agent_name(a) for a in self.completed_agents}
        normalized_current = (
            self._normalize_agent_name(self.current_agent) if self.current_agent else None
        )

        for agent in self.AGENT_ORDER:
            weight = self.AGENT_WEIGHTS.get(agent, 0.0)
            if agent in normalized_completed:
                total += weight
            elif agent == normalized_current:
                agent_prog = self.agent_progress.get(
                    self.current_agent or "", self.agent_progress.get(agent, 0.0)
                )
                total += (agent_prog / 100.0) * weight

        return min(round(total, 2), 100.0)

    def start_agent(self, agent: str, message: str | None = None):
        """Signal that an agent has started."""
        normalized = self._normalize_agent_name(agent)
        self.current_agent = agent
        self.agent_progress[agent] = 0.0
        self.agent_progress[normalized] = 0.0

        event = AgentEvent(
            type=EventType.AGENT_STARTED,
            agent=agent,
            project_id=self.project_id,
            message=message or f"{agent.capitalize()} agent started",
            progress=self._calculate_total_progress(),
            data={"agent": agent, "status": "running"},
        )
        sync_broadcast_agent_update(self.project_id, event)

    def complete_agent(self, agent: str, message: str | None = None):
        """Signal that an agent has completed."""
        normalized = self._normalize_agent_name(agent)
        self.agent_progress[agent] = 100.0
        self.agent_progress[normalized] = 100.0

        if agent not in self.completed_agents:
            self.completed_agents.append(agent)
        if normalized not in self.completed_agents:
            self.completed_agents.append(normalized)

        event = AgentEvent(
            type=EventType.AGENT_COMPLETED,
            agent=agent,
            project_id=self.project_id,
            message=message or f"{agent.capitalize()} agent completed",
            progress=self._calculate_total_progress(),
            data={"agent": agent, "status": "completed"},
        )
        sync_broadcast_agent_update(self.project_id, event)

    def update_progress(self, agent_progress: float, message: str | None = None):
        """Update progress within the current agent."""
        if not self.current_agent:
            return

        self.agent_progress[self.current_agent] = agent_progress
        normalized = self._normalize_agent_name(self.current_agent)
        self.agent_progress[normalized] = agent_progress

        event = create_progress_event(
            project_id=self.project_id,
            agent=self.current_agent,
            progress=self._calculate_total_progress(),
            message=message,
            current=int(agent_progress),
            total=100,
        )
        sync_broadcast_agent_update(self.project_id, event)

    def progress_callback_adapter(self, agent_name: str, message: str, percent: float):
        """
        Adapter to match the callback signature expected by ScholarAgentOrchestrator.
        Signature: callback(agent_name, message, percent)
        """
        normalized_agent = self._normalize_agent_name(agent_name)

        # Handle agent transitions
        if self.current_agent != agent_name and self.current_agent != normalized_agent:
            if self.current_agent and self._normalize_agent_name(self.current_agent) not in [
                self._normalize_agent_name(a) for a in self.completed_agents
            ]:
                self.complete_agent(self.current_agent)

            self.start_agent(agent_name, message)

        if percent >= 100.0:
            self.complete_agent(agent_name, message)
        else:
            self.update_progress(percent, message)

    def log(self, message: str, level: str = "info"):
        """Log a message for the current agent."""
        event = create_log_event(
            project_id=self.project_id,
            agent=self.current_agent or "system",
            message=message,
            level=level,
        )
        sync_broadcast_agent_update(self.project_id, event)

    def discovery_started(self, queries: list[str] | list[dict[str, Any]] | None = None):
        """Broadcast discovery started event."""
        event = create_discovery_started_event(
            project_id=self.project_id,
            queries=queries,
            progress=self._calculate_total_progress(),
        )
        sync_broadcast_agent_update(self.project_id, event)

    def paper_discovered(
        self,
        paper_id: str,
        title: str,
        authors: list[str] | None = None,
        year: int | None = None,
        venue: str | None = None,
        source: str | None = None,
        citation_count: int | None = None,
        relevance_score: float | None = None,
        data: dict[str, Any] | None = None,
    ):
        """Broadcast paper discovered event."""
        event = create_paper_discovered_event(
            project_id=self.project_id,
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            source=source,
            citation_count=citation_count,
            relevance_score=relevance_score,
            data=data,
            progress=self._calculate_total_progress(),
        )
        sync_broadcast_agent_update(self.project_id, event)

    def pdf_parsed(
        self,
        paper_id: str,
        title: str,
        is_full_text: bool = False,
        sections_count: int = 0,
        tables_count: int = 0,
        figures_count: int = 0,
        data: dict[str, Any] | None = None,
    ):
        """Broadcast pdf parsed event."""
        event = create_pdf_parsed_event(
            project_id=self.project_id,
            paper_id=paper_id,
            title=title,
            is_full_text=is_full_text,
            sections_count=sections_count,
            tables_count=tables_count,
            figures_count=figures_count,
            data=data,
            progress=self._calculate_total_progress(),
        )
        sync_broadcast_agent_update(self.project_id, event)

    def matrix_row_added(self, row: Any, data: dict[str, Any] | None = None):
        """Broadcast matrix row added event."""
        event = create_matrix_row_added_event(
            project_id=self.project_id,
            row=row,
            data=data,
            progress=self._calculate_total_progress(),
        )
        sync_broadcast_agent_update(self.project_id, event)

    def thematic_draft_ready(
        self,
        section_count: int = 0,
        debates_count: int = 0,
        gaps_count: int = 0,
        iteration: int = 0,
        sections: list[dict[str, Any]] | None = None,
        data: dict[str, Any] | None = None,
    ):
        """Broadcast thematic draft ready event."""
        event = create_thematic_draft_ready_event(
            project_id=self.project_id,
            section_count=section_count,
            debates_count=debates_count,
            gaps_count=gaps_count,
            iteration=iteration,
            sections=sections,
            data=data,
            progress=self._calculate_total_progress(),
        )
        sync_broadcast_agent_update(self.project_id, event)

    def critic_verdict(
        self,
        score: float,
        should_refine: bool,
        iteration: int = 0,
        dimension_scores: dict[str, float] | None = None,
        weaknesses: list[str] | None = None,
        guidance: str | None = None,
        data: dict[str, Any] | None = None,
    ):
        """Broadcast critic verdict event."""
        event = create_critic_verdict_event(
            project_id=self.project_id,
            score=score,
            should_refine=should_refine,
            iteration=iteration,
            dimension_scores=dimension_scores,
            weaknesses=weaknesses,
            guidance=guidance,
            data=data,
            progress=self._calculate_total_progress(),
        )
        sync_broadcast_agent_update(self.project_id, event)

    def fact_checked(
        self,
        precision_score: float,
        passed: bool,
        entailed_count: int = 0,
        neutral_count: int = 0,
        contradiction_count: int = 0,
        total_propositions: int = 0,
        data: dict[str, Any] | None = None,
    ):
        """Broadcast fact checked citation audit event."""
        event = create_fact_checked_event(
            project_id=self.project_id,
            precision_score=precision_score,
            passed=passed,
            entailed_count=entailed_count,
            neutral_count=neutral_count,
            contradiction_count=contradiction_count,
            total_propositions=total_propositions,
            data=data,
            progress=self._calculate_total_progress(),
        )
        sync_broadcast_agent_update(self.project_id, event)

    def paper_found(self, title: str, data: dict[str, Any] | None = None):
        """Legacy notification that a paper was found."""
        event = create_paper_event(
            project_id=self.project_id,
            event_type=EventType.PAPER_FOUND,
            paper_title=title,
            paper_data=data,
        )
        sync_broadcast_agent_update(self.project_id, event)

    def paper_analyzed(
        self,
        title: str,
        relevance_score: float | None = None,
        current: int | None = None,
        total: int | None = None,
    ):
        """Legacy notification that a paper was analyzed."""
        data: dict[str, Any] = {}
        if relevance_score is not None:
            data["relevance_score"] = relevance_score
        if current is not None:
            data["current"] = current
        if total is not None:
            data["total"] = total

        event = create_paper_event(
            project_id=self.project_id,
            event_type=EventType.PAPER_ANALYZED,
            paper_title=title,
            paper_data=data,
        )
        sync_broadcast_agent_update(self.project_id, event)

    def complete(
        self,
        report: Any = None,
        papers_analyzed: int = 0,
        synthesis_words: int = 0,
    ):
        """Signal project and pipeline completion."""
        # 1. Broadcast legacy completion event
        event_legacy = create_completion_event(
            project_id=self.project_id,
            success=True,
            summary={
                "papers_analyzed": papers_analyzed,
                "synthesis_words": synthesis_words,
                "agents_completed": self.completed_agents,
            },
        )
        sync_broadcast_agent_update(self.project_id, event_legacy)

        # 2. Broadcast standard v3.2 pipeline_completed event
        event_pipeline = create_pipeline_completed_event(
            project_id=self.project_id,
            report=report,
            summary={
                "papers_analyzed": papers_analyzed,
                "synthesis_words": synthesis_words,
                "agents_completed": self.completed_agents,
            },
        )
        sync_broadcast_agent_update(self.project_id, event_pipeline)

    def error(self, error_message: str):
        """Signal project and pipeline error."""
        # 1. Legacy error
        event_legacy = create_completion_event(
            project_id=self.project_id, success=False, error_message=error_message
        )
        sync_broadcast_agent_update(self.project_id, event_legacy)

        # 2. Standard v3.2 pipeline_error
        event_pipeline = create_pipeline_error_event(
            project_id=self.project_id, error_message=error_message
        )
        sync_broadcast_agent_update(self.project_id, event_pipeline)
