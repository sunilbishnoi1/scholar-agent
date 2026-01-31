# WebSocket Event Types and Broadcasting Utilities
# Standardized event format for real-time agent updates

import logging
import json
from enum import Enum
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Any, Dict, List
import os

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events that can be broadcast to clients."""
    
    # Connection events
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    
    # Agent lifecycle events
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_ERROR = "agent_error"
    
    # Progress events
    STATUS_UPDATE = "status"
    PROGRESS_UPDATE = "progress"
    LOG_MESSAGE = "log"
    
    # Paper processing events
    PAPER_FOUND = "paper_found"
    PAPER_ANALYZED = "paper_analyzed"
    
    # Project events
    PROJECT_COMPLETED = "complete"
    PROJECT_ERROR = "error"


@dataclass
class AgentEvent:
    """
    Standardized event format for agent updates.
    
    Attributes:
        type: The type of event
        agent: Which agent generated this event (planner, retriever, analyzer, synthesizer)
        project_id: The project this event relates to
        message: Human-readable message
        progress: Progress percentage (0-100)
        data: Additional event-specific data
        timestamp: When the event occurred
    """
    type: EventType
    agent: Optional[str] = None
    project_id: Optional[str] = None
    message: Optional[str] = None
    progress: Optional[float] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "timestamp": self.timestamp
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
# Event factory functions for common events
# =========================================================================

def create_status_event(
    project_id: str,
    agent: str,
    status: str,
    message: str = None
) -> AgentEvent:
    """Create a status update event."""
    return AgentEvent(
        type=EventType.STATUS_UPDATE,
        agent=agent,
        project_id=project_id,
        message=message or f"{agent} is now {status}",
        data={"status": status}
    )


def create_progress_event(
    project_id: str,
    agent: str,
    progress: float,
    message: str = None,
    current: int = None,
    total: int = None
) -> AgentEvent:
    """Create a progress update event."""
    data = {"progress_percent": progress}
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
        data=data
    )


def create_log_event(
    project_id: str,
    agent: str,
    message: str,
    level: str = "info"
) -> AgentEvent:
    """Create a log message event."""
    return AgentEvent(
        type=EventType.LOG_MESSAGE,
        agent=agent,
        project_id=project_id,
        message=message,
        data={"level": level}
    )


def create_paper_event(
    project_id: str,
    event_type: EventType,
    paper_title: str,
    paper_data: dict = None
) -> AgentEvent:
    """Create a paper-related event."""
    data = {"paper_title": paper_title}
    if paper_data:
        data.update(paper_data)
    
    return AgentEvent(
        type=event_type,
        agent="retriever" if event_type == EventType.PAPER_FOUND else "analyzer",
        project_id=project_id,
        message=f"Paper: {paper_title[:50]}...",
        data=data
    )


def create_completion_event(
    project_id: str,
    success: bool,
    summary: dict = None,
    error_message: str = None
) -> AgentEvent:
    """Create a project completion event."""
    if success:
        return AgentEvent(
            type=EventType.PROJECT_COMPLETED,
            project_id=project_id,
            message="Literature review completed successfully",
            progress=100,
            data=summary or {}
        )
    else:
        return AgentEvent(
            type=EventType.PROJECT_ERROR,
            project_id=project_id,
            message=error_message or "An error occurred during processing",
            data={"error": error_message}
        )


# =========================================================================
# Broadcasting utilities (integrates with Redis pub/sub)
# =========================================================================

async def broadcast_agent_update(
    project_id: str,
    event: AgentEvent,
    use_redis: bool = True
):
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
            from cache import get_cache
            cache = get_cache()
            channel = f"project:{project_id}:updates"
            cache.publish(channel, event_dict)
        except Exception as e:
            logger.warning(f"Redis publish failed: {e}")


def sync_broadcast_agent_update(
    project_id: str,
    event: AgentEvent
):
    """
    Synchronous version of broadcast for use in Celery tasks.
    
    Uses Redis pub/sub to notify WebSocket servers which then
    broadcast to connected clients.
    """
    try:
        from cache import get_cache
        cache = get_cache()
        channel = f"project:{project_id}:updates"
        event_dict = event.to_dict()
        cache.publish(channel, event_dict)
        logger.debug(f"Published event to {channel}: {event.type}")
    except Exception as e:
        logger.warning(f"Failed to broadcast event: {e}")


# =========================================================================
# Agent Progress Tracker (helper class for Celery tasks)
# =========================================================================

class AgentProgressTracker:
    """
    Helper class for tracking and broadcasting agent progress in Celery tasks.
    
    Usage:
        tracker = AgentProgressTracker(project_id)
        
        tracker.start_agent("planner")
        tracker.log("Generating search terms...")
        tracker.update_progress(50)
        tracker.complete_agent("planner")
        
        tracker.start_agent("retriever")
        for i, paper in enumerate(papers):
            tracker.paper_found(paper["title"])
            tracker.update_progress((i + 1) / len(papers) * 100)
        tracker.complete_agent("retriever")
    """
    
    AGENT_ORDER = ["planner", "retriever", "analyzer", "synthesizer"]
    AGENT_WEIGHTS = {
        "planner": 10,      # 10% of total progress
        "retriever": 20,    # 20% of total progress
        "analyzer": 50,     # 50% of total progress
        "synthesizer": 20   # 20% of total progress
    }
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.current_agent: Optional[str] = None
        self.agent_progress: Dict[str, float] = {}
        self.completed_agents: List[str] = []
    
    def _calculate_total_progress(self) -> float:
        """Calculate overall progress based on agent weights."""
        total = 0.0
        
        for agent in self.AGENT_ORDER:
            weight = self.AGENT_WEIGHTS.get(agent, 25)
            
            if agent in self.completed_agents:
                total += weight
            elif agent == self.current_agent:
                agent_progress = self.agent_progress.get(agent, 0)
                total += (agent_progress / 100) * weight
        
        return min(total, 100)
    
    def start_agent(self, agent: str, message: str = None):
        """Signal that an agent has started."""
        self.current_agent = agent
        self.agent_progress[agent] = 0
        
        event = AgentEvent(
            type=EventType.AGENT_STARTED,
            agent=agent,
            project_id=self.project_id,
            message=message or f"{agent.capitalize()} agent started",
            progress=self._calculate_total_progress(),
            data={"agent": agent, "status": "running"}
        )
        sync_broadcast_agent_update(self.project_id, event)
    
    def complete_agent(self, agent: str, message: str = None):
        """Signal that an agent has completed."""
        self.agent_progress[agent] = 100
        if agent not in self.completed_agents:
            self.completed_agents.append(agent)
        
        event = AgentEvent(
            type=EventType.AGENT_COMPLETED,
            agent=agent,
            project_id=self.project_id,
            message=message or f"{agent.capitalize()} agent completed",
            progress=self._calculate_total_progress(),
            data={"agent": agent, "status": "completed"}
        )
        sync_broadcast_agent_update(self.project_id, event)
    
    def update_progress(self, agent_progress: float, message: str = None):
        """Update progress within the current agent."""
        if not self.current_agent:
            return
        
        self.agent_progress[self.current_agent] = agent_progress
        
        event = create_progress_event(
            project_id=self.project_id,
            agent=self.current_agent,
            progress=self._calculate_total_progress(),
            message=message,
            current=int(agent_progress),
            total=100
        )
        sync_broadcast_agent_update(self.project_id, event)
    
    def log(self, message: str, level: str = "info"):
        """Log a message for the current agent."""
        event = create_log_event(
            project_id=self.project_id,
            agent=self.current_agent or "system",
            message=message,
            level=level
        )
        sync_broadcast_agent_update(self.project_id, event)
    
    def paper_found(self, title: str, data: dict = None):
        """Notify that a paper was found."""
        event = create_paper_event(
            project_id=self.project_id,
            event_type=EventType.PAPER_FOUND,
            paper_title=title,
            paper_data=data
        )
        sync_broadcast_agent_update(self.project_id, event)
    
    def paper_analyzed(self, title: str, relevance_score: float = None):
        """Notify that a paper was analyzed."""
        data = {}
        if relevance_score is not None:
            data["relevance_score"] = relevance_score
        
        event = create_paper_event(
            project_id=self.project_id,
            event_type=EventType.PAPER_ANALYZED,
            paper_title=title,
            paper_data=data
        )
        sync_broadcast_agent_update(self.project_id, event)
    
    def complete(self, papers_analyzed: int = 0, synthesis_words: int = 0):
        """Signal project completion."""
        event = create_completion_event(
            project_id=self.project_id,
            success=True,
            summary={
                "papers_analyzed": papers_analyzed,
                "synthesis_words": synthesis_words,
                "agents_completed": self.completed_agents
            }
        )
        sync_broadcast_agent_update(self.project_id, event)
    
    def error(self, error_message: str):
        """Signal project error."""
        event = create_completion_event(
            project_id=self.project_id,
            success=False,
            error_message=error_message
        )
        sync_broadcast_agent_update(self.project_id, event)
