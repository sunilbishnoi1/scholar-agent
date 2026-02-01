# Tests for WebSocket Manager (Phase 3: Production Features)
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


class TestConnectionManager:
    """Tests for the WebSocket ConnectionManager."""

    @pytest.fixture
    def manager(self):
        """Create a ConnectionManager instance."""
        from realtime.manager import ConnectionManager
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.receive_text = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_connect_accepts_websocket(self, manager, mock_websocket):
        """Test that connect accepts the WebSocket connection."""
        result = await manager.connect(mock_websocket, "user-123", "project-456")

        assert result is True
        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_sends_confirmation(self, manager, mock_websocket):
        """Test that connect sends a confirmation message."""
        await manager.connect(mock_websocket, "user-123", "project-456")

        # Should have sent a connected confirmation
        mock_websocket.send_json.assert_called_once()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == "connected"
        assert call_args["project_id"] == "project-456"

    @pytest.mark.asyncio
    async def test_connect_tracks_connection(self, manager, mock_websocket):
        """Test that connect tracks the connection."""
        await manager.connect(mock_websocket, "user-123", "project-456")

        assert manager.get_project_subscriber_count("project-456") == 1

    @pytest.mark.asyncio
    async def test_disconnect_removes_tracking(self, manager, mock_websocket):
        """Test that disconnect removes connection tracking."""
        await manager.connect(mock_websocket, "user-123", "project-456")
        await manager.disconnect(mock_websocket)

        assert manager.get_project_subscriber_count("project-456") == 0

    @pytest.mark.asyncio
    async def test_broadcast_to_project_sends_to_subscribers(self, manager, mock_websocket):
        """Test that broadcast sends messages to all project subscribers."""
        # Connect two websockets to same project
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()

        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        await manager.connect(ws1, "user-1", "project-123")
        await manager.connect(ws2, "user-2", "project-123")

        # Broadcast message
        await manager.broadcast_to_project("project-123", {"type": "test", "data": "hello"})

        # Both should have received the message (plus the initial connected message)
        assert ws1.send_json.call_count >= 2
        assert ws2.send_json.call_count >= 2

    @pytest.mark.asyncio
    async def test_broadcast_excludes_specified_websocket(self, manager, mock_websocket):
        """Test that broadcast can exclude a specific websocket."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()

        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        await manager.connect(ws1, "user-1", "project-123")
        await manager.connect(ws2, "user-2", "project-123")

        # Broadcast excluding ws1
        await manager.broadcast_to_project("project-123", {"type": "test"}, exclude=ws1)

        # ws1 should only have the connected message, ws2 should have both
        assert ws1.send_json.call_count == 1  # Only connected message
        assert ws2.send_json.call_count == 2  # Connected + broadcast

    @pytest.mark.asyncio
    async def test_send_to_user_sends_to_all_user_connections(self, manager, mock_websocket):
        """Test that send_to_user sends to all connections for a user."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()

        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        # Same user, different projects
        await manager.connect(ws1, "user-123", "project-1")
        await manager.connect(ws2, "user-123", "project-2")

        # Send to user
        await manager.send_to_user("user-123", {"type": "notification"})

        # Both connections should receive the message
        assert ws1.send_json.call_count >= 2
        assert ws2.send_json.call_count >= 2

    @pytest.mark.asyncio
    async def test_get_stats_returns_correct_counts(self, manager, mock_websocket):
        """Test that get_stats returns accurate statistics."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()

        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        await manager.connect(ws1, "user-1", "project-1")
        await manager.connect(ws2, "user-2", "project-2")

        stats = manager.get_stats()

        assert stats["active_connections"] == 2
        assert stats["active_projects"] == 2
        assert stats["connections_by_project"]["project-1"] == 1
        assert stats["connections_by_project"]["project-2"] == 1

    @pytest.mark.asyncio
    async def test_subscribe_to_project_adds_subscription(self, manager, mock_websocket):
        """Test that subscribe_to_project adds new subscription."""
        await manager.connect(mock_websocket, "user-123", "project-1")
        await manager.subscribe_to_project(mock_websocket, "project-2")

        assert manager.get_project_subscriber_count("project-1") == 1
        assert manager.get_project_subscriber_count("project-2") == 1


class TestAgentEvents:
    """Tests for agent event creation utilities."""

    def test_create_status_event(self):
        """Test creating a status event."""
        from realtime.events import EventType, create_status_event

        event = create_status_event(
            project_id="proj-123",
            agent="analyzer",
            status="running",
            message="Analyzing papers..."
        )

        assert event.type == EventType.STATUS_UPDATE
        assert event.agent == "analyzer"
        assert event.project_id == "proj-123"
        assert "running" in event.data["status"]

    def test_create_progress_event(self):
        """Test creating a progress event."""
        from realtime.events import EventType, create_progress_event

        event = create_progress_event(
            project_id="proj-123",
            agent="analyzer",
            progress=75.5,
            message="Processing paper 15/20",
            current=15,
            total=20
        )

        assert event.type == EventType.PROGRESS_UPDATE
        assert event.progress == 75.5
        assert event.data["current"] == 15
        assert event.data["total"] == 20

    def test_create_log_event(self):
        """Test creating a log event."""
        from realtime.events import EventType, create_log_event

        event = create_log_event(
            project_id="proj-123",
            agent="synthesizer",
            message="Generating synthesis...",
            level="info"
        )

        assert event.type == EventType.LOG_MESSAGE
        assert event.message == "Generating synthesis..."
        assert event.data["level"] == "info"

    def test_create_completion_event_success(self):
        """Test creating a successful completion event."""
        from realtime.events import EventType, create_completion_event

        event = create_completion_event(
            project_id="proj-123",
            success=True,
            summary={"papers_analyzed": 20, "synthesis_words": 500}
        )

        assert event.type == EventType.PROJECT_COMPLETED
        assert event.progress == 100
        assert event.data["papers_analyzed"] == 20

    def test_create_completion_event_error(self):
        """Test creating an error completion event."""
        from realtime.events import EventType, create_completion_event

        event = create_completion_event(
            project_id="proj-123",
            success=False,
            error_message="API rate limit exceeded"
        )

        assert event.type == EventType.PROJECT_ERROR
        assert "rate limit" in event.data["error"]

    def test_agent_event_to_dict(self):
        """Test converting AgentEvent to dictionary."""
        from realtime.events import AgentEvent, EventType

        event = AgentEvent(
            type=EventType.PROGRESS_UPDATE,
            agent="analyzer",
            project_id="proj-123",
            progress=50.0,
            message="Halfway done"
        )

        event_dict = event.to_dict()

        assert event_dict["type"] == "progress"
        assert event_dict["agent"] == "analyzer"
        assert event_dict["progress"] == 50.0
        assert "timestamp" in event_dict


class TestAgentProgressTracker:
    """Tests for the AgentProgressTracker helper class."""

    def test_progress_tracker_initialization(self):
        """Test that progress tracker initializes correctly."""
        from realtime.events import AgentProgressTracker

        tracker = AgentProgressTracker("project-123")

        assert tracker.project_id == "project-123"
        assert tracker.current_agent is None
        assert tracker.completed_agents == []

    def test_progress_calculation_weights(self):
        """Test that progress is calculated correctly with weights."""
        from realtime.events import AgentProgressTracker

        tracker = AgentProgressTracker("project-123")

        # Complete planner (10% weight)
        tracker.completed_agents = ["planner"]
        tracker.current_agent = "retriever"
        tracker.agent_progress = {"planner": 100, "retriever": 50}

        progress = tracker._calculate_total_progress()

        # Should be: 10% (planner complete) + 50% of 20% (retriever half done) = 20%
        assert progress == 20.0

    def test_progress_tracker_agent_lifecycle(self):
        """Test starting and completing agents."""
        from realtime.events import AgentProgressTracker

        with patch('realtime.events.sync_broadcast_agent_update'):
            tracker = AgentProgressTracker("project-123")

            tracker.start_agent("planner", "Planning search strategy")
            assert tracker.current_agent == "planner"

            tracker.complete_agent("planner")
            assert "planner" in tracker.completed_agents

    def test_progress_tracker_logging(self):
        """Test that logging works correctly."""
        from realtime.events import AgentProgressTracker

        with patch('realtime.events.sync_broadcast_agent_update') as mock_broadcast:
            tracker = AgentProgressTracker("project-123")
            tracker.current_agent = "analyzer"

            tracker.log("Processing paper...", level="info")

            mock_broadcast.assert_called_once()


class TestGetConnectionManagerSingleton:
    """Tests for connection manager singleton pattern."""

    def test_get_connection_manager_returns_same_instance(self):
        """Test that get_connection_manager returns the same instance."""
        from realtime.manager import get_connection_manager

        manager1 = get_connection_manager()
        manager2 = get_connection_manager()

        assert manager1 is manager2
