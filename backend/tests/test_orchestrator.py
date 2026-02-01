# Tests for the LangGraph Orchestrator
# Tests the state machine and routing logic

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from agents.orchestrator import ResearchOrchestrator, create_orchestrator
from agents.state import AgentState, AgentType, create_initial_state


class TestResearchOrchestrator:
    """Tests for the ResearchOrchestrator class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock = Mock()
        mock.chat = Mock(return_value='{"keywords": ["test"]}')
        return mock

    @pytest.fixture
    def orchestrator(self, mock_llm_client):
        """Create an orchestrator instance."""
        return ResearchOrchestrator(mock_llm_client)

    def test_orchestrator_initializes_agents(self, orchestrator):
        """Orchestrator should initialize all required agents."""
        assert orchestrator.planner is not None
        assert orchestrator.retriever is not None
        assert orchestrator.analyzer is not None
        assert orchestrator.synthesizer is not None
        assert orchestrator.quality_checker is not None

    def test_orchestrator_builds_graph(self, orchestrator):
        """Orchestrator should build a LangGraph state machine."""
        assert orchestrator.graph is not None

    def test_should_continue_or_end_completes_when_status_completed(self, orchestrator):
        """Routing should return 'complete' when status is completed."""
        state = create_initial_state(
            project_id="test", user_id="test", title="Test", research_question="Test?"
        )
        state["status"] = "completed"

        result = orchestrator._should_continue_or_end(state)

        assert result == "complete"

    def test_should_continue_or_end_refines_when_needed(self, orchestrator):
        """Routing should return 'refine' when status is needs_refinement."""
        state = create_initial_state(
            project_id="test",
            user_id="test",
            title="Test",
            research_question="Test?",
            max_iterations=3,
        )
        state["status"] = "needs_refinement"
        state["iteration"] = 0

        result = orchestrator._should_continue_or_end(state)

        assert result == "refine"
        # Iteration should be incremented
        assert state["iteration"] == 1

    def test_should_continue_or_end_completes_at_max_iterations(self, orchestrator):
        """Routing should complete when max iterations reached."""
        state = create_initial_state(
            project_id="test",
            user_id="test",
            title="Test",
            research_question="Test?",
            max_iterations=3,
        )
        state["status"] = "needs_refinement"
        state["iteration"] = 3  # At max

        result = orchestrator._should_continue_or_end(state)

        assert result == "complete"

    def test_progress_callback_called(self, mock_llm_client):
        """Progress callback should be called during execution."""
        callback_calls = []

        def progress_callback(agent, message, percent):
            callback_calls.append((agent, message, percent))

        orchestrator = ResearchOrchestrator(mock_llm_client, progress_callback)

        # Manually call progress report
        orchestrator._report_progress("planner", "Testing...", 50)

        assert len(callback_calls) == 1
        assert callback_calls[0] == ("planner", "Testing...", 50)

    def test_progress_callback_handles_errors(self, mock_llm_client):
        """Progress callback errors should not crash the orchestrator."""

        def failing_callback(agent, message, percent):
            raise ValueError("Callback error")

        orchestrator = ResearchOrchestrator(mock_llm_client, failing_callback)

        # Should not raise
        orchestrator._report_progress("planner", "Testing...", 50)


class TestOrchestratorFactory:
    """Tests for the create_orchestrator factory function."""

    def test_create_orchestrator_returns_instance(self):
        """Factory should return an orchestrator instance."""
        mock_llm = Mock()

        orchestrator = create_orchestrator(mock_llm)

        assert isinstance(orchestrator, ResearchOrchestrator)

    def test_create_orchestrator_with_callback(self):
        """Factory should accept progress callback."""
        mock_llm = Mock()
        callback = Mock()

        orchestrator = create_orchestrator(mock_llm, callback)

        assert orchestrator.progress_callback == callback


class TestOrchestratorAgentNodes:
    """Tests for individual agent node runners."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock = Mock()
        mock.chat = Mock(return_value='{"keywords": ["test"]}')
        return mock

    @pytest.fixture
    def orchestrator(self, mock_llm_client):
        """Create an orchestrator instance."""
        return ResearchOrchestrator(mock_llm_client)

    @pytest.fixture
    def sample_state(self):
        """Create a sample state for testing."""
        return create_initial_state(
            project_id="test-project",
            user_id="test-user",
            title="AI in Education",
            research_question="How does AI affect learning?",
        )

    @pytest.mark.asyncio
    async def test_run_planner_node(self, orchestrator, sample_state):
        """Planner node should update state."""
        orchestrator.planner.run = AsyncMock(return_value=sample_state)

        result = await orchestrator._run_planner(sample_state)

        orchestrator.planner.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_retriever_node(self, orchestrator, sample_state):
        """Retriever node should update state."""
        orchestrator.retriever.run = AsyncMock(return_value=sample_state)

        result = await orchestrator._run_retriever(sample_state)

        orchestrator.retriever.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_analyzer_node(self, orchestrator, sample_state):
        """Analyzer node should update state."""
        sample_state["papers"] = [{"title": "Test"}]
        orchestrator.analyzer.run = AsyncMock(return_value=sample_state)

        result = await orchestrator._run_analyzer(sample_state)

        orchestrator.analyzer.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_synthesizer_node(self, orchestrator, sample_state):
        """Synthesizer node should update state."""
        orchestrator.synthesizer.run = AsyncMock(return_value=sample_state)

        result = await orchestrator._run_synthesizer(sample_state)

        orchestrator.synthesizer.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_quality_checker_node(self, orchestrator, sample_state):
        """Quality checker node should update state."""
        orchestrator.quality_checker.run = AsyncMock(return_value=sample_state)

        result = await orchestrator._run_quality_checker(sample_state)

        orchestrator.quality_checker.run.assert_called_once()


class TestOrchestratorRun:
    """Tests for the full orchestrator run method."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock = Mock()
        # Default response for any chat call
        mock.chat = Mock(return_value='{"keywords": ["test"], "subtopics": ["topic"]}')
        return mock

    @pytest.mark.asyncio
    async def test_run_creates_initial_state(self, mock_llm_client):
        """Run should create proper initial state."""
        orchestrator = ResearchOrchestrator(mock_llm_client)

        # Mock the graph to avoid full execution
        mock_state = create_initial_state(
            project_id="test", user_id="user", title="Test", research_question="Test?"
        )
        mock_state["status"] = "completed"
        orchestrator.graph = Mock()
        orchestrator.graph.ainvoke = AsyncMock(return_value=mock_state)

        result = await orchestrator.run(
            project_id="test", user_id="user", title="Test", research_question="Test?"
        )

        assert result["project_id"] == "test"
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_handles_exceptions(self, mock_llm_client):
        """Run should handle exceptions gracefully."""
        orchestrator = ResearchOrchestrator(mock_llm_client)

        orchestrator.graph = Mock()
        orchestrator.graph.ainvoke = AsyncMock(side_effect=ValueError("Test error"))

        result = await orchestrator.run(
            project_id="test", user_id="user", title="Test", research_question="Test?"
        )

        assert result["status"] == "error"
        assert len(result["errors"]) > 0

    def test_run_sync_wrapper(self, mock_llm_client):
        """run_sync should work as synchronous wrapper."""
        orchestrator = ResearchOrchestrator(mock_llm_client)

        # Mock the graph
        mock_state = create_initial_state(
            project_id="test", user_id="user", title="Test", research_question="Test?"
        )
        mock_state["status"] = "completed"
        orchestrator.graph = Mock()
        orchestrator.graph.ainvoke = AsyncMock(return_value=mock_state)

        result = orchestrator.run_sync(
            project_id="test", user_id="user", title="Test", research_question="Test?"
        )

        assert result["status"] == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
