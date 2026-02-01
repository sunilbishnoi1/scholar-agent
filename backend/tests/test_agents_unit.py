# Unit Tests for Agent System
# Tests for the LangGraph-based multi-agent architecture

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from agents.analyzer_agent import PaperAnalyzerAgent
from agents.planner_agent import ResearchPlannerAgent
from agents.quality_checker_agent import QualityCheckerAgent
from agents.retriever_agent import PaperRetrieverAgent
from agents.state import AgentResult, AgentState, AgentType, PaperData, create_initial_state
from agents.synthesizer_agent import SynthesisExecutorAgent
from agents.tools import (
    ToolResult,
    evaluate_synthesis_quality,
    extract_keywords_from_question,
    extract_paper_insights,
    identify_subtopics,
    score_paper_relevance,
)

# ============================================
# FIXTURES
# ============================================


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    mock = Mock()
    mock.chat = Mock(return_value='{"keywords": ["test1", "test2"]}')
    return mock


@pytest.fixture
def sample_state() -> AgentState:
    """Create a sample initial state for testing."""
    return create_initial_state(
        project_id="test-project-123",
        user_id="test-user-456",
        title="AI in Education",
        research_question="How does AI affect student learning outcomes?",
        max_papers=10,
        max_iterations=2,
    )


@pytest.fixture
def sample_papers() -> list[PaperData]:
    """Create sample paper data for testing."""
    return [
        {
            "id": "paper_1",
            "title": "Machine Learning for Student Performance Prediction",
            "abstract": "This paper explores using ML algorithms to predict student grades...",
            "authors": ["John Doe", "Jane Smith"],
            "url": "https://arxiv.org/abs/1234.5678",
            "source": "arXiv",
            "relevance_score": None,
            "analysis": None,
        },
        {
            "id": "paper_2",
            "title": "Deep Learning in Educational Assessment",
            "abstract": "We propose a deep learning model for automated essay scoring...",
            "authors": ["Alice Johnson"],
            "url": "https://semanticscholar.org/paper/abc123",
            "source": "Semantic Scholar",
            "relevance_score": None,
            "analysis": None,
        },
    ]


# ============================================
# STATE TESTS
# ============================================


class TestAgentState:
    """Tests for AgentState creation and management."""

    def test_create_initial_state_has_required_fields(self, sample_state):
        """Initial state should have all required fields."""
        assert sample_state["project_id"] == "test-project-123"
        assert sample_state["user_id"] == "test-user-456"
        assert sample_state["title"] == "AI in Education"
        assert sample_state["research_question"] == "How does AI affect student learning outcomes?"
        assert sample_state["max_papers"] == 10
        assert sample_state["max_iterations"] == 2

    def test_create_initial_state_has_empty_lists(self, sample_state):
        """Initial state should have empty lists for results."""
        assert sample_state["keywords"] == []
        assert sample_state["subtopics"] == []
        assert sample_state["papers"] == []
        assert sample_state["analyzed_papers"] == []
        assert sample_state["errors"] == []

    def test_create_initial_state_has_default_config(self, sample_state):
        """Initial state should have default configuration values."""
        assert sample_state["relevance_threshold"] == 60.0
        assert sample_state["academic_level"] == "graduate"
        assert sample_state["iteration"] == 0
        assert sample_state["status"] == "running"


class TestAgentResult:
    """Tests for AgentResult wrapper."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = AgentResult(success=True, data={"keywords": ["test"]}, metadata={"count": 1})
        assert result.success is True
        assert result.data == {"keywords": ["test"]}
        assert result.error is None

    def test_failed_result(self):
        """Test creating a failed result."""
        result = AgentResult(success=False, data=None, error="LLM call failed")
        assert result.success is False
        assert result.error == "LLM call failed"

    def test_result_to_message(self):
        """Test converting result to AgentMessage."""
        result = AgentResult(success=True, data="test data")
        message = result.to_message("planner", "extract_keywords")

        assert message["agent"] == "planner"
        assert message["action"] == "extract_keywords"
        assert message["content"] == "test data"
        assert "timestamp" in message


# ============================================
# PLANNER AGENT TESTS
# ============================================


class TestResearchPlannerAgent:
    """Tests for the ResearchPlannerAgent."""

    def test_planner_initializes_with_tools(self, mock_llm_client):
        """Planner should register its tools on initialization."""
        planner = ResearchPlannerAgent(mock_llm_client)

        assert "extract_keywords" in planner.tools
        assert "identify_subtopics" in planner.tools
        assert "refine_search_query" in planner.tools

    def test_legacy_generate_initial_plan_success(self, mock_llm_client):
        """Legacy method should parse valid JSON response."""
        mock_llm_client.chat.return_value = """
        {
            "keywords": ["machine learning", "education", "student performance"],
            "subtopics": ["Learning outcomes", "Implementation challenges"]
        }
        """

        planner = ResearchPlannerAgent(mock_llm_client)
        result = planner.generate_initial_plan("How does AI affect learning?", "AI in Education")

        assert "keywords" in result
        assert len(result["keywords"]) == 3
        assert "subtopics" in result
        assert len(result["subtopics"]) == 2

    def test_legacy_generate_initial_plan_handles_markdown(self, mock_llm_client):
        """Legacy method should handle markdown-wrapped JSON."""
        mock_llm_client.chat.return_value = """```json
        {
            "keywords": ["test1", "test2"],
            "subtopics": ["subtopic1"]
        }
        ```"""

        planner = ResearchPlannerAgent(mock_llm_client)
        result = planner.generate_initial_plan("Test?", "Test")

        assert result["keywords"] == ["test1", "test2"]

    def test_legacy_generate_initial_plan_handles_invalid_json(self, mock_llm_client):
        """Legacy method should return empty lists on invalid JSON."""
        mock_llm_client.chat.return_value = "This is not JSON"

        planner = ResearchPlannerAgent(mock_llm_client)
        result = planner.generate_initial_plan("Test?", "Test")

        assert result == {"keywords": [], "subtopics": []}

    def test_fallback_keyword_extraction(self, mock_llm_client):
        """Fallback extraction should produce reasonable keywords."""
        planner = ResearchPlannerAgent(mock_llm_client)

        keywords = planner._fallback_keyword_extraction(
            "How does machine learning affect student performance in higher education?",
            "AI in Education",
        )

        # Should extract meaningful words, excluding stopwords
        assert len(keywords) > 0
        assert "machine" in keywords or "learning" in keywords
        assert "the" not in keywords
        assert "how" not in keywords

    @pytest.mark.asyncio
    async def test_planner_run_updates_state(self, mock_llm_client, sample_state):
        """Running planner should update state with keywords and subtopics."""
        mock_llm_client.chat.side_effect = [
            '{"keywords": ["ai", "education", "learning"]}',
            '{"subtopics": ["Benefits", "Challenges"]}',
        ]

        planner = ResearchPlannerAgent(mock_llm_client)
        result_state = await planner.run(sample_state)

        assert result_state["current_agent"] == AgentType.PLANNER
        assert len(result_state["keywords"]) > 0
        assert "search_strategy" in result_state


# ============================================
# ANALYZER AGENT TESTS
# ============================================


class TestPaperAnalyzerAgent:
    """Tests for the PaperAnalyzerAgent."""

    def test_analyzer_initializes_with_tools(self, mock_llm_client):
        """Analyzer should register its tools on initialization."""
        analyzer = PaperAnalyzerAgent(mock_llm_client)

        assert "score_relevance" in analyzer.tools
        assert "extract_insights" in analyzer.tools

    def test_legacy_analyze_paper(self, mock_llm_client):
        """Legacy method should call LLM with proper prompt."""
        mock_llm_client.chat.return_value = '{"relevance_score": 85}'

        analyzer = PaperAnalyzerAgent(mock_llm_client)
        result = analyzer.analyze_paper(
            "Test Paper", "Test abstract", "Test content", "Test question?"
        )

        assert mock_llm_client.chat.called
        assert result == '{"relevance_score": 85}'

    @pytest.mark.asyncio
    async def test_analyzer_run_filters_by_relevance(
        self, mock_llm_client, sample_state, sample_papers
    ):
        """Analyzer should filter papers by relevance threshold."""
        # First paper: high relevance, second paper: low relevance
        mock_llm_client.chat.side_effect = [
            '{"score": 80, "justification": "Highly relevant"}',
            '{"key_findings": ["finding1"], "methodology": "ML", "limitations": [], "contribution": "Important", "key_quotes": []}',
            '{"score": 30, "justification": "Not relevant"}',
        ]

        sample_state["papers"] = sample_papers
        sample_state["relevance_threshold"] = 50.0

        analyzer = PaperAnalyzerAgent(mock_llm_client)
        result_state = await analyzer.run(sample_state)

        # Should have analyzed both papers
        assert len(result_state["analyzed_papers"]) == 2
        # Only high-relevance paper should be in high_quality_papers
        assert len(result_state["high_quality_papers"]) == 1
        assert result_state["high_quality_papers"][0]["title"] == sample_papers[0]["title"]


# ============================================
# SYNTHESIZER AGENT TESTS
# ============================================


class TestSynthesisExecutorAgent:
    """Tests for the SynthesisExecutorAgent."""

    def test_synthesizer_initializes_with_tools(self, mock_llm_client):
        """Synthesizer should register its tools on initialization."""
        synthesizer = SynthesisExecutorAgent(mock_llm_client)

        assert "synthesize_section" in synthesizer.tools
        assert "identify_research_gaps" in synthesizer.tools

    def test_legacy_synthesize_section(self, mock_llm_client):
        """Legacy method should produce synthesis text."""
        mock_llm_client.chat.return_value = "This is a synthesized literature review section..."

        synthesizer = SynthesisExecutorAgent(mock_llm_client)
        result = synthesizer.synthesize_section(
            "AI Applications", "Paper 1: ... Paper 2: ...", "graduate", 500
        )

        assert "synthesized" in result

    def test_combine_sections(self, mock_llm_client):
        """_combine_sections should create formatted output."""
        synthesizer = SynthesisExecutorAgent(mock_llm_client)

        sections = [
            {"subtopic": "Introduction", "content": "Intro content..."},
            {"subtopic": "Methods", "content": "Methods content..."},
        ]
        gaps = [{"description": "Gap 1", "importance": "High", "directions": "Future work"}]

        combined = synthesizer._combine_sections(sections, gaps, "Test Title", 5)

        assert "# Literature Review: Test Title" in combined
        assert "## Introduction" in combined
        assert "## Methods" in combined
        assert "## Research Gaps" in combined


# ============================================
# QUALITY CHECKER AGENT TESTS
# ============================================


class TestQualityCheckerAgent:
    """Tests for the QualityCheckerAgent."""

    def test_quality_checker_initializes(self, mock_llm_client):
        """Quality checker should initialize with threshold."""
        checker = QualityCheckerAgent(mock_llm_client)

        assert checker.quality_threshold == 70.0
        assert "evaluate_quality" in checker.tools

    @pytest.mark.asyncio
    async def test_quality_checker_passes_good_synthesis(self, mock_llm_client, sample_state):
        """Quality checker should pass synthesis above threshold."""
        mock_llm_client.chat.return_value = json.dumps(
            {
                "overall_score": 85,
                "criteria_scores": {"coherence": 90, "coverage": 80},
                "feedback": "Good synthesis",
                "should_refine": False,
            }
        )

        sample_state["synthesis"] = "This is a comprehensive literature review..."
        sample_state["high_quality_papers"] = [{"title": "Paper 1"}]

        checker = QualityCheckerAgent(mock_llm_client)
        result_state = await checker.run(sample_state)

        assert result_state["quality_score"] == 85
        assert result_state["status"] == "completed"

    @pytest.mark.asyncio
    async def test_quality_checker_requests_refinement(self, mock_llm_client, sample_state):
        """Quality checker should request refinement for low scores."""
        mock_llm_client.chat.return_value = json.dumps(
            {
                "overall_score": 50,
                "criteria_scores": {"coherence": 50},
                "feedback": "Needs improvement",
                "should_refine": True,
            }
        )

        sample_state["synthesis"] = "Poor synthesis..."
        sample_state["high_quality_papers"] = [{"title": "Paper 1"}]
        sample_state["iteration"] = 0
        sample_state["max_iterations"] = 3

        checker = QualityCheckerAgent(mock_llm_client)
        result_state = await checker.run(sample_state)

        assert result_state["quality_score"] == 50
        assert result_state["status"] == "needs_refinement"


# ============================================
# TOOLS TESTS
# ============================================


class TestAgentTools:
    """Tests for individual agent tools."""

    def test_extract_keywords_success(self, mock_llm_client):
        """extract_keywords_from_question should return keywords."""
        mock_llm_client.chat.return_value = '{"keywords": ["ai", "education", "learning"]}'

        result = extract_keywords_from_question(
            mock_llm_client, "How does AI affect education?", "AI in Education"
        )

        assert result.success is True
        assert result.data == ["ai", "education", "learning"]

    def test_extract_keywords_handles_invalid_json(self, mock_llm_client):
        """extract_keywords should handle invalid JSON gracefully."""
        mock_llm_client.chat.return_value = "Not valid JSON"

        result = extract_keywords_from_question(mock_llm_client, "Test?", "Test")

        assert result.success is False
        assert result.data == []

    def test_score_paper_relevance_success(self, mock_llm_client):
        """score_paper_relevance should return score and justification."""
        mock_llm_client.chat.return_value = '{"score": 75, "justification": "Relevant paper"}'

        result = score_paper_relevance(
            mock_llm_client,
            "Test Paper",
            "Test abstract about AI in education",
            "How does AI affect education?",
        )

        assert result.success is True
        assert result.data["score"] == 75
        assert "justification" in result.data

    def test_evaluate_synthesis_quality_success(self, mock_llm_client):
        """evaluate_synthesis_quality should return quality metrics."""
        mock_llm_client.chat.return_value = json.dumps(
            {
                "overall_score": 80,
                "criteria_scores": {"coherence": 85, "coverage": 75},
                "feedback": "Well-structured review",
                "should_refine": False,
            }
        )

        result = evaluate_synthesis_quality(
            mock_llm_client, "This is a literature review...", "Research question?", 10
        )

        assert result.success is True
        assert result.data["overall_score"] == 80
        assert result.data["should_refine"] is False


# ============================================
# RETRIEVER AGENT TESTS
# ============================================


class TestPaperRetrieverAgent:
    """Tests for the PaperRetrieverAgent."""

    def test_retriever_initializes(self):
        """Retriever should initialize without LLM client."""
        retriever = PaperRetrieverAgent()

        assert retriever.paper_retriever is not None
        assert "search_all_sources" in retriever.tools

    def test_deduplicate_papers(self):
        """_deduplicate_papers should remove duplicates by title."""
        retriever = PaperRetrieverAgent()

        papers = [
            {
                "title": "Paper One",
                "abstract": "Abstract 1",
                "authors": [],
                "url": "url1",
                "source": "arXiv",
            },
            {
                "title": "Paper One",
                "abstract": "Abstract 1 duplicate",
                "authors": [],
                "url": "url2",
                "source": "SS",
            },
            {
                "title": "Paper Two",
                "abstract": "Abstract 2",
                "authors": [],
                "url": "url3",
                "source": "arXiv",
            },
        ]

        unique = retriever._deduplicate_papers(papers)

        assert len(unique) == 2
        titles = [p["title"] for p in unique]
        assert "Paper One" in titles
        assert "Paper Two" in titles


# ============================================
# INTEGRATION-STYLE TESTS
# ============================================


class TestAgentIntegration:
    """Integration tests for the agent pipeline."""

    @pytest.mark.asyncio
    async def test_full_agent_pipeline_mock(self, mock_llm_client, sample_state):
        """Test running multiple agents in sequence."""
        # Setup mock responses for each agent
        mock_llm_client.chat.side_effect = [
            # Planner keywords
            '{"keywords": ["ai", "education"]}',
            # Planner subtopics
            '{"subtopics": ["Benefits", "Challenges"]}',
        ]

        # Run planner
        planner = ResearchPlannerAgent(mock_llm_client)
        state = await planner.run(sample_state)

        assert len(state["keywords"]) > 0
        assert state["current_agent"] == AgentType.PLANNER


# ============================================
# RUN TESTS
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
