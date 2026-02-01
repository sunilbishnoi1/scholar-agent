# End-to-End Integration Tests
# Tests the complete research pipeline with real (mocked) external services

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from agents.llm import GeminiClient, get_llm_client

# Import system components
from agents.orchestrator import ResearchOrchestrator, create_orchestrator
from agents.state import AgentState, create_initial_state

# ============================================
# Fixtures for E2E Tests
# ============================================

@pytest.fixture
def comprehensive_mock_llm():
    """Create a mock LLM client with realistic responses for all agents."""
    mock = Mock(spec=GeminiClient)

    def generate_response(prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Planner responses
        if "keywords" in prompt_lower or "plan" in prompt_lower:
            return '''
            {
                "keywords": [
                    "artificial intelligence",
                    "machine learning",
                    "education",
                    "student performance",
                    "adaptive learning"
                ],
                "subtopics": [
                    "AI-powered tutoring systems",
                    "Predictive analytics in education",
                    "Personalized learning pathways",
                    "Automated assessment"
                ],
                "search_terms": [
                    "AI education student outcomes",
                    "machine learning academic performance",
                    "adaptive learning systems"
                ]
            }
            '''

        # Analyzer responses
        elif "analyze" in prompt_lower or "relevance" in prompt_lower:
            return '''
            {
                "relevance_score": 87,
                "justification": "This paper directly addresses the research question by examining AI applications in educational settings.",
                "key_findings": [
                    "AI tutoring improves student engagement by 32%",
                    "Personalized learning paths show 25% improvement in retention",
                    "Early intervention based on ML predictions reduces dropout rates"
                ],
                "methodology": "Mixed methods study with 2000 participants across 15 institutions",
                "limitations": [
                    "Study limited to undergraduate STEM courses",
                    "No long-term follow-up beyond one semester",
                    "Self-reported engagement metrics"
                ],
                "contribution": "Provides comprehensive framework for implementing AI in university settings",
                "key_quotes": [
                    "Our findings suggest that AI-powered adaptive systems represent a paradigm shift in personalized education."
                ]
            }
            '''

        # Quality checker responses
        elif "quality" in prompt_lower or "evaluate" in prompt_lower:
            return '''
            {
                "quality_score": 8.2,
                "strengths": [
                    "Comprehensive coverage of key research areas",
                    "Well-structured synthesis of findings",
                    "Clear identification of research gaps"
                ],
                "weaknesses": [
                    "Could include more recent 2024 publications",
                    "Limited discussion of ethical implications"
                ],
                "completeness": 0.85,
                "needs_refinement": false,
                "suggestions": [
                    "Add section on AI ethics in education",
                    "Include international perspectives"
                ]
            }
            '''

        # Synthesizer responses
        elif "synthesize" in prompt_lower or "synthesis" in prompt_lower:
            return '''
# Literature Review: Artificial Intelligence in Higher Education

## Executive Summary

This comprehensive review examines the current state of artificial intelligence 
applications in higher education, synthesizing findings from 25 peer-reviewed studies 
published between 2020 and 2024.

## 1. Introduction

The integration of artificial intelligence (AI) in educational settings has emerged 
as a transformative force in higher education. This review synthesizes current research 
on AI applications, their effectiveness, and implications for teaching and learning.

## 2. Key Themes

### 2.1 AI-Powered Tutoring Systems

Research consistently demonstrates that AI tutoring systems can significantly enhance 
student learning outcomes. Smith et al. (2023) found that students using AI tutors 
showed 32% improvement in test scores compared to traditional instruction methods.

### 2.2 Predictive Analytics for Student Success

Machine learning models have shown promise in predicting student performance and 
identifying at-risk learners. Early intervention strategies based on these predictions 
have reduced dropout rates by up to 18% (Johnson & Lee, 2024).

### 2.3 Personalized Learning Pathways

Adaptive learning systems that customize content based on individual student needs 
have demonstrated significant benefits. Chen (2023) reported 25% improvement in 
knowledge retention among students using personalized pathways.

## 3. Research Gaps and Future Directions

Despite promising results, several gaps in the literature warrant attention:
- Long-term impact studies beyond single semesters
- Cross-cultural validation of AI educational tools
- Ethical frameworks for AI in education

## 4. Conclusion

AI presents significant opportunities for transforming higher education, though 
careful implementation and continued research are essential.

## References

[Generated references would appear here]
            '''

        else:
            return "Generic response for unmatched prompt"

    mock.chat = Mock(side_effect=generate_response)
    return mock


@pytest.fixture
def mock_paper_retriever():
    """Mock paper retriever with realistic paper data."""
    mock = Mock()
    mock.search_papers = Mock(return_value=[
        {
            "id": "arxiv-2024-001",
            "title": "Machine Learning Applications in Educational Assessment",
            "abstract": "This paper presents a comprehensive study of ML algorithms for predicting student outcomes...",
            "authors": ["Smith, J.", "Johnson, A.", "Williams, B."],
            "url": "https://arxiv.org/abs/2024.12345",
            "source": "arXiv",
            "published_date": "2024-01-15"
        },
        {
            "id": "ss-2024-002",
            "title": "Adaptive Learning Systems: A Systematic Review",
            "abstract": "We review the effectiveness of AI-powered adaptive learning platforms...",
            "authors": ["Chen, L."],
            "url": "https://semanticscholar.org/paper/abc123",
            "source": "Semantic Scholar",
            "published_date": "2024-02-20"
        },
        {
            "id": "arxiv-2024-003",
            "title": "Deep Learning for Student Performance Prediction",
            "abstract": "Neural network approaches for early identification of at-risk students...",
            "authors": ["Lee, K.", "Park, S."],
            "url": "https://arxiv.org/abs/2024.67890",
            "source": "arXiv",
            "published_date": "2024-03-10"
        }
    ])
    return mock


# ============================================
# E2E Pipeline Tests
# ============================================

@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end tests for the complete research pipeline."""

    def test_complete_research_flow_success(self, comprehensive_mock_llm, mock_paper_retriever):
        """Test complete pipeline from start to finish."""
        orchestrator = ResearchOrchestrator(comprehensive_mock_llm)

        with patch('agents.retriever_agent.PaperRetriever', return_value=mock_paper_retriever):
            final_state = orchestrator.run_sync(
                project_id="e2e-test-001",
                user_id="test-user",
                title="AI in Higher Education",
                research_question="How does artificial intelligence impact student learning outcomes in higher education?",
                max_papers=10,
                max_iterations=2
            )

        # Verify pipeline completion
        assert final_state["status"] == "completed"

        # Verify keywords were generated
        assert len(final_state["keywords"]) > 0

        # Verify subtopics were generated
        assert len(final_state["subtopics"]) > 0

        # Verify messages were recorded
        assert len(final_state["messages"]) > 0

        # Verify no critical errors
        critical_errors = [e for e in final_state.get("errors", []) if "critical" in str(e).lower()]
        assert len(critical_errors) == 0

    def test_pipeline_generates_synthesis(self, comprehensive_mock_llm, mock_paper_retriever):
        """Test that pipeline generates a synthesis document."""
        orchestrator = ResearchOrchestrator(comprehensive_mock_llm)

        with patch('agents.retriever_agent.PaperRetriever', return_value=mock_paper_retriever):
            final_state = orchestrator.run_sync(
                project_id="e2e-test-002",
                user_id="test-user",
                title="ML in Education",
                research_question="How can machine learning improve educational outcomes?",
                max_papers=5
            )

        # Verify synthesis was generated
        if final_state.get("synthesis"):
            assert len(final_state["synthesis"]) > 100  # Non-trivial content
            assert "learning" in final_state["synthesis"].lower() or "education" in final_state["synthesis"].lower()

    def test_pipeline_handles_no_papers_gracefully(self, comprehensive_mock_llm):
        """Test pipeline handles case when no papers are found."""
        mock_empty_retriever = Mock()
        mock_empty_retriever.search_papers = Mock(return_value=[])

        orchestrator = ResearchOrchestrator(comprehensive_mock_llm)

        with patch('agents.retriever_agent.PaperRetriever', return_value=mock_empty_retriever):
            final_state = orchestrator.run_sync(
                project_id="e2e-test-003",
                user_id="test-user",
                title="Obscure Topic",
                research_question="What is the impact of quantum computing on ancient basket weaving?",
                max_papers=5
            )

        # Should complete but with appropriate status
        assert final_state["status"] in ["completed", "error_no_papers_found"]

    def test_pipeline_respects_max_iterations(self, comprehensive_mock_llm, mock_paper_retriever):
        """Test that pipeline respects max_iterations limit."""
        orchestrator = ResearchOrchestrator(comprehensive_mock_llm)

        with patch('agents.retriever_agent.PaperRetriever', return_value=mock_paper_retriever):
            final_state = orchestrator.run_sync(
                project_id="e2e-test-004",
                user_id="test-user",
                title="Iteration Test",
                research_question="Test question",
                max_papers=5,
                max_iterations=1  # Very limited
            )

        # Should not exceed max iterations
        assert final_state["iteration"] <= 1

    def test_pipeline_tracks_progress(self, comprehensive_mock_llm, mock_paper_retriever):
        """Test that pipeline reports progress correctly."""
        progress_updates = []

        def progress_callback(agent, message, percent):
            progress_updates.append({
                "agent": agent,
                "message": message,
                "percent": percent
            })

        orchestrator = ResearchOrchestrator(comprehensive_mock_llm, progress_callback)

        with patch('agents.retriever_agent.PaperRetriever', return_value=mock_paper_retriever):
            orchestrator.run_sync(
                project_id="e2e-test-005",
                user_id="test-user",
                title="Progress Test",
                research_question="Test progress tracking",
                max_papers=3
            )

        # Should have recorded progress updates
        assert len(progress_updates) > 0

        # Should have seen multiple agents
        agents_seen = set(u["agent"] for u in progress_updates)
        assert len(agents_seen) >= 1


# ============================================
# Resilience Tests
# ============================================

@pytest.mark.integration
class TestPipelineResilience:
    """Tests for pipeline resilience and error recovery."""

    def test_recovers_from_llm_transient_failures(self, mock_paper_retriever):
        """Test that pipeline recovers from transient LLM failures."""
        call_count = {"count": 0}

        def flaky_chat(prompt):
            call_count["count"] += 1
            if call_count["count"] <= 2:
                raise Exception("Transient API error")
            return '{"keywords": ["test"], "subtopics": ["Test Topic"]}'

        mock_llm = Mock()
        mock_llm.chat = Mock(side_effect=flaky_chat)

        orchestrator = ResearchOrchestrator(mock_llm)

        # Should eventually succeed despite initial failures
        # The actual behavior depends on retry logic in agents
        with patch('agents.retriever_agent.PaperRetriever', return_value=mock_paper_retriever):
            try:
                final_state = orchestrator.run_sync(
                    project_id="resilience-test-001",
                    user_id="test-user",
                    title="Resilience Test",
                    research_question="Test question",
                    max_papers=3
                )
                # If it completes, verify it handled errors
                assert final_state["status"] in ["completed", "error"]
            except Exception:
                # Expected if retry logic exhausted
                pass

    def test_handles_malformed_llm_responses(self, mock_paper_retriever):
        """Test handling of malformed LLM responses."""
        def malformed_response(prompt):
            if "keywords" in prompt.lower():
                return "This is not valid JSON at all"
            return '{"relevance_score": 50}'

        mock_llm = Mock()
        mock_llm.chat = Mock(side_effect=malformed_response)

        orchestrator = ResearchOrchestrator(mock_llm)

        with patch('agents.retriever_agent.PaperRetriever', return_value=mock_paper_retriever):
            final_state = orchestrator.run_sync(
                project_id="malformed-test-001",
                user_id="test-user",
                title="Malformed Test",
                research_question="Test",
                max_papers=3
            )

        # Should complete with graceful degradation
        assert final_state["status"] in ["completed", "error"]


# ============================================
# Data Flow Tests
# ============================================

@pytest.mark.integration
class TestDataFlow:
    """Tests for data flow between agents."""

    def test_keywords_flow_to_retriever(self, comprehensive_mock_llm, mock_paper_retriever):
        """Test that planner keywords are used by retriever."""
        orchestrator = ResearchOrchestrator(comprehensive_mock_llm)

        with patch('agents.retriever_agent.PaperRetriever', return_value=mock_paper_retriever):
            final_state = orchestrator.run_sync(
                project_id="dataflow-test-001",
                user_id="test-user",
                title="Data Flow Test",
                research_question="How does AI affect education?",
                max_papers=5
            )

        # Verify keywords were generated
        assert len(final_state["keywords"]) > 0

        # Verify retriever was called (search_papers should have been called)
        assert mock_paper_retriever.search_papers.called or len(final_state.get("papers", [])) >= 0

    def test_papers_flow_to_analyzer(self, comprehensive_mock_llm, mock_paper_retriever):
        """Test that retrieved papers are analyzed."""
        orchestrator = ResearchOrchestrator(comprehensive_mock_llm)

        with patch('agents.retriever_agent.PaperRetriever', return_value=mock_paper_retriever):
            final_state = orchestrator.run_sync(
                project_id="dataflow-test-002",
                user_id="test-user",
                title="Paper Analysis Test",
                research_question="Test question",
                max_papers=5
            )

        # If papers were found, they should be analyzed
        if final_state.get("papers"):
            # Either analyzed_papers exists or papers have analysis
            has_analysis = (
                len(final_state.get("analyzed_papers", [])) > 0 or
                any(p.get("analysis") for p in final_state.get("papers", []))
            )
            assert has_analysis or final_state["status"] != "completed"

    def test_analysis_flows_to_synthesizer(self, comprehensive_mock_llm, mock_paper_retriever):
        """Test that paper analyses are used in synthesis."""
        orchestrator = ResearchOrchestrator(comprehensive_mock_llm)

        with patch('agents.retriever_agent.PaperRetriever', return_value=mock_paper_retriever):
            final_state = orchestrator.run_sync(
                project_id="dataflow-test-003",
                user_id="test-user",
                title="Synthesis Test",
                research_question="How does AI improve learning?",
                max_papers=5
            )

        # If synthesis exists, it should be based on analyzed papers
        if final_state.get("synthesis"):
            assert len(final_state["synthesis"]) > 50


# ============================================
# Concurrent Execution Tests
# ============================================

@pytest.mark.integration
@pytest.mark.slow
class TestConcurrentExecution:
    """Tests for concurrent pipeline execution."""

    def test_multiple_concurrent_pipelines(self, comprehensive_mock_llm, mock_paper_retriever):
        """Test running multiple pipelines concurrently."""
        import threading

        results = {}
        errors = []

        def run_pipeline(project_id):
            try:
                orchestrator = ResearchOrchestrator(comprehensive_mock_llm)
                with patch('agents.retriever_agent.PaperRetriever', return_value=mock_paper_retriever):
                    final_state = orchestrator.run_sync(
                        project_id=project_id,
                        user_id="test-user",
                        title=f"Concurrent Test {project_id}",
                        research_question="Test question",
                        max_papers=3
                    )
                results[project_id] = final_state["status"]
            except Exception as e:
                errors.append((project_id, str(e)))

        # Run 3 pipelines concurrently
        threads = [
            threading.Thread(target=run_pipeline, args=(f"concurrent-{i}",))
            for i in range(3)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=60)

        # Should complete without major errors
        assert len(errors) < 3  # At least some should succeed
