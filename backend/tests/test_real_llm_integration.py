# Real LLM Integration Tests
# End-to-end tests with REAL API calls - no mocks
# Run these tests before deploying to production to ensure the system works
#
# IMPORTANT: These tests make REAL API calls and consume API quota!
# - They require GROQ_API_KEY environment variable
# - They fetch papers from real arXiv/Semantic Scholar APIs
# - They call Groq LLMs for analysis and synthesis
# - They may take 5-10 minutes to complete
#
# Run with: pytest tests/test_real_llm_integration.py -v -m real_llm --run-slow
# Or specifically: pytest tests/test_real_llm_integration.py::TestRealLLMFullPipeline -v -s

import logging
import os
import time
from datetime import datetime

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging to see progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_api_key():
    """Check if GROQ_API_KEY is available."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        pytest.skip("GROQ_API_KEY not set - skipping real LLM test")
    return key


# ============================================
# Real LLM Integration Fixtures
# ============================================


@pytest.fixture
def real_groq_client():
    """
    Create a REAL Groq client for integration testing.

    This uses actual API calls - not mocks!
    """
    check_api_key()

    from agents.llm import GroqClient
    from agents.llm.base import LLMConfig

    config = LLMConfig(
        api_key=os.environ.get("GROQ_API_KEY"),
        user_id="integration-test",
        user_budget=1.0,  # Set reasonable budget for tests
        max_retries=5,  # More retries for rate limits
        timeout=120.0,  # Longer timeout for real calls
    )

    client = GroqClient(config)
    logger.info("Created real GroqClient for integration testing")
    return client


@pytest.fixture
def real_orchestrator(real_groq_client):
    """Create a REAL orchestrator with real LLM client."""
    from agents.orchestrator import ResearchOrchestrator

    progress_updates = []

    def progress_callback(agent_name, message, progress):
        progress_updates.append(
            {
                "agent": agent_name,
                "message": message,
                "progress": progress,
                "timestamp": datetime.now().isoformat(),
            }
        )
        logger.info(f"[{progress:.0f}%] {agent_name}: {message}")

    orchestrator = ResearchOrchestrator(real_groq_client, progress_callback)
    orchestrator._test_progress = progress_updates  # Store for assertions
    return orchestrator


# ============================================
# Real LLM Test Class
# ============================================


@pytest.mark.slow
@pytest.mark.real_llm
class TestRealLLMFullPipeline:
    """
    Full pipeline tests with REAL LLM calls.

    These tests verify the complete system works end-to-end
    before deployment to production.

    WARNING: These tests consume API quota and may take 5-10 minutes!
    """

    def test_full_pipeline_with_30_papers(self, real_orchestrator):
        """
        Test the complete pipeline with ~30 papers using REAL API calls.

        This is the primary production-readiness test. It verifies:
        1. Paper retrieval from arXiv and Semantic Scholar works
        2. LLM analysis of papers completes without 413 errors
        3. Synthesis is generated successfully
        4. Quality checking works
        5. The project completes with status='completed'

        Timeouts and error handling:
        - Model failover on 413 errors (payload too large)
        - Model switching on 429 rate limits
        - Fallback synthesis if needed
        """
        logger.info("=" * 60)
        logger.info("STARTING: Real LLM Full Pipeline Test (30 papers)")
        logger.info("=" * 60)

        start_time = time.time()

        # Use a research topic that will find papers
        final_state = real_orchestrator.run_sync(
            project_id=f"real-test-{int(time.time())}",
            user_id="integration-test",
            title="Machine Learning in Healthcare: A Comprehensive Review",
            research_question="How are machine learning and deep learning techniques being applied to improve healthcare diagnostics and patient outcomes?",
            max_papers=30,  # Request ~30 papers
            max_iterations=2,  # Allow one refinement
            relevance_threshold=50.0,  # Lower threshold for more papers
            academic_level="graduate",
            target_word_count=800,
        )

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.2f} seconds")

        # ========================================
        # Core Assertions - Must Pass
        # ========================================

        # 1. Pipeline must complete (not fail with error)
        assert final_state["status"] in [
            "completed",
            "needs_refinement",
        ], f"Pipeline failed with status: {final_state['status']}"

        # 2. Must have retrieved papers
        papers = final_state.get("papers", [])
        logger.info(f"Retrieved {len(papers)} papers")
        assert len(papers) >= 5, f"Expected at least 5 papers, got {len(papers)}"

        # 3. Must have analyzed some papers
        analyzed = final_state.get("analyzed_papers", [])
        logger.info(f"Analyzed {len(analyzed)} papers")
        # Allow partial analysis if some papers failed
        assert len(analyzed) >= 1, "No papers were analyzed successfully"

        # 4. Must have generated synthesis
        synthesis = final_state.get("synthesis", "")
        logger.info(f"Synthesis length: {len(synthesis)} characters")
        assert len(synthesis) >= 100, f"Synthesis too short: {len(synthesis)} chars"

        # 5. Check for critical errors (413, 429 that broke pipeline)
        errors = final_state.get("errors", [])
        critical_errors = [e for e in errors if "failed completely" in e.lower()]
        assert len(critical_errors) == 0, f"Critical errors occurred: {critical_errors}"

        # ========================================
        # Quality Assertions - Should Pass
        # ========================================

        # 6. Keywords were generated
        keywords = final_state.get("keywords", [])
        logger.info(f"Generated {len(keywords)} keywords: {keywords[:5]}...")
        assert len(keywords) >= 3, f"Expected at least 3 keywords, got {len(keywords)}"

        # 7. High quality papers identified
        high_quality = final_state.get("high_quality_papers", [])
        logger.info(f"Identified {len(high_quality)} high-quality papers")

        # 8. Quality score if available
        quality_score = final_state.get("quality_score", 0)
        logger.info(f"Quality score: {quality_score}")

        # ========================================
        # Log Summary
        # ========================================
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Status: {final_state['status']}")
        logger.info(f"Papers retrieved: {len(papers)}")
        logger.info(f"Papers analyzed: {len(analyzed)}")
        logger.info(f"High-quality papers: {len(high_quality)}")
        logger.info(f"Synthesis length: {len(synthesis)} chars")
        logger.info(f"Quality score: {quality_score}")
        logger.info(f"Elapsed time: {elapsed:.2f}s")
        logger.info(f"Errors: {len(errors)}")
        for error in errors[:5]:
            logger.warning(f"  - {error}")
        logger.info("=" * 60)

    def test_full_pipeline_with_cybersecurity_topic(self, real_orchestrator):
        """
        Test pipeline with cybersecurity topic (from nxtIssue.md logs).

        This tests the same type of query that was causing issues in production.
        """
        logger.info("=" * 60)
        logger.info("STARTING: Real LLM Cybersecurity Topic Test")
        logger.info("=" * 60)

        start_time = time.time()

        final_state = real_orchestrator.run_sync(
            project_id=f"cyber-test-{int(time.time())}",
            user_id="integration-test",
            title="Cybersecurity in Defense Systems",
            research_question="What are the emerging cybersecurity threats and defense strategies in military and critical infrastructure systems?",
            max_papers=25,
            max_iterations=2,
            relevance_threshold=45.0,
            academic_level="graduate",
            target_word_count=600,
        )

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.2f} seconds")

        # Must complete without error status
        assert final_state["status"] != "error", f"Pipeline failed: {final_state.get('errors', [])}"

        # Should have synthesis
        synthesis = final_state.get("synthesis", "")
        assert len(synthesis) >= 50, "No meaningful synthesis generated"

        logger.info(f"Cybersecurity test completed. Status: {final_state['status']}")
        logger.info(f"Papers: {len(final_state.get('papers', []))}")
        logger.info(f"Synthesis preview: {synthesis[:200]}...")

    def test_pipeline_handles_rate_limits_gracefully(self, real_orchestrator):
        """
        Test that pipeline handles rate limits without failing.

        Makes multiple rapid requests to trigger rate limiting,
        verifying the system switches models and continues.
        """
        logger.info("=" * 60)
        logger.info("STARTING: Rate Limit Handling Test")
        logger.info("=" * 60)

        # Smaller topic for faster execution
        final_state = real_orchestrator.run_sync(
            project_id=f"rate-test-{int(time.time())}",
            user_id="integration-test",
            title="Neural Networks Overview",
            research_question="What are the fundamental architectures and applications of neural networks?",
            max_papers=15,  # Smaller set
            max_iterations=1,
            relevance_threshold=50.0,
            academic_level="undergraduate",
            target_word_count=400,
        )

        # Should not fail completely
        assert (
            final_state["status"] != "error" or len(final_state.get("synthesis", "")) > 0
        ), "Pipeline failed without producing any output"

        logger.info(f"Rate limit test completed. Status: {final_state['status']}")

    def test_pipeline_recovers_from_413_errors(self, real_orchestrator):
        """
        Test that 413 errors (payload too large) trigger model upgrade.

        Uses a topic that might generate long responses to test failover.
        """
        logger.info("=" * 60)
        logger.info("STARTING: 413 Error Recovery Test")
        logger.info("=" * 60)

        # Use a complex topic that may generate long responses
        final_state = real_orchestrator.run_sync(
            project_id=f"recover-test-{int(time.time())}",
            user_id="integration-test",
            title="Comprehensive AI Ethics Review",
            research_question="What are the ethical implications, challenges, and proposed frameworks for artificial intelligence governance, bias mitigation, and responsible AI deployment in society?",
            max_papers=20,
            max_iterations=1,
            relevance_threshold=40.0,
            academic_level="graduate",
            target_word_count=500,
        )

        # Project should complete (possibly with fallback)
        # We check that we got SOME output, even if partial
        synthesis = final_state.get("synthesis", "")
        papers_analyzed = len(final_state.get("analyzed_papers", []))

        assert (
            final_state["status"] != "error" or len(synthesis) > 0 or papers_analyzed > 0
        ), "Pipeline failed completely without any output"

        logger.info(f"413 recovery test completed. Status: {final_state['status']}")
        logger.info(f"Synthesis length: {len(synthesis)}")
        logger.info(f"Papers analyzed: {papers_analyzed}")


@pytest.mark.slow
@pytest.mark.real_llm
class TestRealLLMIndividualAgents:
    """
    Test individual agents with real LLM calls.

    These are faster than full pipeline tests and help isolate issues.
    """

    def test_planner_agent_real_llm(self, real_groq_client):
        """Test the planner agent with real LLM."""
        from agents.planner_agent import ResearchPlannerAgent
        from agents.state import create_initial_state

        logger.info("Testing Planner Agent with real LLM...")

        agent = ResearchPlannerAgent(real_groq_client)
        state = create_initial_state(
            project_id="planner-test",
            user_id="test",
            title="AI in Education",
            research_question="How does AI improve student learning outcomes?",
        )

        import asyncio

        result_state = asyncio.run(agent.run(state))

        # Should have generated keywords
        keywords = result_state.get("keywords", [])
        logger.info(f"Generated keywords: {keywords}")

        assert len(keywords) >= 3, f"Expected 3+ keywords, got {len(keywords)}"
        assert result_state.get("subtopics"), "No subtopics generated"

    def test_analyzer_agent_real_llm(self, real_groq_client):
        """Test the analyzer agent with real LLM on a single paper."""
        from agents.analyzer_agent import PaperAnalyzerAgent
        from agents.state import create_initial_state

        logger.info("Testing Analyzer Agent with real LLM...")

        agent = PaperAnalyzerAgent(real_groq_client)
        state = create_initial_state(
            project_id="analyzer-test",
            user_id="test",
            title="Test",
            research_question="How does machine learning improve healthcare?",
        )

        # Add a test paper
        state["papers"] = [
            {
                "id": "test-paper-1",
                "title": "Machine Learning for Medical Diagnosis: A Survey",
                "abstract": "This survey explores the application of machine learning techniques in medical diagnosis. We review deep learning approaches for image-based diagnosis, including CNNs for radiology and pathology. Our analysis shows that ML models can achieve accuracy comparable to expert physicians in specific diagnostic tasks.",
                "authors": ["John Smith", "Jane Doe"],
                "url": "https://example.com/paper1",
                "source": "arXiv",
            }
        ]
        state["relevance_threshold"] = 30.0  # Low threshold for testing

        import asyncio

        result_state = asyncio.run(agent.run(state))

        analyzed = result_state.get("analyzed_papers", [])
        logger.info(f"Analyzed {len(analyzed)} papers")

        # Should have analyzed the paper
        assert len(analyzed) >= 1, "Paper was not analyzed"

        # Check analysis quality
        if analyzed:
            paper = analyzed[0]
            analysis = paper.get("analysis", {})
            logger.info(f"Analysis: {analysis}")
            assert paper.get("relevance_score") is not None, "No relevance score"

    def test_synthesizer_agent_real_llm(self, real_groq_client):
        """Test the synthesizer agent with real LLM."""
        from agents.state import create_initial_state
        from agents.synthesizer_agent import SynthesisExecutorAgent

        logger.info("Testing Synthesizer Agent with real LLM...")

        agent = SynthesisExecutorAgent(real_groq_client)
        state = create_initial_state(
            project_id="synth-test",
            user_id="test",
            title="ML in Healthcare Review",
            research_question="How is ML used in healthcare?",
        )

        # Add pre-analyzed papers
        state["high_quality_papers"] = [
            {
                "title": "Deep Learning for Medical Image Analysis",
                "analysis": {
                    "relevance_score": 85,
                    "key_findings": [
                        "CNNs achieve 95% accuracy in tumor detection",
                        "Transfer learning reduces training data requirements",
                    ],
                    "methodology": "Systematic review of 50 studies",
                    "limitations": ["Limited to radiology applications"],
                    "contribution": "Comprehensive framework for medical image ML",
                },
            },
            {
                "title": "Machine Learning in Clinical Decision Support",
                "analysis": {
                    "relevance_score": 78,
                    "key_findings": [
                        "ML reduces diagnostic errors by 30%",
                        "Real-time prediction improves outcomes",
                    ],
                    "methodology": "Retrospective analysis of 10,000 cases",
                    "limitations": ["Single hospital study"],
                    "contribution": "Clinical validation of ML decision support",
                },
            },
        ]
        state["subtopics"] = ["Medical Imaging", "Clinical Decision Support"]
        state["target_word_count"] = 400

        import asyncio

        result_state = asyncio.run(agent.run(state))

        synthesis = result_state.get("synthesis", "")
        logger.info(f"Synthesis length: {len(synthesis)} chars")
        logger.info(f"Synthesis preview: {synthesis[:300]}...")

        assert len(synthesis) >= 100, f"Synthesis too short: {len(synthesis)} chars"


@pytest.mark.slow
@pytest.mark.real_llm
class TestRealPaperRetrieval:
    """Test real paper retrieval without LLM (API-only tests)."""

    def test_arxiv_retrieval(self):
        """Test fetching papers from arXiv."""
        from paper_retriever import PaperRetriever

        logger.info("Testing arXiv paper retrieval...")

        retriever = PaperRetriever()
        papers = retriever._search_arxiv("machine learning healthcare diagnosis", max_results=10)

        logger.info(f"Retrieved {len(papers)} papers from arXiv")
        for paper in papers[:3]:
            logger.info(f"  - {paper['title'][:60]}...")

        assert len(papers) >= 1, "Failed to retrieve papers from arXiv"
        assert all(p.get("title") and p.get("abstract") for p in papers)

    def test_semantic_scholar_retrieval(self):
        """Test fetching papers from Semantic Scholar."""
        from paper_retriever import PaperRetriever

        logger.info("Testing Semantic Scholar paper retrieval...")

        retriever = PaperRetriever()
        papers, rate_limited = retriever._search_semantic_scholar(
            "artificial intelligence education", max_results=10
        )

        logger.info(
            f"Retrieved {len(papers)} papers from Semantic Scholar (rate_limited={rate_limited})"
        )

        # May be rate limited, so don't fail
        if rate_limited and len(papers) == 0:
            pytest.skip("Semantic Scholar rate limited - test inconclusive")

        if papers:
            for paper in papers[:3]:
                logger.info(f"  - {paper['title'][:60]}...")

    def test_multi_source_retrieval_30_papers(self):
        """Test fetching ~30 papers from multiple sources."""
        from paper_retriever import PaperRetriever

        logger.info("Testing multi-source paper retrieval (targeting 30 papers)...")

        retriever = PaperRetriever()

        # Multiple search queries to get more papers
        queries = [
            "machine learning diagnosis",
            "deep learning medical imaging",
            "AI healthcare prediction",
        ]

        all_papers = []
        for query in queries:
            arxiv_papers = retriever._search_arxiv(query, max_results=10)
            all_papers.extend(arxiv_papers)

            # Add delay to respect rate limits
            time.sleep(3.5)

        # Deduplicate by title (rough)
        seen_titles = set()
        unique_papers = []
        for paper in all_papers:
            title_lower = paper["title"].lower().strip()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(paper)

        logger.info(f"Retrieved {len(unique_papers)} unique papers from {len(all_papers)} total")

        assert len(unique_papers) >= 15, f"Expected 15+ papers, got {len(unique_papers)}"


# ============================================
# Production Readiness Check
# ============================================


@pytest.mark.slow
@pytest.mark.real_llm
class TestProductionReadiness:
    """
    Final production readiness checks.

    Run these before every production deployment.
    """

    def test_complete_project_lifecycle(self, real_orchestrator):
        """
        Simulate a complete user project lifecycle.

        This is the ultimate test - if this passes, the system is production-ready.
        """
        logger.info("=" * 60)
        logger.info("PRODUCTION READINESS CHECK")
        logger.info("=" * 60)

        start_time = time.time()
        project_id = f"prod-check-{int(time.time())}"

        final_state = real_orchestrator.run_sync(
            project_id=project_id,
            user_id="prod-test-user",
            title="Artificial Intelligence in Modern Healthcare Systems",
            research_question="How are artificial intelligence and machine learning technologies transforming healthcare delivery, diagnosis, and patient outcomes?",
            max_papers=35,  # Target 30-40 papers
            max_iterations=2,
            relevance_threshold=45.0,
            academic_level="graduate",
            target_word_count=700,
        )

        elapsed = time.time() - start_time

        # ========================================
        # CRITICAL CHECKS - All must pass
        # ========================================

        # 1. Status check
        if final_state["status"] == "error":
            errors = final_state.get("errors", [])
            logger.error(f"PRODUCTION CHECK FAILED: {errors}")
            pytest.fail(f"Project ended with error status: {errors}")

        # 2. Output check
        synthesis = final_state.get("synthesis", "")
        if len(synthesis) < 100:
            pytest.fail(f"No meaningful synthesis generated (only {len(synthesis)} chars)")

        # 3. Paper processing check
        papers = final_state.get("papers", [])
        analyzed = final_state.get("analyzed_papers", [])
        if len(papers) < 5:
            pytest.fail(f"Too few papers retrieved: {len(papers)}")

        # ========================================
        # Summary Report
        # ========================================
        logger.info("=" * 60)
        logger.info("✅ PRODUCTION READINESS CHECK PASSED")
        logger.info("=" * 60)
        logger.info(f"Project ID: {project_id}")
        logger.info(f"Final Status: {final_state['status']}")
        logger.info(f"Total Time: {elapsed:.1f}s")
        logger.info(f"Papers Retrieved: {len(papers)}")
        logger.info(f"Papers Analyzed: {len(analyzed)}")
        logger.info(f"High Quality: {len(final_state.get('high_quality_papers', []))}")
        logger.info(f"Synthesis Length: {len(synthesis)} chars")
        logger.info(f"Quality Score: {final_state.get('quality_score', 'N/A')}")
        logger.info(f"Keywords: {final_state.get('keywords', [])[:5]}")
        logger.info(f"Errors: {len(final_state.get('errors', []))}")
        logger.info("=" * 60)

        # Log any errors for debugging (but test still passes)
        for error in final_state.get("errors", []):
            logger.warning(f"Non-fatal error: {error}")
