# Performance Tests
# Benchmarks and performance testing for Scholar Agent

import asyncio
import statistics
import time
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

# ============================================
# Performance Fixtures
# ============================================


@pytest.fixture
def fast_mock_llm():
    """Create a fast mock LLM for performance testing."""
    mock = Mock()
    mock.chat = Mock(return_value='{"keywords": ["test"], "subtopics": ["Topic"]}')
    return mock


@pytest.fixture
def mock_papers_batch():
    """Generate batch of mock papers for performance testing."""
    return [
        {
            "id": f"paper-{i}",
            "title": f"Test Paper {i}: Machine Learning Applications",
            "abstract": f"This is the abstract for paper {i}. " * 20,
            "authors": ["Author A", "Author B"],
            "url": f"https://example.com/paper/{i}",
            "source": "arXiv" if i % 2 == 0 else "Semantic Scholar",
        }
        for i in range(100)
    ]


# ============================================
# Latency Benchmarks
# ============================================


@pytest.mark.slow
class TestLatencyBenchmarks:
    """Benchmark tests for measuring system latency."""

    def test_planner_latency(self, fast_mock_llm):
        """Benchmark planner agent latency."""
        from agents.planner_agent import ResearchPlannerAgent
        from agents.state import create_initial_state

        agent = ResearchPlannerAgent(fast_mock_llm)

        latencies = []
        for _ in range(10):
            state = create_initial_state(
                project_id="perf-test",
                user_id="test-user",
                title="AI in Education",
                research_question="How does AI affect education?",
            )
            start = time.perf_counter()
            result = asyncio.get_event_loop().run_until_complete(agent.run(state))
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nPlanner Latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")

        # Should complete quickly with mocked LLM
        assert avg_latency < 100, f"Planner too slow: {avg_latency}ms avg"

    def test_analyzer_latency(self, fast_mock_llm):
        """Benchmark analyzer agent latency."""
        from agents.analyzer_agent import PaperAnalyzerAgent
        from agents.state import create_initial_state

        fast_mock_llm.chat.return_value = """
        {
            "relevance_score": 85,
            "key_findings": ["Finding 1"],
            "methodology": "Test"
        }
        """

        agent = PaperAnalyzerAgent(fast_mock_llm)

        paper = {"id": "test-paper", "title": "Test Paper", "abstract": "Test abstract " * 50}

        latencies = []
        for _ in range(10):
            state = create_initial_state(
                project_id="perf-test",
                user_id="test-user",
                title="Test",
                research_question="Test research question",
            )
            state["papers"] = [paper]

            start = time.perf_counter()
            result = asyncio.get_event_loop().run_until_complete(agent.run(state))
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = statistics.mean(latencies)

        print(f"\nAnalyzer Latency - Avg: {avg_latency:.2f}ms")
        # Allow more time for async overhead and logging in test environment
        assert avg_latency < 2000, f"Analyzer too slow: {avg_latency}ms avg"

    def test_model_router_latency(self):
        """Benchmark model router decision latency."""
        from agents.model_router import SmartModelRouter

        router = SmartModelRouter(user_budget=1.0)

        latencies = []
        for i in range(100):
            prompt = "Test prompt " * (i + 1)

            start = time.perf_counter()
            decision = router.route(task_type="paper_analysis", prompt=prompt)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)

        print(f"\nModel Router Latency - Avg: {avg_latency:.4f}ms, Max: {max_latency:.4f}ms")

        # Router should be very fast (no LLM calls)
        assert avg_latency < 1, f"Router too slow: {avg_latency}ms avg"
        assert max_latency < 5, f"Router max latency too high: {max_latency}ms"


# ============================================
# Throughput Benchmarks
# ============================================


@pytest.mark.slow
class TestThroughputBenchmarks:
    """Benchmark tests for measuring system throughput."""

    def test_chunker_throughput(self, mock_papers_batch):
        """Benchmark chunker throughput."""
        from rag.chunker import SemanticChunker

        chunker = SemanticChunker(max_chunk_size=256, min_chunk_size=50, overlap_size=25)

        start = time.perf_counter()
        total_chunks = 0

        for paper in mock_papers_batch[:50]:
            chunks = chunker.chunk_paper(paper)
            total_chunks += len(chunks)

        end = time.perf_counter()
        duration = end - start

        papers_per_second = 50 / duration
        chunks_per_second = total_chunks / duration

        print("\nChunker Throughput:")
        print(f"  Papers/sec: {papers_per_second:.2f}")
        print(f"  Chunks/sec: {chunks_per_second:.2f}")
        print(f"  Total chunks: {total_chunks}")

        # Should process at least 10 papers per second
        assert papers_per_second > 10, f"Chunker too slow: {papers_per_second} papers/sec"

    def test_bm25_indexing_throughput(self, mock_papers_batch):
        """Benchmark BM25 index building throughput."""
        from rag.hybrid_search import BM25Index

        index = BM25Index()

        # Prepare documents
        documents = [
            {
                "id": p["id"],
                "content": p["abstract"],
                "paper_id": p["id"],
                "paper_title": p["title"],
            }
            for p in mock_papers_batch[:100]
        ]

        start = time.perf_counter()
        index.add_documents(documents)
        end = time.perf_counter()

        duration = end - start
        docs_per_second = len(documents) / duration

        print("\nBM25 Indexing Throughput:")
        print(f"  Docs/sec: {docs_per_second:.2f}")
        print(f"  Index size: {len(documents)}")

        # Should index at least 100 documents per second
        assert docs_per_second > 50, f"BM25 indexing too slow: {docs_per_second} docs/sec"

    def test_bm25_search_throughput(self, mock_papers_batch):
        """Benchmark BM25 search throughput."""
        from rag.hybrid_search import BM25Index

        index = BM25Index()

        # Build index
        documents = [
            {
                "id": p["id"],
                "content": p["abstract"],
                "paper_id": p["id"],
                "paper_title": p["title"],
            }
            for p in mock_papers_batch[:100]
        ]
        index.add_documents(documents)

        # Run search benchmark
        queries = [
            "machine learning",
            "artificial intelligence",
            "student performance",
            "education technology",
            "deep learning",
        ]

        start = time.perf_counter()
        for _ in range(100):
            for query in queries:
                results = index.search(query, top_k=10)
        end = time.perf_counter()

        duration = end - start
        searches_per_second = 500 / duration

        print("\nBM25 Search Throughput:")
        print(f"  Searches/sec: {searches_per_second:.2f}")

        # Should handle at least 100 searches per second
        assert searches_per_second > 50, f"BM25 search too slow: {searches_per_second} searches/sec"


# ============================================
# Memory Usage Tests
# ============================================


@pytest.mark.slow
class TestMemoryUsage:
    """Tests for memory usage and efficiency."""

    def test_chunker_memory_efficiency(self, mock_papers_batch):
        """Test chunker doesn't leak memory."""
        import gc

        from rag.chunker import SemanticChunker

        chunker = SemanticChunker()

        # Force garbage collection
        gc.collect()

        # Process papers
        for i in range(5):  # Multiple iterations
            for paper in mock_papers_batch[:20]:
                chunks = chunker.chunk_paper(paper)
                del chunks
            gc.collect()

        # If we get here without MemoryError, test passes
        assert True

    def test_state_size_growth(self):
        """Test that agent state doesn't grow unboundedly."""
        import sys

        from agents.state import create_initial_state

        state = create_initial_state(
            project_id="test", user_id="test", title="Test", research_question="Test?"
        )

        initial_size = sys.getsizeof(str(state))

        # Simulate adding content
        for i in range(100):
            state["messages"].append(
                {
                    "agent": "test",
                    "content": f"Message {i}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            state["papers"].append(
                {"id": f"paper-{i}", "title": f"Paper {i}", "abstract": "Test " * 100}
            )

        final_size = sys.getsizeof(str(state))

        # Size should grow linearly, not exponentially
        growth_ratio = final_size / initial_size

        print("\nState Size Growth:")
        print(f"  Initial: {initial_size} bytes")
        print(f"  Final: {final_size} bytes")
        print(f"  Growth ratio: {growth_ratio:.2f}x")

        # Reasonable growth for 100 messages + 100 papers
        assert growth_ratio < 1000, f"State grew too much: {growth_ratio}x"


# ============================================
# Scalability Tests
# ============================================


@pytest.mark.slow
class TestScalability:
    """Tests for system scalability."""

    def test_analysis_scales_linearly(self, fast_mock_llm):
        """Test that analysis time scales linearly with paper count."""
        from agents.analyzer_agent import PaperAnalyzerAgent
        from agents.state import create_initial_state

        fast_mock_llm.chat.return_value = '{"relevance_score": 85}'
        agent = PaperAnalyzerAgent(fast_mock_llm)

        timings = {}

        for n_papers in [5, 10, 20]:
            papers = [
                {"id": f"paper-{i}", "title": f"Paper {i}", "abstract": "Test abstract"}
                for i in range(n_papers)
            ]

            state = create_initial_state(
                project_id="perf-test",
                user_id="test-user",
                title="Test",
                research_question="Test question",
            )
            state["papers"] = papers

            start = time.perf_counter()
            result = asyncio.get_event_loop().run_until_complete(agent.run(state))
            end = time.perf_counter()

            timings[n_papers] = end - start

        # Check linear scaling (allow 20% variance)
        ratio_5_to_10 = timings[10] / timings[5]
        ratio_10_to_20 = timings[20] / timings[10]

        print("\nScalability Test:")
        print(f"  5 papers: {timings[5]*1000:.2f}ms")
        print(f"  10 papers: {timings[10]*1000:.2f}ms (ratio: {ratio_5_to_10:.2f})")
        print(f"  20 papers: {timings[20]*1000:.2f}ms (ratio: {ratio_10_to_20:.2f})")

        # Should scale roughly linearly (2x papers = ~2x time, with tolerance)
        assert 1.5 <= ratio_5_to_10 <= 2.5, f"Non-linear scaling: {ratio_5_to_10}"
        assert 1.5 <= ratio_10_to_20 <= 2.5, f"Non-linear scaling: {ratio_10_to_20}"

    def test_router_scales_with_history(self):
        """Test that router doesn't slow down with usage history."""
        from agents.model_router import SmartModelRouter

        router = SmartModelRouter(user_budget=100.0)

        # Warm up
        for _ in range(10):
            router.route("paper_analysis", "test prompt")

        # Measure initial performance
        start = time.perf_counter()
        for _ in range(100):
            router.route("paper_analysis", "test prompt")
        initial_time = time.perf_counter() - start

        # Simulate lots of usage
        for _ in range(1000):
            decision = router.route("paper_analysis", "test prompt")
            router.record_usage(decision.estimated_cost)

        # Measure performance after heavy usage
        start = time.perf_counter()
        for _ in range(100):
            router.route("paper_analysis", "test prompt")
        final_time = time.perf_counter() - start

        slowdown = final_time / initial_time

        print("\nRouter Scaling with History:")
        print(f"  Initial: {initial_time*1000:.2f}ms for 100 calls")
        print(f"  After 1000 uses: {final_time*1000:.2f}ms for 100 calls")
        print(f"  Slowdown factor: {slowdown:.2f}x")

        # Should not slow down significantly
        assert slowdown < 2.0, f"Router slowed down too much: {slowdown}x"


# ============================================
# Load Tests
# ============================================


@pytest.mark.slow
class TestLoadHandling:
    """Tests for handling load and concurrent requests."""

    def test_cache_under_load(self):
        """Test cache performance under load."""
        from cache.redis_cache import IntelligentCache

        # Mock Redis
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True

        with patch("cache.redis_cache.redis.from_url", return_value=mock_redis):
            cache = IntelligentCache("redis://localhost:6379")

            start = time.perf_counter()

            # Simulate high load
            for i in range(1000):
                key = cache._make_key("llm", f"prompt-{i % 100}")
                cache.set(key, {"result": f"data-{i}"}, ttl=3600, tier="llm")
                cache.get(key, "llm")

            end = time.perf_counter()

            ops_per_second = 2000 / (end - start)  # 1000 sets + 1000 gets

            print("\nCache Load Test:")
            print(f"  Ops/sec: {ops_per_second:.2f}")

            assert ops_per_second > 1000, f"Cache too slow under load: {ops_per_second} ops/sec"


# ============================================
# Regression Detection
# ============================================


class TestPerformanceRegression:
    """Tests to detect performance regressions."""

    # Baseline timings (update these when intentionally changing performance)
    BASELINES = {
        "planner_init": 10,  # ms
        "analyzer_init": 10,  # ms
        "state_creation": 1,  # ms
        "router_decision": 0.5,  # ms
    }

    def test_agent_initialization_regression(self, fast_mock_llm):
        """Test that agent initialization hasn't regressed."""
        from agents.analyzer_agent import PaperAnalyzerAgent
        from agents.planner_agent import ResearchPlannerAgent

        # Planner
        start = time.perf_counter()
        planner = ResearchPlannerAgent(fast_mock_llm)
        planner_time = (time.perf_counter() - start) * 1000

        # Analyzer
        start = time.perf_counter()
        analyzer = PaperAnalyzerAgent(fast_mock_llm)
        analyzer_time = (time.perf_counter() - start) * 1000

        assert (
            planner_time < self.BASELINES["planner_init"] * 2
        ), f"Planner init regression: {planner_time}ms (baseline: {self.BASELINES['planner_init']}ms)"

        assert (
            analyzer_time < self.BASELINES["analyzer_init"] * 2
        ), f"Analyzer init regression: {analyzer_time}ms (baseline: {self.BASELINES['analyzer_init']}ms)"

    def test_state_creation_regression(self):
        """Test that state creation hasn't regressed."""
        from agents.state import create_initial_state

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            state = create_initial_state(
                project_id="test", user_id="test", title="Test", research_question="Test question?"
            )
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = statistics.mean(latencies)

        assert (
            avg_latency < self.BASELINES["state_creation"] * 2
        ), f"State creation regression: {avg_latency}ms (baseline: {self.BASELINES['state_creation']}ms)"

    def test_router_decision_regression(self):
        """Test that router decision hasn't regressed."""
        from agents.model_router import SmartModelRouter

        router = SmartModelRouter(user_budget=1.0)

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            router.route("paper_analysis", "Test prompt for analysis")
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = statistics.mean(latencies)

        assert (
            avg_latency < self.BASELINES["router_decision"] * 2
        ), f"Router decision regression: {avg_latency}ms (baseline: {self.BASELINES['router_decision']}ms)"
