"""
Unit and integration tests for HybridSearchEngine and BM25Index.

Verifies:
- BM25 tokenization, inverted indexing, and Robertson-Spärck Jones smoothed IDF
- Exact mathematical calculation of Section-Weighted Reciprocal Rank Fusion (RRF):
    RRF(d) = sum_{m in {vector, bm25}} (1 / (60 + rank_m(d))) * W(section_type)
- Default section multipliers: RESULTS (1.4x), METHODOLOGY (1.3x), LIMITATIONS (1.2x), ABSTRACT (1.1x), GENERAL (1.0x)
- Custom section_weights overrides
- Graceful resilience to empty queries, whitespace, empty indexes, and missing projects
- Project-isolated BM25 and vector retrieval
"""

from unittest.mock import MagicMock

import pytest

from backend.rag.hybrid_search import (
    DEFAULT_SECTION_MULTIPLIERS,
    BM25Index,
    HybridSearchEngine,
    HybridSearchResult,
    get_search_engine,
)
from backend.rag.vector_store import AcademicVectorStore, SearchResult


class TestBM25Index:
    """Test standalone BM25 inverted index and scoring."""

    @pytest.fixture
    def sample_docs(self):
        return [
            {
                "chunk_id": "c1",
                "content": "Deep reinforcement learning for autonomous scientific discovery agents.",
                "paper_id": "p1",
                "paper_title": "RL Agents",
                "chunk_type": "methodology",
            },
            {
                "chunk_id": "c2",
                "content": "Empirical evaluation and results across multi-agent benchmark datasets.",
                "paper_id": "p2",
                "paper_title": "Benchmarking",
                "chunk_type": "results",
            },
            {
                "chunk_id": "c3",
                "content": "Limitations include high compute cost and latency in distributed multi-agent systems.",
                "paper_id": "p3",
                "paper_title": "Scaling Limits",
                "chunk_type": "limitations",
            },
        ]

    def test_bm25_index_and_search(self, sample_docs):
        index = BM25Index(k1=1.5, b=0.75)
        index.add_documents(sample_docs)

        assert index.total_docs == 3
        assert index.avg_doc_length > 0

        # Query matching document 1
        results = index.search("reinforcement learning autonomous", top_k=2)
        assert len(results) > 0
        assert results[0][0] == "c1"
        assert results[0][1] > 0.0

        # Query matching document 3
        results_lim = index.search("compute cost limitations", top_k=2)
        assert len(results_lim) > 0
        assert results_lim[0][0] == "c3"

    def test_bm25_empty_query_and_empty_index(self):
        index = BM25Index()
        assert index.search("test", top_k=5) == []

        index.add_documents([{"chunk_id": "c1", "content": "Sample content"}])
        assert index.search("", top_k=5) == []
        assert index.search("   ", top_k=5) == []
        assert index.search("nonexistent_word_xyz", top_k=5) == []

    def test_bm25_get_document_and_clear(self, sample_docs):
        index = BM25Index()
        index.add_documents(sample_docs)

        doc = index.get_document("c2")
        assert doc is not None
        assert doc["paper_title"] == "Benchmarking"

        doc_by_int = index.get_document(0)
        assert doc_by_int is not None
        assert doc_by_int["chunk_id"] == "c1"

        index.clear()
        assert index.total_docs == 0
        assert index.get_document("c2") is None


class TestWeightedRRFMath:
    """Test exact mathematical formulation of Section-Weighted Reciprocal Rank Fusion."""

    def test_controlled_rrf_scoring_math(self):
        engine = HybridSearchEngine(rrf_k=60)

        # Chunk A: RESULTS (weight 1.4), vector rank 2, BM25 rank 1
        # Raw RRF = (1/(60+2) + 1/(60+1)) = (1/62 + 1/61) = 0.016129032 + 0.016393442 = 0.032522474
        # Weighted RRF = 0.032522474 * 1.4 = 0.045531464
        expected_rrf_a = (1.0 / 62.0 + 1.0 / 61.0) * 1.4

        # Chunk B: METHODOLOGY (weight 1.3), vector rank 1, not in BM25
        # Raw RRF = (1/(60+1)) = 1/61 = 0.016393442
        # Weighted RRF = 0.016393442 * 1.3 = 0.021311475
        expected_rrf_b = (1.0 / 61.0) * 1.3

        vec_results = [
            SearchResult(
                chunk_id="chunk_b",
                content="Methodology content",
                paper_id="p_b",
                paper_title="Title B",
                chunk_type="methodology",
                score=0.92,
                weight=1.3,
            ),
            SearchResult(
                chunk_id="chunk_a",
                content="Results content",
                paper_id="p_a",
                paper_title="Title A",
                chunk_type="results",
                score=0.88,
                weight=1.4,
            ),
        ]
        bm25_results = [("chunk_a", 4.5)]

        fused = engine._reciprocal_rank_fusion(
            vector_results=vec_results,
            bm25_results=bm25_results,
            bm25_index=None,
        )

        assert len(fused) == 2
        # First should be chunk_a due to double presence + 1.4x multiplier
        assert fused[0].chunk_id == "chunk_a"
        assert fused[0].chunk_type == "results"
        assert abs(fused[0].rrf_score - expected_rrf_a) < 1e-6

        assert fused[1].chunk_id == "chunk_b"
        assert fused[1].chunk_type == "methodology"
        assert abs(fused[1].rrf_score - expected_rrf_b) < 1e-6


class TestSectionImportanceMultipliers:
    """Test section importance multipliers and rank tie-breaking."""

    def test_section_multiplier_ranking_hierarchy(self):
        engine = HybridSearchEngine(rrf_k=60)

        # 5 chunks with identical vector score and identical rank 1 in isolated scenarios
        # We test get_section_multiplier directly
        assert engine.get_section_multiplier("results") == 1.4
        assert engine.get_section_multiplier("methodology") == 1.3
        assert engine.get_section_multiplier("limitations") == 1.2
        assert engine.get_section_multiplier("abstract") == 1.1
        assert engine.get_section_multiplier("tables") == 1.1
        assert engine.get_section_multiplier("general") == 1.0
        assert engine.get_section_multiplier("introduction") == 1.0

        # Now test 5 identical-rank items fused
        vec_results = [
            SearchResult(f"c_{t}", f"Content {t}", "p", "T", t, 0.9, 1.0)
            for t in ["general", "abstract", "limitations", "methodology", "results"]
        ]
        # In vector results, ranks are 1 to 5. Let's make ranks identical by putting them all in BM25 at rank 1 each separately
        # Or test when each has same raw rank in vector:
        # If order in vector is [results (rank 1), methodology (rank 2), limitations (rank 3)...],
        # let's test if section multipliers break equal base RRF:
        fused = engine._reciprocal_rank_fusion(
            vector_results=vec_results,
            bm25_results=[],
            bm25_index=None,
        )
        assert len(fused) == 5

    def test_custom_section_weights_override(self):
        engine = HybridSearchEngine(rrf_k=60)

        custom_weights = {"general": 3.0, "results": 0.5}
        mult_general = engine.get_section_multiplier("general", section_weights=custom_weights)
        mult_results = engine.get_section_multiplier("results", section_weights=custom_weights)

        assert mult_general == 3.0
        assert mult_results == 0.5

        # Test case-insensitivity in custom weights
        mult_general_upper = engine.get_section_multiplier(
            "GENERAL", section_weights={"General": 2.5}
        )
        assert mult_general_upper == 2.5


class TestHybridSearchEngineEdgeCases:
    """Test boundary conditions and empty index resilience."""

    @pytest.fixture
    def search_engine_with_mock_store(self):
        mock_store = MagicMock(spec=AcademicVectorStore)
        mock_store.search.return_value = []
        return HybridSearchEngine(vector_store=mock_store)

    def test_empty_query_returns_empty_list(self, search_engine_with_mock_store):
        assert search_engine_with_mock_store.search("", project_id="proj_1") == []
        assert search_engine_with_mock_store.search("   ", project_id="proj_1") == []
        assert search_engine_with_mock_store.search("query", project_id="proj_1", top_k=0) == []
        assert search_engine_with_mock_store.search("query", project_id="proj_1", top_k=-5) == []

    def test_empty_project_returns_empty_list(self, search_engine_with_mock_store):
        results = search_engine_with_mock_store.search(
            query="multi agent synthesis", project_id="nonexistent_proj"
        )
        assert results == []

    def test_project_isolation(self):
        engine = HybridSearchEngine(vector_store=None)

        # Index project A
        docs_a = [
            {
                "chunk_id": "pA_c1",
                "content": "Quantum computing error mitigation strategies.",
                "paper_id": "pA",
                "paper_title": "Quantum Paper",
                "chunk_type": "methodology",
            }
        ]
        engine.index_project_documents("project_A", docs_a)

        # Index project B
        docs_b = [
            {
                "chunk_id": "pB_c1",
                "content": "Neuroscience models of human working memory.",
                "paper_id": "pB",
                "paper_title": "Neuroscience Paper",
                "chunk_type": "results",
            }
        ]
        engine.index_project_documents("project_B", docs_b)

        # Search project A for quantum
        res_a = engine.search("quantum error", project_id="project_A")
        assert len(res_a) == 1
        assert res_a[0].chunk_id == "pA_c1"

        # Search project B for quantum -> should return empty
        res_b = engine.search("quantum error", project_id="project_B")
        assert len(res_b) == 0

        # Clear project A
        engine.clear_project_index("project_A")
        assert engine.search("quantum error", project_id="project_A") == []

    def test_chunk_type_filtering(self):
        engine = HybridSearchEngine(vector_store=None)
        docs = [
            {
                "chunk_id": "c1",
                "content": "Evaluation results on ImageNet show 92% accuracy.",
                "chunk_type": "results",
            },
            {
                "chunk_id": "c2",
                "content": "Evaluation methodology includes 5-fold cross validation.",
                "chunk_type": "methodology",
            },
        ]
        engine.index_project_documents("proj_filt", docs)

        # Filter only results
        res_results = engine.search("evaluation", project_id="proj_filt", chunk_types=["results"])
        assert len(res_results) == 1
        assert res_results[0].chunk_id == "c1"

    def test_singleton_getter(self):
        engine1 = get_search_engine()
        engine2 = get_search_engine()
        assert engine1 is engine2
        assert isinstance(engine1, HybridSearchEngine)
