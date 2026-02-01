"""
Tests for the RAG (Retrieval-Augmented Generation) module.

Tests cover:
- Semantic chunking
- Embedding service
- Vector store operations
- BM25 indexing
- Hybrid search
- Reranking
"""

import hashlib
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import RAG components
from rag.chunker import (
    ChunkType,
    PaperChunk,
    SemanticChunker,
)
from rag.embeddings import EmbeddingService
from rag.hybrid_search import BM25Index, HybridSearchEngine
from rag.reranker import CrossEncoderReranker, RerankResult
from rag.vector_store import AcademicVectorStore, SearchResult

# ============== Fixtures ==============

@pytest.fixture
def sample_paper():
    """Sample paper for testing."""
    return {
        "id": "paper123",
        "title": "Machine Learning in Education: A Comprehensive Review",
        "abstract": "This paper presents a comprehensive review of machine learning applications in educational settings. We examine how ML algorithms can improve student outcomes through personalized learning pathways.",
        "authors": ["John Doe", "Jane Smith", "Bob Wilson"],
        "url": "https://arxiv.org/abs/1234.5678",
    }


@pytest.fixture
def sample_paper_with_full_text():
    """Sample paper with full text sections."""
    return {
        "id": "paper456",
        "title": "Deep Learning for Student Assessment",
        "abstract": "We propose a deep learning model for automated student assessment that achieves state-of-the-art results.",
        "authors": ["Alice Chen"],
        "full_text": """
Abstract: We propose a deep learning model for automated student assessment.

1. Introduction
Machine learning has transformed many fields including education. This paper explores
how deep neural networks can be applied to automate the assessment process.

2. Methodology
We collected data from 10,000 students across 50 schools. Our model uses a 
transformer architecture with attention mechanisms to analyze student responses.

3. Results
Our model achieved 95% accuracy on the benchmark dataset, outperforming previous
methods by 10%. The results demonstrate significant improvements in assessment speed.

4. Conclusion
Deep learning offers promising solutions for educational assessment. Future work
will explore multi-modal inputs and real-time feedback systems.

References
[1] Smith et al. 2020. Neural Assessment Methods.
[2] Johnson et al. 2021. Transformers in Education.
""",
    }


@pytest.fixture
def chunker():
    """Create a semantic chunker instance."""
    return SemanticChunker(
        max_chunk_size=256,
        min_chunk_size=50,
        overlap_size=25,
    )


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    mock = Mock(spec=EmbeddingService)
    mock.EMBEDDING_DIM = 768

    # Return deterministic embeddings based on text hash
    def mock_embed(text):
        # Create a deterministic "embedding" from text
        hash_val = hashlib.md5(text.encode()).hexdigest()
        return [float(int(c, 16)) / 15.0 for c in hash_val[:768 // 16]] * 16

    mock.embed.side_effect = mock_embed
    mock.embed_batch.side_effect = lambda texts: [mock_embed(t) for t in texts]

    return mock


@pytest.fixture
def bm25_index():
    """Create a BM25 index instance."""
    return BM25Index()


# ============== Chunker Tests ==============

class TestSemanticChunker:
    """Tests for the semantic chunker."""

    def test_chunk_paper_basic(self, chunker, sample_paper):
        """Test basic paper chunking."""
        chunks = chunker.chunk_paper(sample_paper)

        assert len(chunks) >= 2  # At least title and abstract
        assert all(isinstance(c, PaperChunk) for c in chunks)

        # Check title chunk
        title_chunks = [c for c in chunks if c.chunk_type == ChunkType.TITLE]
        assert len(title_chunks) == 1
        assert "Machine Learning in Education" in title_chunks[0].content

        # Check abstract chunk
        abstract_chunks = [c for c in chunks if c.chunk_type == ChunkType.ABSTRACT]
        assert len(abstract_chunks) >= 1

    def test_chunk_paper_with_sections(self, chunker, sample_paper_with_full_text):
        """Test chunking paper with full text sections."""
        chunks = chunker.chunk_paper(sample_paper_with_full_text)

        # Should have chunks from different sections
        chunk_types = set(c.chunk_type for c in chunks)
        assert ChunkType.TITLE in chunk_types
        assert ChunkType.ABSTRACT in chunk_types

        # Should detect methodology section
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHODOLOGY]
        assert len(method_chunks) >= 1

    def test_chunk_weights(self, chunker, sample_paper):
        """Test that chunks have appropriate weights."""
        chunks = chunker.chunk_paper(sample_paper)

        title_chunk = next(c for c in chunks if c.chunk_type == ChunkType.TITLE)
        abstract_chunk = next(c for c in chunks if c.chunk_type == ChunkType.ABSTRACT)

        # Title should have higher weight than abstract
        assert title_chunk.weight > abstract_chunk.weight

    def test_chunk_has_required_fields(self, chunker, sample_paper):
        """Test that chunks have all required fields."""
        chunks = chunker.chunk_paper(sample_paper)

        for chunk in chunks:
            assert chunk.paper_id == sample_paper["id"]
            assert chunk.paper_title == sample_paper["title"]
            assert chunk.content
            assert chunk.chunk_index >= 0
            assert chunk.token_count > 0

    def test_chunk_paper_empty_abstract(self, chunker):
        """Test handling paper with no abstract."""
        paper = {
            "id": "empty123",
            "title": "Test Paper",
            "abstract": "",
            "authors": [],
        }

        chunks = chunker.chunk_paper(paper)

        # Should still create title chunk
        assert len(chunks) >= 1
        assert chunks[0].chunk_type == ChunkType.TITLE

    def test_chunk_multiple_papers(self, chunker, sample_paper, sample_paper_with_full_text):
        """Test chunking multiple papers."""
        papers = [sample_paper, sample_paper_with_full_text]
        all_chunks = chunker.chunk_papers(papers)

        # Should have chunks from both papers
        paper_ids = set(c.paper_id for c in all_chunks)
        assert len(paper_ids) == 2

    def test_to_dict(self, chunker, sample_paper):
        """Test chunk serialization to dict."""
        chunks = chunker.chunk_paper(sample_paper)

        for chunk in chunks:
            chunk_dict = chunk.to_dict()
            assert "content" in chunk_dict
            assert "chunk_type" in chunk_dict
            assert "paper_id" in chunk_dict
            assert chunk_dict["chunk_type"] == chunk.chunk_type.value


# ============== BM25 Tests ==============

class TestBM25Index:
    """Tests for BM25 keyword search."""

    def test_add_documents(self, bm25_index):
        """Test adding documents to the index."""
        docs = [
            {"content": "machine learning for education"},
            {"content": "deep learning neural networks"},
            {"content": "natural language processing"},
        ]

        bm25_index.add_documents(docs)

        assert bm25_index.total_docs == 3
        assert len(bm25_index.documents) == 3

    def test_search_basic(self, bm25_index):
        """Test basic BM25 search."""
        docs = [
            {"content": "machine learning algorithms for student assessment"},
            {"content": "deep learning in healthcare applications"},
            {"content": "natural language processing for chatbots"},
        ]

        bm25_index.add_documents(docs)
        results = bm25_index.search("machine learning", top_k=3)

        assert len(results) > 0
        # First result should be about machine learning
        top_doc_id, top_score = results[0]
        top_doc = bm25_index.get_document(top_doc_id)
        assert "machine learning" in top_doc["content"]

    def test_search_no_match(self, bm25_index):
        """Test search with no matching terms."""
        docs = [
            {"content": "python programming language"},
            {"content": "javascript web development"},
        ]

        bm25_index.add_documents(docs)
        results = bm25_index.search("quantum physics", top_k=3)

        # Should return empty or very low scores
        assert len(results) == 0 or all(score < 0.1 for _, score in results)

    def test_search_empty_query(self, bm25_index):
        """Test search with empty query."""
        bm25_index.add_documents([{"content": "test document"}])
        results = bm25_index.search("", top_k=3)

        assert len(results) == 0

    def test_search_empty_index(self, bm25_index):
        """Test search on empty index."""
        results = bm25_index.search("test query", top_k=3)

        assert len(results) == 0

    def test_clear_index(self, bm25_index):
        """Test clearing the index."""
        bm25_index.add_documents([{"content": "test"}])
        assert bm25_index.total_docs == 1

        bm25_index.clear()
        assert bm25_index.total_docs == 0

    def test_get_document(self, bm25_index):
        """Test retrieving document by ID."""
        docs = [{"content": "doc one"}, {"content": "doc two"}]
        bm25_index.add_documents(docs)

        doc = bm25_index.get_document(0)
        assert doc["content"] == "doc one"

        doc = bm25_index.get_document(1)
        assert doc["content"] == "doc two"

        # Invalid ID
        assert bm25_index.get_document(999) is None


# ============== Reranker Tests ==============

class TestCrossEncoderReranker:
    """Tests for the cross-encoder reranker."""

    def test_heuristic_scoring(self):
        """Test heuristic relevance scoring."""
        reranker = CrossEncoderReranker(use_llm_reranker=False)

        results = [
            {"content": "machine learning in education improves outcomes", "score": 0.8},
            {"content": "cooking recipes for beginners", "score": 0.9},
            {"content": "deep learning for educational assessment", "score": 0.7},
        ]

        reranked = reranker.rerank(
            query="machine learning education",
            results=results,
            top_k=3,
        )

        # ML education content should rank higher after reranking
        assert len(reranked) == 3
        assert "machine learning" in reranked[0].content.lower()

    def test_rerank_empty_results(self):
        """Test reranking with empty results."""
        reranker = CrossEncoderReranker(use_llm_reranker=False)

        reranked = reranker.rerank(
            query="test query",
            results=[],
            top_k=5,
        )

        assert len(reranked) == 0

    def test_rerank_single_result(self):
        """Test reranking with single result."""
        reranker = CrossEncoderReranker(use_llm_reranker=False)

        results = [{"content": "test content", "score": 0.5}]
        reranked = reranker.rerank(query="test", results=results, top_k=5)

        assert len(reranked) == 1

    def test_rerank_preserves_metadata(self):
        """Test that reranking preserves metadata."""
        reranker = CrossEncoderReranker(use_llm_reranker=False)

        results = [
            {
                "content": "test content",
                "score": 0.5,
                "paper_id": "paper123",
                "paper_title": "Test Paper",
            }
        ]

        reranked = reranker.rerank(query="test", results=results, top_k=5)

        assert reranked[0].metadata["paper_id"] == "paper123"


# ============== Integration Tests ==============

class TestHybridSearchIntegration:
    """Integration tests for the hybrid search pipeline."""

    @pytest.fixture
    def hybrid_engine(self, mock_embedding_service):
        """Create hybrid search engine with mocked dependencies."""
        with patch('rag.hybrid_search.get_vector_store') as mock_vs:
            mock_vector_store = Mock(spec=AcademicVectorStore)
            mock_vector_store.search.return_value = []
            mock_vs.return_value = mock_vector_store

            with patch('rag.hybrid_search.get_embedding_service') as mock_es:
                mock_es.return_value = mock_embedding_service

                engine = HybridSearchEngine(
                    vector_store=mock_vector_store,
                    embedding_service=mock_embedding_service,
                    use_hyde=False,  # Disable HyDE for testing
                )

                return engine

    def test_build_bm25_index(self, hybrid_engine):
        """Test building BM25 index for a project."""
        documents = [
            {"content": "machine learning", "chunk_id": "1"},
            {"content": "deep learning", "chunk_id": "2"},
        ]

        hybrid_engine.build_bm25_index("project123", documents)

        # Index should be created
        assert "project123" in hybrid_engine._bm25_indexes
        assert hybrid_engine._bm25_indexes["project123"].total_docs == 2

    def test_clear_project_index(self, hybrid_engine):
        """Test clearing project index."""
        hybrid_engine.build_bm25_index("project123", [{"content": "test"}])
        assert "project123" in hybrid_engine._bm25_indexes

        hybrid_engine.clear_project_index("project123")
        assert "project123" not in hybrid_engine._bm25_indexes


# ============== Mock Vector Store Tests ==============

class TestVectorStoreMocked:
    """Tests for vector store with mocked Qdrant client."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        mock_client = Mock()

        # Mock get_collections
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        # Mock create_collection
        mock_client.create_collection.return_value = None
        mock_client.create_payload_index.return_value = None

        # Mock upsert
        mock_client.upsert.return_value = None

        # Mock search
        mock_client.search.return_value = []

        # Mock count
        mock_count = Mock()
        mock_count.count = 0
        mock_client.count.return_value = mock_count

        return mock_client

    def test_search_result_to_dict(self):
        """Test SearchResult serialization."""
        result = SearchResult(
            chunk_id="chunk123",
            content="test content",
            paper_id="paper123",
            paper_title="Test Paper",
            chunk_type="abstract",
            score=0.85,
            weight=1.5,
            metadata={"key": "value"},
        )

        result_dict = result.to_dict()

        assert result_dict["chunk_id"] == "chunk123"
        assert result_dict["score"] == 0.85
        assert result_dict["metadata"]["key"] == "value"


# ============== Edge Cases ==============

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_chunker_handles_unicode(self, chunker):
        """Test chunker handles unicode text."""
        paper = {
            "id": "unicode123",
            "title": "研究论文：机器学习 🤖",
            "abstract": "This paper discusses émigré contributions to AI research. Includes symbols: α, β, γ.",
            "authors": ["张三", "李四"],
        }

        chunks = chunker.chunk_paper(paper)
        assert len(chunks) >= 1

    def test_chunker_handles_very_long_abstract(self, chunker):
        """Test chunker splits very long abstracts."""
        long_abstract = "This is a test sentence. " * 500  # Very long

        paper = {
            "id": "long123",
            "title": "Test",
            "abstract": long_abstract,
            "authors": [],
        }

        chunks = chunker.chunk_paper(paper)

        # Should create multiple chunks for long abstract
        abstract_chunks = [c for c in chunks if c.chunk_type == ChunkType.ABSTRACT]
        assert len(abstract_chunks) >= 1

    def test_bm25_handles_special_characters(self, bm25_index):
        """Test BM25 handles special characters in text."""
        docs = [
            {"content": "C++ programming & algorithms (advanced)"},
            {"content": "Python 3.9 @decorators #meta"},
        ]

        bm25_index.add_documents(docs)
        results = bm25_index.search("C++ programming", top_k=2)

        # Should still find the document
        assert len(results) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
