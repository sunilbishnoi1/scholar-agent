"""
RAG (Retrieval-Augmented Generation) Module for Scholar Agent.

This module provides:
- Vector store integration with Qdrant
- Section-aware semantic chunking for academic papers
- Hybrid search (dense + sparse/BM25) with section-weighted RRF
- Query expansion with HyDE
- Cross-encoder reranking
"""

from .chunker import (
    SECTION_WEIGHTS,
    ChunkType,
    PaperChunk,
    SectionAwareChunker,
    SemanticChunker,
    create_chunker,
)
from .embeddings import EmbeddingService, get_embedding_service
from .hybrid_search import (
    DEFAULT_SECTION_MULTIPLIERS,
    BM25Index,
    HybridSearchEngine,
    HybridSearchResult,
    get_search_engine,
)
from .reranker import CrossEncoderReranker, RerankResult
from .service import RAGService, get_rag_service
from .vector_store import AcademicVectorStore, SearchResult, get_vector_store

__all__ = [
    # Vector Store
    "AcademicVectorStore",
    "get_vector_store",
    "SearchResult",
    # Chunker
    "SectionAwareChunker",
    "SemanticChunker",
    "create_chunker",
    "ChunkType",
    "PaperChunk",
    "SECTION_WEIGHTS",
    # Embeddings
    "EmbeddingService",
    "get_embedding_service",
    # Hybrid Search
    "HybridSearchEngine",
    "get_search_engine",
    "HybridSearchResult",
    "BM25Index",
    "DEFAULT_SECTION_MULTIPLIERS",
    # Reranker
    "CrossEncoderReranker",
    "RerankResult",
    # Service
    "RAGService",
    "get_rag_service",
]
