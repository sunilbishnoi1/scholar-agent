"""
RAG (Retrieval-Augmented Generation) Module for Scholar Agent.

This module provides:
- Vector store integration with Qdrant
- Semantic chunking for academic papers
- Hybrid search (dense + sparse/BM25)
- Query expansion with HyDE
- Cross-encoder reranking
"""

from .vector_store import AcademicVectorStore, get_vector_store, SearchResult
from .chunker import SemanticChunker, ChunkType, PaperChunk
from .embeddings import EmbeddingService, get_embedding_service
from .hybrid_search import HybridSearchEngine, get_search_engine, HybridSearchResult
from .reranker import CrossEncoderReranker, RerankResult
from .service import RAGService, get_rag_service

__all__ = [
    # Vector Store
    "AcademicVectorStore",
    "get_vector_store",
    "SearchResult",
    # Chunker
    "SemanticChunker",
    "ChunkType",
    "PaperChunk",
    # Embeddings
    "EmbeddingService",
    "get_embedding_service",
    # Hybrid Search
    "HybridSearchEngine",
    "get_search_engine",
    "HybridSearchResult",
    # Reranker
    "CrossEncoderReranker",
    "RerankResult",
    # Service
    "RAGService",
    "get_rag_service",
]
