"""
RAG Service - High-level interface for RAG operations.

Provides a simplified API for:
- Ingesting papers into the vector store
- Performing semantic search
- Managing project data
"""

import logging
from typing import Any

from .chunker import SemanticChunker
from .embeddings import EmbeddingService, get_embedding_service
from .hybrid_search import HybridSearchEngine, get_search_engine
from .reranker import CrossEncoderReranker
from .vector_store import AcademicVectorStore, get_vector_store

logger = logging.getLogger(__name__)


class RAGService:
    """
    High-level RAG service for the Scholar Agent.
    
    Provides a unified interface for all RAG operations.
    
    Usage:
        rag = RAGService()
        
        # Ingest papers
        rag.ingest_papers(papers, project_id="project123")
        
        # Search
        results = rag.search("machine learning in education", project_id="project123")
    """

    def __init__(
        self,
        vector_store: AcademicVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        search_engine: HybridSearchEngine | None = None,
        chunker: SemanticChunker | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        """
        Initialize the RAG service.
        
        Args:
            vector_store: Vector store instance (auto-created if None)
            embedding_service: Embedding service instance (auto-created if None)
            search_engine: Hybrid search engine (auto-created if None)
            chunker: Semantic chunker (auto-created if None)
            reranker: Reranker instance (auto-created if None)
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.search_engine = search_engine
        self.chunker = chunker or SemanticChunker()
        self.reranker = reranker or CrossEncoderReranker(use_llm_reranker=True)

        # Lazy initialization
        self._initialized = False

        logger.info("RAGService created (lazy initialization)")

    def _ensure_initialized(self):
        """Ensure all services are initialized."""
        if self._initialized:
            return

        try:
            if self.vector_store is None:
                self.vector_store = get_vector_store()

            if self.embedding_service is None:
                self.embedding_service = get_embedding_service()

            if self.search_engine is None:
                self.search_engine = get_search_engine()

            self._initialized = True
            logger.info("RAGService fully initialized")

        except Exception as e:
            logger.error(f"Failed to initialize RAG services: {e}")
            raise

    def ingest_papers(
        self,
        papers: list[dict],
        project_id: str,
        rebuild_bm25: bool = True,
    ) -> dict[str, Any]:
        """
        Ingest papers into the RAG system.
        
        Args:
            papers: List of paper dictionaries with title, abstract, etc.
            project_id: Project ID for isolation
            rebuild_bm25: Whether to rebuild the BM25 index
            
        Returns:
            Dict with ingestion statistics
        """
        self._ensure_initialized()

        if not papers:
            return {"chunks_ingested": 0, "papers_processed": 0}

        logger.info(f"Ingesting {len(papers)} papers for project {project_id}")

        # Chunk papers
        all_chunks = self.chunker.chunk_papers(papers)

        # Ingest into vector store
        chunks_ingested = self.vector_store.ingest_chunks(all_chunks, project_id)

        # Build BM25 index
        if rebuild_bm25 and self.search_engine:
            documents = [
                {
                    "content": chunk.content,
                    "chunk_id": f"{chunk.paper_id}_{chunk.chunk_index}",
                    "paper_id": chunk.paper_id,
                    "paper_title": chunk.paper_title,
                    "chunk_type": chunk.chunk_type.value,
                    "weight": chunk.weight,
                }
                for chunk in all_chunks
            ]
            self.search_engine.index_project_documents(project_id, documents)

        stats = {
            "chunks_ingested": chunks_ingested,
            "papers_processed": len(papers),
            "avg_chunks_per_paper": chunks_ingested / max(len(papers), 1),
        }

        logger.info(f"Ingestion complete: {stats}")

        return stats

    def search(
        self,
        query: str,
        project_id: str,
        top_k: int = 10,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        chunk_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant content.
        
        Args:
            query: Search query
            project_id: Project ID to search within
            top_k: Number of results to return
            use_hybrid: Use hybrid search (vector + BM25)
            use_reranker: Apply reranking to results
            chunk_types: Filter by chunk types
            
        Returns:
            List of search results as dictionaries
        """
        self._ensure_initialized()

        logger.debug(f"Searching for '{query[:50]}...' in project {project_id}")

        if use_hybrid and self.search_engine:
            # Use hybrid search
            results = self.search_engine.search(
                query=query,
                project_id=project_id,
                top_k=top_k,
                use_reranker=use_reranker,
                chunk_types=chunk_types,
            )
            return [r.to_dict() for r in results]
        else:
            # Use vector-only search
            results = self.vector_store.search(
                query=query,
                project_id=project_id,
                top_k=top_k,
                chunk_types=chunk_types,
            )
            return [r.to_dict() for r in results]

    def search_similar_papers(
        self,
        paper_id: str,
        project_id: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Find papers similar to a given paper.
        
        Args:
            paper_id: ID of the source paper
            project_id: Project ID
            top_k: Number of similar papers to return
            
        Returns:
            List of similar paper results
        """
        self._ensure_initialized()

        # First, get the abstract of the source paper
        results = self.vector_store.search(
            query="",  # Will be overridden
            project_id=project_id,
            top_k=1,
            chunk_types=["abstract"],
        )

        # Filter to find source paper's abstract
        source_abstract = None
        for r in results:
            if r.paper_id == paper_id:
                source_abstract = r.content
                break

        if not source_abstract:
            logger.warning(f"Could not find abstract for paper {paper_id}")
            return []

        # Search using the abstract as query
        results = self.search(
            query=source_abstract,
            project_id=project_id,
            top_k=top_k + 5,  # Get extra to filter out self
            use_hybrid=True,
            chunk_types=["abstract"],
        )

        # Filter out the source paper and deduplicate
        seen_papers = {paper_id}
        similar_papers = []

        for result in results:
            if result["paper_id"] not in seen_papers:
                seen_papers.add(result["paper_id"])
                similar_papers.append(result)

                if len(similar_papers) >= top_k:
                    break

        return similar_papers

    def get_context_for_synthesis(
        self,
        research_question: str,
        project_id: str,
        max_chunks: int = 20,
        max_tokens: int = 8000,
    ) -> str:
        """
        Get relevant context for synthesis.
        
        Retrieves and formats chunks for use in synthesis prompts.
        
        Args:
            research_question: The research question
            project_id: Project ID
            max_chunks: Maximum chunks to retrieve
            max_tokens: Approximate token limit
            
        Returns:
            Formatted context string
        """
        self._ensure_initialized()

        # Search with high recall
        results = self.search(
            query=research_question,
            project_id=project_id,
            top_k=max_chunks,
            use_hybrid=True,
            use_reranker=True,
        )

        # Format context with source tracking
        context_parts = []
        total_chars = 0
        char_limit = max_tokens * 4  # Approximate chars from tokens

        for i, result in enumerate(results, 1):
            content = result["content"]
            paper_title = result.get("paper_title", "Unknown")
            chunk_type = result.get("chunk_type", "general")

            # Create formatted entry
            entry = f"\n[Source {i}: {paper_title} ({chunk_type})]\n{content}\n"

            if total_chars + len(entry) > char_limit:
                break

            context_parts.append(entry)
            total_chars += len(entry)

        context = "".join(context_parts)

        logger.debug(
            f"Retrieved {len(context_parts)} chunks ({total_chars} chars) for synthesis"
        )

        return context

    def delete_project_data(self, project_id: str) -> dict[str, Any]:
        """
        Delete all RAG data for a project.
        
        Args:
            project_id: Project ID to delete
            
        Returns:
            Dict with deletion stats
        """
        self._ensure_initialized()

        logger.info(f"Deleting RAG data for project {project_id}")

        # Delete from vector store
        self.vector_store.delete_project_data(project_id)

        # Clear BM25 index
        if self.search_engine:
            self.search_engine.clear_project_index(project_id)

        return {"project_id": project_id, "deleted": True}

    def get_project_stats(self, project_id: str) -> dict[str, Any]:
        """
        Get RAG statistics for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            Dict with project statistics
        """
        self._ensure_initialized()

        return self.vector_store.get_project_stats(project_id)

    def get_embedding_stats(self) -> dict[str, Any]:
        """Get embedding service statistics."""
        self._ensure_initialized()
        return self.embedding_service.get_stats()


# Singleton instance
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Get or create the global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
