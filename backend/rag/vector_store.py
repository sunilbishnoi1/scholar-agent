"""
Qdrant Vector Store for Academic Papers.

Handles storage and retrieval of paper embeddings using Qdrant.
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SearchParams,
    VectorParams,
)

from .chunker import PaperChunk, SemanticChunker
from .embeddings import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""
    chunk_id: str
    content: str
    paper_id: str
    paper_title: str
    chunk_type: str
    score: float
    weight: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "chunk_type": self.chunk_type,
            "score": self.score,
            "weight": self.weight,
            "metadata": self.metadata,
        }


class AcademicVectorStore:
    """
    Vector store for academic papers using Qdrant.
    
    Features:
    - Automatic collection management
    - Chunk deduplication via content hashing
    - Project-level isolation
    - Metadata filtering
    - Weighted scoring
    """

    COLLECTION_NAME = "academic_papers"

    def __init__(
        self,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        embedding_service: EmbeddingService | None = None,
        collection_name: str | None = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            qdrant_url: Qdrant server URL (defaults to QDRANT_URL env var)
            qdrant_api_key: Qdrant API key (defaults to QDRANT_API_KEY env var)
            embedding_service: Embedding service instance
            collection_name: Override default collection name
        """
        self.qdrant_url = qdrant_url or os.environ.get(
            "QDRANT_URL", "http://localhost:6333"
        )
        self.qdrant_api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY")

        # Initialize Qdrant client
        if self.qdrant_api_key:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )
        else:
            self.client = QdrantClient(url=self.qdrant_url)

        # Embedding service
        self.embedding_service = embedding_service or get_embedding_service()
        self.embedding_dim = self.embedding_service.EMBEDDING_DIM

        # Collection name
        self.collection_name = collection_name or self.COLLECTION_NAME

        # Ensure collection exists
        self._ensure_collection()

        logger.info(
            f"AcademicVectorStore initialized (url={self.qdrant_url}, "
            f"collection={self.collection_name})"
        )

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                    # Enable payload indexing for filtering
                    optimizers_config=qdrant_models.OptimizersConfigDiff(
                        indexing_threshold=10000,
                    ),
                )

                # Create payload indexes for efficient filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="project_id",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="paper_id",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="chunk_type",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                )

                logger.info(f"Created collection '{self.collection_name}'")
            else:
                logger.debug(f"Collection '{self.collection_name}' already exists")

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def _generate_chunk_id(self, chunk: PaperChunk, project_id: str) -> str:
        """Generate unique ID for a chunk (for deduplication)."""
        content_hash = hashlib.md5(
            f"{project_id}:{chunk.paper_id}:{chunk.chunk_type.value}:{chunk.content[:200]}".encode()
        ).hexdigest()
        return content_hash

    def ingest_chunks(
        self,
        chunks: list[PaperChunk],
        project_id: str,
        batch_size: int = 100,
    ) -> int:
        """
        Ingest paper chunks into the vector store.
        
        Args:
            chunks: List of PaperChunk objects
            project_id: Project ID for isolation
            batch_size: Batch size for embedding and upsert
            
        Returns:
            Number of chunks ingested
        """
        if not chunks:
            return 0

        logger.info(f"Ingesting {len(chunks)} chunks for project {project_id}")

        total_ingested = 0

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Generate embeddings
            texts = [chunk.content for chunk in batch]
            embeddings = self.embedding_service.embed_batch(texts)

            # Create points
            points = []
            for chunk, embedding in zip(batch, embeddings):
                chunk_id = self._generate_chunk_id(chunk, project_id)

                payload = {
                    "project_id": project_id,
                    "paper_id": chunk.paper_id,
                    "paper_title": chunk.paper_title,
                    "content": chunk.content,
                    "chunk_type": chunk.chunk_type.value,
                    "chunk_index": chunk.chunk_index,
                    "weight": chunk.weight,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                    "ingested_at": datetime.utcnow().isoformat(),
                }

                points.append(PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload,
                ))

            # Upsert to Qdrant (handles duplicates automatically)
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

            total_ingested += len(points)
            logger.debug(f"Ingested batch {i // batch_size + 1}: {len(points)} chunks")

        logger.info(f"Successfully ingested {total_ingested} chunks")
        return total_ingested

    def ingest_papers(
        self,
        papers: list[dict],
        project_id: str,
        chunker: SemanticChunker | None = None,
    ) -> int:
        """
        Chunk and ingest papers into the vector store.
        
        Args:
            papers: List of paper dictionaries
            project_id: Project ID for isolation
            chunker: Optional custom chunker
            
        Returns:
            Number of chunks ingested
        """
        if not papers:
            return 0

        # Use default chunker if not provided
        if chunker is None:
            chunker = SemanticChunker()

        # Chunk all papers
        all_chunks = chunker.chunk_papers(papers)

        # Ingest chunks
        return self.ingest_chunks(all_chunks, project_id)

    def search(
        self,
        query: str,
        project_id: str,
        top_k: int = 10,
        chunk_types: list[str] | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search for relevant chunks using vector similarity.
        
        Args:
            query: Search query text
            project_id: Project ID to search within
            top_k: Number of results to return
            chunk_types: Filter by chunk types (e.g., ["abstract", "methodology"])
            score_threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed(query)

        # Build filter
        filter_conditions = [
            FieldCondition(
                key="project_id",
                match=MatchValue(value=project_id),
            )
        ]

        if chunk_types:
            filter_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=qdrant_models.MatchAny(any=chunk_types),
                )
            )

        search_filter = Filter(must=filter_conditions)

        # Search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=search_filter,
            limit=top_k,
            score_threshold=score_threshold,
            search_params=SearchParams(
                hnsw_ef=128,  # Higher for better recall
                exact=False,
            ),
        ).points

        # Convert to SearchResult objects
        search_results = []
        for hit in results:
            payload = hit.payload
            search_results.append(SearchResult(
                chunk_id=str(hit.id),
                content=payload.get("content", ""),
                paper_id=payload.get("paper_id", ""),
                paper_title=payload.get("paper_title", ""),
                chunk_type=payload.get("chunk_type", "general"),
                score=hit.score,
                weight=payload.get("weight", 1.0),
                metadata=payload.get("metadata", {}),
            ))

        logger.debug(f"Search returned {len(search_results)} results for project {project_id}")

        return search_results

    def search_with_weighted_score(
        self,
        query: str,
        project_id: str,
        top_k: int = 10,
        **kwargs
    ) -> list[SearchResult]:
        """
        Search with weight-adjusted scores.
        
        Combines vector similarity with chunk importance weights.
        """
        results = self.search(query, project_id, top_k=top_k * 2, **kwargs)

        # Apply weight adjustments
        for result in results:
            result.score = result.score * result.weight

        # Re-sort by weighted score
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def delete_project_data(self, project_id: str) -> int:
        """
        Delete all chunks for a project.
        
        Args:
            project_id: Project ID to delete
            
        Returns:
            Number of points deleted
        """
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="project_id",
                            match=MatchValue(value=project_id),
                        )
                    ]
                )
            ),
        )

        logger.info(f"Deleted data for project {project_id}")
        return result

    def get_project_stats(self, project_id: str) -> dict[str, Any]:
        """Get statistics for a project's chunks."""
        # Count total chunks
        count_result = self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="project_id",
                        match=MatchValue(value=project_id),
                    )
                ]
            ),
        )

        return {
            "project_id": project_id,
            "total_chunks": count_result.count,
            "collection_name": self.collection_name,
        }

    def get_collection_info(self) -> dict[str, Any]:
        """Get collection information and statistics."""
        info = self.client.get_collection(self.collection_name)

        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
            "optimizer_status": info.optimizer_status.status.value,
        }


# Singleton instance
_vector_store: AcademicVectorStore | None = None


def get_vector_store() -> AcademicVectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = AcademicVectorStore()
    return _vector_store
