"""
Hybrid Search Engine for RAG Pipeline.

Combines dense vector search with sparse keyword search (BM25)
using Reciprocal Rank Fusion for optimal retrieval.
"""

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .embeddings import EmbeddingService, get_embedding_service
from .reranker import CrossEncoderReranker
from .vector_store import AcademicVectorStore, SearchResult, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""
    chunk_id: str
    content: str
    paper_id: str
    paper_title: str
    chunk_type: str
    vector_score: float
    bm25_score: float
    rrf_score: float
    final_score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "chunk_type": self.chunk_type,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "rrf_score": self.rrf_score,
            "final_score": self.final_score,
            "metadata": self.metadata,
        }


class BM25Index:
    """
    Simple BM25 implementation for keyword search.
    
    BM25 (Best Matching 25) is a bag-of-words retrieval function
    that ranks documents based on query term frequency.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b

        # Index storage
        self.documents: list[dict] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0

        # Inverted index: term -> list of (doc_id, term_freq)
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization with lowercasing and basic cleaning."""
        # Remove special characters, keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split and filter empty tokens
        tokens = [t.strip() for t in text.split() if t.strip()]
        return tokens

    def add_documents(self, documents: list[dict]):
        """
        Add documents to the BM25 index.
        
        Args:
            documents: List of dicts with 'content' and optional metadata
        """
        start_idx = len(self.documents)

        for i, doc in enumerate(documents):
            doc_id = start_idx + i
            content = doc.get("content", "")
            tokens = self._tokenize(content)

            # Store document
            self.documents.append(doc)
            self.doc_lengths.append(len(tokens))

            # Count term frequencies
            term_freqs: dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            # Update inverted index
            for term, freq in term_freqs.items():
                self.inverted_index[term].append((doc_id, freq))
                self.doc_freqs[term] += 1

        # Update statistics
        self.total_docs = len(self.documents)
        self.avg_doc_length = sum(self.doc_lengths) / max(self.total_docs, 1)

        logger.debug(f"Added {len(documents)} documents to BM25 index (total: {self.total_docs})")

    def _calculate_idf(self, term: str) -> float:
        """Calculate inverse document frequency for a term."""
        n = self.total_docs
        df = self.doc_freqs.get(term, 0)

        if df == 0:
            return 0.0

        # Standard IDF formula with smoothing
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Search the index using BM25 scoring.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        if self.total_docs == 0:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Calculate scores for each document
        scores: dict[int, float] = defaultdict(float)

        for token in query_tokens:
            idf = self._calculate_idf(token)

            if idf == 0:
                continue

            # Get documents containing this term
            for doc_id, tf in self.inverted_index.get(token, []):
                doc_length = self.doc_lengths[doc_id]

                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )

                scores[doc_id] += idf * (numerator / denominator)

        # Sort by score and return top_k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results[:top_k]

    def get_document(self, doc_id: int) -> dict | None:
        """Get document by ID."""
        if 0 <= doc_id < len(self.documents):
            return self.documents[doc_id]
        return None

    def clear(self):
        """Clear the index."""
        self.documents.clear()
        self.doc_freqs.clear()
        self.doc_lengths.clear()
        self.inverted_index.clear()
        self.avg_doc_length = 0.0
        self.total_docs = 0


class HybridSearchEngine:
    """
    Hybrid search combining dense vectors and sparse keywords.
    
    Features:
    - Dense vector search using Qdrant
    - Sparse keyword search using BM25
    - Query expansion with HyDE (Hypothetical Document Embeddings)
    - Reciprocal Rank Fusion (RRF) for combining results
    - Optional cross-encoder reranking
    """

    def __init__(
        self,
        vector_store: AcademicVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        reranker: CrossEncoderReranker | None = None,
        rrf_k: int = 60,
        use_hyde: bool = True,
    ):
        """
        Initialize the hybrid search engine.
        
        Args:
            vector_store: Vector store instance
            embedding_service: Embedding service instance
            reranker: Optional reranker for final reranking
            rrf_k: RRF constant (higher = more weight to lower ranks)
            use_hyde: Whether to use HyDE query expansion
        """
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or get_embedding_service()
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.use_hyde = use_hyde

        # BM25 indexes per project
        self._bm25_indexes: dict[str, BM25Index] = {}

        logger.info(f"HybridSearchEngine initialized (rrf_k={rrf_k}, use_hyde={use_hyde})")

    def _get_or_create_bm25_index(self, project_id: str) -> BM25Index:
        """Get or create BM25 index for a project."""
        if project_id not in self._bm25_indexes:
            self._bm25_indexes[project_id] = BM25Index()
        return self._bm25_indexes[project_id]

    def build_bm25_index(
        self,
        project_id: str,
        documents: list[dict],
    ):
        """
        Build BM25 index for a project.
        
        Args:
            project_id: Project ID
            documents: List of documents with 'content' field
        """
        index = self._get_or_create_bm25_index(project_id)
        index.clear()
        index.add_documents(documents)

        logger.info(f"Built BM25 index for project {project_id} with {len(documents)} documents")

    def _expand_query_hyde(self, query: str) -> str:
        """
        Expand query using HyDE (Hypothetical Document Embeddings).
        
        Generates a hypothetical answer to the query and uses it
        for retrieval, which often improves recall.
        """
        import os

        import requests

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return query

        try:
            prompt = f"""Given this research question, write a brief academic paragraph that might answer it.
Write as if this is from a research paper abstract. Be specific and technical.

Research Question: {query}

Brief academic paragraph:"""

            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200}
                },
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                hypothetical = result["candidates"][0]["content"]["parts"][0]["text"]
                # Combine original query with hypothetical for better coverage
                expanded = f"{query} {hypothetical}"
                logger.debug(f"HyDE expanded query from {len(query)} to {len(expanded)} chars")
                return expanded

        except Exception as e:
            logger.warning(f"HyDE expansion failed, using original query: {e}")

        return query

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[tuple[str, float]]],
    ) -> list[tuple[str, float]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each list
        
        Args:
            ranked_lists: List of (doc_id, score) lists
            
        Returns:
            Combined (doc_id, rrf_score) list sorted by score
        """
        rrf_scores: dict[str, float] = defaultdict(float)

        for ranked_list in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked_list, start=1):
                rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank)

        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results

    def search(
        self,
        query: str,
        project_id: str,
        top_k: int = 10,
        use_reranker: bool = True,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        chunk_types: list[str] | None = None,
    ) -> list[HybridSearchResult]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Search query
            project_id: Project ID to search within
            top_k: Number of results to return
            use_reranker: Whether to apply reranking
            vector_weight: Weight for vector search in RRF
            bm25_weight: Weight for BM25 in RRF
            chunk_types: Optional filter for chunk types
            
        Returns:
            List of HybridSearchResult objects
        """
        # Expand query with HyDE if enabled
        expanded_query = query
        if self.use_hyde:
            expanded_query = self._expand_query_hyde(query)

        # 1. Vector search
        vector_results = self.vector_store.search(
            query=expanded_query,
            project_id=project_id,
            top_k=top_k * 2,  # Get more for fusion
            chunk_types=chunk_types,
        )

        # Create lookup map
        results_map: dict[str, SearchResult] = {
            r.chunk_id: r for r in vector_results
        }

        # 2. BM25 search (if index exists)
        bm25_index = self._bm25_indexes.get(project_id)
        bm25_results = []

        if bm25_index and bm25_index.total_docs > 0:
            bm25_raw = bm25_index.search(query, top_k=top_k * 2)

            for doc_id, score in bm25_raw:
                doc = bm25_index.get_document(doc_id)
                if doc:
                    chunk_id = doc.get("chunk_id", str(doc_id))
                    bm25_results.append((chunk_id, score))

                    # Add to results map if not already present
                    if chunk_id not in results_map:
                        results_map[chunk_id] = SearchResult(
                            chunk_id=chunk_id,
                            content=doc.get("content", ""),
                            paper_id=doc.get("paper_id", ""),
                            paper_title=doc.get("paper_title", ""),
                            chunk_type=doc.get("chunk_type", "general"),
                            score=0.0,
                            weight=doc.get("weight", 1.0),
                            metadata=doc.get("metadata", {}),
                        )

        # 3. Reciprocal Rank Fusion
        vector_ranked = [(r.chunk_id, r.score) for r in vector_results]

        if bm25_results:
            # Combine using RRF
            rrf_results = self._reciprocal_rank_fusion([vector_ranked, bm25_results])
        else:
            # Just use vector results
            rrf_results = vector_ranked

        # Create lookup for scores
        vector_scores = {r.chunk_id: r.score for r in vector_results}
        bm25_scores = dict(bm25_results) if bm25_results else {}
        rrf_scores = dict(rrf_results)

        # 4. Build hybrid results
        hybrid_results = []
        for chunk_id, rrf_score in rrf_results[:top_k * 2]:
            if chunk_id not in results_map:
                continue

            result = results_map[chunk_id]

            hybrid_results.append(HybridSearchResult(
                chunk_id=chunk_id,
                content=result.content,
                paper_id=result.paper_id,
                paper_title=result.paper_title,
                chunk_type=result.chunk_type,
                vector_score=vector_scores.get(chunk_id, 0.0),
                bm25_score=bm25_scores.get(chunk_id, 0.0),
                rrf_score=rrf_score,
                final_score=rrf_score,
                metadata=result.metadata,
            ))

        # 5. Optional reranking
        if use_reranker and self.reranker and hybrid_results:
            results_for_rerank = [
                {
                    "content": r.content,
                    "score": r.rrf_score,
                    "weight": results_map[r.chunk_id].weight if r.chunk_id in results_map else 1.0,
                    **r.to_dict()
                }
                for r in hybrid_results
            ]

            reranked = self.reranker.rerank(query, results_for_rerank, top_k=top_k)

            # Update final scores from reranker
            rerank_map = {r.metadata["chunk_id"]: r.combined_score for r in reranked}

            for result in hybrid_results:
                if result.chunk_id in rerank_map:
                    result.final_score = rerank_map[result.chunk_id]

            # Re-sort by final score
            hybrid_results.sort(key=lambda x: x.final_score, reverse=True)

        logger.info(
            f"Hybrid search for '{query[:50]}...' returned {len(hybrid_results[:top_k])} results "
            f"(vector: {len(vector_results)}, bm25: {len(bm25_results)})"
        )

        return hybrid_results[:top_k]

    def index_project_documents(
        self,
        project_id: str,
        documents: list[dict],
    ):
        """
        Index documents for both vector and BM25 search.
        
        This should be called after ingesting papers into the vector store.
        
        Args:
            project_id: Project ID
            documents: List of document dicts with content and metadata
        """
        # Build BM25 index
        self.build_bm25_index(project_id, documents)

    def clear_project_index(self, project_id: str):
        """Clear BM25 index for a project."""
        if project_id in self._bm25_indexes:
            self._bm25_indexes[project_id].clear()
            del self._bm25_indexes[project_id]
            logger.info(f"Cleared BM25 index for project {project_id}")


# Singleton instance
_search_engine: HybridSearchEngine | None = None


def get_search_engine() -> HybridSearchEngine:
    """Get or create the global hybrid search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = HybridSearchEngine()
    return _search_engine
