"""
Hybrid Search Engine for RAG Pipeline.

Combines dense vector search with sparse keyword search (BM25)
using Reciprocal Rank Fusion (RRF) and Section Importance Multipliers.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .embeddings import EmbeddingService, get_embedding_service
from .reranker import CrossEncoderReranker
from .vector_store import AcademicVectorStore, SearchResult, get_vector_store

logger = logging.getLogger(__name__)

# Section importance multipliers per v3.2 architecture contracts
DEFAULT_SECTION_MULTIPLIERS: dict[str, float] = {
    "results": 1.4,
    "methodology": 1.3,
    "limitations": 1.2,
    "abstract": 1.1,
    "tables": 1.1,
    "introduction": 1.0,
    "discussion": 1.0,
    "conclusion": 1.0,
    "general": 1.0,
}


@dataclass
class HybridSearchResult:
    """Result from hybrid search with RRF and section weighting."""

    chunk_id: str
    content: str
    paper_id: str
    paper_title: str
    chunk_type: str
    vector_score: float
    bm25_score: float
    rrf_score: float
    final_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def section_type(self) -> str:
        """Alias for chunk_type."""
        return self.chunk_type

    @property
    def anchor(self) -> str:
        """Get anchor tag from metadata or fallback."""
        return self.metadata.get("anchor", self.metadata.get("anchor_tag", f"[ref_{self.paper_id}]"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
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
    In-memory BM25 index for sparse keyword retrieval.
    Implements Robertson-Spärck Jones BM25 with Lucene smoothed IDF.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Document length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b

        # Index storage
        self.documents: list[dict[str, Any]] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.chunk_id_to_idx: dict[str, int] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric words."""
        if not text:
            return []
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return [t.strip() for t in text.split() if t.strip()]

    def add_documents(self, documents: list[dict[str, Any]]):
        """
        Add documents to the BM25 index.

        Args:
            documents: List of dicts with 'content', 'chunk_id', and optional metadata.
        """
        if not documents:
            return

        start_idx = len(self.documents)
        for i, doc in enumerate(documents):
            doc_idx = start_idx + i
            chunk_id = str(doc.get("chunk_id", f"doc_{doc_idx}"))
            content = doc.get("content", "")
            tokens = self._tokenize(content)

            self.documents.append(doc)
            self.doc_lengths.append(len(tokens))
            self.chunk_id_to_idx[chunk_id] = doc_idx

            term_freqs: dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            for term, freq in term_freqs.items():
                self.inverted_index[term].append((doc_idx, freq))
                self.doc_freqs[term] += 1

        self.total_docs = len(self.documents)
        self.avg_doc_length = sum(self.doc_lengths) / max(self.total_docs, 1)
        logger.debug(f"BM25 index updated: {self.total_docs} total docs (avg len: {self.avg_doc_length:.1f})")

    def _calculate_idf(self, term: str) -> float:
        """Calculate Lucene-smoothed inverse document frequency."""
        n = self.total_docs
        df = self.doc_freqs.get(term, 0)
        if df == 0 or n == 0:
            return 0.0
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Search the BM25 index.

        Args:
            query: Search query text
            top_k: Number of top results to return

        Returns:
            List of (chunk_id, bm25_score) tuples sorted descending by score.
        """
        if self.total_docs == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)
        avg_dl = max(self.avg_doc_length, 1.0)

        for token in query_tokens:
            idf = self._calculate_idf(token)
            if idf <= 0.0:
                continue

            for doc_idx, tf in self.inverted_index.get(token, []):
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_dl))
                scores[doc_idx] += idf * (numerator / denominator)

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_idx, score in sorted_results[:top_k]:
            doc = self.documents[doc_idx]
            chunk_id = str(doc.get("chunk_id", str(doc_idx)))
            results.append((chunk_id, score))
        return results

    def get_document(self, chunk_id_or_idx: str | int) -> dict[str, Any] | None:
        """Retrieve document dictionary by chunk_id or index."""
        if isinstance(chunk_id_or_idx, int):
            if 0 <= chunk_id_or_idx < len(self.documents):
                return self.documents[chunk_id_or_idx]
            return None
        if isinstance(chunk_id_or_idx, str):
            if chunk_id_or_idx in self.chunk_id_to_idx:
                idx = self.chunk_id_to_idx[chunk_id_or_idx]
                return self.documents[idx]
            try:
                idx = int(chunk_id_or_idx)
                if 0 <= idx < len(self.documents):
                    return self.documents[idx]
            except ValueError:
                pass
        return None

    def clear(self):
        """Clear the BM25 index completely."""
        self.documents.clear()
        self.doc_freqs.clear()
        self.doc_lengths.clear()
        self.inverted_index.clear()
        self.chunk_id_to_idx.clear()
        self.avg_doc_length = 0.0
        self.total_docs = 0


class HybridSearchEngine:
    """
    Hybrid Search Engine combining dense vector search and sparse BM25
    with Reciprocal Rank Fusion (RRF) and Section Importance Multipliers.
    """

    def __init__(
        self,
        vector_store: AcademicVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        reranker: CrossEncoderReranker | None = None,
        rrf_k: int = 60,
        use_hyde: bool = False,
        section_multipliers: dict[str, float] | None = None,
    ):
        """
        Initialize the Hybrid Search Engine.

        Args:
            vector_store: Vector store instance (defaults to global AcademicVectorStore)
            embedding_service: Embedding service instance
            reranker: Optional cross-encoder reranker
            rrf_k: RRF smoothing constant (default: 60)
            use_hyde: Whether to enable HyDE query expansion (default: False)
            section_multipliers: Custom overrides for section importance weights
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.use_hyde = use_hyde
        self.section_multipliers = {
            **DEFAULT_SECTION_MULTIPLIERS,
            **(section_multipliers or {}),
        }
        self._bm25_indexes: dict[str, BM25Index] = {}

        logger.info(f"HybridSearchEngine initialized (rrf_k={rrf_k}, use_hyde={use_hyde})")

    def _ensure_vector_store(self) -> AcademicVectorStore | None:
        """Lazy load vector store if not provided."""
        if self.vector_store is None:
            try:
                self.vector_store = get_vector_store()
            except Exception as e:
                logger.warning(f"Vector store unavailable: {e}")
        return self.vector_store

    def _get_or_create_bm25_index(self, project_id: str) -> BM25Index:
        """Get or create project-isolated BM25 index."""
        if project_id not in self._bm25_indexes:
            self._bm25_indexes[project_id] = BM25Index()
        return self._bm25_indexes[project_id]

    def build_bm25_index(self, project_id: str, documents: list[dict[str, Any]]):
        """Build and replace BM25 index for a project."""
        index = self._get_or_create_bm25_index(project_id)
        index.clear()
        index.add_documents(documents)
        logger.info(f"Built BM25 index for project '{project_id}' with {len(documents)} documents")

    def index_project_documents(self, project_id: str, documents: list[dict[str, Any]]):
        """Index documents into BM25 for a project."""
        self.build_bm25_index(project_id, documents)

    def clear_project_index(self, project_id: str):
        """Clear and remove BM25 index for a project."""
        if project_id in self._bm25_indexes:
            self._bm25_indexes[project_id].clear()
            del self._bm25_indexes[project_id]
            logger.info(f"Cleared BM25 index for project '{project_id}'")

    def get_section_multiplier(
        self,
        chunk_type: str,
        section_weights: dict[str, float] | None = None,
    ) -> float:
        """
        Get the section importance multiplier for a given chunk type.

        Default multipliers:
        - results: 1.4x
        - methodology: 1.3x
        - limitations: 1.2x
        - abstract: 1.1x
        - tables: 1.1x
        - general: 1.0x
        """
        norm_type = str(chunk_type).lower().strip() if chunk_type else "general"
        if section_weights:
            for k, v in section_weights.items():
                if str(k).lower().strip() == norm_type:
                    return float(v)
        return self.section_multipliers.get(norm_type, 1.0)

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[tuple[str, float]],
        bm25_index: BM25Index | None,
        section_weights: dict[str, float] | None = None,
    ) -> list[HybridSearchResult]:
        """
        Combine dense vector and sparse BM25 results using weighted RRF:
        RRF(d) = sum_{m in {vector, bm25}} (1 / (k + rank_m(d))) * W(section_type)
        """
        vector_map: dict[str, tuple[int, SearchResult]] = {
            r.chunk_id: (rank, r) for rank, r in enumerate(vector_results, start=1)
        }
        bm25_map: dict[str, tuple[int, float]] = {
            chunk_id: (rank, score) for rank, (chunk_id, score) in enumerate(bm25_results, start=1)
        }

        all_chunk_ids = set(vector_map.keys()) | set(bm25_map.keys())
        if not all_chunk_ids:
            return []

        results: list[HybridSearchResult] = []

        for chunk_id in all_chunk_ids:
            raw_rrf = 0.0
            vec_score = 0.0
            bm_score = 0.0
            content = ""
            paper_id = ""
            paper_title = ""
            chunk_type = "general"
            metadata: dict[str, Any] = {}

            if chunk_id in vector_map:
                rank_v, vec_res = vector_map[chunk_id]
                raw_rrf += 1.0 / (self.rrf_k + rank_v)
                vec_score = vec_res.score
                content = vec_res.content
                paper_id = vec_res.paper_id
                paper_title = vec_res.paper_title
                chunk_type = vec_res.chunk_type
                metadata = dict(vec_res.metadata) if vec_res.metadata else {}

            if chunk_id in bm25_map:
                rank_b, bm_score = bm25_map[chunk_id]
                raw_rrf += 1.0 / (self.rrf_k + rank_b)
                if not content and bm25_index:
                    doc = bm25_index.get_document(chunk_id)
                    if doc:
                        content = doc.get("content", "")
                        paper_id = doc.get("paper_id", "")
                        paper_title = doc.get("paper_title", "")
                        chunk_type = doc.get("chunk_type", "general")
                        metadata = dict(doc.get("metadata", {}))

            multiplier = self.get_section_multiplier(chunk_type, section_weights)
            weighted_rrf = raw_rrf * multiplier

            results.append(
                HybridSearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    paper_id=paper_id,
                    paper_title=paper_title,
                    chunk_type=chunk_type,
                    vector_score=vec_score,
                    bm25_score=bm_score,
                    rrf_score=weighted_rrf,
                    final_score=weighted_rrf,
                    metadata=metadata,
                )
            )

        # Sort descending by final_score, then tie-break by vector_score and bm25_score
        results.sort(key=lambda x: (x.final_score, x.vector_score, x.bm25_score), reverse=True)
        return results

    def search(
        self,
        query: str,
        project_id: str,
        top_k: int = 10,
        section_weights: dict[str, float] | None = None,
        use_reranker: bool = False,
        chunk_types: list[str] | None = None,
        **kwargs: Any,
    ) -> list[HybridSearchResult]:
        """
        Perform hybrid search combining vector similarity and BM25 keywords with RRF.

        Args:
            query: Search query string
            project_id: Project ID for isolation
            top_k: Number of final results to return
            section_weights: Custom section importance multipliers
            use_reranker: Whether to apply cross-encoder reranking
            chunk_types: Optional list of chunk types to filter by

        Returns:
            List of HybridSearchResult objects sorted descending by score.
        """
        if not query or not query.strip() or top_k <= 0:
            return []

        candidate_k = max(top_k * 2, 20)
        vs = self._ensure_vector_store()

        # 1. Vector Search
        vector_results: list[SearchResult] = []
        if vs is not None:
            try:
                vector_results = vs.search(
                    query=query,
                    project_id=project_id,
                    top_k=candidate_k,
                    chunk_types=chunk_types,
                )
            except Exception as e:
                logger.warning(f"Vector search failed for project '{project_id}': {e}")
                vector_results = []

        # 2. BM25 Search
        bm25_index = self._bm25_indexes.get(project_id)
        bm25_results: list[tuple[str, float]] = []
        if bm25_index and bm25_index.total_docs > 0:
            try:
                raw_bm25 = bm25_index.search(query, top_k=candidate_k)
                if chunk_types:
                    norm_types = {ct.lower().strip() for ct in chunk_types}
                    filtered = []
                    for cid, score in raw_bm25:
                        doc = bm25_index.get_document(cid)
                        if doc and doc.get("chunk_type", "general").lower().strip() in norm_types:
                            filtered.append((cid, score))
                    bm25_results = filtered
                else:
                    bm25_results = raw_bm25
            except Exception as e:
                logger.warning(f"BM25 search failed for project '{project_id}': {e}")
                bm25_results = []

        # 3. Fuse via Section-Weighted RRF
        hybrid_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            bm25_index=bm25_index,
            section_weights=section_weights,
        )

        # 4. Optional Cross-Encoder Reranking
        if use_reranker and self.reranker and hybrid_results:
            try:
                results_for_rerank = [
                    {
                        "content": r.content,
                        "score": r.rrf_score,
                        "weight": self.get_section_multiplier(r.chunk_type, section_weights),
                        **r.to_dict(),
                    }
                    for r in hybrid_results[:candidate_k]
                ]
                reranked = self.reranker.rerank(query, results_for_rerank, top_k=top_k)
                rerank_map = {r.metadata["chunk_id"]: r.combined_score for r in reranked if "chunk_id" in r.metadata}

                for h_res in hybrid_results:
                    if h_res.chunk_id in rerank_map:
                        h_res.final_score = rerank_map[h_res.chunk_id]

                hybrid_results.sort(key=lambda x: x.final_score, reverse=True)
            except Exception as e:
                logger.warning(f"Reranking failed, preserving RRF scores: {e}")

        logger.info(
            f"Hybrid search for '{query[:40]}...' in project '{project_id}' returned "
            f"{len(hybrid_results[:top_k])} results (vector: {len(vector_results)}, bm25: {len(bm25_results)})"
        )
        return hybrid_results[:top_k]


# Singleton instance
_search_engine: HybridSearchEngine | None = None


def get_search_engine() -> HybridSearchEngine:
    """Get or create the global hybrid search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = HybridSearchEngine()
    return _search_engine
