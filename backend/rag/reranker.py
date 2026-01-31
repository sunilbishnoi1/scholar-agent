"""
Cross-Encoder Reranker for RAG Pipeline.

Uses a cross-encoder model to rerank search results for better precision.
Falls back to score-based sorting if model is unavailable.
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking."""
    content: str
    original_score: float
    rerank_score: float
    combined_score: float
    metadata: dict


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving search precision.
    
    Uses Google's Gemini for reranking when available,
    with fallback to heuristic scoring.
    """
    
    def __init__(
        self,
        use_llm_reranker: bool = True,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the reranker.
        
        Args:
            use_llm_reranker: Whether to use LLM for reranking
            api_key: API key for Gemini (defaults to GEMINI_API_KEY)
        """
        self.use_llm_reranker = use_llm_reranker
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        logger.info(f"CrossEncoderReranker initialized (llm_reranker={use_llm_reranker})")

    def _calculate_heuristic_score(
        self,
        query: str,
        content: str,
        original_score: float,
        weight: float = 1.0
    ) -> float:
        """
        Calculate a heuristic relevance score.
        
        Considers:
        - Original vector similarity score
        - Keyword overlap
        - Content weight (section importance)
        """
        query_terms = set(query.lower().split())
        content_lower = content.lower()
        
        # Keyword overlap score
        keyword_hits = sum(1 for term in query_terms if term in content_lower)
        keyword_score = keyword_hits / max(len(query_terms), 1)
        
        # Exact phrase bonus
        phrase_bonus = 0.1 if query.lower() in content_lower else 0.0
        
        # Combine scores
        combined = (
            original_score * 0.6 +
            keyword_score * 0.25 +
            weight * 0.1 +
            phrase_bonus * 0.05
        )
        
        return min(combined, 1.0)

    def _batch_rerank_with_llm(
        self,
        query: str,
        candidates: List[Tuple[str, float, dict]],
    ) -> List[float]:
        """
        Rerank using LLM (batch approach for efficiency).
        
        Returns list of relevance scores (0-1).
        """
        if not self.api_key:
            logger.warning("No API key for LLM reranking, using heuristic")
            return [self._calculate_heuristic_score(query, c[0], c[1]) for c in candidates]
        
        try:
            import requests
            
            # Build prompt for batch reranking
            candidates_text = "\n".join([
                f"[{i+1}] {content[:500]}..."  # Truncate for context limit
                for i, (content, _, _) in enumerate(candidates[:20])  # Limit to 20
            ])
            
            prompt = f"""Rate the relevance of each document snippet to the query.
Query: "{query}"

Documents:
{candidates_text}

For each document, output ONLY a relevance score from 0.0 to 1.0, one per line.
Output format (one score per line):
0.85
0.72
...
"""
            
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200}
                },
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code}")
            
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # Parse scores
            scores = []
            for line in text.strip().split("\n"):
                try:
                    score = float(line.strip())
                    scores.append(min(max(score, 0.0), 1.0))
                except ValueError:
                    scores.append(0.5)  # Default score
            
            # Pad if needed
            while len(scores) < len(candidates):
                scores.append(0.5)
            
            return scores[:len(candidates)]
            
        except Exception as e:
            logger.warning(f"LLM reranking failed, using heuristic: {e}")
            return [self._calculate_heuristic_score(query, c[0], c[1]) for c in candidates]

    def rerank(
        self,
        query: str,
        results: List[dict],
        top_k: int = 10,
        score_weight: float = 0.4,
        rerank_weight: float = 0.6,
    ) -> List[RerankResult]:
        """
        Rerank search results.
        
        Args:
            query: Search query
            results: List of search results (must have 'content', 'score', etc.)
            top_k: Number of results to return
            score_weight: Weight for original score
            rerank_weight: Weight for rerank score
            
        Returns:
            List of reranked results
        """
        if not results:
            return []
        
        # Prepare candidates
        candidates = [
            (r.get("content", ""), r.get("score", 0.5), r)
            for r in results
        ]
        
        # Get rerank scores
        if self.use_llm_reranker and len(candidates) > 3:
            rerank_scores = self._batch_rerank_with_llm(query, candidates)
        else:
            rerank_scores = [
                self._calculate_heuristic_score(
                    query, c[0], c[1], c[2].get("weight", 1.0)
                )
                for c in candidates
            ]
        
        # Combine scores and create results
        reranked = []
        for (content, original_score, metadata), rerank_score in zip(candidates, rerank_scores):
            combined = (
                original_score * score_weight +
                rerank_score * rerank_weight
            )
            
            reranked.append(RerankResult(
                content=content,
                original_score=original_score,
                rerank_score=rerank_score,
                combined_score=combined,
                metadata=metadata,
            ))
        
        # Sort by combined score
        reranked.sort(key=lambda x: x.combined_score, reverse=True)
        
        logger.debug(f"Reranked {len(results)} results, returning top {top_k}")
        
        return reranked[:top_k]


def create_reranker(use_llm: bool = True) -> CrossEncoderReranker:
    """Factory function to create a reranker."""
    return CrossEncoderReranker(use_llm_reranker=use_llm)
