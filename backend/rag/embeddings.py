"""
Embedding Service for RAG Pipeline.

Uses Google's text-embedding model (Gemini) for generating embeddings.
Includes caching to avoid redundant API calls.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""

    text: str
    embedding: list[float]
    model: str
    token_count: int


class EmbeddingService:
    """
    Service for generating text embeddings using Google's embedding API.

    Features:
    - Batch embedding for efficiency
    - In-memory caching with LRU
    - Automatic retry with exponential backoff
    - Token counting for cost tracking
    """

    # Google's text-embedding-004 model
    MODEL_NAME = "text-embedding-004"
    EMBEDDING_DIM = 768  # Dimension for text-embedding-004

    # Rate limiting
    MAX_BATCH_SIZE = 100
    REQUESTS_PER_MINUTE = 1500

    def __init__(self, api_key: str | None = None, cache_size: int = 10000):
        """
        Initialize the embedding service.

        Args:
            api_key: Google API key (defaults to GEMINI_API_KEY env var)
            cache_size: Maximum number of embeddings to cache
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL_NAME}:embedContent"
        self.batch_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL_NAME}:batchEmbedContents"

        # Simple in-memory cache
        self._cache: dict = {}
        self._cache_size = cache_size

        # Stats tracking
        self.total_tokens = 0
        self.total_requests = 0
        self.cache_hits = 0

        logger.info(f"EmbeddingService initialized with model {self.MODEL_NAME}")

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key from text."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (approximately 4 chars per token)."""
        return len(text) // 4

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            self.cache_hits += 1
            return self._cache[cache_key]

        # Make API request
        embedding = self._call_embedding_api(text)

        # Cache result
        if len(self._cache) < self._cache_size:
            self._cache[cache_key] = embedding

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Separate cached and uncached texts
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
                self.cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached texts in batches
        if uncached_texts:
            embeddings = self._batch_embed_api(uncached_texts)

            for idx, embedding in zip(uncached_indices, embeddings):
                results[idx] = embedding
                # Cache the result
                cache_key = self._get_cache_key(texts[idx])
                if len(self._cache) < self._cache_size:
                    self._cache[cache_key] = embedding

        return results

    def _call_embedding_api(self, text: str, retries: int = 3) -> list[float]:
        """Make a single embedding API call with retry logic."""
        headers = {"Content-Type": "application/json"}
        payload = {"model": f"models/{self.MODEL_NAME}", "content": {"parts": [{"text": text}]}}

        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}?key={self.api_key}", headers=headers, json=payload, timeout=30
                )

                if response.status_code == 429:  # Rate limited
                    wait_time = 2**attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                result = response.json()

                self.total_requests += 1
                self.total_tokens += self._estimate_tokens(text)

                # Handle response format: {"embedding": {"values": [...]}} or {"values": [...]}
                if "embedding" in result:
                    return result["embedding"]["values"]
                elif "values" in result:
                    return result["values"]
                else:
                    raise ValueError(f"Unexpected embedding response format: {result}")

            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    logger.error(f"Embedding API call failed after {retries} attempts: {e}")
                    raise
                time.sleep(2**attempt)

        raise RuntimeError("Failed to get embedding")

    def _batch_embed_api(self, texts: list[str], retries: int = 3) -> list[list[float]]:
        """Make batch embedding API calls."""
        all_embeddings = []

        # Process in chunks of MAX_BATCH_SIZE
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i : i + self.MAX_BATCH_SIZE]
            batch_embeddings = self._call_batch_api(batch, retries)
            all_embeddings.extend(batch_embeddings)

            # Small delay between batches to avoid rate limiting
            if i + self.MAX_BATCH_SIZE < len(texts):
                time.sleep(0.1)

        return all_embeddings

    def _call_batch_api(self, texts: list[str], retries: int = 3) -> list[list[float]]:
        """Make a batch embedding API call."""
        headers = {"Content-Type": "application/json"}

        requests_list = [
            {"model": f"models/{self.MODEL_NAME}", "content": {"parts": [{"text": text}]}}
            for text in texts
        ]

        payload = {"requests": requests_list}

        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.batch_url}?key={self.api_key}",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 429:  # Rate limited
                    wait_time = 2**attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                result = response.json()

                self.total_requests += 1
                self.total_tokens += sum(self._estimate_tokens(t) for t in texts)

                # Handle batch response format
                embeddings = []
                for item in result.get("embeddings", []):
                    if "embedding" in item:
                        embeddings.append(item["embedding"]["values"])
                    elif "values" in item:
                        embeddings.append(item["values"])
                    else:
                        logger.warning(f"Unexpected embedding item format: {item}")
                        embeddings.append([0.0] * self.EMBEDDING_DIM)  # Fallback

                return embeddings

            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    logger.error(f"Batch embedding API call failed: {e}")
                    raise
                time.sleep(2**attempt)

        raise RuntimeError("Failed to get batch embeddings")

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "cache_size": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.total_requests),
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")


# Singleton instance for reuse
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
