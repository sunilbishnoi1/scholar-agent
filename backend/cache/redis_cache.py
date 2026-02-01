# Redis-based Intelligent Caching for AI Operations
# Multi-tier caching strategy for LLM responses, embeddings, and search results

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import wraps
from typing import Any, TypeVar

import redis

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "hit_rate": self.hit_rate,
            "total_requests": self.hits + self.misses,
        }


class CacheTier:
    """Cache tier configuration with TTL defaults."""

    # LLM Response Cache - short TTL for identical prompts
    LLM_RESPONSE = "llm"
    LLM_RESPONSE_TTL = 1800  # 30 minutes

    # Paper Metadata Cache - medium TTL for API responses
    PAPER_METADATA = "paper"
    PAPER_METADATA_TTL = 86400  # 24 hours

    # Embedding Cache - long TTL for computed embeddings
    EMBEDDING = "embed"
    EMBEDDING_TTL = 604800  # 7 days

    # Search Result Cache - short TTL for search queries
    SEARCH = "search"
    SEARCH_TTL = 3600  # 1 hour

    # User Session Cache - for model router state
    SESSION = "session"
    SESSION_TTL = 86400  # 24 hours


class IntelligentCache:
    """
    Multi-tier caching for AI operations.

    Tiers:
    1. LLM Response Cache - Cache identical prompts (short TTL)
    2. Paper Metadata Cache - Cache API responses (medium TTL)
    3. Embedding Cache - Cache computed embeddings (long TTL)
    4. Search Result Cache - Cache search queries (short TTL)
    5. Session Cache - Cache user session data like router state

    Features:
    - Automatic serialization/deserialization
    - Cache key hashing for consistent lookups
    - Statistics tracking for monitoring
    - Graceful degradation when Redis is unavailable
    """

    def __init__(self, redis_url: str | None = None):
        """
        Initialize the cache with Redis connection.

        Args:
            redis_url: Redis connection URL (defaults to env var)
        """
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._redis: redis.Redis | None = None
        self._connected = False

        # Per-tier statistics
        self._stats: dict[str, CacheStats] = {
            CacheTier.LLM_RESPONSE: CacheStats(),
            CacheTier.PAPER_METADATA: CacheStats(),
            CacheTier.EMBEDDING: CacheStats(),
            CacheTier.SEARCH: CacheStats(),
            CacheTier.SESSION: CacheStats(),
        }

        self._connect()

    def _connect(self) -> bool:
        """Establish Redis connection with error handling."""
        try:
            self._redis = redis.from_url(
                self.redis_url,
                decode_responses=False,  # Handle bytes for embeddings
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            # Test connection
            self._redis.ping()
            self._connected = True
            logger.info(f"Redis cache connected: {self.redis_url}")
            return True
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}. Cache will be disabled.")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected Redis error: {e}")
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        """Check if Redis is available."""
        if not self._connected or not self._redis:
            return False
        try:
            self._redis.ping()
            return True
        except:
            self._connected = False
            return False

    def _make_key(self, tier: str, content: str, namespace: str | None = None) -> str:
        """
        Create a cache key from content hash.

        Args:
            tier: Cache tier (llm, embed, search, etc.)
            content: Content to hash
            namespace: Optional namespace (e.g., project_id, user_id)

        Returns:
            Cache key string
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if namespace:
            return f"scholar:{tier}:{namespace}:{content_hash}"
        return f"scholar:{tier}:{content_hash}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            return json.dumps(value).encode("utf-8")
        except (TypeError, ValueError):
            # For non-JSON-serializable objects (like numpy arrays)
            import pickle

            return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            import pickle

            return pickle.loads(data)

    def get(self, key: str, tier: str = CacheTier.LLM_RESPONSE) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key
            tier: Cache tier for statistics

        Returns:
            Cached value or None
        """
        if not self.is_connected:
            self._stats[tier].misses += 1
            return None

        try:
            data = self._redis.get(key)
            if data:
                self._stats[tier].hits += 1
                return self._deserialize(data)
            self._stats[tier].misses += 1
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self._stats[tier].errors += 1
            self._stats[tier].misses += 1
            return None

    def set(self, key: str, value: Any, ttl: int, tier: str = CacheTier.LLM_RESPONSE) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            tier: Cache tier for statistics

        Returns:
            True if successful
        """
        if not self.is_connected:
            return False

        try:
            serialized = self._serialize(value)
            self._redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self._stats[tier].errors += 1
            return False

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not self.is_connected:
            return False
        try:
            self._redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    def get_or_compute(
        self,
        tier: str,
        content: str,
        compute_fn: Callable[[], T],
        ttl: int | None = None,
        namespace: str | None = None,
    ) -> T:
        """
        Get from cache or compute and cache result.

        Args:
            tier: Cache tier
            content: Content to generate cache key from
            compute_fn: Function to compute value if not cached
            ttl: Time-to-live (defaults to tier default)
            namespace: Optional namespace for key

        Returns:
            Cached or computed value
        """
        key = self._make_key(tier, content, namespace)

        # Try cache first
        cached = self.get(key, tier)
        if cached is not None:
            logger.debug(f"Cache hit for {tier}: {key[:50]}...")
            return cached

        # Compute value
        logger.debug(f"Cache miss for {tier}: {key[:50]}...")
        result = compute_fn()

        # Determine TTL
        if ttl is None:
            ttl_map = {
                CacheTier.LLM_RESPONSE: CacheTier.LLM_RESPONSE_TTL,
                CacheTier.PAPER_METADATA: CacheTier.PAPER_METADATA_TTL,
                CacheTier.EMBEDDING: CacheTier.EMBEDDING_TTL,
                CacheTier.SEARCH: CacheTier.SEARCH_TTL,
                CacheTier.SESSION: CacheTier.SESSION_TTL,
            }
            ttl = ttl_map.get(tier, 3600)

        # Cache result
        self.set(key, result, ttl, tier)

        return result

    async def get_or_compute_async(
        self,
        tier: str,
        content: str,
        compute_fn: Callable[[], Any],
        ttl: int | None = None,
        namespace: str | None = None,
    ) -> Any:
        """Async version of get_or_compute."""
        key = self._make_key(tier, content, namespace)

        # Try cache first
        cached = self.get(key, tier)
        if cached is not None:
            return cached

        # Compute value (handle both sync and async functions)
        if asyncio.iscoroutinefunction(compute_fn):
            result = await compute_fn()
        else:
            result = compute_fn()

        # Determine TTL
        if ttl is None:
            ttl_map = {
                CacheTier.LLM_RESPONSE: CacheTier.LLM_RESPONSE_TTL,
                CacheTier.PAPER_METADATA: CacheTier.PAPER_METADATA_TTL,
                CacheTier.EMBEDDING: CacheTier.EMBEDDING_TTL,
                CacheTier.SEARCH: CacheTier.SEARCH_TTL,
                CacheTier.SESSION: CacheTier.SESSION_TTL,
            }
            ttl = ttl_map.get(tier, 3600)

        # Cache result
        self.set(key, result, ttl, tier)

        return result

    # =========================================================================
    # Decorator-based caching for common patterns
    # =========================================================================

    def cache_llm_response(self, ttl: int = CacheTier.LLM_RESPONSE_TTL):
        """
        Decorator to cache LLM responses.

        Usage:
            @cache.cache_llm_response()
            def generate_text(prompt: str) -> str:
                return llm.chat(prompt)
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(prompt: str, *args, **kwargs) -> Any:
                key = self._make_key(CacheTier.LLM_RESPONSE, prompt)

                cached = self.get(key, CacheTier.LLM_RESPONSE)
                if cached is not None:
                    logger.debug(f"LLM cache hit for prompt: {prompt[:50]}...")
                    return cached

                result = func(prompt, *args, **kwargs)
                self.set(key, result, ttl, CacheTier.LLM_RESPONSE)
                return result

            @wraps(func)
            async def async_wrapper(prompt: str, *args, **kwargs) -> Any:
                key = self._make_key(CacheTier.LLM_RESPONSE, prompt)

                cached = self.get(key, CacheTier.LLM_RESPONSE)
                if cached is not None:
                    return cached

                result = await func(prompt, *args, **kwargs)
                self.set(key, result, ttl, CacheTier.LLM_RESPONSE)
                return result

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return wrapper

        return decorator

    def cache_embedding(self, ttl: int = CacheTier.EMBEDDING_TTL):
        """Decorator to cache embeddings (long TTL)."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(text: str, *args, **kwargs) -> Any:
                key = self._make_key(CacheTier.EMBEDDING, text)

                cached = self.get(key, CacheTier.EMBEDDING)
                if cached is not None:
                    return cached

                result = func(text, *args, **kwargs)
                self.set(key, result, ttl, CacheTier.EMBEDDING)
                return result

            @wraps(func)
            async def async_wrapper(text: str, *args, **kwargs) -> Any:
                key = self._make_key(CacheTier.EMBEDDING, text)

                cached = self.get(key, CacheTier.EMBEDDING)
                if cached is not None:
                    return cached

                result = await func(text, *args, **kwargs)
                self.set(key, result, ttl, CacheTier.EMBEDDING)
                return result

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return wrapper

        return decorator

    def cache_search(self, ttl: int = CacheTier.SEARCH_TTL, namespace: str | None = None):
        """Decorator to cache search results."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(query: str, *args, **kwargs) -> Any:
                # Include relevant kwargs in cache key
                cache_content = f"{query}:{args!s}:{kwargs!s}"
                key = self._make_key(CacheTier.SEARCH, cache_content, namespace)

                cached = self.get(key, CacheTier.SEARCH)
                if cached is not None:
                    return cached

                result = func(query, *args, **kwargs)
                self.set(key, result, ttl, CacheTier.SEARCH)
                return result

            @wraps(func)
            async def async_wrapper(query: str, *args, **kwargs) -> Any:
                cache_content = f"{query}:{args!s}:{kwargs!s}"
                key = self._make_key(CacheTier.SEARCH, cache_content, namespace)

                cached = self.get(key, CacheTier.SEARCH)
                if cached is not None:
                    return cached

                result = await func(query, *args, **kwargs)
                self.set(key, result, ttl, CacheTier.SEARCH)
                return result

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return wrapper

        return decorator

    # =========================================================================
    # User session management (for model router state)
    # =========================================================================

    def save_user_session(self, user_id: str, data: dict, ttl: int = CacheTier.SESSION_TTL) -> bool:
        """Save user session data (e.g., model router state)."""
        key = f"scholar:session:{user_id}"
        data["_updated_at"] = datetime.utcnow().isoformat()
        return self.set(key, data, ttl, CacheTier.SESSION)

    def get_user_session(self, user_id: str) -> dict | None:
        """Get user session data."""
        key = f"scholar:session:{user_id}"
        return self.get(key, CacheTier.SESSION)

    def update_user_spending(self, user_id: str, amount: float) -> float | None:
        """
        Atomically update user spending and return new total.
        Uses Redis INCRBYFLOAT for atomicity.
        """
        if not self.is_connected:
            return None

        key = f"scholar:spending:{user_id}"
        try:
            # INCRBYFLOAT returns the new value
            new_total = self._redis.incrbyfloat(key, amount)
            # Set expiry if this is a new key (monthly reset)
            self._redis.expire(key, 30 * 24 * 3600)  # 30 days
            return float(new_total)
        except Exception as e:
            logger.error(f"Failed to update spending for user {user_id}: {e}")
            return None

    def get_user_spending(self, user_id: str) -> float:
        """Get current user spending."""
        if not self.is_connected:
            return 0.0

        key = f"scholar:spending:{user_id}"
        try:
            value = self._redis.get(key)
            return float(value) if value else 0.0
        except:
            return 0.0

    # =========================================================================
    # Pub/Sub for real-time updates
    # =========================================================================

    def publish(self, channel: str, message: dict) -> bool:
        """Publish a message to a Redis channel."""
        if not self.is_connected:
            return False

        try:
            self._redis.publish(channel, json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")
            return False

    def get_pubsub(self):
        """Get a pubsub object for subscribing to channels."""
        if not self.is_connected:
            return None
        return self._redis.pubsub()

    # =========================================================================
    # Statistics and monitoring
    # =========================================================================

    def get_stats(self) -> dict:
        """Get cache statistics for all tiers."""
        return {tier: stats.to_dict() for tier, stats in self._stats.items()}

    def get_tier_stats(self, tier: str) -> dict:
        """Get statistics for a specific tier."""
        if tier in self._stats:
            return self._stats[tier].to_dict()
        return {}

    def clear_tier(self, tier: str) -> int:
        """Clear all keys for a specific tier. Returns count of deleted keys."""
        if not self.is_connected:
            return 0

        try:
            pattern = f"scholar:{tier}:*"
            keys = self._redis.keys(pattern)
            if keys:
                return self._redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Failed to clear tier {tier}: {e}")
            return 0

    def health_check(self) -> dict:
        """Return cache health status."""
        return {
            "connected": self.is_connected,
            "redis_url": self.redis_url.split("@")[-1] if "@" in self.redis_url else self.redis_url,
            "stats": self.get_stats(),
        }


# =========================================================================
# Singleton instance
# =========================================================================

_cache_instance: IntelligentCache | None = None


def get_cache() -> IntelligentCache:
    """Get or create the singleton cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = IntelligentCache()
    return _cache_instance


def reset_cache():
    """Reset the cache instance (useful for testing)."""
    global _cache_instance
    _cache_instance = None
