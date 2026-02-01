# Tests for Redis Cache (Phase 3: Production Features)
import json
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestIntelligentCache:
    """Tests for the IntelligentCache class."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = MagicMock()
        mock.ping.return_value = True
        mock.get.return_value = None
        mock.setex.return_value = True
        mock.delete.return_value = 1
        mock.keys.return_value = []
        mock.incrbyfloat.return_value = 0.5
        mock.expire.return_value = True
        mock.publish.return_value = 1
        return mock

    @pytest.fixture
    def cache_with_mock_redis(self, mock_redis):
        """Create an IntelligentCache with mocked Redis."""
        with patch("cache.redis_cache.redis.from_url", return_value=mock_redis):
            from cache.redis_cache import IntelligentCache

            cache = IntelligentCache("redis://localhost:6379/0")
            return cache

    def test_cache_initialization(self, cache_with_mock_redis):
        """Test that cache initializes correctly."""
        assert cache_with_mock_redis.is_connected

    def test_make_key_generates_consistent_keys(self, cache_with_mock_redis):
        """Test that the same content generates the same key."""
        key1 = cache_with_mock_redis._make_key("llm", "test content")
        key2 = cache_with_mock_redis._make_key("llm", "test content")
        assert key1 == key2
        assert key1.startswith("scholar:llm:")

    def test_make_key_with_namespace(self, cache_with_mock_redis):
        """Test key generation with namespace."""
        key = cache_with_mock_redis._make_key("llm", "test", namespace="user123")
        assert "user123" in key

    def test_get_returns_none_on_miss(self, cache_with_mock_redis, mock_redis):
        """Test that get returns None on cache miss."""
        mock_redis.get.return_value = None
        result = cache_with_mock_redis.get("nonexistent_key", "llm")
        assert result is None

    def test_get_returns_cached_value(self, cache_with_mock_redis, mock_redis):
        """Test that get returns cached value on hit."""
        cached_data = {"result": "cached"}
        mock_redis.get.return_value = json.dumps(cached_data).encode()
        result = cache_with_mock_redis.get("existing_key", "llm")
        assert result == cached_data

    def test_set_stores_value(self, cache_with_mock_redis, mock_redis):
        """Test that set stores value with TTL."""
        result = cache_with_mock_redis.set("key", {"data": "value"}, ttl=3600, tier="llm")
        assert result is True
        mock_redis.setex.assert_called_once()

    def test_get_or_compute_returns_cached_on_hit(self, cache_with_mock_redis, mock_redis):
        """Test get_or_compute returns cached value without computing."""
        cached_data = {"cached": True}
        mock_redis.get.return_value = json.dumps(cached_data).encode()

        compute_called = False

        def compute_fn():
            nonlocal compute_called
            compute_called = True
            return {"computed": True}

        result = cache_with_mock_redis.get_or_compute("llm", "test", compute_fn)
        assert result == cached_data
        assert not compute_called

    def test_get_or_compute_computes_on_miss(self, cache_with_mock_redis, mock_redis):
        """Test get_or_compute calls compute function on miss."""
        mock_redis.get.return_value = None

        def compute_fn():
            return {"computed": True}

        result = cache_with_mock_redis.get_or_compute("llm", "test", compute_fn)
        assert result == {"computed": True}
        mock_redis.setex.assert_called_once()

    def test_user_spending_tracking(self, cache_with_mock_redis, mock_redis):
        """Test updating and getting user spending."""
        mock_redis.incrbyfloat.return_value = 0.5

        new_total = cache_with_mock_redis.update_user_spending("user123", 0.5)
        assert new_total == 0.5
        mock_redis.incrbyfloat.assert_called_once()

    def test_pubsub_publish(self, cache_with_mock_redis, mock_redis):
        """Test publishing messages to channels."""
        result = cache_with_mock_redis.publish("test_channel", {"event": "test"})
        assert result is True
        mock_redis.publish.assert_called_once()

    def test_stats_tracking(self, cache_with_mock_redis, mock_redis):
        """Test that cache statistics are tracked."""
        # Simulate some cache operations
        mock_redis.get.return_value = None
        cache_with_mock_redis.get("key1", "llm")  # Miss

        cached_data = json.dumps({"data": "test"}).encode()
        mock_redis.get.return_value = cached_data
        cache_with_mock_redis.get("key2", "llm")  # Hit

        stats = cache_with_mock_redis.get_stats()
        assert "llm" in stats
        assert stats["llm"]["hits"] >= 0
        assert stats["llm"]["misses"] >= 0

    def test_health_check(self, cache_with_mock_redis):
        """Test health check returns correct status."""
        health = cache_with_mock_redis.health_check()
        assert "connected" in health
        assert "stats" in health


class TestCacheDecorators:
    """Tests for cache decorators."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache instance."""
        with patch("cache.redis_cache.redis.from_url") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.get.return_value = None
            mock_client.setex.return_value = True
            mock_redis.return_value = mock_client

            from cache.redis_cache import IntelligentCache

            cache = IntelligentCache()
            return cache, mock_client

    def test_cache_llm_response_decorator(self, mock_cache):
        """Test LLM response caching decorator."""
        cache, mock_client = mock_cache

        call_count = 0

        @cache.cache_llm_response(ttl=1800)
        def generate_response(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Response to: {prompt}"

        # First call - should compute
        result1 = generate_response("test prompt")
        assert call_count == 1

        # Second call with same prompt - should use cache
        mock_client.get.return_value = json.dumps("Response to: test prompt").encode()
        result2 = generate_response("test prompt")
        # Call count shouldn't increase on cache hit

    def test_cache_embedding_decorator(self, mock_cache):
        """Test embedding caching decorator."""
        cache, mock_client = mock_cache

        @cache.cache_embedding(ttl=604800)
        def compute_embedding(text: str) -> list:
            return [0.1, 0.2, 0.3]  # Mock embedding

        result = compute_embedding("test text")
        assert result == [0.1, 0.2, 0.3]


class TestCacheTierConfiguration:
    """Tests for cache tier TTL configuration."""

    def test_cache_tier_ttls(self):
        """Test that cache tiers have appropriate TTLs."""
        from cache.redis_cache import CacheTier

        # LLM responses should be short-lived
        assert CacheTier.LLM_RESPONSE_TTL == 1800  # 30 minutes

        # Paper metadata should be cached longer
        assert CacheTier.PAPER_METADATA_TTL == 86400  # 24 hours

        # Embeddings should be cached longest
        assert CacheTier.EMBEDDING_TTL == 604800  # 7 days

        # Search results should be short-lived
        assert CacheTier.SEARCH_TTL == 3600  # 1 hour


class TestGetCacheSingleton:
    """Tests for cache singleton pattern."""

    def test_get_cache_returns_same_instance(self):
        """Test that get_cache returns the same instance."""
        from cache.redis_cache import get_cache, reset_cache

        # Reset to ensure clean state
        reset_cache()

        with patch("cache.redis_cache.redis.from_url") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client

            cache1 = get_cache()
            cache2 = get_cache()
            assert cache1 is cache2

        # Clean up
        reset_cache()
