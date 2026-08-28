"""
Thread-safe and Distributed Sliding Window Rate Limiter for LLM API calls.
Enforces per-provider, per-key, and per-model Requests-Per-Minute (RPM) limits.
Supports multi-process Celery environments via Redis and in-memory smoothing fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ProviderRateLimiter:
    """
    Sliding window rate limiter with burst smoothing.
    
    Features:
    1. Distributed coordination via Redis (when available) for Celery multi-worker syncing.
    2. In-memory sliding window + burst-smoothing interval as a rock-solid fallback.
    3. Thread-safe and asyncio-safe (non-blocking sleep outside lock).
    """

    def __init__(self, key: str = "default", max_rpm: int = 14, min_interval: float | None = None) -> None:
        self.key = key
        self.max_rpm = max_rpm
        # Minimum spacing between requests to smooth out bursts (e.g. 60/14 ≈ 4.28s, with slight headroom)
        self.min_interval = min_interval if min_interval is not None else (60.0 / max_rpm if max_rpm > 0 else 0.0)
        self._timestamps: list[float] = []
        self._last_request_time: float = 0.0
        self._lock = threading.Lock()

    def _get_redis_client(self):
        """Safely retrieve connected Redis instance if available."""
        try:
            from cache.redis_cache import get_cache
            cache = get_cache()
            if cache and cache.is_connected and cache._redis:
                return cache._redis
        except Exception:
            pass
        return None

    def _acquire_redis(self, client) -> float:
        """Execute distributed rate limiting via Redis Sorted Set."""
        redis_key = f"scholar:ratelimit:{self.key}"
        window = 60.0

        while True:
            now = time.time()
            pipe = client.pipeline()
            # 1. Remove timestamps older than 60 seconds
            pipe.zremrangebyscore(redis_key, "-inf", now - window)
            # 2. Count requests in current window
            pipe.zcard(redis_key)
            # 3. Get oldest timestamp
            pipe.zrange(redis_key, 0, 0, withscores=True)
            # 4. Set expiry on key
            pipe.expire(redis_key, int(window) + 10)
            
            try:
                results = pipe.execute()
                current_count = results[1]
                oldest_entries = results[2]

                if current_count < self.max_rpm:
                    # Slot available - add current timestamp
                    client.zadd(redis_key, {str(now): now})
                    return 0.0

                # Must wait until oldest timestamp exits the window
                if oldest_entries:
                    oldest_time = float(oldest_entries[0][1])
                    sleep_needed = max(0.1, window - (now - oldest_time) + 0.2)
                else:
                    sleep_needed = 2.0

                logger.info(
                    f"Redis Rate Limit reached for '{self.key}' ({current_count}/{self.max_rpm} RPM). "
                    f"Throttling for {sleep_needed:.2f}s..."
                )
                time.sleep(sleep_needed)
            except Exception as e:
                logger.warning(f"Redis rate limiter error ({e}), falling back to in-memory limiter.")
                return self._acquire_memory()

    def _acquire_memory(self) -> float:
        """In-memory sliding window rate limiting with burst smoothing."""
        waited = 0.0

        while True:
            sleep_needed = 0.0
            with self._lock:
                now = time.time()
                # Purge timestamps older than 60s
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]

                # Check RPM quota
                if len(self._timestamps) >= self.max_rpm:
                    oldest = self._timestamps[0]
                    sleep_needed = max(0.1, 60.0 - (now - oldest) + 0.2)
                else:
                    # Enforce burst smoothing (spacing requests evenly)
                    time_since_last = now - self._last_request_time
                    if self.min_interval > 0 and time_since_last < self.min_interval:
                        sleep_needed = self.min_interval - time_since_last

                if sleep_needed <= 0:
                    record_time = time.time()
                    self._timestamps.append(record_time)
                    self._last_request_time = record_time
                    return waited

            # Sleep OUTSIDE the lock to prevent freezing other threads
            logger.info(
                f"In-memory rate limiter throttling '{self.key}' for {sleep_needed:.2f}s "
                f"({len(self._timestamps)}/{self.max_rpm} RPM)..."
            )
            time.sleep(sleep_needed)
            waited += sleep_needed

    def acquire(self, key_name: str = "") -> float:
        """
        Block synchronously until a request slot is available under the RPM limit.
        Returns the duration waited in seconds.
        """
        if self.max_rpm <= 0:
            return 0.0

        redis_client = self._get_redis_client()
        if redis_client:
            return self._acquire_redis(redis_client)
        return self._acquire_memory()

    async def acquire_async(self, key_name: str = "") -> float:
        """
        Async version of acquire using asyncio.sleep instead of time.sleep.
        """
        if self.max_rpm <= 0:
            return 0.0

        waited = 0.0
        while True:
            sleep_needed = 0.0
            with self._lock:
                now = time.time()
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]

                if len(self._timestamps) >= self.max_rpm:
                    oldest = self._timestamps[0]
                    sleep_needed = max(0.1, 60.0 - (now - oldest) + 0.2)
                else:
                    time_since_last = now - self._last_request_time
                    if self.min_interval > 0 and time_since_last < self.min_interval:
                        sleep_needed = self.min_interval - time_since_last

                if sleep_needed <= 0:
                    record_time = time.time()
                    self._timestamps.append(record_time)
                    self._last_request_time = record_time
                    return waited

            logger.info(
                f"Async rate limiter throttling '{self.key}' for {sleep_needed:.2f}s..."
            )
            await asyncio.sleep(sleep_needed)
            waited += sleep_needed


# Backward-compatible alias
ModelRateLimiter = ProviderRateLimiter

# Global rate limiters cache
_rate_limiters: Dict[str, ProviderRateLimiter] = {}
_lock = threading.Lock()


def get_rate_limiter(key: str, max_rpm: int = 14) -> ProviderRateLimiter:
    """Get or create the singleton rate limiter for a specific provider/model/key."""
    # Normalize key (e.g. gemini keys share the same quota)
    normalized_key = key.lower().strip()
    with _lock:
        if normalized_key not in _rate_limiters:
            _rate_limiters[normalized_key] = ProviderRateLimiter(key=normalized_key, max_rpm=max_rpm)
        return _rate_limiters[normalized_key]


def get_provider_limiter(provider: str = "gemini", api_key: Optional[str] = None, max_rpm: int = 14) -> ProviderRateLimiter:
    """
    Get or create a unified rate limiter keyed on provider and API key hash.
    Ensures all models under the same API key share the single 14 RPM quota.
    """
    if api_key:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8]
        limiter_key = f"{provider}:{key_hash}"
    else:
        limiter_key = provider
    return get_rate_limiter(limiter_key, max_rpm=max_rpm)

