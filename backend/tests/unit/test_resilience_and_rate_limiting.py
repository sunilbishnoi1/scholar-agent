"""
Unit Tests for ProviderRateLimiter, CircuitBreaker 429 Exclusion, and Multi-Agent Resilience.
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from agents.error_handling import (
    CircuitBreaker,
    CircuitBreakerOpen,
    ErrorCategory,
    ErrorSeverity,
    RetryableError,
)
from agents.llm.rate_limiter import (
    ProviderRateLimiter,
    get_provider_limiter,
    get_rate_limiter,
)


@pytest.mark.unit
class TestProviderRateLimiter:
    """Test sliding window rate limiting and burst smoothing."""

    def test_in_memory_rate_limiter_rpm_enforcement(self, monkeypatch):
        """Verify limiter enforces RPM limit and spaces requests."""
        limiter = ProviderRateLimiter(key="test_gemini", max_rpm=5, min_interval=0.05)
        monkeypatch.setattr(limiter, "_get_redis_client", lambda: None)

        for _ in range(5):
            limiter.acquire("test_gemini")

        start = time.time()
        with limiter._lock:
            if limiter._timestamps:
                limiter._timestamps[0] = time.time() - 59.8

        waited = limiter.acquire("test_gemini")
        elapsed = time.time() - start
        assert elapsed >= 0.04

    def test_get_provider_limiter_key_hashing(self):
        """Verify API key hashing shares limiter instance."""
        lim1 = get_provider_limiter("gemini", api_key="secret_key_123", max_rpm=14)
        lim2 = get_provider_limiter("gemini", api_key="secret_key_123", max_rpm=14)
        assert lim1 is lim2

        lim3 = get_provider_limiter("gemini", api_key="different_key_456", max_rpm=14)
        assert lim3 is not lim1


@pytest.mark.unit
class TestCircuitBreakerRateLimitExclusion:
    """Verify that 429 Rate Limit errors do not trip CircuitBreaker into OPEN state."""

    def test_429_does_not_trip_circuit_breaker(self):
        """Test that repeated 429 errors bypass failure increment."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0, name="test_cb")

        def failing_function():
            raise RetryableError(
                "Rate limit exceeded (429)",
                category=ErrorCategory.RATE_LIMIT,
                severity=ErrorSeverity.LOW,
            )

        for _ in range(5):
            with pytest.raises(RetryableError):
                cb.call(failing_function)

        assert cb.state == CircuitBreaker.State.CLOSED
        assert cb.failure_count == 0

    def test_server_errors_trip_circuit_breaker(self):
        """Test that real server errors (500) still trip the CircuitBreaker."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0, name="test_cb_server")

        def server_error_function():
            raise RetryableError(
                "Internal Server Error (500)",
                category=ErrorCategory.SERVER_ERROR,
                severity=ErrorSeverity.HIGH,
            )

        for _ in range(3):
            with pytest.raises(RetryableError):
                cb.call(server_error_function)

        assert cb.state == CircuitBreaker.State.OPEN
        with pytest.raises(CircuitBreakerOpen):
            cb.call(server_error_function)

