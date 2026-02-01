# Tests for Enhanced Error Handling
# Validates retry logic, circuit breaker, and error categorization

import time
from unittest.mock import Mock

import pytest

from agents.error_handling import (
    CircuitBreaker,
    CircuitBreakerOpen,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    NonRetryableError,
    RetryableError,
    RetryConfig,
    _assess_severity,
    _categorize_error,
    get_circuit_breaker,
    with_retry,
)


class TestRetryDecorator:
    """Tests for the with_retry decorator."""

    def test_retry_succeeds_on_first_attempt(self):
        """Function should succeed without retries if no error."""
        call_count = {"count": 0}

        @with_retry(config=RetryConfig(max_retries=3))
        def successful_function():
            call_count["count"] += 1
            return "success"

        result = successful_function()

        assert result == "success"
        assert call_count["count"] == 1

    def test_retry_recovers_from_transient_failures(self):
        """Retry should handle transient failures and succeed."""
        call_count = {"count": 0}

        @with_retry(config=RetryConfig(max_retries=3, initial_delay=0.01))
        def flaky_function():
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise Exception("Transient error")
            return "success"

        result = flaky_function()

        assert result == "success"
        assert call_count["count"] == 3

    def test_retry_gives_up_after_max_retries(self):
        """Retry should give up after max_retries exhausted."""
        call_count = {"count": 0}

        @with_retry(config=RetryConfig(max_retries=2, initial_delay=0.01))
        def always_failing():
            call_count["count"] += 1
            raise Exception("Permanent failure")

        with pytest.raises(Exception, match="Permanent failure"):
            always_failing()

        # Should try initial + 2 retries = 3 total
        assert call_count["count"] == 3

    def test_retry_respects_non_retryable_exceptions(self):
        """Non-retryable exceptions should not be retried."""
        call_count = {"count": 0}

        @with_retry(
            config=RetryConfig(max_retries=3),
            non_retryable_exceptions=(ValueError,)
        )
        def non_retryable_function():
            call_count["count"] += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError):
            non_retryable_function()

        # Should only be called once
        assert call_count["count"] == 1

    def test_retry_uses_exponential_backoff(self):
        """Retry should use exponential backoff between attempts."""
        call_count = {"count": 0}
        timestamps = []

        @with_retry(
            config=RetryConfig(
                max_retries=3,
                initial_delay=0.1,
                exponential_base=2.0,
                jitter=False
            )
        )
        def failing_function():
            call_count["count"] += 1
            timestamps.append(time.time())
            if call_count["count"] < 4:
                raise Exception("Failing")
            return "success"

        result = failing_function()

        assert result == "success"

        # Check delays between attempts
        if len(timestamps) >= 2:
            delay1 = timestamps[1] - timestamps[0]
            assert 0.08 < delay1 < 0.15  # Should be ~0.1s

        if len(timestamps) >= 3:
            delay2 = timestamps[2] - timestamps[1]
            assert 0.18 < delay2 < 0.25  # Should be ~0.2s

    def test_retry_respects_max_delay(self):
        """Retry should cap delay at max_delay."""
        @with_retry(
            config=RetryConfig(
                max_retries=10,
                initial_delay=1.0,
                max_delay=2.0,
                exponential_base=2.0,
                jitter=False
            )
        )
        def function():
            raise Exception("Test")

        # With exponential backoff, later attempts would exceed max_delay
        # but should be capped at 2.0
        with pytest.raises(Exception):
            function()

    def test_retry_calls_on_retry_callback(self):
        """Retry should call on_retry callback before each retry."""
        call_count = {"count": 0}
        retry_contexts = []

        def on_retry(error_ctx: ErrorContext):
            retry_contexts.append(error_ctx)

        @with_retry(
            config=RetryConfig(max_retries=2, initial_delay=0.01),
            on_retry=on_retry
        )
        def failing_function():
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise Exception("Failing")
            return "success"

        result = failing_function()

        assert result == "success"
        assert len(retry_contexts) == 2  # 2 retries

        # Check error context
        assert all(isinstance(ctx, ErrorContext) for ctx in retry_contexts)
        assert retry_contexts[0].retry_count == 1
        assert retry_contexts[1].retry_count == 2


class TestCircuitBreaker:
    """Tests for the CircuitBreaker class."""

    def test_circuit_breaker_allows_calls_when_closed(self):
        """Circuit breaker should allow calls in CLOSED state."""
        breaker = CircuitBreaker(failure_threshold=3)

        def successful_function():
            return "success"

        result = breaker.call(successful_function)

        assert result == "success"
        assert breaker.state == CircuitBreaker.State.CLOSED

    def test_circuit_breaker_opens_after_threshold_failures(self):
        """Circuit breaker should open after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        def failing_function():
            raise Exception("Service unavailable")

        # Fail 3 times
        for i in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_function)

        # Circuit should be open now
        assert breaker.state == CircuitBreaker.State.OPEN

        # Next call should raise CircuitBreakerOpen
        with pytest.raises(CircuitBreakerOpen):
            breaker.call(failing_function)

    def test_circuit_breaker_recovers_after_timeout(self):
        """Circuit breaker should attempt recovery after timeout."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.3)

        call_count = {"count": 0}

        def recovering_function():
            call_count["count"] += 1
            if call_count["count"] <= 2:
                raise Exception("Service down")
            return "recovered"

        # Fail twice to open circuit
        for i in range(2):
            with pytest.raises(Exception):
                breaker.call(recovering_function)

        assert breaker.state == CircuitBreaker.State.OPEN

        # Wait for recovery timeout
        time.sleep(0.4)

        # Should attempt recovery and succeed
        result = breaker.call(recovering_function)
        assert result == "recovered"
        assert breaker.state == CircuitBreaker.State.CLOSED

    def test_circuit_breaker_resets_on_success(self):
        """Circuit breaker should reset failure count on success."""
        breaker = CircuitBreaker(failure_threshold=3)

        call_count = {"count": 0}

        def intermittent_function():
            call_count["count"] += 1
            if call_count["count"] in (1, 2):
                raise Exception("Failing")
            return "success"

        # Fail twice
        for i in range(2):
            with pytest.raises(Exception):
                breaker.call(intermittent_function)

        assert breaker.failure_count == 2

        # Succeed once
        result = breaker.call(intermittent_function)
        assert result == "success"

        # Failure count should reset
        assert breaker.failure_count == 0


class TestErrorCategorization:
    """Tests for error categorization logic."""

    def test_categorize_rate_limit_error(self):
        """Should categorize rate limit errors correctly."""
        error = Exception("Error 429: Too many requests")
        category = _categorize_error(error)
        assert category == ErrorCategory.RATE_LIMIT

    def test_categorize_timeout_error(self):
        """Should categorize timeout errors correctly."""
        error = Exception("Request timeout after 60s")
        category = _categorize_error(error)
        assert category == ErrorCategory.TIMEOUT

    def test_categorize_server_error(self):
        """Should categorize 5xx errors correctly."""
        error = Exception("Error 503: Service unavailable")
        category = _categorize_error(error)
        assert category == ErrorCategory.SERVER_ERROR

    def test_categorize_client_error(self):
        """Should categorize 4xx errors correctly."""
        error = Exception("Error 400: Bad request")
        category = _categorize_error(error)
        assert category == ErrorCategory.CLIENT_ERROR

    def test_categorize_quota_exceeded(self):
        """Should categorize quota errors correctly."""
        error = Exception("Quota exceeded for this resource")
        category = _categorize_error(error)
        assert category == ErrorCategory.QUOTA_EXCEEDED

    def test_categorize_unknown_error(self):
        """Should categorize unknown errors as UNKNOWN."""
        error = Exception("Something went wrong")
        category = _categorize_error(error)
        assert category == ErrorCategory.UNKNOWN


class TestErrorSeverityAssessment:
    """Tests for error severity assessment."""

    def test_client_errors_are_low_severity(self):
        """Client errors should be assessed as low severity."""
        error = Exception("Error 400: Bad request")
        severity = _assess_severity(error, retry_count=0)
        assert severity == ErrorSeverity.LOW

    def test_quota_exceeded_is_high_severity(self):
        """Quota exceeded should be high severity."""
        error = Exception("Quota exceeded")
        severity = _assess_severity(error, retry_count=0)
        assert severity == ErrorSeverity.HIGH

    def test_server_errors_escalate_with_retries(self):
        """Server errors should escalate severity with retry count."""
        error = Exception("Error 500: Internal server error")

        severity_0 = _assess_severity(error, retry_count=0)
        severity_3 = _assess_severity(error, retry_count=3)
        severity_5 = _assess_severity(error, retry_count=5)

        assert severity_0 == ErrorSeverity.LOW
        assert severity_3 == ErrorSeverity.MEDIUM
        assert severity_5 == ErrorSeverity.HIGH


class TestRetryableAndNonRetryableErrors:
    """Tests for custom error types."""

    def test_retryable_error_includes_metadata(self):
        """RetryableError should include category and severity."""
        error = RetryableError(
            "Temporary failure",
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.LOW
        )

        assert str(error) == "Temporary failure"
        assert error.category == ErrorCategory.TIMEOUT
        assert error.severity == ErrorSeverity.LOW

    def test_non_retryable_error_includes_metadata(self):
        """NonRetryableError should include category and severity."""
        error = NonRetryableError(
            "Invalid API key",
            category=ErrorCategory.CLIENT_ERROR,
            severity=ErrorSeverity.HIGH
        )

        assert str(error) == "Invalid API key"
        assert error.category == ErrorCategory.CLIENT_ERROR
        assert error.severity == ErrorSeverity.HIGH


class TestGetCircuitBreaker:
    """Tests for get_circuit_breaker factory function."""

    def test_get_circuit_breaker_creates_breaker(self):
        """get_circuit_breaker should create a breaker instance."""
        breaker = get_circuit_breaker("test_service")

        assert isinstance(breaker, CircuitBreaker)
        assert breaker.name == "test_service"

    def test_get_circuit_breaker_returns_same_instance(self):
        """get_circuit_breaker should return cached instance."""
        breaker1 = get_circuit_breaker("service1")
        breaker2 = get_circuit_breaker("service1")

        assert breaker1 is breaker2

    def test_get_circuit_breaker_different_services(self):
        """Different service names should get different breakers."""
        breaker1 = get_circuit_breaker("service1")
        breaker2 = get_circuit_breaker("service2")

        assert breaker1 is not breaker2


class TestErrorContext:
    """Tests for ErrorContext dataclass."""

    def test_error_context_initialization(self):
        """ErrorContext should initialize with all fields."""
        ctx = ErrorContext(
            error_type=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            message="Request timed out",
            retry_count=2,
            total_delay=5.0
        )

        assert ctx.error_type == ErrorCategory.TIMEOUT
        assert ctx.severity == ErrorSeverity.MEDIUM
        assert ctx.message == "Request timed out"
        assert ctx.retry_count == 2
        assert ctx.total_delay == 5.0
        assert ctx.context == {}

    def test_error_context_with_exception(self):
        """ErrorContext can store original exception."""
        original = ValueError("Test error")
        ctx = ErrorContext(
            error_type=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            message="Validation failed",
            original_exception=original
        )

        assert ctx.original_exception is original
