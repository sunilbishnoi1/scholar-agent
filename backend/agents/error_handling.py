# Enhanced Error Handling and Retry Logic
# Provides robust error handling with exponential backoff, circuit breaker, and comprehensive error types

import time
import logging
import functools
from typing import Optional, Callable, Any, TypeVar, cast
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(str, Enum):
    """Error severity levels for prioritization and alerting."""
    LOW = "low"           # Transient, will likely succeed on retry
    MEDIUM = "medium"     # May need intervention but not critical
    HIGH = "high"         # Requires immediate attention
    CRITICAL = "critical" # System-level failure


class ErrorCategory(str, Enum):
    """Categories of errors for handling strategy."""
    RATE_LIMIT = "rate_limit"           # API rate limit exceeded
    TIMEOUT = "timeout"                  # Request timed out
    SERVER_ERROR = "server_error"        # 5xx errors
    CLIENT_ERROR = "client_error"        # 4xx errors (non-retryable)
    NETWORK = "network"                  # Network connectivity issues
    VALIDATION = "validation"            # Input validation failed
    QUOTA_EXCEEDED = "quota_exceeded"    # Usage quota exceeded
    UNKNOWN = "unknown"                  # Unclassified error


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 5
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0     # seconds
    exponential_base: float = 2.0
    jitter: bool = True         # Add randomness to prevent thundering herd


@dataclass
class ErrorContext:
    """Rich error context for debugging and monitoring."""
    error_type: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception] = None
    retry_count: int = 0
    total_delay: float = 0.0
    context: dict = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    class State(str, Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"
    
    def __init__(
        self, 
        failure_threshold: int = 5, 
        recovery_timeout: float = 60.0,
        name: str = "default"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying to recover
            name: Identifier for this circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = self.State.CLOSED
        
        logger.info(f"CircuitBreaker '{name}' initialized (threshold={failure_threshold})")
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        if self.state == self.State.OPEN:
            # Check if we should try to recover
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info(f"CircuitBreaker '{self.name}': Attempting recovery (HALF_OPEN)")
                self.state = self.State.HALF_OPEN
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Will retry in {self.recovery_timeout - (time.time() - self.last_failure_time):.1f}s"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Record successful call."""
        if self.state == self.State.HALF_OPEN:
            logger.info(f"CircuitBreaker '{self.name}': Recovery successful (CLOSED)")
            self.state = self.State.CLOSED
        # Always reset failure count on success
        self.failure_count = 0
    
    def _on_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            logger.error(
                f"CircuitBreaker '{self.name}': Threshold exceeded ({self.failure_count} failures) - OPENING circuit"
            )
            self.state = self.State.OPEN


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


class RetryableError(Exception):
    """Base class for errors that should be retried."""
    def __init__(self, message: str, category: ErrorCategory, severity: ErrorSeverity):
        super().__init__(message)
        self.category = category
        self.severity = severity


class NonRetryableError(Exception):
    """Base class for errors that should NOT be retried."""
    def __init__(self, message: str, category: ErrorCategory, severity: ErrorSeverity):
        super().__init__(message)
        self.category = category
        self.severity = severity


def with_retry(
    config: Optional[RetryConfig] = None,
    retryable_exceptions: tuple = (Exception,),
    non_retryable_exceptions: tuple = (NonRetryableError,),
    on_retry: Optional[Callable[[ErrorContext], None]] = None
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        config: Retry configuration
        retryable_exceptions: Tuple of exception types to retry
        non_retryable_exceptions: Tuple of exception types to never retry
        on_retry: Optional callback called before each retry
        
    Example:
        @with_retry(config=RetryConfig(max_retries=3))
        def fetch_data():
            return api.get("/data")
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            total_delay = 0.0
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except non_retryable_exceptions as e:
                    # Don't retry these
                    logger.error(f"{func.__name__} failed with non-retryable error: {e}")
                    raise
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        # Last attempt, raise
                        logger.error(
                            f"{func.__name__} failed after {config.max_retries} retries: {e}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.initial_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        import random
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    total_delay += delay
                    
                    # Create error context
                    error_ctx = ErrorContext(
                        error_type=_categorize_error(e),
                        severity=_assess_severity(e, attempt),
                        message=str(e),
                        original_exception=e,
                        retry_count=attempt + 1,
                        total_delay=total_delay,
                        context={"function": func.__name__}
                    )
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{config.max_retries} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(error_ctx)
                    
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__name__} failed without exception (should not happen)")
        
        return wrapper
    return decorator


async def with_retry_async(
    config: Optional[RetryConfig] = None,
    retryable_exceptions: tuple = (Exception,),
    non_retryable_exceptions: tuple = (NonRetryableError,),
    on_retry: Optional[Callable[[ErrorContext], None]] = None
):
    """
    Async version of with_retry decorator.
    
    Example:
        @with_retry_async(config=RetryConfig(max_retries=3))
        async def fetch_data():
            return await api.get("/data")
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception: Optional[Exception] = None
            total_delay = 0.0
            
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except non_retryable_exceptions as e:
                    logger.error(f"{func.__name__} failed with non-retryable error: {e}")
                    raise
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        logger.error(
                            f"{func.__name__} failed after {config.max_retries} retries: {e}"
                        )
                        raise
                    
                    delay = min(
                        config.initial_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    if config.jitter:
                        import random
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    total_delay += delay
                    
                    error_ctx = ErrorContext(
                        error_type=_categorize_error(e),
                        severity=_assess_severity(e, attempt),
                        message=str(e),
                        original_exception=e,
                        retry_count=attempt + 1,
                        total_delay=total_delay,
                        context={"function": func.__name__}
                    )
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{config.max_retries} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    if on_retry:
                        on_retry(error_ctx)
                    
                    await asyncio.sleep(delay)
            
            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__name__} failed without exception")
        
        return wrapper
    return decorator


def _categorize_error(exception: Exception) -> ErrorCategory:
    """Categorize exception into error types."""
    error_str = str(exception).lower()
    exception_name = type(exception).__name__.lower()
    
    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
        return ErrorCategory.RATE_LIMIT
    elif "timeout" in error_str or "timeout" in exception_name:
        return ErrorCategory.TIMEOUT
    elif "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
        return ErrorCategory.SERVER_ERROR
    elif "400" in error_str or "401" in error_str or "403" in error_str or "404" in error_str:
        return ErrorCategory.CLIENT_ERROR
    elif "quota" in error_str or "limit exceeded" in error_str:
        return ErrorCategory.QUOTA_EXCEEDED
    elif "network" in error_str or "connection" in error_str:
        return ErrorCategory.NETWORK
    elif "validation" in error_str:
        return ErrorCategory.VALIDATION
    else:
        return ErrorCategory.UNKNOWN


def _assess_severity(exception: Exception, retry_count: int) -> ErrorSeverity:
    """Assess error severity based on type and retry count."""
    category = _categorize_error(exception)
    
    # Client errors are generally not severe (user input issue)
    if category == ErrorCategory.CLIENT_ERROR:
        return ErrorSeverity.LOW
    
    # Rate limits and timeouts are transient
    if category in (ErrorCategory.RATE_LIMIT, ErrorCategory.TIMEOUT):
        return ErrorSeverity.LOW if retry_count < 3 else ErrorSeverity.MEDIUM
    
    # Quota exceeded is concerning
    if category == ErrorCategory.QUOTA_EXCEEDED:
        return ErrorSeverity.HIGH
    
    # Server errors escalate with retries
    if category == ErrorCategory.SERVER_ERROR:
        if retry_count < 2:
            return ErrorSeverity.LOW
        elif retry_count < 4:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.HIGH
    
    # Unknown errors are concerning
    return ErrorSeverity.MEDIUM


# Global circuit breakers for common services
_circuit_breakers = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.
    
    Args:
        name: Circuit breaker identifier
        **kwargs: Additional arguments for CircuitBreaker constructor
        
    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
    return _circuit_breakers[name]
