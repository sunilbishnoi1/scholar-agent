# Groq LLM Client
# Implementation of the BaseLLMClient for Groq's API
# Groq offers very fast inference with generous free tier (30 RPM)

from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
import requests
import threading
from collections import deque
from typing import Optional, Dict, Any, Deque

from agents.llm.base import BaseLLMClient, LLMConfig, LLMResponse
from agents.llm.model_config import ModelTier, GROQ_MODELS, get_model_config
from agents.error_handling import (
    with_retry,
    RetryConfig,
    get_circuit_breaker,
    ErrorCategory,
    RetryableError,
    NonRetryableError
)

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for Groq API.
    
    Implements both RPM (requests per minute) and TPM (tokens per minute) limiting.
    Based on Groq's FREE TIER rate limits:
    - llama-3.3-70b-versatile: 30 RPM, 12K TPM, 1K RPD
    - llama-3.1-8b-instant: 30 RPM, 6K TPM, 14.4K RPD
    
    IMPORTANT: These are strict limits. Exceeding TPM causes 413 errors.
    """
    
    # Groq FREE TIER rate limits (strict - do not exceed)
    MODEL_LIMITS = {
        "llama-3.3-70b-versatile": {"rpm": 30, "tpm": 12000, "rpd": 1000},
        "llama-3.1-8b-instant": {"rpm": 30, "tpm": 6000, "rpd": 14400},
        "qwen/qwen3-32b": {"rpm": 60, "tpm": 6000, "rpd": 1000},
        # Default conservative limits
        "default": {"rpm": 20, "tpm": 6000, "rpd": 1000}
    }
    
    def __init__(self, safety_margin: float = 0.8):
        """
        Initialize rate limiter.
        
        Args:
            safety_margin: Use this fraction of the limit (0.8 = 80% of limit)
        """
        self.safety_margin = safety_margin
        self._lock = threading.Lock()
        
        # Track request timestamps per model
        self._request_times: Dict[str, Deque[float]] = {}
        self._token_usage: Dict[str, Deque[tuple]] = {}  # (timestamp, tokens)
        self._daily_requests: Dict[str, int] = {}
        self._daily_reset: float = time.time()
        
        # Global rate limit state from headers
        self._remaining_requests: Optional[int] = None
        self._remaining_tokens: Optional[int] = None
        self._reset_requests: Optional[float] = None
        self._reset_tokens: Optional[float] = None
    
    def get_limits(self, model: str) -> dict:
        """Get rate limits for a model, with safety margin applied."""
        limits = self.MODEL_LIMITS.get(model, self.MODEL_LIMITS["default"])
        return {
            "rpm": int(limits["rpm"] * self.safety_margin),
            "tpm": int(limits["tpm"] * self.safety_margin),
            "rpd": int(limits["rpd"] * self.safety_margin)
        }
    
    def update_from_headers(self, headers: dict) -> None:
        """Update rate limit state from response headers."""
        with self._lock:
            if "x-ratelimit-remaining-requests" in headers:
                self._remaining_requests = int(headers["x-ratelimit-remaining-requests"])
            if "x-ratelimit-remaining-tokens" in headers:
                self._remaining_tokens = int(headers["x-ratelimit-remaining-tokens"])
            if "x-ratelimit-reset-requests" in headers:
                # Parse duration like "2m59.56s"
                self._reset_requests = self._parse_reset_time(headers["x-ratelimit-reset-requests"])
            if "x-ratelimit-reset-tokens" in headers:
                self._reset_tokens = self._parse_reset_time(headers["x-ratelimit-reset-tokens"])
    
    def _parse_reset_time(self, reset_str: str) -> float:
        """Parse reset time string like '2m59.56s' to seconds."""
        try:
            total_seconds = 0.0
            if 'm' in reset_str:
                parts = reset_str.split('m')
                total_seconds += float(parts[0]) * 60
                reset_str = parts[1] if len(parts) > 1 else ""
            if 's' in reset_str:
                total_seconds += float(reset_str.replace('s', ''))
            return total_seconds
        except:
            return 60.0  # Default to 1 minute
    
    def wait_if_needed(self, model: str, estimated_tokens: int = 1000) -> float:
        """
        Wait if we're close to rate limits.
        
        Args:
            model: The model being used
            estimated_tokens: Estimated tokens for this request
            
        Returns:
            Time waited in seconds
        """
        with self._lock:
            limits = self.get_limits(model)
            now = time.time()
            
            # Reset daily counters if needed (every 24h)
            if now - self._daily_reset > 86400:
                self._daily_requests = {}
                self._daily_reset = now
            
            # Initialize tracking for this model
            if model not in self._request_times:
                self._request_times[model] = deque(maxlen=100)
                self._token_usage[model] = deque(maxlen=100)
                self._daily_requests[model] = 0
            
            # Clean old entries (older than 1 minute)
            minute_ago = now - 60
            while self._request_times[model] and self._request_times[model][0] < minute_ago:
                self._request_times[model].popleft()
            while self._token_usage[model] and self._token_usage[model][0][0] < minute_ago:
                self._token_usage[model].popleft()
            
            # Calculate current usage
            rpm_current = len(self._request_times[model])
            tpm_current = sum(t[1] for t in self._token_usage[model])
            rpd_current = self._daily_requests.get(model, 0)
            
            wait_time = 0.0
            
            # Check daily limit first
            if rpd_current >= limits["rpd"]:
                # Daily limit reached - this is severe
                logger.warning(f"Daily request limit reached for {model} ({rpd_current}/{limits['rpd']})")
                # Wait until next day reset, but cap at 5 minutes for now
                wait_time = min(300, 86400 - (now - self._daily_reset))
            
            # Check RPM limit
            if rpm_current >= limits["rpm"]:
                # Find when the oldest request will expire
                oldest = self._request_times[model][0] if self._request_times[model] else now
                rpm_wait = max(0, 60 - (now - oldest)) + 1  # +1 for safety
                wait_time = max(wait_time, rpm_wait)
                logger.info(f"RPM limit reached for {model}, waiting {rpm_wait:.1f}s")
            
            # Check TPM limit  
            if tpm_current + estimated_tokens >= limits["tpm"]:
                # Find when enough tokens will expire
                oldest = self._token_usage[model][0][0] if self._token_usage[model] else now
                tpm_wait = max(0, 60 - (now - oldest)) + 1
                wait_time = max(wait_time, tpm_wait)
                logger.info(f"TPM limit approaching for {model}, waiting {tpm_wait:.1f}s")
            
            # Use server-reported limits if available and more restrictive
            if self._remaining_requests is not None and self._remaining_requests <= 1:
                if self._reset_requests:
                    wait_time = max(wait_time, self._reset_requests + 0.5)
                    logger.info(f"Server reports low remaining requests, waiting {self._reset_requests:.1f}s")
            
            if self._remaining_tokens is not None and self._remaining_tokens < estimated_tokens:
                if self._reset_tokens:
                    wait_time = max(wait_time, self._reset_tokens + 0.5)
                    logger.info(f"Server reports low remaining tokens, waiting {self._reset_tokens:.1f}s")
        
        # Do the wait outside the lock
        if wait_time > 0:
            logger.info(f"Rate limiter waiting {wait_time:.1f}s for {model}")
            time.sleep(wait_time)
        
        return wait_time
    
    def record_request(self, model: str, tokens_used: int) -> None:
        """Record a completed request for rate tracking."""
        with self._lock:
            now = time.time()
            
            if model not in self._request_times:
                self._request_times[model] = deque(maxlen=100)
                self._token_usage[model] = deque(maxlen=100)
            
            self._request_times[model].append(now)
            self._token_usage[model].append((now, tokens_used))
            self._daily_requests[model] = self._daily_requests.get(model, 0) + 1
    
    def get_status(self, model: str) -> dict:
        """Get current rate limit status for a model."""
        with self._lock:
            limits = self.get_limits(model)
            now = time.time()
            
            # Clean old entries
            minute_ago = now - 60
            if model in self._request_times:
                while self._request_times[model] and self._request_times[model][0] < minute_ago:
                    self._request_times[model].popleft()
            if model in self._token_usage:
                while self._token_usage[model] and self._token_usage[model][0][0] < minute_ago:
                    self._token_usage[model].popleft()
            
            return {
                "model": model,
                "rpm_used": len(self._request_times.get(model, [])),
                "rpm_limit": limits["rpm"],
                "tpm_used": sum(t[1] for t in self._token_usage.get(model, [])),
                "tpm_limit": limits["tpm"],
                "rpd_used": self._daily_requests.get(model, 0),
                "rpd_limit": limits["rpd"],
                "server_remaining_requests": self._remaining_requests,
                "server_remaining_tokens": self._remaining_tokens
            }


# Global rate limiter instance
_rate_limiter: Optional[TokenBucketRateLimiter] = None

def get_rate_limiter() -> TokenBucketRateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        # Use 95% of limits - let server headers guide us if we're too aggressive
        _rate_limiter = TokenBucketRateLimiter(safety_margin=0.95)
    return _rate_limiter


class GroqClient(BaseLLMClient):
    """
    Groq API client with smart routing and error handling.
    
    Groq Features:
    - Very fast inference (fastest in the industry)
    - Generous free tier: 30 RPM, 14,400 RPD
    - OpenAI-compatible API format
    - Great for Llama and Mixtral models
    
    Models:
    - llama-3.1-8b-instant: Fast, cheap, good for simple tasks
    - llama-3.3-70b-versatile: Powerful, good for complex tasks
    - mixtral-8x7b-32768: Good balance of speed and quality
    """
    
    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
    PROVIDER_NAME = "groq"
    
    def _setup_client(self) -> None:
        """Set up the Groq client."""
        self.api_key = self.config.api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set. Groq client will not be functional.")
        
        # Budget tracking
        self.user_budget = self.config.user_budget
        self.user_id = self.config.user_id
        self.spent = 0.0
        
        # Rate limiter (shared instance)
        self.rate_limiter = get_rate_limiter()
        
        # Circuit breaker for API calls
        self.circuit_breaker = get_circuit_breaker(
            "groq_api",
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        # Retry configuration - more aggressive for rate limits
        self.retry_config = RetryConfig(
            max_retries=self.config.max_retries,
            initial_delay=2.0,  # Start with longer delay
            max_delay=120.0,    # Allow longer waits for rate limits
            exponential_base=2.0,
            jitter=True
        )
        
        logger.info(
            f"GroqClient initialized (user={self.user_id}, "
            f"budget=${self.user_budget:.4f})"
        )
    
    def get_provider_name(self) -> str:
        return self.PROVIDER_NAME
    
    def _select_model(
        self,
        task_type: str,
        complexity_hint: Optional[str] = None,
        max_latency_ms: Optional[int] = None
    ) -> tuple[str, ModelTier]:
        """
        Select the appropriate model based on task requirements.
        
        Model Selection Strategy (optimized for Groq free tier):
        - Use llama-3.1-8b-instant for most tasks (14.4K RPD vs 1K RPD for 70B)
        - Only use llama-3.3-70b-versatile for truly complex tasks
        - Check rate limit status and downgrade if near limits
        
        Returns:
            Tuple of (model_name, tier)
        """
        # Task type to tier mapping
        # Note: Most tasks use FAST_CHEAP now due to much better rate limits
        TASK_ROUTING = {
            # Simple tasks -> Fast/Cheap (8B model)
            "extract_keywords": ModelTier.FAST_CHEAP,
            "classify_relevance": ModelTier.FAST_CHEAP,
            "simple_parsing": ModelTier.FAST_CHEAP,
            "format_output": ModelTier.FAST_CHEAP,
            "general": ModelTier.FAST_CHEAP,  # Default to 8B for general tasks
            
            # Medium tasks -> Also use Fast/Cheap for better rate limits
            "paper_analysis": ModelTier.BALANCED,  # 8B is sufficient for analysis
            "summarization": ModelTier.BALANCED,
            "search_planning": ModelTier.BALANCED,
            "relevance_scoring": ModelTier.FAST_CHEAP,
            
            # Complex tasks -> Powerful (70B) - use sparingly!
            "synthesis": ModelTier.POWERFUL,
            "quality_evaluation": ModelTier.BALANCED,  # Downgraded to save 70B quota
            "research_gap_identification": ModelTier.POWERFUL,
            "complex_reasoning": ModelTier.POWERFUL,
        }
        
        # Start with task-based routing
        tier = TASK_ROUTING.get(task_type, ModelTier.FAST_CHEAP)
        
        # Check rate limit status and potentially downgrade
        model_name = GROQ_MODELS.get(tier, GROQ_MODELS[ModelTier.FAST_CHEAP]).name
        rate_status = self.rate_limiter.get_status(model_name)
        
        # If the selected model is near daily limit, downgrade
        rpd_usage_pct = rate_status["rpd_used"] / rate_status["rpd_limit"] if rate_status["rpd_limit"] > 0 else 0
        if rpd_usage_pct >= 0.8 and tier == ModelTier.POWERFUL:
            logger.warning(f"70B model near daily limit ({rate_status['rpd_used']}/{rate_status['rpd_limit']}), using 8B")
            tier = ModelTier.BALANCED
        
        # Budget check - downgrade if running low
        budget_usage = self.spent / self.user_budget if self.user_budget > 0 else 0
        if budget_usage >= 0.8:
            tier = ModelTier.FAST_CHEAP
            logger.warning(f"Budget constraint: Using {tier.value} (${self.spent:.4f}/${self.user_budget:.4f} spent)")
        
        # Complexity hint override
        if complexity_hint == "high" and tier != ModelTier.POWERFUL and budget_usage < 0.8:
            tier = ModelTier.POWERFUL
        elif complexity_hint == "low":
            tier = ModelTier.FAST_CHEAP
        
        # Latency constraint
        if max_latency_ms:
            model_config = GROQ_MODELS.get(tier)
            if model_config and model_config.base_latency_ms > max_latency_ms:
                # Downgrade for latency
                if tier == ModelTier.POWERFUL:
                    tier = ModelTier.BALANCED
                elif tier == ModelTier.BALANCED:
                    tier = ModelTier.FAST_CHEAP
        
        model_config = GROQ_MODELS.get(tier, GROQ_MODELS[ModelTier.BALANCED])
        return model_config.name, tier
    
    def chat(
        self,
        prompt: str,
        task_type: str = "general",
        complexity_hint: Optional[str] = None,
        max_latency_ms: Optional[int] = None,
        **kwargs
    ) -> str:
        """Send a chat request and return the response text."""
        response = self.chat_with_response(
            prompt=prompt,
            task_type=task_type,
            complexity_hint=complexity_hint,
            max_latency_ms=max_latency_ms,
            **kwargs
        )
        return response.text
    
    def chat_with_response(
        self,
        prompt: str,
        task_type: str = "general",
        complexity_hint: Optional[str] = None,
        max_latency_ms: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Send a chat request and return full response with metadata."""
        
        # Select model
        model_name, tier = self._select_model(task_type, complexity_hint, max_latency_ms)
        logger.info(f"Groq routing: {task_type} -> {model_name} (tier: {tier.value})")
        
        start_time = time.time()
        
        try:
            result = self._make_api_call_with_retry(model_name, prompt, **kwargs)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract token usage from response
            input_tokens = result.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = result.get("usage", {}).get("completion_tokens", 0)
            total_tokens = result.get("usage", {}).get("total_tokens", 0)
            
            # Calculate cost
            model_config = GROQ_MODELS.get(tier)
            if model_config:
                cost = model_config.estimate_cost(input_tokens, output_tokens)
            else:
                cost = 0.0
            
            # Record usage
            self.spent += cost
            
            # Extract response text
            text = result["choices"][0]["message"]["content"]
            
            return LLMResponse(
                text=text,
                model=model_name,
                provider=self.PROVIDER_NAME,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost=cost,
                latency_ms=latency_ms,
                metadata={
                    "tier": tier.value,
                    "task_type": task_type,
                    "finish_reason": result["choices"][0].get("finish_reason"),
                }
            )
            
        except Exception as e:
            logger.error(f"Groq chat request failed: {e}")
            raise
    
    def _make_api_call_with_retry(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Make API call with retry logic, rate limiting, and circuit breaker."""
        
        # Estimate tokens for rate limiting (rough: 4 chars = 1 token)
        # Use a more realistic estimate - most responses are <500 tokens
        max_response_tokens = min(kwargs.get("max_tokens", 1024), 1024)
        estimated_tokens = len(prompt) // 4 + max_response_tokens
        
        # Wait if needed to respect rate limits BEFORE making the request
        self.rate_limiter.wait_if_needed(model, estimated_tokens)
        
        @with_retry(
            config=self.retry_config,
            retryable_exceptions=(
                requests.exceptions.HTTPError,
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                RetryableError
            ),
            non_retryable_exceptions=(NonRetryableError,)
        )
        def _call_api():
            return self.circuit_breaker.call(self._do_api_request, model, prompt, **kwargs)
        
        result = _call_api()
        
        # Record the request for rate tracking
        total_tokens = result.get("usage", {}).get("total_tokens", estimated_tokens)
        self.rate_limiter.record_request(model, total_tokens)
        
        return result
    
    def _do_api_request(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Execute the actual API request to Groq."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Respect model context limits and max payload size
        max_tokens = kwargs.get("max_tokens", 4096)
        
        # Groq has a ~128K context limit for most models
        # Truncate prompt if too long (estimate: 4 chars = 1 token)
        max_prompt_tokens = 100000  # Leave room for response
        prompt_tokens_estimate = len(prompt) // 4
        
        if prompt_tokens_estimate > max_prompt_tokens:
            logger.warning(
                f"Prompt too long ({prompt_tokens_estimate} est. tokens), "
                f"truncating to ~{max_prompt_tokens} tokens"
            )
            # Truncate to approximately max_prompt_tokens * 4 characters
            max_chars = max_prompt_tokens * 4
            prompt = prompt[:max_chars] + "\n\n[Content truncated due to length limits...]"
        
        # Build request payload (OpenAI-compatible format)
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": max_tokens,
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            data["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            data["stop"] = kwargs["stop"]
        
        try:
            resp = requests.post(
                self.BASE_URL,
                headers=headers,
                json=data,
                timeout=self.config.timeout
            )
            
            # Update rate limiter with response headers (even on errors)
            self.rate_limiter.update_from_headers(dict(resp.headers))
            
            resp.raise_for_status()
            return resp.json()
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            
            # Update rate limiter even on errors
            self.rate_limiter.update_from_headers(dict(e.response.headers))
            
            if status_code == 429:
                # Rate limit - retryable
                retry_after = e.response.headers.get("retry-after", "60")
                raise RetryableError(
                    f"Rate limit exceeded. Retry after {retry_after}s",
                    category=ErrorCategory.RATE_LIMIT,
                    severity="low"
                )
            elif status_code == 413:
                # Payload too large - this is a prompt size issue
                # Try to extract more info
                try:
                    error_detail = e.response.json().get("error", {}).get("message", "Payload too large")
                except:
                    error_detail = "Payload too large"
                raise NonRetryableError(
                    f"Payload too large ({status_code}): {error_detail}. "
                    f"Prompt has ~{len(prompt)//4} tokens. Consider reducing input size.",
                    category=ErrorCategory.VALIDATION,
                    severity="medium"
                )
            elif status_code >= 500:
                raise RetryableError(
                    f"Groq server error ({status_code}): {e}",
                    category=ErrorCategory.SERVER_ERROR,
                    severity="medium"
                )
            elif status_code in (401, 403):
                raise NonRetryableError(
                    f"Authentication failed ({status_code}): Check GROQ_API_KEY",
                    category=ErrorCategory.CLIENT_ERROR,
                    severity="high"
                )
            elif status_code == 400:
                # Try to extract error message
                try:
                    error_detail = e.response.json().get("error", {}).get("message", str(e))
                except:
                    error_detail = str(e)
                raise NonRetryableError(
                    f"Invalid request ({status_code}): {error_detail}",
                    category=ErrorCategory.VALIDATION,
                    severity="medium"
                )
            else:
                raise NonRetryableError(
                    f"Client error ({status_code}): {e}",
                    category=ErrorCategory.CLIENT_ERROR,
                    severity="medium"
                )
                
        except requests.exceptions.Timeout as e:
            raise RetryableError(
                f"Request timeout: {e}",
                category=ErrorCategory.TIMEOUT,
                severity="low"
            )
            
        except requests.exceptions.ConnectionError as e:
            raise RetryableError(
                f"Network error: {e}",
                category=ErrorCategory.NETWORK,
                severity="medium"
            )
            
        except Exception as e:
            logger.error(f"Unexpected error in Groq request: {e}", exc_info=True)
            raise RetryableError(
                f"Unexpected error: {e}",
                category=ErrorCategory.UNKNOWN,
                severity="medium"
            )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics including rate limit status."""
        # Get rate limit status for common models
        rate_stats = {
            "llama-3.1-8b-instant": self.rate_limiter.get_status("llama-3.1-8b-instant"),
            "llama-3.3-70b-versatile": self.rate_limiter.get_status("llama-3.3-70b-versatile"),
        }
        
        return {
            "provider": self.PROVIDER_NAME,
            "user_id": self.user_id,
            "budget": self.user_budget,
            "spent": self.spent,
            "remaining": self.user_budget - self.spent,
            "usage_percent": (self.spent / self.user_budget * 100) if self.user_budget > 0 else 0,
            "rate_limits": rate_stats
        }
    
    def reset_budget(self, new_budget: Optional[float] = None) -> None:
        """Reset budget tracking."""
        if new_budget is not None:
            self.user_budget = new_budget
        self.spent = 0.0
        logger.info(f"Groq budget reset to ${self.user_budget:.4f}")
    
    def is_available(self) -> bool:
        """Check if Groq is configured and available."""
        return bool(self.api_key)
