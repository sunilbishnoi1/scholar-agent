# Groq LLM Client
# Implementation of the BaseLLMClient for Groq's API
# Groq offers very fast inference with generous free tier (30 RPM)
#
# KEY PRINCIPLE: NEVER WAIT FOR RATE LIMITS - IMMEDIATELY SWITCH MODELS
# We have 3 models (llama-8b, llama-70b, qwen-32b) - use them all before waiting.

import logging
import os
import threading
import time
from collections import deque
from typing import Any

import requests

from agents.error_handling import (
    ErrorCategory,
    NonRetryableError,
    RetryableError,
    RetryConfig,
    get_circuit_breaker,
    with_retry,
)
from agents.llm.base import BaseLLMClient, LLMResponse
from agents.llm.failover import FailoverReason, get_failover_manager
from agents.llm.model_config import GROQ_MODELS, ModelTier

logger = logging.getLogger(__name__)


class PreemptiveSwitchNeeded(Exception):
    """Raised when rate limits suggest switching models before making a request."""

    pass


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
        "default": {"rpm": 20, "tpm": 6000, "rpd": 1000},
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
        self._request_times: dict[str, deque[float]] = {}
        self._token_usage: dict[str, deque[tuple]] = {}  # (timestamp, tokens)
        self._daily_requests: dict[str, int] = {}
        self._daily_reset: float = time.time()

        # Global rate limit state from headers
        self._remaining_requests: int | None = None
        self._remaining_tokens: int | None = None
        self._reset_requests: float | None = None
        self._reset_tokens: float | None = None

    def get_limits(self, model: str) -> dict:
        """Get rate limits for a model, with safety margin applied."""
        limits = self.MODEL_LIMITS.get(model, self.MODEL_LIMITS["default"])
        return {
            "rpm": int(limits["rpm"] * self.safety_margin),
            "tpm": int(limits["tpm"] * self.safety_margin),
            "rpd": int(limits["rpd"] * self.safety_margin),
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
            if "m" in reset_str:
                parts = reset_str.split("m")
                total_seconds += float(parts[0]) * 60
                reset_str = parts[1] if len(parts) > 1 else ""
            if "s" in reset_str:
                total_seconds += float(reset_str.replace("s", ""))
            return total_seconds
        except:
            return 60.0  # Default to 1 minute

    def wait_if_needed(self, model: str, estimated_tokens: int = 1000) -> tuple[float, bool]:
        """
        Check if we're close to rate limits.

        IMPORTANT: This method NO LONGER WAITS for rate limits.
        Instead, it returns information so the caller can switch to another model.
        We only wait if ALL models are rate limited.

        Args:
            model: The model being used
            estimated_tokens: Estimated tokens for this request

        Returns:
            Tuple of (wait_time_if_all_blocked, should_switch_model)
            - wait_time: Time to wait if all models are blocked (otherwise 0)
            - should_switch: True if we should preemptively switch to another model
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

            should_switch = False

            # Check daily limit first
            if rpd_current >= limits["rpd"]:
                logger.warning(
                    f"Daily request limit reached for {model} ({rpd_current}/{limits['rpd']})"
                )
                should_switch = True

            # Check RPM limit - if at limit, switch models instead of waiting
            if rpm_current >= limits["rpm"]:
                logger.info(f"RPM limit reached for {model}, signaling to switch models")
                should_switch = True

            # Check TPM limit - if approaching, switch models
            if tpm_current + estimated_tokens >= limits["tpm"]:
                logger.info(f"TPM limit approaching for {model}, signaling to switch models")
                should_switch = True

            # Check server-reported limits
            if self._remaining_requests is not None and self._remaining_requests <= 1:
                logger.info(f"Server reports low remaining requests for {model}")
                should_switch = True

            if self._remaining_tokens is not None and self._remaining_tokens < estimated_tokens:
                logger.info(f"Server reports low remaining tokens for {model}")
                should_switch = True

        # Return 0 wait time - we don't wait, we switch models
        return 0.0, should_switch

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
                "server_remaining_tokens": self._remaining_tokens,
            }


# Global rate limiter instance
_rate_limiter: TokenBucketRateLimiter | None = None


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

        # Failover manager for dynamic model switching
        self.failover_manager = get_failover_manager()

        # Circuit breaker for API calls
        self.circuit_breaker = get_circuit_breaker(
            "groq_api", failure_threshold=5, recovery_timeout=60.0
        )

        # Retry configuration - more aggressive for rate limits
        self.retry_config = RetryConfig(
            max_retries=self.config.max_retries,
            initial_delay=2.0,  # Start with longer delay
            max_delay=120.0,  # Allow longer waits for rate limits
            exponential_base=2.0,
            jitter=True,
        )

        logger.info(
            f"GroqClient initialized (user={self.user_id}, " f"budget=${self.user_budget:.4f})"
        )

    def get_provider_name(self) -> str:
        return self.PROVIDER_NAME

    def _select_model(
        self, task_type: str, complexity_hint: str | None = None, max_latency_ms: int | None = None
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
        rpd_usage_pct = (
            rate_status["rpd_used"] / rate_status["rpd_limit"]
            if rate_status["rpd_limit"] > 0
            else 0
        )
        if rpd_usage_pct >= 0.8 and tier == ModelTier.POWERFUL:
            logger.warning(
                f"70B model near daily limit ({rate_status['rpd_used']}/{rate_status['rpd_limit']}), using 8B"
            )
            tier = ModelTier.BALANCED

        # Budget check - downgrade if running low
        budget_usage = self.spent / self.user_budget if self.user_budget > 0 else 0
        if budget_usage >= 0.8:
            tier = ModelTier.FAST_CHEAP
            logger.warning(
                f"Budget constraint: Using {tier.value} (${self.spent:.4f}/${self.user_budget:.4f} spent)"
            )

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
        complexity_hint: str | None = None,
        max_latency_ms: int | None = None,
        **kwargs,
    ) -> str:
        """Send a chat request and return the response text."""
        response = self.chat_with_response(
            prompt=prompt,
            task_type=task_type,
            complexity_hint=complexity_hint,
            max_latency_ms=max_latency_ms,
            **kwargs,
        )
        return response.text

    def chat_with_response(
        self,
        prompt: str,
        task_type: str = "general",
        complexity_hint: str | None = None,
        max_latency_ms: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a chat request and return full response with metadata."""

        # Select model based on task type
        model_name, tier = self._select_model(task_type, complexity_hint, max_latency_ms)

        # Check failover manager for model availability
        estimated_tokens = len(prompt) // 4
        failover_decision = self.failover_manager.get_available_model(
            preferred_model=model_name,
            prompt_tokens=estimated_tokens,
        )

        # Use the model selected by failover manager
        actual_model = failover_decision.selected_model
        if failover_decision.was_failover:
            logger.warning(
                f"Failover active: {model_name} -> {actual_model} "
                f"(reason: {failover_decision.reason})"
            )

        logger.info(f"Groq routing: {task_type} -> {actual_model} (original: {model_name})")

        start_time = time.time()

        try:
            result, used_model = self._make_api_call_with_failover(actual_model, prompt, **kwargs)

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract token usage from response
            input_tokens = result.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = result.get("usage", {}).get("completion_tokens", 0)
            total_tokens = result.get("usage", {}).get("total_tokens", 0)

            # Calculate cost - find config for the actually used model
            cost = 0.0
            for _t, config in GROQ_MODELS.items():
                if config.name == used_model:
                    cost = config.estimate_cost(input_tokens, output_tokens)
                    break

            # Record usage
            self.spent += cost

            # Record success for the model that worked
            self.failover_manager.record_success(used_model)

            # Extract response text
            text = result["choices"][0]["message"]["content"]

            return LLMResponse(
                text=text,
                model=used_model,
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
                    "original_model": model_name,
                    "was_failover": used_model != model_name,
                },
            )

        except Exception as e:
            logger.error(f"Groq chat request failed after all failover attempts: {e}")
            raise

    def _make_api_call_with_failover(
        self, model: str, prompt: str, max_failover_attempts: int = 6, **kwargs
    ) -> tuple[dict[str, Any], str]:
        """
        Make API call with automatic failover on specific errors.

        KEY PRINCIPLE: NEVER WAIT FOR RATE LIMITS - USE ANOTHER MODEL IMMEDIATELY.
        We have 3 models, try all of them before giving up or waiting.

        This method handles:
        - Preemptive rate limit switches (before hitting 429)
        - 413 Payload Too Large: Failover to model with larger context
        - 429 Rate Limit: Immediate failover to alternative model
        - Server errors: Retry with different model
        - Payload chunking: If all models fail with 413, reduce payload size

        Args:
            model: Initial model to try
            prompt: The prompt to send
            max_failover_attempts: Maximum number of failover attempts (increased to 6 for 3 models x 2)
            **kwargs: Additional arguments for the API call

        Returns:
            Tuple of (API response dict, model name that succeeded)
        """
        current_model = model
        original_prompt = prompt
        attempt = 0
        last_error = None
        tried_models = set()
        all_models_tried_with_413 = False

        while attempt < max_failover_attempts:
            tried_models.add(current_model)

            try:
                # Allow preemptive switch only if we haven't tried all models
                allow_switch = len(tried_models) < len(self.failover_manager.MODEL_PRIORITY)
                result = self._make_single_api_call(
                    current_model, prompt, allow_preemptive_switch=allow_switch, **kwargs
                )
                return result, current_model

            except PreemptiveSwitchNeeded:
                # Rate limits approaching - switch to another model immediately
                failover = self.failover_manager.handle_failure(
                    model=current_model,
                    reason=FailoverReason.TPM_APPROACHING,
                    skip_cooldown=True,  # Don't penalize for preemptive switch
                )
                if (
                    failover.selected_model != current_model
                    and failover.selected_model not in tried_models
                ):
                    logger.info(
                        f"Preemptive switch: {current_model} -> {failover.selected_model} "
                        f"(rate limits approaching)"
                    )
                    current_model = failover.selected_model
                    attempt += 1
                    continue
                else:
                    # All models at rate limit - try current one anyway (might work)
                    logger.warning(f"All models at rate limits, trying {current_model} anyway")
                    try:
                        result = self._make_single_api_call(
                            current_model, prompt, allow_preemptive_switch=False, **kwargs
                        )
                        return result, current_model
                    except Exception as inner_e:
                        last_error = inner_e
                        attempt += 1
                        continue

            except NonRetryableError as e:
                # Check if this is a failover-able error
                if "413" in str(e) or "payload too large" in str(e).lower():
                    # Track that this model failed with 413
                    tried_models.add(current_model)

                    # Check if we've tried ALL models with 413
                    all_models = set(self.failover_manager.MODEL_PRIORITY)
                    all_tried_with_413 = tried_models >= all_models

                    if all_tried_with_413 and not all_models_tried_with_413:
                        # All models failed with 413 - need to reduce payload
                        logger.warning(
                            f"All {len(all_models)} models failed with 413. "
                            f"Attempting to reduce payload size."
                        )
                        reduced_prompt = self._reduce_prompt_size(original_prompt)
                        if len(reduced_prompt) < len(prompt):
                            logger.warning(
                                f"Reducing prompt from ~{len(prompt)//4} to "
                                f"~{len(reduced_prompt)//4} tokens"
                            )
                            prompt = reduced_prompt
                            # Reset and try all models again with smaller prompt
                            tried_models.clear()
                            current_model = model  # Start with original model
                            all_models_tried_with_413 = True
                            attempt += 1
                            continue
                        else:
                            # Can't reduce further - raise the error
                            logger.error("Cannot reduce prompt further. Failing.")
                            raise

                    # 413 error - try model with higher TPM
                    failover = self.failover_manager.handle_failure(
                        model=current_model,
                        reason=FailoverReason.PAYLOAD_TOO_LARGE_413,
                    )

                    # Check if failover gives us a different, untried model
                    if (
                        failover.selected_model != current_model
                        and failover.selected_model not in tried_models
                    ):
                        logger.info(
                            f"413 error on {current_model}, failing over to "
                            f"{failover.selected_model}"
                        )
                        current_model = failover.selected_model
                        attempt += 1
                        last_error = e
                        continue
                    else:
                        # Failover returned same model or already tried model
                        # Try to find any untried model
                        for m in self.failover_manager.MODEL_PRIORITY:
                            if m not in tried_models:
                                logger.info(f"413 error, trying next untried model: {m}")
                                current_model = m
                                attempt += 1
                                last_error = e
                                break
                        else:
                            # All models tried, will be caught in next iteration
                            attempt += 1
                            continue

                    continue

                # Not a failover-able error, re-raise
                raise

            except RetryableError as e:
                error_str = str(e).lower()

                # Determine failover reason
                if "429" in str(e) or "rate limit" in error_str:
                    failover_reason = FailoverReason.RATE_LIMIT_429
                    # Try to extract retry-after from error message
                    retry_after = self._extract_retry_after(str(e))
                elif "timeout" in error_str:
                    failover_reason = FailoverReason.TIMEOUT
                    retry_after = None
                elif "500" in str(e) or "502" in str(e) or "503" in str(e):
                    failover_reason = FailoverReason.SERVER_ERROR
                    retry_after = None
                else:
                    # Unknown retryable error - still try failover
                    failover_reason = FailoverReason.SERVER_ERROR
                    retry_after = None

                # Get failover decision
                failover = self.failover_manager.handle_failure(
                    model=current_model,
                    reason=failover_reason,
                    retry_after=retry_after,
                )

                if failover.selected_model != current_model:
                    logger.info(
                        f"{failover_reason.value} on {current_model}, "
                        f"failing over to {failover.selected_model}"
                    )
                    current_model = failover.selected_model
                    attempt += 1
                    last_error = e

                    # Small delay before trying new model
                    time.sleep(0.5)
                    continue
                elif failover.cooldown_remaining > 0:
                    # All models in cooldown, wait and retry
                    wait_time = min(failover.cooldown_remaining, 30.0)
                    logger.warning(f"All models in cooldown, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    attempt += 1
                    last_error = e
                    continue

                # No failover available, re-raise
                raise

            except Exception as e:
                # Unexpected error, try failover as server error
                logger.warning(f"Unexpected error on {current_model}: {e}")
                failover = self.failover_manager.handle_failure(
                    model=current_model,
                    reason=FailoverReason.SERVER_ERROR,
                )

                if failover.selected_model != current_model and attempt < max_failover_attempts - 1:
                    current_model = failover.selected_model
                    attempt += 1
                    last_error = e
                    continue

                raise

        # All attempts exhausted
        if last_error:
            raise last_error
        raise RuntimeError(
            f"Failed to get response after {max_failover_attempts} failover attempts"
        )

    def _make_single_api_call(
        self, model: str, prompt: str, allow_preemptive_switch: bool = True, **kwargs
    ) -> dict[str, Any]:
        """
        Make a single API call with rate limit checking but no failover.

        Args:
            model: Model to use
            prompt: The prompt to send
            allow_preemptive_switch: If True, raise special exception to switch models preemptively

        Returns:
            API response dict

        Raises:
            PreemptiveSwitchNeeded: If rate limits suggest switching models
            NonRetryableError: If prompt is too large for the model's TPM limit
        """

        # Estimate tokens for rate limiting
        max_response_tokens = min(kwargs.get("max_tokens", 1024), 1024)
        estimated_tokens = len(prompt) // 4 + max_response_tokens

        # Get TPM limit for this model (with safety margin already applied)
        model_limits = self.rate_limiter.get_limits(model)
        model_tpm = model_limits.get("tpm", 6000)

        # PRE-VALIDATE: Check if prompt is too large for this model's TPM BEFORE sending
        # This catches oversized prompts early and triggers failover to a model with higher limits
        if estimated_tokens > model_tpm:
            logger.warning(
                f"Prompt too large for {model}: ~{estimated_tokens} tokens > {model_tpm} TPM. "
                f"Will trigger failover to model with higher limits."
            )
            raise NonRetryableError(
                f"Payload too large (413): Prompt has ~{estimated_tokens} tokens but {model} "
                f"TPM limit is {model_tpm}. Consider reducing input size.",
                category=ErrorCategory.VALIDATION,
                severity="medium",
            )

        # Check rate limits - DON'T WAIT, just check if we should switch
        _, should_switch = self.rate_limiter.wait_if_needed(model, estimated_tokens)

        if should_switch and allow_preemptive_switch:
            # Signal to failover manager that we should preemptively switch
            raise PreemptiveSwitchNeeded(
                f"Rate limits approaching for {model}, preemptively switching"
            )

        # Make the actual API call (with circuit breaker)
        result = self.circuit_breaker.call(self._do_api_request, model, prompt, **kwargs)

        # Record the request for rate tracking
        total_tokens = result.get("usage", {}).get("total_tokens", estimated_tokens)
        self.rate_limiter.record_request(model, total_tokens)

        return result

    def _reduce_prompt_size(self, prompt: str, target_reduction: float = 0.5) -> str:
        """
        Reduce prompt size intelligently when facing 413 errors.

        Strategy:
        1. Try to identify and truncate the data/content section while keeping instructions
        2. Summarize long sections instead of just cutting them off
        3. Keep the most important parts (beginning and end) of long content

        Args:
            prompt: Original prompt
            target_reduction: Target reduction ratio (0.5 = reduce to 50%)

        Returns:
            Reduced prompt
        """
        original_length = len(prompt)
        target_length = int(original_length * target_reduction)

        # If prompt is small enough, return as-is
        if original_length <= 20000:  # ~5000 tokens
            return prompt

        logger.info(
            f"Reducing prompt from ~{original_length//4} tokens " f"to ~{target_length//4} tokens"
        )

        # Try to find patterns that indicate data sections
        # Common patterns: "Papers:", "Analyses:", "Content:", etc.
        data_markers = [
            "\n\nAnalyzed Papers:",
            "\nPapers:",
            "\nContent:",
            "\nData:",
            "---\n\n",
            "\n\n---",
        ]

        # Find where the main content/data starts
        split_point = -1
        for marker in data_markers:
            pos = prompt.find(marker)
            if pos > 0:
                split_point = pos
                break

        if split_point > 0:
            # Split into instructions and data
            instructions = prompt[:split_point]
            data = prompt[split_point:]

            # Calculate how much data we can keep
            available_for_data = target_length - len(instructions) - 200  # 200 for safety margin

            if available_for_data > 0:
                # Keep beginning and end of data, add truncation notice
                keep_start = available_for_data // 2
                keep_end = available_for_data // 2

                truncated_data = (
                    data[:keep_start]
                    + "\n\n[... Content truncated to fit token limits. "
                    + f"Removed ~{(len(data) - available_for_data)//4} tokens ...]\n\n"
                    + data[-keep_end:]
                )

                return instructions + truncated_data

        # Fallback: simple truncation from the middle
        keep_start = target_length // 2
        keep_end = target_length // 2

        return (
            prompt[:keep_start]
            + "\n\n[... Content truncated to fit token limits ...]\n\n"
            + prompt[-keep_end:]
        )

    def _extract_retry_after(self, error_message: str) -> float | None:
        """Extract retry-after value from error message."""
        import re

        # Look for patterns like "Retry after 60s" or "retry-after: 30"
        patterns = [
            r"retry.?after[:\s]+(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*s(?:econds?)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, error_message.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        return None

    def _make_api_call_with_retry(self, model: str, prompt: str, **kwargs) -> dict[str, Any]:
        """Legacy method - now calls _make_api_call_with_failover for backward compatibility."""
        result, _ = self._make_api_call_with_failover(model, prompt, **kwargs)
        return result

    def _do_api_request(self, model: str, prompt: str, **kwargs) -> dict[str, Any]:
        """Execute the actual API request to Groq."""

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

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
            "messages": [{"role": "user", "content": prompt}],
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
                self.BASE_URL, headers=headers, json=data, timeout=self.config.timeout
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
                    severity="low",
                )
            elif status_code == 413:
                # Payload too large - this is a prompt size issue
                # Try to extract more info
                try:
                    error_detail = (
                        e.response.json().get("error", {}).get("message", "Payload too large")
                    )
                except:
                    error_detail = "Payload too large"
                raise NonRetryableError(
                    f"Payload too large ({status_code}): {error_detail}. "
                    f"Prompt has ~{len(prompt)//4} tokens. Consider reducing input size.",
                    category=ErrorCategory.VALIDATION,
                    severity="medium",
                )
            elif status_code >= 500:
                raise RetryableError(
                    f"Groq server error ({status_code}): {e}",
                    category=ErrorCategory.SERVER_ERROR,
                    severity="medium",
                )
            elif status_code in (401, 403):
                raise NonRetryableError(
                    f"Authentication failed ({status_code}): Check GROQ_API_KEY",
                    category=ErrorCategory.CLIENT_ERROR,
                    severity="high",
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
                    severity="medium",
                )
            else:
                raise NonRetryableError(
                    f"Client error ({status_code}): {e}",
                    category=ErrorCategory.CLIENT_ERROR,
                    severity="medium",
                )

        except requests.exceptions.Timeout as e:
            raise RetryableError(
                f"Request timeout: {e}", category=ErrorCategory.TIMEOUT, severity="low"
            )

        except requests.exceptions.ConnectionError as e:
            raise RetryableError(
                f"Network error: {e}", category=ErrorCategory.NETWORK, severity="medium"
            )

        except Exception as e:
            logger.error(f"Unexpected error in Groq request: {e}", exc_info=True)
            raise RetryableError(
                f"Unexpected error: {e}", category=ErrorCategory.UNKNOWN, severity="medium"
            )

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics including rate limit and failover status."""
        # Get rate limit status for all models
        rate_stats = {
            "llama-3.1-8b-instant": self.rate_limiter.get_status("llama-3.1-8b-instant"),
            "llama-3.3-70b-versatile": self.rate_limiter.get_status("llama-3.3-70b-versatile"),
            "qwen/qwen3-32b": self.rate_limiter.get_status("qwen/qwen3-32b"),
        }

        # Get failover status
        failover_stats = self.failover_manager.get_status()

        return {
            "provider": self.PROVIDER_NAME,
            "user_id": self.user_id,
            "budget": self.user_budget,
            "spent": self.spent,
            "remaining": self.user_budget - self.spent,
            "usage_percent": (self.spent / self.user_budget * 100) if self.user_budget > 0 else 0,
            "rate_limits": rate_stats,
            "failover_status": failover_stats,
        }

    def reset_budget(self, new_budget: float | None = None) -> None:
        """Reset budget tracking."""
        if new_budget is not None:
            self.user_budget = new_budget
        self.spent = 0.0
        logger.info(f"Groq budget reset to ${self.user_budget:.4f}")

    def is_available(self) -> bool:
        """Check if Groq is configured and available."""
        return bool(self.api_key)
