# Model Failover Manager
# Handles dynamic model switching when primary models fail due to rate limits or payload issues
# Production-grade failover system for Groq models
#
# KEY PRINCIPLE: NEVER WAIT FOR RATE LIMITS - IMMEDIATELY USE ANOTHER MODEL
# We have 3 models available, use them all before waiting.

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agents.llm.model_config import GROQ_MODELS, ModelConfig, ModelTier

logger = logging.getLogger(__name__)


class FailoverReason(StrEnum):
    """Reasons for model failover."""

    RATE_LIMIT_429 = "rate_limit_429"  # 429 Too Many Requests
    PAYLOAD_TOO_LARGE_413 = "payload_413"  # 413 Payload Too Large
    CONTEXT_LENGTH = "context_length"  # Context window exceeded
    SERVER_ERROR = "server_error"  # 5xx errors
    TIMEOUT = "timeout"  # Request timeout
    COOLDOWN = "cooldown"  # Model in cooldown period
    TPM_APPROACHING = "tpm_approaching"  # TPM limit approaching (preemptive switch)


@dataclass
class ModelCooldownState:
    """Tracks cooldown state for a specific model."""

    model_name: str
    cooldown_until: float = 0.0  # Unix timestamp when cooldown ends
    consecutive_failures: int = 0
    last_failure_reason: FailoverReason | None = None
    last_failure_time: float = 0.0

    def is_in_cooldown(self) -> bool:
        """Check if model is currently in cooldown."""
        return time.time() < self.cooldown_until

    def remaining_cooldown(self) -> float:
        """Get remaining cooldown time in seconds."""
        return max(0, self.cooldown_until - time.time())


@dataclass
class FailoverDecision:
    """Result of a failover decision."""

    original_model: str
    selected_model: str
    reason: str
    was_failover: bool
    cooldown_remaining: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelFailoverManager:
    """
    Manages dynamic model failover based on errors and rate limits.

    Failover Strategy:
    1. 413 Payload Too Large on 8B model -> Upgrade to 70B (larger context handling)
    2. 429 Rate Limit on any Llama model -> Switch to Qwen temporarily
    3. 429 Rate Limit on Qwen -> Wait and retry with original model
    4. Consecutive failures -> Increase cooldown exponentially

    The manager tracks cooldown periods for each model and automatically
    selects the best available model based on current state.

    Thread-safe implementation for production use.
    """

    # Cooldown durations in seconds
    # IMPORTANT: Groq rate limits reset every 60 seconds (per model).
    # The `retry-after` header typically shows 2-60 seconds.
    # Keep cooldowns SHORT to allow quick model cycling.
    BASE_COOLDOWN = 10.0  # Base cooldown after rate limit (was 30s, now 10s)
    MAX_COOLDOWN = 60.0  # Max cooldown = Groq's 1-minute reset window (was 300s)
    COOLDOWN_MULTIPLIER = 1.5  # Gentler backoff since we cycle models (was 2.0)

    # Failover chains: what model to use when the primary fails
    # Key: (original_model, failure_reason) -> fallback_model
    #
    # TPM LIMITS (critical for 413 handling):
    # - llama-3.1-8b-instant: 6K TPM
    # - llama-3.3-70b-versatile: 12K TPM (HIGHEST - use for large payloads!)
    # - qwen/qwen3-32b: 6K TPM
    #
    # 413 STRATEGY: Always try llama-70b (12K TPM) before qwen (6K TPM)
    FAILOVER_CHAIN = {
        # 413 Payload Too Large: upgrade to model with higher TPM
        # 8b (6K) -> 70b (12K) - upgrade to higher TPM
        ("llama-3.1-8b-instant", FailoverReason.PAYLOAD_TOO_LARGE_413): "llama-3.3-70b-versatile",
        # 70b (12K) -> qwen (6K) - 70b has highest TPM, if still fails try qwen
        ("llama-3.3-70b-versatile", FailoverReason.PAYLOAD_TOO_LARGE_413): "qwen/qwen3-32b",
        # qwen (6K) -> 70b (12K) - try 70b since it has higher TPM
        ("qwen/qwen3-32b", FailoverReason.PAYLOAD_TOO_LARGE_413): "llama-3.3-70b-versatile",
        # 429 Rate Limit: switch to different model family immediately
        ("llama-3.1-8b-instant", FailoverReason.RATE_LIMIT_429): "qwen/qwen3-32b",
        ("llama-3.3-70b-versatile", FailoverReason.RATE_LIMIT_429): "qwen/qwen3-32b",
        ("qwen/qwen3-32b", FailoverReason.RATE_LIMIT_429): "llama-3.1-8b-instant",
        # TPM approaching - preemptive switch to avoid hitting limits
        ("llama-3.1-8b-instant", FailoverReason.TPM_APPROACHING): "llama-3.3-70b-versatile",
        ("llama-3.3-70b-versatile", FailoverReason.TPM_APPROACHING): "qwen/qwen3-32b",
        ("qwen/qwen3-32b", FailoverReason.TPM_APPROACHING): "llama-3.1-8b-instant",
        # Context length issues: upgrade to larger context model
        ("llama-3.1-8b-instant", FailoverReason.CONTEXT_LENGTH): "llama-3.3-70b-versatile",
        ("qwen/qwen3-32b", FailoverReason.CONTEXT_LENGTH): "llama-3.3-70b-versatile",
        # Server errors: try different model
        ("llama-3.1-8b-instant", FailoverReason.SERVER_ERROR): "qwen/qwen3-32b",
        ("llama-3.3-70b-versatile", FailoverReason.SERVER_ERROR): "qwen/qwen3-32b",
        ("qwen/qwen3-32b", FailoverReason.SERVER_ERROR): "llama-3.1-8b-instant",
    }

    # Model priority order for general selection
    # Note: For synthesis/summarization, prefer 70b and qwen over 8b
    MODEL_PRIORITY = [
        "llama-3.1-8b-instant",  # Highest priority (best rate limits) for simple tasks
        "llama-3.3-70b-versatile",  # Powerful - use for complex tasks
        "qwen/qwen3-32b",  # Good quality fallback
    ]

    # Models preferred for synthesis tasks (need better quality/larger context)
    SYNTHESIS_MODEL_PRIORITY = [
        "llama-3.3-70b-versatile",  # Best for synthesis (12K TPM)
        "qwen/qwen3-32b",  # Good quality alternative
        "llama-3.1-8b-instant",  # Last resort for synthesis
    ]

    # Cooldown for preemptive switches (when TPM approaching, not actual 429)
    # This prevents immediately retrying the same model on the next request
    PREEMPTIVE_COOLDOWN = 15.0  # 15 seconds - gives other models priority

    def __init__(self):
        """Initialize the failover manager."""
        self._lock = threading.Lock()
        self._cooldown_states: dict[str, ModelCooldownState] = {}

        # Initialize states for all known models
        for model_name in self.MODEL_PRIORITY:
            self._cooldown_states[model_name] = ModelCooldownState(model_name=model_name)

        logger.info("ModelFailoverManager initialized with models: %s", self.MODEL_PRIORITY)

    def _check_and_clear_expired_cooldowns(self) -> int:
        """
        Check all models and clear expired cooldowns.

        This ensures models become available again once their rate limit window passes.
        Groq rate limits reset every 60 seconds, so this is critical for model cycling.

        Returns:
            Number of models that became available
        """
        cleared_count = 0
        current_time = time.time()

        for model_name, state in self._cooldown_states.items():
            if state.cooldown_until > 0 and current_time >= state.cooldown_until:
                # Cooldown expired - model is available again
                logger.info(
                    f"Model {model_name} cooldown expired, marking as available "
                    f"(was in cooldown for {state.last_failure_reason.value if state.last_failure_reason else 'unknown'} reason)"
                )
                state.cooldown_until = 0.0
                # Reset consecutive failures to give model a fresh start
                # (keep count if last failure was recent - within 30 seconds)
                if current_time - state.last_failure_time > 30:
                    state.consecutive_failures = 0
                cleared_count += 1

        return cleared_count

    def get_model_for_synthesis(
        self,
        prompt_tokens: int = 0,
        max_wait_seconds: float = 30.0,
    ) -> FailoverDecision:
        """
        Get the best available model for synthesis tasks.

        Synthesis tasks require higher quality output, so we prefer:
        1. llama-3.3-70b-versatile (best quality, 12K TPM)
        2. qwen/qwen3-32b (good quality alternative)
        3. llama-3.1-8b-instant (last resort)

        This method will wait briefly for a good model rather than immediately
        falling back to 8b.

        Args:
            prompt_tokens: Estimated tokens in the prompt
            max_wait_seconds: Maximum time to wait for a good model

        Returns:
            FailoverDecision with the selected model
        """
        start_time = time.time()

        while time.time() - start_time < max_wait_seconds:
            with self._lock:
                # First, clear any expired cooldowns
                self._check_and_clear_expired_cooldowns()

                # Try synthesis-preferred models in order
                for model_name in self.SYNTHESIS_MODEL_PRIORITY:
                    state = self._cooldown_states.get(model_name)

                    if state and not state.is_in_cooldown():
                        # Check context window
                        model_config = self._get_model_config(model_name)
                        if model_config and prompt_tokens > 0:
                            if prompt_tokens > model_config.context_window * 0.9:
                                continue  # Model can't handle this prompt size

                        logger.info(f"Selected {model_name} for synthesis task")
                        return FailoverDecision(
                            original_model=self.SYNTHESIS_MODEL_PRIORITY[0],
                            selected_model=model_name,
                            reason="Best available model for synthesis",
                            was_failover=model_name != self.SYNTHESIS_MODEL_PRIORITY[0],
                        )

                # All models in cooldown - find shortest wait
                shortest_wait = float("inf")
                for model_name in self.SYNTHESIS_MODEL_PRIORITY:
                    state = self._cooldown_states.get(model_name)
                    if state:
                        remaining = state.remaining_cooldown()
                        shortest_wait = min(shortest_wait, remaining)

            # If shortest wait is very short, wait for it
            if shortest_wait <= 5.0:
                logger.info(f"Waiting {shortest_wait:.1f}s for model to become available")
                time.sleep(min(shortest_wait + 0.5, 5.0))
            else:
                # Wait a bit and check again (models might recover)
                time.sleep(2.0)

        # Timeout - return whatever is available
        with self._lock:
            self._check_and_clear_expired_cooldowns()

            # Return best available or shortest wait
            for model_name in self.SYNTHESIS_MODEL_PRIORITY:
                state = self._cooldown_states.get(model_name)
                if state and not state.is_in_cooldown():
                    return FailoverDecision(
                        original_model=self.SYNTHESIS_MODEL_PRIORITY[0],
                        selected_model=model_name,
                        reason="Best available after timeout",
                        was_failover=True,
                    )

            # All still in cooldown - return shortest wait model
            shortest_wait = float("inf")
            best_model = self.SYNTHESIS_MODEL_PRIORITY[0]
            for model_name in self.SYNTHESIS_MODEL_PRIORITY:
                state = self._cooldown_states.get(model_name)
                if state and state.remaining_cooldown() < shortest_wait:
                    shortest_wait = state.remaining_cooldown()
                    best_model = model_name

            logger.warning(
                f"All synthesis models in cooldown after {max_wait_seconds}s wait. "
                f"Using {best_model} (cooldown: {shortest_wait:.1f}s remaining)"
            )
            return FailoverDecision(
                original_model=self.SYNTHESIS_MODEL_PRIORITY[0],
                selected_model=best_model,
                reason="All models in cooldown, using shortest wait",
                was_failover=True,
                cooldown_remaining=shortest_wait,
            )

    def get_available_model(
        self,
        preferred_model: str,
        prompt_tokens: int = 0,
    ) -> FailoverDecision:
        """
        Get the best available model, considering cooldowns.

        Args:
            preferred_model: The originally requested model
            prompt_tokens: Estimated tokens in the prompt (for context check)

        Returns:
            FailoverDecision with the selected model
        """
        with self._lock:
            # First, clear any expired cooldowns to ensure accurate availability
            self._check_and_clear_expired_cooldowns()

            # Check if preferred model is available
            state = self._cooldown_states.get(preferred_model)
            if state and not state.is_in_cooldown():
                # Check context window if we have token info
                model_config = self._get_model_config(preferred_model)
                if model_config and prompt_tokens > 0:
                    if prompt_tokens > model_config.context_window * 0.9:
                        # Prompt is too large for this model, try larger one
                        return self._find_larger_context_model(preferred_model, prompt_tokens)

                return FailoverDecision(
                    original_model=preferred_model,
                    selected_model=preferred_model,
                    reason="Model available",
                    was_failover=False,
                )

            # Preferred model is in cooldown, find alternative
            cooldown_remaining = state.remaining_cooldown() if state else 0

            # Try models in priority order
            for model_name in self.MODEL_PRIORITY:
                if model_name == preferred_model:
                    continue

                alt_state = self._cooldown_states.get(model_name)
                if alt_state and not alt_state.is_in_cooldown():
                    logger.info(
                        f"Failover: {preferred_model} in cooldown ({cooldown_remaining:.1f}s), "
                        f"using {model_name}"
                    )
                    return FailoverDecision(
                        original_model=preferred_model,
                        selected_model=model_name,
                        reason="Primary model in cooldown, using fallback",
                        was_failover=True,
                        cooldown_remaining=cooldown_remaining,
                        metadata={
                            "original_cooldown_reason": (
                                state.last_failure_reason.value
                                if state and state.last_failure_reason
                                else None
                            )
                        },
                    )

            # All models in cooldown - return the one with shortest remaining time
            shortest_cooldown = float("inf")
            best_model = preferred_model

            for model_name in self.MODEL_PRIORITY:
                model_state = self._cooldown_states.get(model_name)
                if model_state:
                    remaining = model_state.remaining_cooldown()
                    if remaining < shortest_cooldown:
                        shortest_cooldown = remaining
                        best_model = model_name

            logger.warning(
                f"All models in cooldown. Using {best_model} "
                f"(cooldown ends in {shortest_cooldown:.1f}s)"
            )

            return FailoverDecision(
                original_model=preferred_model,
                selected_model=best_model,
                reason="All models in cooldown, using shortest wait",
                was_failover=True,
                cooldown_remaining=shortest_cooldown,
            )

    def handle_failure(
        self,
        model: str,
        reason: FailoverReason,
        retry_after: float | None = None,
        skip_cooldown: bool = False,
        preemptive_cooldown: bool = False,
    ) -> FailoverDecision:
        """
        Handle a model failure and determine the next model to use.

        IMPORTANT: For rate limits and 413 errors, we immediately switch models
        without waiting. We only wait if ALL THREE models are unavailable.

        Args:
            model: The model that failed
            reason: Reason for failure
            retry_after: Server-suggested retry time (from headers)
            skip_cooldown: If True, don't put model in cooldown (for preemptive switches)
            preemptive_cooldown: If True, apply short cooldown (for TPM approaching)

        Returns:
            FailoverDecision with the fallback model to use.
            If metadata contains 'needs_chunking': True, the caller should reduce payload size.
        """
        with self._lock:
            # Update cooldown state for failed model
            state = self._cooldown_states.get(model)
            if not state:
                state = ModelCooldownState(model_name=model)
                self._cooldown_states[model] = state

            state.consecutive_failures += 1
            state.last_failure_reason = reason
            state.last_failure_time = time.time()

            # Calculate cooldown duration - shorter for rate limits since we have alternatives
            if preemptive_cooldown:
                # Preemptive switch - use short cooldown so model isn't immediately retried
                cooldown_duration = self.PREEMPTIVE_COOLDOWN
            elif skip_cooldown:
                cooldown_duration = 0.0
            elif retry_after and retry_after > 0:
                # Use server suggestion but cap it - we have other models!
                cooldown_duration = min(retry_after, 60.0)
            else:
                # Shorter exponential backoff - we want to try other models quickly
                cooldown_duration = min(
                    self.BASE_COOLDOWN
                    * (self.COOLDOWN_MULTIPLIER ** (state.consecutive_failures - 1)),
                    self.MAX_COOLDOWN,
                )

            # Set cooldown unless explicitly skipped (preemptive_cooldown DOES set cooldown)
            if not skip_cooldown or preemptive_cooldown:
                state.cooldown_until = time.time() + cooldown_duration

            logger.warning(
                f"Model {model} failed ({reason.value}). "
                f"Cooldown: {cooldown_duration:.1f}s. "
                f"Consecutive failures: {state.consecutive_failures}"
            )

            # Determine fallback model
            fallback_key = (model, reason)
            fallback_model = self.FAILOVER_CHAIN.get(fallback_key)

            # Special case: 413 on all models means we need to chunk the payload
            if fallback_model is None and reason == FailoverReason.PAYLOAD_TOO_LARGE_413:
                logger.warning(
                    "All models failed with 413 for this payload. "
                    "Signaling to reduce payload size."
                )
                return FailoverDecision(
                    original_model=model,
                    selected_model=model,  # Return same model - caller will reduce size
                    reason="All models failed with 413 - need to reduce payload",
                    was_failover=False,
                    cooldown_remaining=0,
                    metadata={
                        "needs_chunking": True,
                        "failure_reason": reason.value,
                    },
                )

            if fallback_model:
                # Check if fallback is available
                fallback_state = self._cooldown_states.get(fallback_model)
                if fallback_state and fallback_state.is_in_cooldown():
                    # Fallback also in cooldown, find any available model
                    return self._find_any_available_model(model, reason)

                logger.info(f"Failover: {model} -> {fallback_model} (reason: {reason.value})")

                return FailoverDecision(
                    original_model=model,
                    selected_model=fallback_model,
                    reason=f"Failover due to {reason.value}",
                    was_failover=True,
                    cooldown_remaining=cooldown_duration,
                    metadata={
                        "failure_reason": reason.value,
                        "consecutive_failures": state.consecutive_failures,
                    },
                )
            else:
                # No specific failover rule, find any available
                return self._find_any_available_model(model, reason)

    def record_success(self, model: str) -> None:
        """
        Record a successful request, resetting consecutive failure count.

        Args:
            model: The model that succeeded
        """
        with self._lock:
            state = self._cooldown_states.get(model)
            if state:
                state.consecutive_failures = 0
                logger.debug(f"Model {model} succeeded, reset failure count")

    def clear_cooldown(self, model: str) -> None:
        """
        Manually clear cooldown for a model.

        Args:
            model: The model to clear cooldown for
        """
        with self._lock:
            state = self._cooldown_states.get(model)
            if state:
                state.cooldown_until = 0.0
                state.consecutive_failures = 0
                logger.info(f"Cleared cooldown for model {model}")

    def get_status(self) -> dict[str, Any]:
        """Get current status of all models."""
        with self._lock:
            status = {}
            for model_name, state in self._cooldown_states.items():
                status[model_name] = {
                    "in_cooldown": state.is_in_cooldown(),
                    "cooldown_remaining": state.remaining_cooldown(),
                    "consecutive_failures": state.consecutive_failures,
                    "last_failure_reason": (
                        state.last_failure_reason.value if state.last_failure_reason else None
                    ),
                    "last_failure_time": state.last_failure_time,
                }
            return status

    def wait_for_model(
        self,
        preferred_model: str,
        max_wait_seconds: float = 120.0,
        check_interval: float = 5.0,
    ) -> FailoverDecision:
        """
        Wait for a specific model to become available (for critical tasks like final synthesis).

        This is used when a high-quality result is critical and we're willing to wait
        for the best model to become available rather than using a degraded fallback.

        Strategy:
        1. First try the preferred model (e.g., llama-3.3-70b for final synthesis)
        2. If in cooldown, wait up to max_wait_seconds for it to become available
        3. If still unavailable, try any available model in priority order
        4. Returns the best available model after waiting

        Args:
            preferred_model: The model we want (e.g., "llama-3.3-70b-versatile")
            max_wait_seconds: Maximum time to wait for the preferred model
            check_interval: How often to check if model is available

        Returns:
            FailoverDecision with the selected model
        """
        start_time = time.time()
        elapsed = 0.0

        while elapsed < max_wait_seconds:
            with self._lock:
                # Clear expired cooldowns before checking
                self._check_and_clear_expired_cooldowns()

                # Check if preferred model is available
                state = self._cooldown_states.get(preferred_model)
                if state and not state.is_in_cooldown():
                    logger.info(
                        f"wait_for_model: {preferred_model} is available after "
                        f"{elapsed:.1f}s wait"
                    )
                    return FailoverDecision(
                        original_model=preferred_model,
                        selected_model=preferred_model,
                        reason="Preferred model available after wait",
                        was_failover=False,
                        metadata={"wait_time": elapsed},
                    )

                remaining_cooldown = state.remaining_cooldown() if state else 0

            # If cooldown will end within our wait window, wait for it
            if remaining_cooldown > 0 and remaining_cooldown < (max_wait_seconds - elapsed):
                wait_time = min(remaining_cooldown + 1.0, check_interval)
                logger.info(
                    f"wait_for_model: {preferred_model} in cooldown for "
                    f"{remaining_cooldown:.1f}s more, waiting {wait_time:.1f}s"
                )
                time.sleep(wait_time)
                elapsed = time.time() - start_time
                continue

            # Check other models in priority order
            # Give preference to powerful models for critical tasks
            priority_for_critical = [
                "llama-3.3-70b-versatile",  # Powerful - best for synthesis
                "qwen/qwen3-32b",  # Medium quality
                "llama-3.1-8b-instant",  # Fast but lower quality
            ]

            with self._lock:
                # Clear expired cooldowns again before checking alternatives
                self._check_and_clear_expired_cooldowns()

                for model_name in priority_for_critical:
                    if model_name == preferred_model:
                        continue
                    alt_state = self._cooldown_states.get(model_name)
                    if alt_state and not alt_state.is_in_cooldown():
                        logger.info(
                            f"wait_for_model: Using alternative {model_name} "
                            f"(preferred {preferred_model} still in cooldown)"
                        )
                        return FailoverDecision(
                            original_model=preferred_model,
                            selected_model=model_name,
                            reason=f"Used alternative while waiting for {preferred_model}",
                            was_failover=True,
                            metadata={"wait_time": elapsed},
                        )

            # All models in cooldown, wait a bit and retry
            time.sleep(check_interval)
            elapsed = time.time() - start_time

        # Timeout - return best available model
        logger.warning(
            f"wait_for_model: Timeout ({max_wait_seconds}s) waiting for {preferred_model}"
        )

        # Return whatever is available (or will be available soonest)
        return self.get_available_model(preferred_model)

    def _find_any_available_model(
        self,
        failed_model: str,
        reason: FailoverReason,
    ) -> FailoverDecision:
        """Find any available model when specific failover rules don't apply."""
        # Clear expired cooldowns first - this is critical for model cycling!
        self._check_and_clear_expired_cooldowns()

        for model_name in self.MODEL_PRIORITY:
            if model_name == failed_model:
                continue

            state = self._cooldown_states.get(model_name)
            if state and not state.is_in_cooldown():
                return FailoverDecision(
                    original_model=failed_model,
                    selected_model=model_name,
                    reason=f"Failover to available model after {reason.value}",
                    was_failover=True,
                    metadata={"failure_reason": reason.value},
                )

        # All models in cooldown, return one with shortest wait
        shortest_wait = float("inf")
        best_model = failed_model

        for model_name in self.MODEL_PRIORITY:
            state = self._cooldown_states.get(model_name)
            if state:
                remaining = state.remaining_cooldown()
                if remaining < shortest_wait:
                    shortest_wait = remaining
                    best_model = model_name

        return FailoverDecision(
            original_model=failed_model,
            selected_model=best_model,
            reason="All models in cooldown, using shortest wait",
            was_failover=True,
            cooldown_remaining=shortest_wait,
        )

    def _find_larger_context_model(
        self,
        current_model: str,
        prompt_tokens: int,
    ) -> FailoverDecision:
        """Find a model with larger context window."""
        # Models sorted by context window (largest first)
        models_by_context = sorted(
            self.MODEL_PRIORITY,
            key=lambda m: (
                self._get_model_config(m).context_window if self._get_model_config(m) else 0
            ),
            reverse=True,
        )

        for model_name in models_by_context:
            if model_name == current_model:
                continue

            config = self._get_model_config(model_name)
            state = self._cooldown_states.get(model_name)

            if config and prompt_tokens < config.context_window * 0.9:
                if state and not state.is_in_cooldown():
                    return FailoverDecision(
                        original_model=current_model,
                        selected_model=model_name,
                        reason="Upgraded to larger context model",
                        was_failover=True,
                        metadata={"prompt_tokens": prompt_tokens},
                    )

        # No suitable model found, return original
        return FailoverDecision(
            original_model=current_model,
            selected_model=current_model,
            reason="No larger context model available",
            was_failover=False,
        )

    def _get_model_config(self, model_name: str) -> ModelConfig | None:
        """Get model configuration by name."""
        for _tier, config in GROQ_MODELS.items():
            if config.name == model_name:
                return config
        return None


# Global singleton instance
_failover_manager: ModelFailoverManager | None = None


def get_failover_manager() -> ModelFailoverManager:
    """Get or create the global failover manager instance."""
    global _failover_manager
    if _failover_manager is None:
        _failover_manager = ModelFailoverManager()
    return _failover_manager
