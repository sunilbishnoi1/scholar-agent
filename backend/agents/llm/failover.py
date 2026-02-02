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
from enum import Enum
from typing import Any

from agents.llm.model_config import GROQ_MODELS, ModelConfig, ModelTier

logger = logging.getLogger(__name__)


class FailoverReason(str, Enum):
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
    BASE_COOLDOWN = 30.0  # Base cooldown after rate limit
    MAX_COOLDOWN = 300.0  # Maximum cooldown (5 minutes)
    COOLDOWN_MULTIPLIER = 2.0  # Exponential backoff multiplier

    # Failover chains: what model to use when the primary fails
    # Key: (original_model, failure_reason) -> fallback_model
    # EXTENDED: Now includes 413 failover for ALL models, including Qwen
    FAILOVER_CHAIN = {
        # 413 Payload Too Large: upgrade to larger model OR try smaller chunks
        ("llama-3.1-8b-instant", FailoverReason.PAYLOAD_TOO_LARGE_413): "llama-3.3-70b-versatile",
        ("llama-3.3-70b-versatile", FailoverReason.PAYLOAD_TOO_LARGE_413): "qwen/qwen3-32b",
        ("qwen/qwen3-32b", FailoverReason.PAYLOAD_TOO_LARGE_413): None,  # Signal to chunk/reduce
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
    MODEL_PRIORITY = [
        "llama-3.1-8b-instant",  # Highest priority (best rate limits)
        "llama-3.3-70b-versatile",  # Powerful but limited quota
        "qwen/qwen3-32b",  # Fallback option
    ]

    def __init__(self):
        """Initialize the failover manager."""
        self._lock = threading.Lock()
        self._cooldown_states: dict[str, ModelCooldownState] = {}

        # Initialize states for all known models
        for model_name in self.MODEL_PRIORITY:
            self._cooldown_states[model_name] = ModelCooldownState(model_name=model_name)

        logger.info("ModelFailoverManager initialized with models: %s", self.MODEL_PRIORITY)

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
            if skip_cooldown:
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

            if not skip_cooldown:
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

    def _find_any_available_model(
        self,
        failed_model: str,
        reason: FailoverReason,
    ) -> FailoverDecision:
        """Find any available model when specific failover rules don't apply."""
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
