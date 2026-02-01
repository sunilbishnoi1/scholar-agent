# Model Configuration
# Centralized configuration for all supported LLM models across providers
# This makes it easy to add new models or update pricing

from dataclasses import dataclass
from enum import Enum


class ModelTier(str, Enum):
    """
    Model tiers representing different cost/performance tradeoffs.
    Used by the router to select appropriate models for tasks.
    """
    FAST_CHEAP = "fast_cheap"       # Simple tasks, lowest cost
    BALANCED = "balanced"            # Most tasks, balanced cost/performance
    POWERFUL = "powerful"            # Complex reasoning, highest quality


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str                           # Model name/ID
    provider: str                       # Provider name (groq, gemini, openai)
    tier: ModelTier                     # Performance tier
    cost_per_1k_input: float           # Cost per 1K input tokens (USD)
    cost_per_1k_output: float          # Cost per 1K output tokens (USD)
    base_latency_ms: int               # Estimated base latency
    context_window: int                # Max context window size
    supports_streaming: bool = True    # Whether streaming is supported

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost


# =============================================================================
# GROQ MODELS
# Rate limits (Free tier, from Groq docs):
#   - llama-3.3-70b-versatile: 30 RPM, 1K RPD, 12K TPM
#   - llama-3.1-8b-instant: 30 RPM, 14.4K RPD, 6K TPM
#   - qwen/qwen3-32b: 60 RPM, 1K RPD, 6K TPM
# Note: llama-3.1-8b-instant has MUCH higher daily limits (14.4K vs 1K)
# =============================================================================

GROQ_MODELS: dict[ModelTier, ModelConfig] = {
    ModelTier.FAST_CHEAP: ModelConfig(
        name="llama-3.1-8b-instant",      # Best for high volume: 14.4K RPD!
        provider="groq",
        tier=ModelTier.FAST_CHEAP,
        cost_per_1k_input=0.00005,        # $0.05 / 1M tokens
        cost_per_1k_output=0.00008,       # $0.08 / 1M tokens
        base_latency_ms=100,              # Very fast!
        context_window=131072,
    ),
    ModelTier.BALANCED: ModelConfig(
        name="llama-3.1-8b-instant",       # Use 8B for balanced too (better rate limits)
        provider="groq",
        tier=ModelTier.BALANCED,
        cost_per_1k_input=0.00005,
        cost_per_1k_output=0.00008,
        base_latency_ms=100,
        context_window=131072,
    ),
    ModelTier.POWERFUL: ModelConfig(
        name="llama-3.3-70b-versatile",   # 70B only for truly complex tasks
        provider="groq",
        tier=ModelTier.POWERFUL,
        cost_per_1k_input=0.00059,        # $0.59 / 1M tokens
        cost_per_1k_output=0.00079,       # $0.79 / 1M tokens
        base_latency_ms=300,
        context_window=131072,
    ),
}


# =============================================================================
# GEMINI MODELS (Free tier: 15 RPM for flash, 2 RPM for pro)
# Google's Gemini models
# =============================================================================

GEMINI_MODELS: dict[ModelTier, ModelConfig] = {
    ModelTier.FAST_CHEAP: ModelConfig(
        name="gemini-2.0-flash-lite",
        provider="gemini",
        tier=ModelTier.FAST_CHEAP,
        cost_per_1k_input=0.00001,      # Very cheap
        cost_per_1k_output=0.00001,
        base_latency_ms=200,
        context_window=1048576,
    ),
    ModelTier.BALANCED: ModelConfig(
        name="gemini-2.0-flash",
        provider="gemini",
        tier=ModelTier.BALANCED,
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0001,
        base_latency_ms=500,
        context_window=1048576,
    ),
    ModelTier.POWERFUL: ModelConfig(
        name="gemini-1.5-pro",
        provider="gemini",
        tier=ModelTier.POWERFUL,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.001,
        base_latency_ms=1500,
        context_window=2097152,
    ),
}


# =============================================================================
# OPENAI MODELS (for future use)
# =============================================================================

OPENAI_MODELS: dict[ModelTier, ModelConfig] = {
    ModelTier.FAST_CHEAP: ModelConfig(
        name="gpt-4o-mini",
        provider="openai",
        tier=ModelTier.FAST_CHEAP,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        base_latency_ms=300,
        context_window=128000,
    ),
    ModelTier.BALANCED: ModelConfig(
        name="gpt-4o",
        provider="openai",
        tier=ModelTier.BALANCED,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        base_latency_ms=600,
        context_window=128000,
    ),
    ModelTier.POWERFUL: ModelConfig(
        name="gpt-4o",
        provider="openai",
        tier=ModelTier.POWERFUL,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        base_latency_ms=800,
        context_window=128000,
    ),
}


# =============================================================================
# Provider to Models mapping
# =============================================================================

PROVIDER_MODELS = {
    "groq": GROQ_MODELS,
    "gemini": GEMINI_MODELS,
    "openai": OPENAI_MODELS,
}


def get_model_config(provider: str, tier: ModelTier) -> ModelConfig | None:
    """
    Get model configuration for a provider and tier.
    
    Args:
        provider: Provider name (groq, gemini, openai)
        tier: Model tier
        
    Returns:
        ModelConfig or None if not found
    """
    models = PROVIDER_MODELS.get(provider.lower())
    if models:
        return models.get(tier)
    return None


def get_all_models_for_provider(provider: str) -> dict[ModelTier, ModelConfig]:
    """Get all model configurations for a provider."""
    return PROVIDER_MODELS.get(provider.lower(), {})
