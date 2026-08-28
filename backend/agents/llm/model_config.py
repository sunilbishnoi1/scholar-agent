"""
Model Tier Mappings and Configuration Constants for Gemini, DeepSeek, Groq, and OpenAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.llm.base import ModelTier


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    provider: str
    tier: ModelTier
    cost_per_1k_input: float
    cost_per_1k_output: float
    base_latency_ms: int
    context_window: int
    supports_streaming: bool = True
    max_rpm: int = 60

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for request tokens."""
        return ((input_tokens / 1000) * self.cost_per_1k_input) + ((output_tokens / 1000) * self.cost_per_1k_output)


GEMINI_MODELS: dict[ModelTier, ModelConfig] = {
    ModelTier.FAST: ModelConfig(
        name="gemini-3.5-flash-lite",
        provider="gemini",
        tier=ModelTier.FAST,
        cost_per_1k_input=0.00005,
        cost_per_1k_output=0.0002,
        base_latency_ms=200,
        context_window=1048576,
        max_rpm=14,
    ),
    ModelTier.REASONING: ModelConfig(
        name="gemini-3.5-flash-lite",
        provider="gemini",
        tier=ModelTier.REASONING,
        cost_per_1k_input=0.00005,
        cost_per_1k_output=0.0002,
        base_latency_ms=400,
        context_window=1048576,
        max_rpm=14,
    ),
    ModelTier.STRUCTURED_NLI: ModelConfig(
        name="gemini-3.5-flash-lite",
        provider="gemini",
        tier=ModelTier.STRUCTURED_NLI,
        cost_per_1k_input=0.00005,
        cost_per_1k_output=0.0002,
        base_latency_ms=200,
        context_window=1048576,
        max_rpm=14,
    ),
}

DEEPSEEK_MODELS: dict[ModelTier, ModelConfig] = {
    ModelTier.FAST: ModelConfig(
        name="deepseek-chat",
        provider="deepseek",
        tier=ModelTier.FAST,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        base_latency_ms=400,
        context_window=65536,
    ),
    ModelTier.REASONING: ModelConfig(
        name="deepseek-reasoner",
        provider="deepseek",
        tier=ModelTier.REASONING,
        cost_per_1k_input=0.00055,
        cost_per_1k_output=0.00219,
        base_latency_ms=1200,
        context_window=65536,
    ),
    ModelTier.STRUCTURED_NLI: ModelConfig(
        name="deepseek-chat",
        provider="deepseek",
        tier=ModelTier.STRUCTURED_NLI,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        base_latency_ms=400,
        context_window=65536,
    ),
}

GROQ_MODELS: dict[ModelTier, ModelConfig] = {
    ModelTier.FAST: ModelConfig(
        name="llama-3.1-8b-instant",
        provider="groq",
        tier=ModelTier.FAST,
        cost_per_1k_input=0.00005,
        cost_per_1k_output=0.00008,
        base_latency_ms=100,
        context_window=131072,
    ),
    ModelTier.REASONING: ModelConfig(
        name="llama-3.3-70b-versatile",
        provider="groq",
        tier=ModelTier.REASONING,
        cost_per_1k_input=0.00059,
        cost_per_1k_output=0.00079,
        base_latency_ms=400,
        context_window=131072,
    ),
    ModelTier.STRUCTURED_NLI: ModelConfig(
        name="llama-3.1-8b-instant",
        provider="groq",
        tier=ModelTier.STRUCTURED_NLI,
        cost_per_1k_input=0.00005,
        cost_per_1k_output=0.00008,
        base_latency_ms=100,
        context_window=131072,
    ),
}

PROVIDER_MODELS: dict[str, dict[ModelTier, ModelConfig]] = {
    "gemini": GEMINI_MODELS,
    "deepseek": DEEPSEEK_MODELS,
    "groq": GROQ_MODELS,
}


def get_model_config(provider: str, tier: str | ModelTier) -> ModelConfig | None:
    """Get model configuration for provider and tier."""
    model_tier = ModelTier.from_str(tier)
    provider_dict = PROVIDER_MODELS.get(provider.lower())
    if provider_dict:
        return provider_dict.get(model_tier)
    return None
