"""
Base LLM Client Interface.
Defines the abstract BaseLLMClient interface, ModelTier enum, LLMConfig, and LLMResponse.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ModelTier(StrEnum):
    """
    Model tiers representing performance and reasoning capabilities.
    """

    FAST = "fast"
    REASONING = "reasoning"
    STRUCTURED_NLI = "structured_nli"

    @classmethod
    def from_str(cls, value: str | ModelTier | None) -> ModelTier:
        """Convert string to ModelTier with backward compatibility support."""
        if value is None:
            return cls.FAST
        if isinstance(value, cls):
            return value

        val_lower = str(value).lower().strip()
        if val_lower in ("fast", "fast_cheap", "cheap", "balanced"):
            return cls.FAST
        elif val_lower in ("reasoning", "powerful", "deep", "r1", "thinking"):
            return cls.REASONING
        elif val_lower in ("structured_nli", "nli", "auditor", "audit", "verification"):
            return cls.STRUCTURED_NLI
        return cls.FAST


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    api_key: str | None = None
    base_url: str | None = None
    user_budget: float = 1.0
    user_id: str = "default"
    enable_router: bool = True
    timeout: int = 60
    max_retries: int = 5
    temperature: float = 0.7
    max_tokens: int = 4096

    # Provider-specific settings
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    text: str
    model: str
    provider: str

    # Token usage
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    # Cost tracking
    estimated_cost: float | None = None

    # Metadata
    latency_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLMClient(ABC):
    """
    Abstract base class for all unified LLM providers.
    Supports high-capacity text generation and native/fallback structured outputs.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._setup_client()

    @abstractmethod
    def _setup_client(self) -> None:
        """Provider-specific initialization."""
        pass

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        model_tier: str | ModelTier = ModelTier.FAST,
        **kwargs: Any,
    ) -> str:
        """
        Generate raw text response from the model.

        Args:
            prompt: User/task prompt.
            system_prompt: Optional system instruction.
            model_tier: Target model tier (FAST, REASONING, STRUCTURED_NLI).
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            Generated response string.
        """
        return self.chat(prompt, system_prompt=system_prompt, model_tier=model_tier, **kwargs)

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str = "",
        model_tier: str | ModelTier = ModelTier.FAST,
        **kwargs: Any,
    ) -> T:
        """
        Generate structured output adhering strictly to a Pydantic schema.

        Args:
            prompt: User/task prompt.
            schema: Target Pydantic BaseModel subclass.
            system_prompt: Optional system instruction.
            model_tier: Target model tier (FAST, REASONING, STRUCTURED_NLI).
            **kwargs: Additional parameters.

        Returns:
            Instantiated and validated Pydantic model instance of type T.
        """
        from agents.llm.structured_output import parse_and_validate

        raw_text = self.generate_text(prompt, system_prompt=system_prompt, model_tier=model_tier, **kwargs)
        return parse_and_validate(raw_text, schema)

    def chat(self, prompt: str, **kwargs: Any) -> str:
        """
        Send a chat request to the LLM.
        """
        return self.generate_text(prompt, **kwargs)

    def chat_with_response(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Send a chat request and get full response container with metadata.
        """
        text = self.chat(prompt, **kwargs)
        return LLMResponse(
            text=text,
            model="default",
            provider=self.get_provider_name(),
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
            total_tokens=(len(prompt) + len(text)) // 4,
            estimated_cost=self.estimate_cost(prompt),
        )

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider identifier ('gemini', 'deepseek', 'groq', 'mock')."""
        pass

    def is_available(self) -> bool:
        """Check if provider credentials and network configurations are available."""
        return self.config.api_key is not None

    @abstractmethod
    def get_usage_stats(self) -> dict[str, Any]:
        """Return usage and cost statistics."""
        pass

    @abstractmethod
    def reset_budget(self, new_budget: float | None = None) -> None:
        """Reset budget tracking."""
        pass

    def estimate_cost(self, prompt: str, model_tier: str | ModelTier = ModelTier.FAST) -> float:
        """Estimate token cost for prompt."""
        tokens = len(prompt) // 4
        return (tokens / 1000) * 0.0001
