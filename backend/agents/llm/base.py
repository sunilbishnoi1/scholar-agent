# Base LLM Client Interface
# Abstract base class that all LLM providers must implement
# This allows easy switching between providers (Groq, Gemini, OpenAI, etc.)

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    api_key: str | None = None
    user_budget: float = 1.0
    user_id: str = "default"
    enable_router: bool = True
    timeout: int = 60
    max_retries: int = 5

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
    Abstract base class for all LLM provider clients.

    All LLM providers (Groq, Gemini, OpenAI, Anthropic, etc.) must implement
    this interface to work with our agent system.

    This abstraction enables:
    - Easy provider switching via configuration
    - Consistent API across all providers
    - Unified error handling and retry logic
    - Cost and usage tracking
    """

    def __init__(self, config: LLMConfig | None = None):
        """
        Initialize the LLM client.

        Args:
            config: Configuration for the client
        """
        self.config = config or LLMConfig()
        self._setup_client()

    @abstractmethod
    def _setup_client(self) -> None:
        """
        Provider-specific setup logic.
        Called during initialization to set up API keys, clients, etc.
        """
        pass

    @abstractmethod
    def chat(
        self,
        prompt: str,
        task_type: str = "general",
        complexity_hint: str | None = None,
        max_latency_ms: int | None = None,
        **kwargs,
    ) -> str:
        """
        Send a chat request to the LLM.

        This is the main method that agents use to interact with the LLM.

        Args:
            prompt: The prompt text to send
            task_type: Type of task for routing (e.g., "synthesis", "extract_keywords")
            complexity_hint: Optional complexity hint ("low", "medium", "high")
            max_latency_ms: Optional maximum latency requirement
            **kwargs: Provider-specific parameters

        Returns:
            Response text from the LLM

        Raises:
            RetryableError: For transient errors after retries exhausted
            NonRetryableError: For permanent errors (e.g., invalid API key)
        """
        pass

    @abstractmethod
    def chat_with_response(
        self,
        prompt: str,
        task_type: str = "general",
        complexity_hint: str | None = None,
        max_latency_ms: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a chat request and get full response with metadata.

        Use this when you need token counts, cost estimates, etc.

        Args:
            prompt: The prompt text to send
            task_type: Type of task for routing
            complexity_hint: Optional complexity hint
            max_latency_ms: Optional maximum latency requirement
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with text and metadata
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name (e.g., 'groq', 'gemini', 'openai')."""
        pass

    @abstractmethod
    def get_usage_stats(self) -> dict[str, Any]:
        """
        Get usage statistics for this client.

        Returns:
            Dictionary with usage stats (spent, remaining budget, etc.)
        """
        pass

    @abstractmethod
    def reset_budget(self, new_budget: float | None = None) -> None:
        """
        Reset budget tracking.

        Args:
            new_budget: Optional new budget amount
        """
        pass

    def is_available(self) -> bool:
        """
        Check if the provider is available (API key configured, etc.).

        Returns:
            True if the provider can be used
        """
        return self.config.api_key is not None

    def estimate_cost(self, prompt: str, task_type: str = "general") -> float:
        """
        Estimate the cost of a request before making it.

        Args:
            prompt: The prompt text
            task_type: Type of task

        Returns:
            Estimated cost in USD
        """
        # Default implementation - providers should override
        tokens = len(prompt) // 4  # Rough estimate
        return tokens * 0.00001  # Very rough cost estimate
