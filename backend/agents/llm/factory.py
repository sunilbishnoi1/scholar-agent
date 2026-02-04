# LLM Client Factory
# Central factory for creating LLM clients based on configuration
# This is the main entry point for getting an LLM client

import logging
import os
from enum import StrEnum
from typing import Any

from agents.llm.base import BaseLLMClient, LLMConfig

logger = logging.getLogger(__name__)


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    GROQ = "groq"
    GEMINI = "gemini"
    OPENAI = "openai"  # For future use

    @classmethod
    def from_string(cls, value: str) -> "LLMProvider":
        """Convert string to provider enum."""
        try:
            return cls(value.lower())
        except ValueError:
            logger.warning(f"Unknown provider '{value}', defaulting to GROQ")
            return cls.GROQ


# Global default provider - can be changed at runtime
_default_provider: LLMProvider = LLMProvider.GROQ

# Cache for client instances (singleton pattern per provider)
_client_cache: dict[str, BaseLLMClient] = {}


def get_default_provider() -> LLMProvider:
    """
    Get the default LLM provider.

    Priority:
    1. Environment variable LLM_PROVIDER
    2. Global default (GROQ)

    Returns:
        The default provider
    """
    env_provider = os.environ.get("LLM_PROVIDER", "").lower()
    if env_provider:
        return LLMProvider.from_string(env_provider)
    return _default_provider


def set_default_provider(provider: LLMProvider) -> None:
    """
    Set the default LLM provider.

    Args:
        provider: The provider to use as default
    """
    global _default_provider
    _default_provider = provider
    logger.info(f"Default LLM provider set to: {provider.value}")


def get_llm_client(
    provider: LLMProvider | None = None,
    config: LLMConfig | None = None,
    force_new: bool = False,
    **kwargs,
) -> BaseLLMClient:
    """
    Get an LLM client instance.

    This is the main factory function for getting LLM clients.
    By default, it returns a cached singleton instance.

    Args:
        provider: LLM provider to use (defaults to environment/global default)
        config: Configuration for the client
        force_new: If True, create a new instance instead of using cache
        **kwargs: Additional config options passed to LLMConfig

    Returns:
        An LLM client instance

    Example:
        # Use default provider (Groq by default, or from LLM_PROVIDER env var)
        client = get_llm_client()

        # Explicitly use Groq
        client = get_llm_client(provider=LLMProvider.GROQ)

        # With custom config
        client = get_llm_client(
            provider=LLMProvider.GROQ,
            config=LLMConfig(user_budget=2.0, user_id="user123")
        )

        # Quick config via kwargs
        client = get_llm_client(user_budget=2.0, user_id="user123")
    """
    # Determine provider
    if provider is None:
        provider = get_default_provider()

    # Build config
    if config is None:
        config = LLMConfig(**kwargs) if kwargs else LLMConfig()

    # Cache key based on provider and user_id
    cache_key = f"{provider.value}:{config.user_id}"

    # Return cached instance if available and not forcing new
    if not force_new and cache_key in _client_cache:
        cached = _client_cache[cache_key]
        logger.debug(f"Returning cached {provider.value} client for {config.user_id}")
        return cached

    # Create new instance
    client = _create_client(provider, config)

    # Cache the instance
    _client_cache[cache_key] = client

    return client


def _create_client(provider: LLMProvider, config: LLMConfig) -> BaseLLMClient:
    """Create a new LLM client instance."""

    if provider == LLMProvider.GROQ:
        from agents.llm.groq_client import GroqClient

        return GroqClient(config)

    elif provider == LLMProvider.GEMINI:
        from agents.llm.gemini import GeminiProvider

        return GeminiProvider(config)

    elif provider == LLMProvider.OPENAI:
        # Future: implement OpenAI provider
        raise NotImplementedError("OpenAI provider not yet implemented")

    else:
        raise ValueError(f"Unknown provider: {provider}")


def clear_client_cache() -> None:
    """Clear the client cache. Useful for testing."""
    global _client_cache
    _client_cache = {}
    logger.info("LLM client cache cleared")


def get_available_providers() -> list[LLMProvider]:
    """
    Get list of providers that are configured and available.

    Returns:
        List of available providers
    """
    available = []

    if os.environ.get("GROQ_API_KEY"):
        available.append(LLMProvider.GROQ)

    if os.environ.get("GEMINI_API_KEY"):
        available.append(LLMProvider.GEMINI)

    if os.environ.get("OPENAI_API_KEY"):
        available.append(LLMProvider.OPENAI)

    return available


def get_best_available_provider() -> LLMProvider | None:
    """
    Get the best available provider based on configuration.

    Priority:
    1. LLM_PROVIDER environment variable (if set and available)
    2. Groq (best free tier)
    3. Gemini
    4. OpenAI

    Returns:
        Best available provider or None if none configured
    """
    available = get_available_providers()

    if not available:
        return None

    # Check env var preference
    env_provider = os.environ.get("LLM_PROVIDER", "").lower()
    if env_provider:
        try:
            preferred = LLMProvider(env_provider)
            if preferred in available:
                return preferred
        except ValueError:
            pass

    # Priority order
    priority = [LLMProvider.GROQ, LLMProvider.GEMINI, LLMProvider.OPENAI]

    for provider in priority:
        if provider in available:
            return provider

    return available[0] if available else None


# =============================================================================
# Backward Compatibility - GeminiClient alias
# =============================================================================


class GeminiClient(BaseLLMClient):
    """
    Backward-compatible GeminiClient class.

    This is a wrapper that maintains the old GeminiClient API but uses
    the new provider architecture under the hood. It now defaults to
    using Groq (better free tier) but can be configured to use any provider.

    For new code, use get_llm_client() instead.
    """

    def __init__(
        self,
        api_key: str | None = None,
        user_budget: float = 1.0,
        user_id: str = "default",
        enable_router: bool = True,
        provider: str | None = None,
    ):
        """
        Initialize a backward-compatible client.

        Args:
            api_key: API key (deprecated, use env vars instead)
            user_budget: Budget for this user session
            user_id: User identifier for tracking
            enable_router: Whether to enable smart model routing (ignored, always enabled)
            provider: Override provider (defaults to best available)
        """
        # Determine which provider to use
        if provider:
            llm_provider = LLMProvider.from_string(provider)
        else:
            llm_provider = get_best_available_provider()
            if llm_provider is None:
                # Default to Groq if nothing configured
                llm_provider = LLMProvider.GROQ
                logger.warning(
                    "No API keys configured! Set GROQ_API_KEY or GEMINI_API_KEY. "
                    "Defaulting to Groq."
                )

        # Build config
        config = LLMConfig(
            api_key=api_key, user_budget=user_budget, user_id=user_id, enable_router=enable_router
        )

        # Get the actual client
        self._client = get_llm_client(provider=llm_provider, config=config, force_new=True)
        self.config = config

        logger.info(f"GeminiClient (compat) initialized with provider: {llm_provider.value}")

    def _setup_client(self) -> None:
        """Not used - setup handled by delegated client."""
        pass

    def chat(
        self,
        prompt: str,
        task_type: str = "general",
        complexity_hint: str | None = None,
        max_latency_ms: int | None = None,
        **kwargs,
    ) -> str:
        """Send a chat request. Delegates to the underlying provider."""
        return self._client.chat(
            prompt=prompt,
            task_type=task_type,
            complexity_hint=complexity_hint,
            max_latency_ms=max_latency_ms,
            **kwargs,
        )

    def chat_with_response(
        self,
        prompt: str,
        task_type: str = "general",
        complexity_hint: str | None = None,
        max_latency_ms: int | None = None,
        **kwargs,
    ):
        """Send a chat request with full response. Delegates to the underlying provider."""
        return self._client.chat_with_response(
            prompt=prompt,
            task_type=task_type,
            complexity_hint=complexity_hint,
            max_latency_ms=max_latency_ms,
            **kwargs,
        )

    def get_provider_name(self) -> str:
        """Get the actual provider name being used."""
        return self._client.get_provider_name()

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return self._client.get_usage_stats()

    def reset_budget(self, new_budget: float | None = None) -> None:
        """Reset budget tracking."""
        self._client.reset_budget(new_budget)

    def is_available(self) -> bool:
        """Check if the provider is available."""
        return self._client.is_available()
