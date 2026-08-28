"""
LLM Client Factory with Mock Support and Provider Resolution.
"""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel

from agents.llm.base import BaseLLMClient, LLMConfig, LLMResponse, ModelTier

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMProvider(StrEnum):
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    OPENAI = "openai"
    MOCK = "mock"

    @classmethod
    def from_string(cls, value: str | None) -> LLMProvider:
        if not value:
            return cls.GEMINI
        try:
            return cls(value.lower().strip())
        except ValueError:
            logger.warning(f"Unknown provider '{value}', defaulting to GEMINI")
            return cls.GEMINI


_client_cache: dict[str, BaseLLMClient] = {}
_default_provider: LLMProvider = LLMProvider.GEMINI


class MockLLMClient(BaseLLMClient):
    """
    Deterministic Mock LLM Client for offline unit and integration tests.
    """

    PROVIDER_NAME = "mock"

    def _setup_client(self) -> None:
        self.mock_text_responses: list[str] = []
        self.mock_structured_responses: dict[type, Any] = {}
        self.call_history: list[dict[str, Any]] = []

    def get_provider_name(self) -> str:
        return self.PROVIDER_NAME

    def is_available(self) -> bool:
        return True

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        model_tier: str | ModelTier = ModelTier.FAST,
        **kwargs: Any,
    ) -> str:
        self.call_history.append({
            "type": "text",
            "prompt": prompt,
            "system_prompt": system_prompt,
            "tier": model_tier,
            "kwargs": kwargs,
        })
        if self.mock_text_responses:
            return self.mock_text_responses.pop(0)
        return "# Synthetic Scientific Response\n\nMock LLM synthesis response text for testing."

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str = "",
        model_tier: str | ModelTier = ModelTier.FAST,
        **kwargs: Any,
    ) -> T:
        self.call_history.append({
            "type": "structured",
            "schema": schema.__name__,
            "prompt": prompt,
            "tier": model_tier,
            "kwargs": kwargs,
        })
        if schema in self.mock_structured_responses:
            return self.mock_structured_responses[schema]
        return schema.model_construct()

    def get_usage_stats(self) -> dict[str, Any]:
        return {"provider": "mock", "spent": 0.0, "calls": len(self.call_history)}

    def reset_budget(self, new_budget: float | None = None) -> None:
        pass


class GeminiClient(BaseLLMClient):
    """
    Backward-compatible client adapter.
    Delegates to the active default provider or GeminiProvider.
    """

    def _setup_client(self) -> None:
        self._delegate = get_llm_client(provider=LLMProvider.GEMINI, config=self.config)

    def get_provider_name(self) -> str:
        return "gemini"

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        model_tier: str | ModelTier = ModelTier.FAST,
        **kwargs: Any,
    ) -> str:
        return self._delegate.generate_text(
            prompt, system_prompt=system_prompt, model_tier=model_tier, **kwargs
        )

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str = "",
        model_tier: str | ModelTier = ModelTier.FAST,
        **kwargs: Any,
    ) -> T:
        return self._delegate.generate_structured(
            prompt, schema=schema, system_prompt=system_prompt, model_tier=model_tier, **kwargs
        )

    def chat(self, prompt: str, **kwargs: Any) -> str:
        return self._delegate.chat(prompt, **kwargs)

    def chat_with_response(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return self._delegate.chat_with_response(prompt, **kwargs)

    def is_available(self) -> bool:
        return self._delegate.is_available()

    def get_usage_stats(self) -> dict[str, Any]:
        return self._delegate.get_usage_stats()

    def reset_budget(self, new_budget: float | None = None) -> None:
        self._delegate.reset_budget(new_budget)


def get_default_provider() -> LLMProvider:
    global _default_provider
    if _default_provider != LLMProvider.GEMINI:
        return _default_provider
    env_provider = os.environ.get("LLM_PROVIDER", "").lower()
    if env_provider:
        return LLMProvider.from_string(env_provider)
    best = get_best_available_provider()
    if best != LLMProvider.MOCK:
        return best
    return _default_provider


def set_default_provider(provider: LLMProvider | str) -> None:
    global _default_provider
    if isinstance(provider, str):
        provider = LLMProvider.from_string(provider)
    _default_provider = provider
    logger.info(f"Default LLM provider set to: {provider.value}")


def get_available_providers() -> list[LLMProvider]:
    available: list[LLMProvider] = [LLMProvider.MOCK]
    if os.environ.get("GEMINI_API_KEY"):
        available.append(LLMProvider.GEMINI)
    if os.environ.get("DEEPSEEK_API_KEY"):
        available.append(LLMProvider.DEEPSEEK)
    if os.environ.get("GROQ_API_KEY"):
        available.append(LLMProvider.GROQ)
    if os.environ.get("OPENAI_API_KEY"):
        available.append(LLMProvider.OPENAI)
    return available


def get_best_available_provider() -> LLMProvider:
    available = get_available_providers()
    env_provider = os.environ.get("LLM_PROVIDER", "").lower()
    if env_provider:
        try:
            preferred = LLMProvider(env_provider)
            if preferred in available:
                return preferred
        except ValueError:
            pass

    priority = [LLMProvider.GEMINI, LLMProvider.DEEPSEEK, LLMProvider.GROQ, LLMProvider.OPENAI, LLMProvider.MOCK]
    for p in priority:
        if p in available:
            return p
    return LLMProvider.MOCK


def get_llm_client(
    provider: LLMProvider | str | None = None,
    config: LLMConfig | None = None,
    force_new: bool = False,
    **kwargs: Any,
) -> BaseLLMClient:
    """
    Main factory entrypoint for retrieving cached or new LLM clients.
    """
    if provider is None:
        provider = get_default_provider()
    elif isinstance(provider, str):
        provider = LLMProvider.from_string(provider)

    if config is None:
        config = LLMConfig(**kwargs) if kwargs else LLMConfig()

    cache_key = f"{provider.value}:{config.user_id}"
    if not force_new and cache_key in _client_cache:
        return _client_cache[cache_key]

    client = _create_client(provider, config)
    _client_cache[cache_key] = client
    return client


def _create_client(provider: LLMProvider, config: LLMConfig) -> BaseLLMClient:
    if provider == LLMProvider.GEMINI:
        from agents.llm.gemini_provider import GeminiProvider
        return GeminiProvider(config)
    elif provider == LLMProvider.DEEPSEEK:
        from agents.llm.deepseek_provider import DeepSeekProvider
        return DeepSeekProvider(config)
    elif provider == LLMProvider.GROQ:
        from agents.llm.groq_client import GroqClient
        return GroqClient(config)
    elif provider == LLMProvider.MOCK:
        return MockLLMClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def clear_client_cache() -> None:
    global _client_cache
    _client_cache = {}
    logger.info("LLM client cache cleared")
