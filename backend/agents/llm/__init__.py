# LLM Provider Module
# Abstraction layer for multiple LLM providers (Groq, Gemini, OpenAI, etc.)
# Makes it easy to switch providers with minimal code changes

from agents.llm.base import BaseLLMClient, LLMConfig, LLMResponse
from agents.llm.factory import (
    GeminiClient,  # Backward-compatible wrapper
    LLMProvider,
    clear_client_cache,
    get_available_providers,
    get_best_available_provider,
    get_default_provider,
    get_llm_client,
    set_default_provider,
)
from agents.llm.gemini import GeminiProvider
from agents.llm.groq_client import GroqClient
from agents.llm.model_config import (
    GEMINI_MODELS,
    GROQ_MODELS,
    ModelConfig,
    ModelTier,
    get_model_config,
)

__all__ = [
    # Base
    "BaseLLMClient",
    "LLMResponse",
    "LLMConfig",

    # Providers
    "GroqClient",
    "GeminiProvider",

    # Factory
    "get_llm_client",
    "LLMProvider",
    "get_default_provider",
    "set_default_provider",
    "get_available_providers",
    "get_best_available_provider",
    "clear_client_cache",

    # Backward compatible
    "GeminiClient",

    # Model Config
    "ModelTier",
    "ModelConfig",
    "GROQ_MODELS",
    "GEMINI_MODELS",
    "get_model_config",
]
