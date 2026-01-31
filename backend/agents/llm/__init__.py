# LLM Provider Module
# Abstraction layer for multiple LLM providers (Groq, Gemini, OpenAI, etc.)
# Makes it easy to switch providers with minimal code changes

from agents.llm.base import BaseLLMClient, LLMResponse, LLMConfig
from agents.llm.groq_client import GroqClient
from agents.llm.gemini import GeminiProvider
from agents.llm.factory import (
    get_llm_client,
    LLMProvider,
    get_default_provider,
    set_default_provider,
    GeminiClient,  # Backward-compatible wrapper
    get_available_providers,
    get_best_available_provider,
    clear_client_cache,
)
from agents.llm.model_config import (
    ModelTier,
    ModelConfig,
    GROQ_MODELS,
    GEMINI_MODELS,
    get_model_config
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
