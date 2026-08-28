"""
Unified LLM Provider Package.
Exports BaseLLMClient, ModelTier, providers, factory methods, and structured output utilities.
"""

from agents.llm.base import BaseLLMClient, LLMConfig, LLMResponse, ModelTier
from agents.llm.deepseek_provider import DeepSeekProvider
from agents.llm.factory import (
    GeminiClient,
    LLMProvider,
    MockLLMClient,
    clear_client_cache,
    get_available_providers,
    get_best_available_provider,
    get_default_provider,
    get_llm_client,
    set_default_provider,
)
from agents.llm.gemini_provider import GeminiProvider
from agents.llm.groq_client import GroqClient
from agents.llm.model_config import (
    DEEPSEEK_MODELS,
    GEMINI_MODELS,
    GROQ_MODELS,
    ModelConfig,
    get_model_config,
)
from agents.llm.structured_output import (
    StructuredOutputError,
    clean_json_markdown,
    extract_json_substring,
    parse_and_validate,
    repair_json_syntax,
    to_gemini_schema,
    to_openai_response_format,
)

__all__ = [
    # Base
    "BaseLLMClient",
    "LLMConfig",
    "LLMResponse",
    "ModelTier",
    # Providers
    "GeminiProvider",
    "DeepSeekProvider",
    "GroqClient",
    "MockLLMClient",
    "GeminiClient",
    # Factory
    "LLMProvider",
    "get_llm_client",
    "get_default_provider",
    "set_default_provider",
    "get_available_providers",
    "get_best_available_provider",
    "clear_client_cache",
    # Structured Outputs
    "StructuredOutputError",
    "parse_and_validate",
    "clean_json_markdown",
    "extract_json_substring",
    "repair_json_syntax",
    "to_gemini_schema",
    "to_openai_response_format",
    # Model Config
    "ModelConfig",
    "GEMINI_MODELS",
    "DEEPSEEK_MODELS",
    "GROQ_MODELS",
    "get_model_config",
]
