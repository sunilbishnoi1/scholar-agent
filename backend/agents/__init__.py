# Agents Module
# Exposes the new LangGraph-based agent architecture

from agents.analyzer import PaperAnalyzerAgent as LegacyAnalyzerAgent
from agents.analyzer_agent import PaperAnalyzerAgent
from agents.base import BaseAgent, ToolEnabledAgent

# Error Handling (Phase 1 - Enhanced Retry Logic)
from agents.error_handling import (
    CircuitBreaker,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    NonRetryableError,
    RetryableError,
    RetryConfig,
    get_circuit_breaker,
    with_retry,
)

# New LLM Provider Architecture (Phase 2 - Multi-provider support)
from agents.llm import (
    GEMINI_MODELS,
    GROQ_MODELS,
    # Base classes
    BaseLLMClient,
    GeminiProvider,
    # Providers
    GroqClient,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    ModelConfig,
    # Model configuration
    ModelTier,
    get_default_provider,
    # Factory (main entry point)
    get_llm_client,
    get_model_config,
    set_default_provider,
)

# Backward-compatible GeminiClient (now uses Groq by default)
from agents.llm.factory import GeminiClient

# Model Router (Phase 1 - Cost-Aware Selection)
from agents.model_router import RoutingDecision, SmartModelRouter, get_router
from agents.observability import (
    AgentTrace,
    AgentTracer,
    LLMTrace,
    StructuredLogger,
    trace_llm,
    trace_step,
    tracer,
)
from agents.orchestrator import ResearchOrchestrator, create_orchestrator

# Legacy imports for backward compatibility
from agents.planner import ResearchPlannerAgent as LegacyPlannerAgent
from agents.planner_agent import ResearchPlannerAgent
from agents.quality_checker_agent import QualityCheckerAgent
from agents.retriever_agent import PaperRetrieverAgent
from agents.state import (
    AgentMessage,
    AgentResult,
    AgentState,
    AgentType,
    PaperData,
    create_initial_state,
)
from agents.synthesizer import SynthesisExecutorAgent as LegacySynthesizerAgent
from agents.synthesizer_agent import SynthesisExecutorAgent

__all__ = [
    # State
    "AgentState",
    "AgentType",
    "AgentResult",
    "AgentMessage",
    "PaperData",
    "create_initial_state",

    # Base classes
    "BaseAgent",
    "ToolEnabledAgent",

    # Agents
    "ResearchPlannerAgent",
    "PaperRetrieverAgent",
    "PaperAnalyzerAgent",
    "SynthesisExecutorAgent",
    "QualityCheckerAgent",

    # Orchestrator
    "ResearchOrchestrator",
    "create_orchestrator",

    # Observability
    "AgentTracer",
    "tracer",
    "trace_llm",
    "trace_step",
    "LLMTrace",
    "AgentTrace",
    "StructuredLogger",

    # Model Router (Legacy - Phase 1)
    "SmartModelRouter",
    "RoutingDecision",
    "get_router",

    # Error Handling (Phase 1)
    "with_retry",
    "RetryConfig",
    "CircuitBreaker",
    "get_circuit_breaker",
    "ErrorCategory",
    "ErrorSeverity",
    "RetryableError",
    "NonRetryableError",
    "ErrorContext",

    # LLM Providers (Phase 2 - New Architecture)
    "BaseLLMClient",
    "LLMResponse",
    "LLMConfig",
    "GroqClient",
    "GeminiProvider",
    "get_llm_client",
    "LLMProvider",
    "get_default_provider",
    "set_default_provider",
    "ModelTier",
    "ModelConfig",
    "GROQ_MODELS",
    "GEMINI_MODELS",
    "get_model_config",

    # Legacy/Client (backward compatible)
    "GeminiClient",
]
