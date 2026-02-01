# Observability Module - Tracing and Metrics for Agents
# Provides structured logging, tracing, and performance metrics

import functools
import json
import logging
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class LogLevel(str, Enum):
    """Log levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LLMTrace:
    """Structured trace data for LLM calls."""
    trace_id: str
    agent_name: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    prompt_preview: str  # First 200 chars
    response_preview: str  # First 200 chars
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class AgentTrace:
    """Structured trace data for agent execution."""
    trace_id: str
    agent_name: str
    action: str
    duration_ms: float
    success: bool
    input_summary: str
    output_summary: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class StructuredLogger:
    """
    Structured logging for agent observability.
    
    Outputs JSON-formatted logs that can be parsed by log aggregation
    systems like ELK, Datadog, or CloudWatch.
    """

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Add JSON handler if not already present
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            self.logger.addHandler(handler)

    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal log method with structured data."""
        log_data = {
            "message": message,
            "timestamp": datetime.now(UTC).isoformat(),
            **kwargs
        }

        log_method = getattr(self.logger, level.value.lower())
        log_method(json.dumps(log_data))

    def debug(self, message: str, **kwargs):
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def log_llm_call(self, trace: LLMTrace):
        """Log an LLM call with full trace data."""
        self.info(
            "LLM_CALL",
            trace=asdict(trace),
            agent=trace.agent_name,
            model=trace.model,
            tokens=trace.total_tokens,
            latency_ms=trace.latency_ms,
            success=trace.success
        )

    def log_agent_step(self, trace: AgentTrace):
        """Log an agent execution step."""
        self.info(
            "AGENT_STEP",
            trace=asdict(trace),
            agent=trace.agent_name,
            action=trace.action,
            duration_ms=trace.duration_ms,
            success=trace.success
        )


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            # Try to parse the message as JSON
            log_data = json.loads(record.getMessage())
            log_data["level"] = record.levelname
            log_data["logger"] = record.name
            return json.dumps(log_data)
        except json.JSONDecodeError:
            # Fall back to standard formatting
            return json.dumps({
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "timestamp": datetime.now(UTC).isoformat()
            })


class AgentTracer:
    """
    Tracer for agent operations.
    
    Provides decorators and context managers for tracing:
    - LLM calls with token counts and costs
    - Agent workflow steps
    - Tool invocations
    """

    # Cost per 1K tokens for different models (approximate)
    COST_PER_1K_TOKENS = {
        "gemini-2.0-flash-lite": 0.00001,
        "gemini-2.5-flash-lite": 0.00001,
        "gemini-2.0-flash": 0.0001,
        "gemini-1.5-pro": 0.001,
        "default": 0.0001
    }

    def __init__(self, service_name: str = "scholar-agent"):
        self.service_name = service_name
        self.logger = StructuredLogger(f"{service_name}.tracer")
        self._trace_counter = 0

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        self._trace_counter += 1
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return f"{self.service_name}-{timestamp}-{self._trace_counter:06d}"

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (4 chars ≈ 1 token)."""
        return len(text) // 4

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate estimated cost for an LLM call."""
        cost_per_1k = self.COST_PER_1K_TOKENS.get(model, self.COST_PER_1K_TOKENS["default"])
        total_tokens = prompt_tokens + completion_tokens
        return (total_tokens / 1000) * cost_per_1k

    def trace_llm_call(self, agent_name: str, model: str = "gemini-2.5-flash-lite"):
        """
        Decorator to trace LLM calls.
        
        Usage:
            @tracer.trace_llm_call("analyzer", "gemini-2.0-flash")
            def analyze_paper(self, prompt: str) -> str:
                return self.llm_client.chat(prompt)
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                trace_id = self._generate_trace_id()
                start_time = time.time()

                # Extract prompt from args/kwargs
                prompt = ""
                if args:
                    prompt = str(args[0]) if len(args) > 0 else ""
                if "prompt" in kwargs:
                    prompt = str(kwargs["prompt"])

                prompt_tokens = self._estimate_tokens(prompt)

                try:
                    result = func(*args, **kwargs)

                    completion_tokens = self._estimate_tokens(str(result))
                    latency_ms = (time.time() - start_time) * 1000

                    trace = LLMTrace(
                        trace_id=trace_id,
                        agent_name=agent_name,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        latency_ms=latency_ms,
                        cost_usd=self._calculate_cost(model, prompt_tokens, completion_tokens),
                        prompt_preview=prompt[:200],
                        response_preview=str(result)[:200],
                        success=True
                    )

                    self.logger.log_llm_call(trace)

                    return result

                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000

                    trace = LLMTrace(
                        trace_id=trace_id,
                        agent_name=agent_name,
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        total_tokens=prompt_tokens,
                        latency_ms=latency_ms,
                        cost_usd=0.0,
                        prompt_preview=prompt[:200],
                        response_preview="",
                        success=False,
                        error=str(e)
                    )

                    self.logger.log_llm_call(trace)
                    raise

            return wrapper
        return decorator

    def trace_agent_step(self, agent_name: str, action: str):
        """
        Decorator to trace agent workflow steps.
        
        Usage:
            @tracer.trace_agent_step("planner", "generate_keywords")
            async def generate_keywords(self, state):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._trace_step_async(func, agent_name, action, *args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._trace_step_sync(func, agent_name, action, *args, **kwargs)

            # Return appropriate wrapper based on function type
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator

    async def _trace_step_async(self, func, agent_name, action, *args, **kwargs):
        """Async tracing helper."""
        trace_id = self._generate_trace_id()
        start_time = time.time()

        input_summary = self._summarize_input(args, kwargs)

        try:
            result = await func(*args, **kwargs)

            trace = AgentTrace(
                trace_id=trace_id,
                agent_name=agent_name,
                action=action,
                duration_ms=(time.time() - start_time) * 1000,
                success=True,
                input_summary=input_summary,
                output_summary=self._summarize_output(result)
            )

            self.logger.log_agent_step(trace)
            return result

        except Exception as e:
            trace = AgentTrace(
                trace_id=trace_id,
                agent_name=agent_name,
                action=action,
                duration_ms=(time.time() - start_time) * 1000,
                success=False,
                input_summary=input_summary,
                output_summary="",
                error=str(e)
            )

            self.logger.log_agent_step(trace)
            raise

    def _trace_step_sync(self, func, agent_name, action, *args, **kwargs):
        """Sync tracing helper."""
        trace_id = self._generate_trace_id()
        start_time = time.time()

        input_summary = self._summarize_input(args, kwargs)

        try:
            result = func(*args, **kwargs)

            trace = AgentTrace(
                trace_id=trace_id,
                agent_name=agent_name,
                action=action,
                duration_ms=(time.time() - start_time) * 1000,
                success=True,
                input_summary=input_summary,
                output_summary=self._summarize_output(result)
            )

            self.logger.log_agent_step(trace)
            return result

        except Exception as e:
            trace = AgentTrace(
                trace_id=trace_id,
                agent_name=agent_name,
                action=action,
                duration_ms=(time.time() - start_time) * 1000,
                success=False,
                input_summary=input_summary,
                output_summary="",
                error=str(e)
            )

            self.logger.log_agent_step(trace)
            raise

    def _summarize_input(self, args, kwargs) -> str:
        """Create a brief summary of function inputs."""
        parts = []
        for i, arg in enumerate(args[:3]):  # First 3 args
            parts.append(f"arg{i}={str(arg)[:50]}")
        for key in list(kwargs.keys())[:3]:  # First 3 kwargs
            parts.append(f"{key}={str(kwargs[key])[:50]}")
        return ", ".join(parts)

    def _summarize_output(self, result) -> str:
        """Create a brief summary of function output."""
        result_str = str(result)
        return result_str[:200] if len(result_str) > 200 else result_str

    @contextmanager
    def trace_operation(self, operation_name: str, **metadata):
        """
        Context manager for tracing arbitrary operations.
        
        Usage:
            with tracer.trace_operation("fetch_papers", source="arxiv"):
                papers = fetch_from_arxiv(query)
        """
        trace_id = self._generate_trace_id()
        start_time = time.time()

        self.logger.info(
            f"OPERATION_START: {operation_name}",
            trace_id=trace_id,
            operation=operation_name,
            **metadata
        )

        try:
            yield trace_id

            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                f"OPERATION_END: {operation_name}",
                trace_id=trace_id,
                operation=operation_name,
                duration_ms=duration_ms,
                success=True,
                **metadata
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                f"OPERATION_FAILED: {operation_name}",
                trace_id=trace_id,
                operation=operation_name,
                duration_ms=duration_ms,
                success=False,
                error=str(e),
                traceback=traceback.format_exc(),
                **metadata
            )
            raise


# Global tracer instance
tracer = AgentTracer()


# Convenience decorators using the global tracer
def trace_llm(agent_name: str, model: str = "gemini-2.5-flash-lite"):
    """Convenience decorator for LLM call tracing."""
    return tracer.trace_llm_call(agent_name, model)


def trace_step(agent_name: str, action: str):
    """Convenience decorator for agent step tracing."""
    return tracer.trace_agent_step(agent_name, action)
