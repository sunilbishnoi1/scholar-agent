# Tests for the Observability Module
# Tests tracing, logging, and metrics functionality

import pytest
from unittest.mock import Mock, patch
import json
import time

from agents.observability import (
    AgentTracer,
    StructuredLogger,
    LLMTrace,
    AgentTrace,
    tracer,
    trace_llm,
    trace_step
)


class TestStructuredLogger:
    """Tests for StructuredLogger."""
    
    def test_logger_creates_json_output(self, caplog):
        """Logger should create JSON-formatted output."""
        logger = StructuredLogger("test_logger")
        
        with caplog.at_level("INFO"):
            logger.info("Test message", key1="value1", key2=42)
        
        # The log should be parseable as JSON
        # Note: caplog captures the raw message
        assert "Test message" in caplog.text
        
    def test_log_llm_call(self):
        """Logger should accept LLMTrace objects."""
        logger = StructuredLogger("test_logger")
        
        trace = LLMTrace(
            trace_id="test-123",
            agent_name="planner",
            model="gemini-2.0-flash",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=500.0,
            cost_usd=0.00015,
            prompt_preview="Test prompt...",
            response_preview="Test response...",
            success=True
        )
        
        # Should not raise
        logger.log_llm_call(trace)
        
    def test_log_agent_step(self):
        """Logger should accept AgentTrace objects."""
        logger = StructuredLogger("test_logger")
        
        trace = AgentTrace(
            trace_id="test-456",
            agent_name="analyzer",
            action="score_relevance",
            duration_ms=250.0,
            success=True,
            input_summary="paper_title=Test Paper",
            output_summary="score=85"
        )
        
        # Should not raise
        logger.log_agent_step(trace)


class TestAgentTracer:
    """Tests for AgentTracer."""
    
    def test_generate_trace_id_unique(self):
        """Tracer should generate unique trace IDs."""
        tracer = AgentTracer("test-service")
        
        id1 = tracer._generate_trace_id()
        id2 = tracer._generate_trace_id()
        
        assert id1 != id2
        assert "test-service" in id1
        
    def test_estimate_tokens(self):
        """Token estimation should be roughly 4 chars per token."""
        tracer = AgentTracer()
        
        # 100 characters should be about 25 tokens
        text = "a" * 100
        tokens = tracer._estimate_tokens(text)
        
        assert tokens == 25
        
    def test_calculate_cost_known_model(self):
        """Cost calculation should use known model rates."""
        tracer = AgentTracer()
        
        # gemini-2.0-flash-lite: $0.00001 per 1K tokens
        cost = tracer._calculate_cost("gemini-2.0-flash-lite", 500, 500)
        
        # 1000 tokens * ($0.00001 / 1000) = $0.00001
        assert cost == pytest.approx(0.00001, rel=0.01)
        
    def test_calculate_cost_unknown_model(self):
        """Cost calculation should use default rate for unknown models."""
        tracer = AgentTracer()
        
        cost = tracer._calculate_cost("unknown-model", 1000, 1000)
        
        # Should use default rate
        assert cost > 0
        
    def test_trace_llm_call_decorator_success(self):
        """LLM call decorator should track successful calls."""
        test_tracer = AgentTracer("test")
        
        @test_tracer.trace_llm_call("test_agent", "test-model")
        def mock_llm_call(prompt: str) -> str:
            return "response text"
        
        result = mock_llm_call("test prompt")
        
        assert result == "response text"
        
    def test_trace_llm_call_decorator_failure(self):
        """LLM call decorator should track failed calls."""
        test_tracer = AgentTracer("test")
        
        @test_tracer.trace_llm_call("test_agent", "test-model")
        def failing_llm_call(prompt: str) -> str:
            raise ValueError("API error")
        
        with pytest.raises(ValueError):
            failing_llm_call("test prompt")
            
    @pytest.mark.asyncio
    async def test_trace_agent_step_async(self):
        """Agent step decorator should work with async functions."""
        test_tracer = AgentTracer("test")
        
        @test_tracer.trace_agent_step("test_agent", "test_action")
        async def async_agent_step(value: int) -> int:
            return value * 2
        
        result = await async_agent_step(5)
        
        assert result == 10
        
    def test_trace_agent_step_sync(self):
        """Agent step decorator should work with sync functions."""
        test_tracer = AgentTracer("test")
        
        @test_tracer.trace_agent_step("test_agent", "test_action")
        def sync_agent_step(value: int) -> int:
            return value * 2
        
        result = sync_agent_step(5)
        
        assert result == 10
        
    def test_trace_operation_context_manager(self):
        """trace_operation context manager should track operations."""
        test_tracer = AgentTracer("test")
        
        with test_tracer.trace_operation("test_operation", source="test"):
            result = 1 + 1
        
        assert result == 2
        
    def test_trace_operation_captures_exceptions(self):
        """trace_operation should log exceptions."""
        test_tracer = AgentTracer("test")
        
        with pytest.raises(ValueError):
            with test_tracer.trace_operation("failing_operation"):
                raise ValueError("Test error")


class TestGlobalTracer:
    """Tests for the global tracer instance and decorators."""
    
    def test_global_tracer_exists(self):
        """Global tracer should be available."""
        assert tracer is not None
        assert isinstance(tracer, AgentTracer)
        
    def test_trace_llm_convenience_decorator(self):
        """trace_llm decorator should work."""
        @trace_llm("test_agent")
        def test_func(prompt: str) -> str:
            return "response"
        
        result = test_func("prompt")
        assert result == "response"
        
    def test_trace_step_convenience_decorator(self):
        """trace_step decorator should work."""
        @trace_step("test_agent", "test_action")
        def test_func(value: int) -> int:
            return value + 1
        
        result = test_func(5)
        assert result == 6


class TestLLMTrace:
    """Tests for LLMTrace dataclass."""
    
    def test_llm_trace_creation(self):
        """LLMTrace should store all required fields."""
        trace = LLMTrace(
            trace_id="trace-123",
            agent_name="planner",
            model="gemini-2.0-flash",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=500.0,
            cost_usd=0.00015,
            prompt_preview="Test...",
            response_preview="Response...",
            success=True
        )
        
        assert trace.trace_id == "trace-123"
        assert trace.agent_name == "planner"
        assert trace.total_tokens == 150
        assert trace.success is True
        assert trace.error is None
        
    def test_llm_trace_with_error(self):
        """LLMTrace should store error information."""
        trace = LLMTrace(
            trace_id="trace-456",
            agent_name="analyzer",
            model="gemini-2.0-flash",
            prompt_tokens=100,
            completion_tokens=0,
            total_tokens=100,
            latency_ms=100.0,
            cost_usd=0.0,
            prompt_preview="Test...",
            response_preview="",
            success=False,
            error="API rate limit exceeded"
        )
        
        assert trace.success is False
        assert trace.error == "API rate limit exceeded"


class TestAgentTrace:
    """Tests for AgentTrace dataclass."""
    
    def test_agent_trace_creation(self):
        """AgentTrace should store all required fields."""
        trace = AgentTrace(
            trace_id="trace-789",
            agent_name="synthesizer",
            action="synthesize_section",
            duration_ms=2500.0,
            success=True,
            input_summary="subtopic=Introduction",
            output_summary="Generated 500 words..."
        )
        
        assert trace.trace_id == "trace-789"
        assert trace.action == "synthesize_section"
        assert trace.duration_ms == 2500.0
        assert trace.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
