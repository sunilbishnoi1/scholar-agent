# Unit Tests for LLM Client Architecture & Structured Outputs
# Covers BaseLLMClient interface, MockLLMClient, structured output generation, JSON cleaning, and error handling

import json
from typing import Literal
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field, ValidationError

from agents.llm.base import BaseLLMClient, LLMConfig, LLMResponse
from agents.schemas import Citation, PaperAnalysis, PlannerOutput, ReportStatus
from conftest import DeterministicMockLLMClient


# Test schemas for structured output parsing
class SimpleMetric(BaseModel):
    name: str
    value: float
    unit: str = "percent"


class NestedBenchmarkResult(BaseModel):
    benchmark_name: str
    metrics: list[SimpleMetric]
    passed: bool
    importance: Literal["high", "medium", "low"] = "high"


@pytest.mark.unit
class TestBaseLLMClientInterface:
    """Test the BaseLLMClient contract and abstract methods."""

    def test_base_llm_client_cannot_be_instantiated_directly(self):
        """Verify BaseLLMClient is an abstract class with required abstract methods."""
        with pytest.raises(TypeError):
            BaseLLMClient()  # type: ignore

    def test_custom_subclass_must_implement_all_abstract_methods(self):
        """Verify a concrete subclass implementing all abstract methods instantiates cleanly."""

        class MinimalConcreteClient(BaseLLMClient):
            def _setup_client(self) -> None:
                pass

            def chat(self, prompt: str, **kwargs) -> str:
                return "response"

            def chat_with_response(self, prompt: str, **kwargs) -> LLMResponse:
                return LLMResponse(text="response", model="test", provider="test")

            def get_provider_name(self) -> str:
                return "concrete_test"

            def get_usage_stats(self) -> dict:
                return {}

            def reset_budget(self, new_budget=None) -> None:
                pass

        client = MinimalConcreteClient()
        assert client.get_provider_name() == "concrete_test"
        assert client.chat("hello") == "response"

    def test_llm_config_defaults(self):
        """Verify LLMConfig initializes with reasonable defaults."""
        config = LLMConfig()
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.user_budget == 1.0
        assert config.enable_router is True

    def test_llm_response_dataclass(self):
        """Verify LLMResponse fields and metadata handling."""
        resp = LLMResponse(
            text="Synthetic text",
            model="gemini-2.0-flash",
            provider="gemini",
            input_tokens=500,
            output_tokens=150,
            total_tokens=650,
            estimated_cost=0.0001,
            latency_ms=250,
            metadata={"finish_reason": "STOP"},
        )
        assert resp.total_tokens == 650
        assert resp.metadata["finish_reason"] == "STOP"


@pytest.mark.unit
class TestDeterministicMockLLMClient:
    """Test the DeterministicMockLLMClient fixture and deterministic synthesis."""

    def test_mock_generate_text_default_and_custom(self, mock_llm_client: DeterministicMockLLMClient):
        """Verify generate_text returns deterministic output and custom queued responses."""
        # Default response
        res1 = mock_llm_client.generate_text("Explain transformer scaling laws", system_prompt="You are a scientist.")
        assert "# Synthetic Scientific Response" in res1
        assert len(mock_llm_client.calls) == 1
        assert mock_llm_client.calls[0]["system_prompt"] == "You are a scientist."

        # Custom response
        mock_llm_client.set_text_response("Custom scientific conclusion.")
        res2 = mock_llm_client.generate_text("Next prompt")
        assert res2 == "Custom scientific conclusion."

    def test_mock_generate_structured_simple_schema(self, mock_llm_client: DeterministicMockLLMClient):
        """Verify generate_structured automatically creates valid instance for SimpleMetric."""
        metric = mock_llm_client.generate_structured(
            prompt="Extract accuracy metric",
            schema=SimpleMetric,
            model_tier="fast",
        )
        assert isinstance(metric, SimpleMetric)
        assert isinstance(metric.name, str)
        assert isinstance(metric.value, float)
        assert metric.unit == "percent"

    def test_mock_generate_structured_nested_schema(self, mock_llm_client: DeterministicMockLLMClient):
        """Verify generate_structured creates valid nested models with Literal and lists."""
        result = mock_llm_client.generate_structured(
            prompt="Extract benchmark results for PubMedQA",
            schema=NestedBenchmarkResult,
            model_tier="reasoning",
        )
        assert isinstance(result, NestedBenchmarkResult)
        assert len(result.metrics) > 0
        assert isinstance(result.metrics[0], SimpleMetric)
        assert result.importance in ["high", "medium", "low"]
        assert isinstance(result.passed, bool)

    def test_mock_generate_structured_with_explicit_override(self, mock_llm_client: DeterministicMockLLMClient):
        """Verify registering a custom canned response overrides default synthetic generation."""
        custom_metric = SimpleMetric(name="Exact Match", value=98.5, unit="score")
        mock_llm_client.set_structured_response(SimpleMetric, custom_metric)

        output = mock_llm_client.generate_structured("Get metric", schema=SimpleMetric)
        assert output.name == "Exact Match"
        assert output.value == 98.5

    def test_mock_generate_structured_pydantic_agent_schemas(self, mock_llm_client: DeterministicMockLLMClient):
        """Verify generate_structured synthesizes valid models from agents.schemas."""
        citation = mock_llm_client.generate_structured("Generate citation", schema=Citation)
        assert isinstance(citation, Citation)
        assert len(citation.authors) > 0
        assert citation.relevance_score >= 0

        analysis = mock_llm_client.generate_structured("Analyze paper", schema=PaperAnalysis)
        assert isinstance(analysis, PaperAnalysis)
        assert 0 <= analysis.relevance_score <= 100

    def test_mock_error_injection(self, mock_llm_client: DeterministicMockLLMClient):
        """Verify error injection causes expected exception and resets cleanly."""
        mock_llm_client.set_error(ConnectionError("Simulated LLM API timeout"))

        with pytest.raises(ConnectionError, match="Simulated LLM API timeout"):
            mock_llm_client.generate_text("Prompt that will fail")

        # After error raised, subsequent call should succeed
        res = mock_llm_client.generate_text("Recovered prompt")
        assert "Synthetic" in res

    def test_mock_legacy_chat_methods(self, mock_llm_client: DeterministicMockLLMClient):
        """Verify legacy chat() and chat_with_response() methods."""
        chat_out = mock_llm_client.chat("Give me keywords")
        assert "keywords" in chat_out

        resp = mock_llm_client.chat_with_response("Give me keywords with metadata")
        assert isinstance(resp, LLMResponse)
        assert resp.provider == "mock"
        assert resp.total_tokens == 200


@pytest.mark.unit
class TestStructuredOutputParsingAndSanitization:
    """Test parsing, cleaning markdown fences, and validating structured responses."""

    def test_parse_clean_json_string(self):
        """Verify parsing standard clean JSON string."""
        raw_json = '{"name": "F1-Score", "value": 91.4, "unit": "percent"}'
        parsed = SimpleMetric.model_validate_json(raw_json)
        assert parsed.name == "F1-Score"
        assert parsed.value == 91.4

    def test_parse_json_with_markdown_fences(self):
        """Verify handling and extracting JSON enclosed in ```json ... ``` codeblocks."""
        raw_llm_output = (
            "Here is the requested structured JSON data:\n"
            "```json\n"
            "{\n"
            '  "name": "ROUGE-L",\n'
            '  "value": 45.2,\n'
            '  "unit": "score"\n'
            "}\n"
            "```\n"
            "Hope this helps!"
        )

        def extract_and_validate(raw_text: str, schema: type[BaseModel]) -> BaseModel:
            text = raw_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return schema.model_validate_json(text)

        parsed = extract_and_validate(raw_llm_output, SimpleMetric)
        assert parsed.name == "ROUGE-L"
        assert parsed.value == 45.2

    def test_parse_invalid_schema_payload_raises(self):
        """Verify ValidationError when JSON is valid but does not conform to schema."""
        raw_json = '{"name": "Accuracy", "value": "not_a_float", "unit": "percent"}'
        with pytest.raises(ValidationError):
            SimpleMetric.model_validate_json(raw_json)
