"""
Unit Tests for LLM Providers, Model Config, and Factory Resolution.
"""

import os
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel

from agents.llm import (
    DEEPSEEK_MODELS,
    GEMINI_MODELS,
    GROQ_MODELS,
    BaseLLMClient,
    DeepSeekProvider,
    GeminiClient,
    GeminiProvider,
    GroqClient,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    MockLLMClient,
    ModelConfig,
    ModelTier,
    clear_client_cache,
    get_available_providers,
    get_best_available_provider,
    get_default_provider,
    get_llm_client,
    get_model_config,
    set_default_provider,
)
from agents.schemas import EvidenceMatrixRow


class DummySchema(BaseModel):
    title: str
    score: float


@pytest.mark.unit
class TestModelConfigAndTiers:
    """Test model configurations, pricing formulas, and tier mappings."""

    def test_model_tier_from_str(self):
        """Verify ModelTier conversion from standard and legacy aliases."""
        assert ModelTier.from_str("fast") == ModelTier.FAST
        assert ModelTier.from_str("fast_cheap") == ModelTier.FAST
        assert ModelTier.from_str("reasoning") == ModelTier.REASONING
        assert ModelTier.from_str("deep") == ModelTier.REASONING
        assert ModelTier.from_str("structured_nli") == ModelTier.STRUCTURED_NLI
        assert ModelTier.from_str("auditor") == ModelTier.STRUCTURED_NLI
        assert ModelTier.from_str(None) == ModelTier.FAST

    def test_get_model_config_gemini_deepseek_groq(self):
        """Verify model configuration lookup for each provider."""
        gemini_cfg = get_model_config("gemini", ModelTier.FAST)
        assert gemini_cfg is not None
        assert gemini_cfg.name == "gemini-3.5-flash-lite"
        assert gemini_cfg.context_window > 100000

        deepseek_cfg = get_model_config("deepseek", ModelTier.REASONING)
        assert deepseek_cfg is not None
        assert deepseek_cfg.name == "deepseek-reasoner"

        groq_cfg = get_model_config("groq", ModelTier.FAST)
        assert groq_cfg is not None
        assert groq_cfg.name == "llama-3.1-8b-instant"

    def test_model_config_cost_estimation(self):
        """Verify token cost calculation."""
        cfg = ModelConfig(
            name="test-model",
            provider="test",
            tier=ModelTier.FAST,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            base_latency_ms=100,
            context_window=10000,
        )
        cost = cfg.estimate_cost(input_tokens=1000, output_tokens=2000)
        assert cost == pytest.approx(0.001 + 0.004)


@pytest.mark.unit
class TestLLMFactoryAndResolution:
    """Test factory client instantiation, caching, and provider fallback."""

    def setup_method(self):
        clear_client_cache()

    def test_get_llm_client_mock_provider(self):
        """Verify factory returns MockLLMClient when requested."""
        client = get_llm_client(provider=LLMProvider.MOCK, config=LLMConfig(user_id="u1"))
        assert isinstance(client, MockLLMClient)
        assert client.get_provider_name() == "mock"

        # Verify caching
        cached_client = get_llm_client(provider=LLMProvider.MOCK, config=LLMConfig(user_id="u1"))
        assert cached_client is client

    def test_set_and_get_default_provider(self):
        """Verify setting default provider changes default factory resolution."""
        set_default_provider(LLMProvider.MOCK)
        assert get_default_provider() == LLMProvider.MOCK

        # Reset
        set_default_provider(LLMProvider.GEMINI)
        assert get_default_provider() == LLMProvider.GEMINI

    def test_gemini_client_adapter(self):
        """Verify backward compatible GeminiClient delegates appropriately."""
        client = GeminiClient()
        assert client.get_provider_name() == "gemini"

    def test_get_available_and_best_providers(self):
        """Verify discovery of configured providers."""
        providers = get_available_providers()
        assert LLMProvider.MOCK in providers

        best = get_best_available_provider()
        assert isinstance(best, LLMProvider)


@pytest.mark.unit
class TestGeminiProviderMockedAPI:
    """Test GeminiProvider payload generation and structured output parsing."""

    @patch("requests.post")
    def test_gemini_generate_text_mocked_http(self, mock_post):
        """Verify GeminiProvider generates text via REST endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Gemini text response"}]}}]
        }
        mock_post.return_value = mock_response

        client = GeminiProvider(LLMConfig(api_key="dummy_gemini_key"))
        text = client.generate_text("Explain quantum entanglement", system_prompt="You are an expert.")
        assert text == "Gemini text response"
        assert mock_post.called

    @patch("requests.post")
    def test_gemini_generate_structured_mocked_http(self, mock_post):
        """Verify GeminiProvider generates validated Pydantic models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"title": "Test Title", "score": 92.5}'}]}}]
        }
        mock_post.return_value = mock_response

        client = GeminiProvider(LLMConfig(api_key="dummy_gemini_key"))
        obj = client.generate_structured("Extract structured data", schema=DummySchema)
        assert isinstance(obj, DummySchema)
        assert obj.title == "Test Title"
        assert obj.score == 92.5


@pytest.mark.unit
class TestDeepSeekProviderMockedAPI:
    """Test DeepSeekProvider completions and structured JSON modes."""

    def test_deepseek_generate_text_and_structured(self):
        """Verify DeepSeekProvider with mocked OpenAI client."""
        client = DeepSeekProvider(LLMConfig(api_key="dummy_deepseek_key"))

        mock_choice = MagicMock()
        mock_choice.message.content = '{"title": "DeepSeek Title", "score": 88.0}'
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch.object(client.client.chat.completions, "create", return_value=mock_completion):
            # Text generation
            text = client.generate_text("Prompt for DeepSeek")
            assert "DeepSeek Title" in text

            # Structured generation
            obj = client.generate_structured("Prompt for DeepSeek", schema=DummySchema)
            assert isinstance(obj, DummySchema)
            assert obj.title == "DeepSeek Title"
            assert obj.score == 88.0


@pytest.mark.unit
class TestGroqClientMockedAPI:
    """Test GroqClient completions and structured JSON output."""

    def test_groq_generate_text_and_structured(self):
        """Verify GroqClient with mocked OpenAI client."""
        client = GroqClient(LLMConfig(api_key="dummy_groq_key"))

        mock_choice = MagicMock()
        mock_choice.message.content = '{"title": "Groq Title", "score": 90.0}'
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch.object(client.client.chat.completions, "create", return_value=mock_completion):
            text = client.generate_text("Prompt for Groq")
            assert "Groq Title" in text

            obj = client.generate_structured("Prompt for Groq", schema=DummySchema)
            assert isinstance(obj, DummySchema)
            assert obj.title == "Groq Title"
            assert obj.score == 90.0
