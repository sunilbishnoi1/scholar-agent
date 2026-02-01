# Gemini LLM Provider
# Implementation of the BaseLLMClient for Google's Gemini API
# Refactored from the original gemini_client.py for the new provider architecture

import logging
import os
import time
from typing import Any

import requests

from agents.error_handling import (
    ErrorCategory,
    NonRetryableError,
    RetryableError,
    RetryConfig,
    get_circuit_breaker,
    with_retry,
)
from agents.llm.base import BaseLLMClient, LLMResponse
from agents.llm.model_config import GEMINI_MODELS, ModelTier

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMClient):
    """
    Google Gemini API client with smart routing and error handling.

    Gemini Features:
    - Large context windows (up to 2M tokens)
    - Good for complex reasoning tasks
    - Competitive pricing

    Note: Free tier has strict rate limits (15 RPM for flash, 2 RPM for pro)

    Models:
    - gemini-2.0-flash-lite: Fastest, cheapest
    - gemini-2.0-flash: Good balance
    - gemini-1.5-pro: Most powerful
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    PROVIDER_NAME = "gemini"

    def _setup_client(self) -> None:
        """Set up the Gemini client."""
        self.api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY")

        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set. Gemini client will not be functional.")

        # Budget tracking
        self.user_budget = self.config.user_budget
        self.user_id = self.config.user_id
        self.spent = 0.0

        # Circuit breaker
        self.circuit_breaker = get_circuit_breaker(
            "gemini_api", failure_threshold=5, recovery_timeout=60.0
        )

        # Retry config
        self.retry_config = RetryConfig(
            max_retries=self.config.max_retries,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
        )

        logger.info(
            f"GeminiProvider initialized (user={self.user_id}, " f"budget=${self.user_budget:.4f})"
        )

    def get_provider_name(self) -> str:
        return self.PROVIDER_NAME

    def _select_model(
        self, task_type: str, complexity_hint: str | None = None, max_latency_ms: int | None = None
    ) -> tuple[str, ModelTier]:
        """Select the appropriate Gemini model."""

        TASK_ROUTING = {
            "extract_keywords": ModelTier.FAST_CHEAP,
            "classify_relevance": ModelTier.FAST_CHEAP,
            "simple_parsing": ModelTier.FAST_CHEAP,
            "format_output": ModelTier.FAST_CHEAP,
            "paper_analysis": ModelTier.BALANCED,
            "summarization": ModelTier.BALANCED,
            "search_planning": ModelTier.BALANCED,
            "relevance_scoring": ModelTier.BALANCED,
            "synthesis": ModelTier.POWERFUL,
            "quality_evaluation": ModelTier.POWERFUL,
            "research_gap_identification": ModelTier.POWERFUL,
            "complex_reasoning": ModelTier.POWERFUL,
        }

        tier = TASK_ROUTING.get(task_type, ModelTier.BALANCED)

        # Budget check
        budget_usage = self.spent / self.user_budget if self.user_budget > 0 else 0
        if budget_usage >= 0.8:
            tier = ModelTier.FAST_CHEAP
            logger.warning(f"Budget constraint: Using {tier.value}")

        # Complexity override
        if complexity_hint == "high" and tier != ModelTier.POWERFUL and budget_usage < 0.8:
            tier = ModelTier.POWERFUL
        elif complexity_hint == "low":
            tier = ModelTier.FAST_CHEAP

        # Latency constraint
        if max_latency_ms:
            model_config = GEMINI_MODELS.get(tier)
            if model_config and model_config.base_latency_ms > max_latency_ms:
                if tier == ModelTier.POWERFUL:
                    tier = ModelTier.BALANCED
                elif tier == ModelTier.BALANCED:
                    tier = ModelTier.FAST_CHEAP

        model_config = GEMINI_MODELS.get(tier, GEMINI_MODELS[ModelTier.BALANCED])
        return model_config.name, tier

    def _get_model_url(self, model_name: str) -> str:
        """Get the API URL for a model."""
        return f"{self.BASE_URL}/{model_name}:generateContent"

    def chat(
        self,
        prompt: str,
        task_type: str = "general",
        complexity_hint: str | None = None,
        max_latency_ms: int | None = None,
        **kwargs,
    ) -> str:
        """Send a chat request and return the response text."""
        response = self.chat_with_response(
            prompt=prompt,
            task_type=task_type,
            complexity_hint=complexity_hint,
            max_latency_ms=max_latency_ms,
            **kwargs,
        )
        return response.text

    def chat_with_response(
        self,
        prompt: str,
        task_type: str = "general",
        complexity_hint: str | None = None,
        max_latency_ms: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a chat request and return full response with metadata."""

        model_name, tier = self._select_model(task_type, complexity_hint, max_latency_ms)
        logger.info(f"Gemini routing: {task_type} -> {model_name} (tier: {tier.value})")

        start_time = time.time()

        try:
            result = self._make_api_call_with_retry(model_name, prompt, **kwargs)

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract text from Gemini response format
            try:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                logger.warning(f"Unexpected response structure: {result}")
                text = str(result)

            # Gemini doesn't always return token counts in the same way
            usage_metadata = result.get("usageMetadata", {})
            input_tokens = usage_metadata.get("promptTokenCount", len(prompt) // 4)
            output_tokens = usage_metadata.get("candidatesTokenCount", len(text) // 4)
            total_tokens = usage_metadata.get("totalTokenCount", input_tokens + output_tokens)

            # Calculate cost
            model_config = GEMINI_MODELS.get(tier)
            if model_config:
                cost = model_config.estimate_cost(input_tokens, output_tokens)
            else:
                cost = 0.0

            self.spent += cost

            return LLMResponse(
                text=text,
                model=model_name,
                provider=self.PROVIDER_NAME,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost=cost,
                latency_ms=latency_ms,
                metadata={
                    "tier": tier.value,
                    "task_type": task_type,
                },
            )

        except Exception as e:
            logger.error(f"Gemini chat request failed: {e}")
            raise

    def _make_api_call_with_retry(self, model: str, prompt: str, **kwargs) -> dict[str, Any]:
        """Make API call with retry logic and circuit breaker."""

        @with_retry(
            config=self.retry_config,
            retryable_exceptions=(
                requests.exceptions.HTTPError,
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                RetryableError,
            ),
            non_retryable_exceptions=(NonRetryableError,),
        )
        def _call_api():
            return self.circuit_breaker.call(self._do_api_request, model, prompt, **kwargs)

        return _call_api()

    def _do_api_request(self, model: str, prompt: str, **kwargs) -> dict[str, Any]:
        """Execute the actual API request to Gemini."""

        url = self._get_model_url(model)
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}

        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 4096),
            },
        }

        if "top_p" in kwargs:
            data["generationConfig"]["topP"] = kwargs["top_p"]

        try:
            resp = requests.post(
                url, headers=headers, params=params, json=data, timeout=self.config.timeout
            )
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code

            if status_code == 429:
                raise RetryableError(
                    f"Rate limit exceeded: {e}", category=ErrorCategory.RATE_LIMIT, severity="low"
                )
            elif status_code >= 500:
                raise RetryableError(
                    f"Server error ({status_code}): {e}",
                    category=ErrorCategory.SERVER_ERROR,
                    severity="medium",
                )
            elif status_code in (401, 403):
                raise NonRetryableError(
                    f"Authentication failed ({status_code}): Check GEMINI_API_KEY",
                    category=ErrorCategory.CLIENT_ERROR,
                    severity="high",
                )
            elif status_code == 400:
                raise NonRetryableError(
                    f"Invalid request ({status_code}): {e}",
                    category=ErrorCategory.VALIDATION,
                    severity="medium",
                )
            else:
                raise NonRetryableError(
                    f"Client error ({status_code}): {e}",
                    category=ErrorCategory.CLIENT_ERROR,
                    severity="medium",
                )

        except requests.exceptions.Timeout as e:
            raise RetryableError(
                f"Request timeout: {e}", category=ErrorCategory.TIMEOUT, severity="low"
            )

        except requests.exceptions.ConnectionError as e:
            raise RetryableError(
                f"Network error: {e}", category=ErrorCategory.NETWORK, severity="medium"
            )

        except Exception as e:
            logger.error(f"Unexpected error in Gemini request: {e}", exc_info=True)
            raise RetryableError(
                f"Unexpected error: {e}", category=ErrorCategory.UNKNOWN, severity="medium"
            )

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "provider": self.PROVIDER_NAME,
            "user_id": self.user_id,
            "budget": self.user_budget,
            "spent": self.spent,
            "remaining": self.user_budget - self.spent,
            "usage_percent": (self.spent / self.user_budget * 100) if self.user_budget > 0 else 0,
        }

    def reset_budget(self, new_budget: float | None = None) -> None:
        """Reset budget tracking."""
        if new_budget is not None:
            self.user_budget = new_budget
        self.spent = 0.0
        logger.info(f"Gemini budget reset to ${self.user_budget:.4f}")

    def is_available(self) -> bool:
        """Check if Gemini is configured and available."""
        return bool(self.api_key)
