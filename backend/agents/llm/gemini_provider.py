"""
Google Gemini 2.0 Provider with native response_schema and response_mime_type.
"""

from __future__ import annotations

import logging
import os
from typing import Any, TypeVar

import requests
from pydantic import BaseModel

from agents.error_handling import (
    ErrorCategory,
    NonRetryableError,
    RetryableError,
    RetryConfig,
    get_circuit_breaker,
    with_retry,
)
from agents.llm.base import BaseLLMClient, LLMConfig, LLMResponse, ModelTier
from agents.llm.model_config import GEMINI_MODELS
from agents.llm.rate_limiter import get_provider_limiter, get_rate_limiter
from agents.llm.structured_output import parse_and_validate, to_gemini_schema

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class GeminiProvider(BaseLLMClient):
    """
    Google Gemini 2.0 Provider.
    Implements BaseLLMClient with native structured outputs (response_schema).
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    PROVIDER_NAME = "gemini"

    def _setup_client(self) -> None:
        self.api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set. GeminiProvider will not be functional without key.")

        self.spent = 0.0
        self.user_budget = self.config.user_budget
        self.user_id = self.config.user_id

        self.circuit_breaker = get_circuit_breaker(
            "gemini_api", failure_threshold=5, recovery_timeout=60.0
        )
        self.retry_config = RetryConfig(
            max_retries=self.config.max_retries,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
        )

    def get_provider_name(self) -> str:
        return self.PROVIDER_NAME

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _select_model(self, model_tier: str | ModelTier) -> str:
        tier = ModelTier.from_str(model_tier)
        model_cfg = GEMINI_MODELS.get(tier, GEMINI_MODELS[ModelTier.FAST])
        return model_cfg.name

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        model_tier: str | ModelTier = ModelTier.FAST,
        **kwargs: Any,
    ) -> str:
        model_name = self._select_model(model_tier)
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        result = self._execute_request_with_retry(model_name, payload)
        return self._extract_text_from_result(result)

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str = "",
        model_tier: str | ModelTier = ModelTier.FAST,
        **kwargs: Any,
    ) -> T:
        model_name = self._select_model(model_tier)
        gemini_schema = to_gemini_schema(schema)

        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=kwargs.get("temperature", 0.2),  # Lower temp for structured output precision
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            response_schema=gemini_schema,
            response_mime_type="application/json",
        )

        result = self._execute_request_with_retry(model_name, payload)
        raw_text = self._extract_text_from_result(result)
        return parse_and_validate(raw_text, schema)

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_schema: dict[str, Any] | None = None,
        response_mime_type: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        if response_mime_type:
            payload["generationConfig"]["responseMimeType"] = response_mime_type

        if response_schema:
            payload["generationConfig"]["responseSchema"] = response_schema

        return payload

    def _execute_request_with_retry(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
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
        def _call() -> dict[str, Any]:
            return self.circuit_breaker.call(self._raw_post, model, payload)

        return _call()

    def _raw_post(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Enforce unified provider rate limits across all models on this key (<= 14 RPM)
        limiter = get_provider_limiter("gemini", api_key=self.api_key, max_rpm=14)
        limiter.acquire("gemini")

        url = f"{self.BASE_URL}/{model}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}

        try:
            resp = requests.post(url, headers=headers, params=params, json=payload, timeout=self.config.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 500
            if status == 429:
                # Check for Retry-After header or delay hint
                retry_after_hdr = e.response.headers.get("Retry-After") if e.response is not None else None
                backoff_hint = f" (Retry-After: {retry_after_hdr}s)" if retry_after_hdr else ""
                logger.warning(f"Gemini Rate Limit (429) encountered{backoff_hint}. Backing off gracefully...")
                raise RetryableError(f"Gemini Rate Limit (429): {e}{backoff_hint}", category=ErrorCategory.RATE_LIMIT, severity="low")
            elif status >= 500:
                raise RetryableError(f"Gemini Server Error ({status}): {e}", category=ErrorCategory.SERVER_ERROR, severity="medium")
            elif status in (401, 403):
                raise NonRetryableError(f"Gemini Auth Failure ({status})", category=ErrorCategory.CLIENT_ERROR, severity="high")
            else:
                raise NonRetryableError(f"Gemini Client Error ({status}): {e}", category=ErrorCategory.CLIENT_ERROR, severity="medium")
        except requests.exceptions.Timeout as e:
            raise RetryableError(f"Gemini Timeout: {e}", category=ErrorCategory.TIMEOUT, severity="low")
        except requests.exceptions.ConnectionError as e:
            raise RetryableError(f"Gemini Connection Error: {e}", category=ErrorCategory.NETWORK, severity="medium")

    def _extract_text_from_result(self, result: dict[str, Any]) -> str:
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            logger.warning(f"Unexpected Gemini response structure: {result}")
            return str(result)

    def get_usage_stats(self) -> dict[str, Any]:
        return {
            "provider": self.PROVIDER_NAME,
            "user_id": self.user_id,
            "budget": self.user_budget,
            "spent": self.spent,
            "remaining": max(0.0, self.user_budget - self.spent),
        }

    def reset_budget(self, new_budget: float | None = None) -> None:
        if new_budget is not None:
            self.user_budget = new_budget
        self.spent = 0.0
