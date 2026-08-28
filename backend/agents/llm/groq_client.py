"""
Groq LLM Client.
Implementation of BaseLLMClient for Groq's high-speed inference API.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TypeVar

from openai import OpenAI
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
from agents.llm.model_config import GROQ_MODELS
from agents.llm.structured_output import parse_and_validate

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class GroqClient(BaseLLMClient):
    """
    Groq LLM Client Provider.
    Supports ultra-fast inference with Llama 3.1 and Llama 3.3 models.
    """

    PROVIDER_NAME = "groq"

    def _setup_client(self) -> None:
        self.api_key = self.config.api_key or os.environ.get("GROQ_API_KEY")
        self.base_url = self.config.base_url or os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

        if not self.api_key:
            logger.warning("GROQ_API_KEY not set. GroqClient will not be functional without key.")

        self.client = OpenAI(
            api_key=self.api_key or "gsk-dummy-key",
            base_url=self.base_url,
            timeout=float(self.config.timeout),
        )
        self.spent = 0.0
        self.user_budget = self.config.user_budget
        self.user_id = self.config.user_id

        self.circuit_breaker = get_circuit_breaker(
            "groq_api", failure_threshold=5, recovery_timeout=60.0
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
        model_cfg = GROQ_MODELS.get(tier, GROQ_MODELS[ModelTier.FAST])
        return model_cfg.name

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        model_tier: str | ModelTier = ModelTier.FAST,
        **kwargs: Any,
    ) -> str:
        model_name = self._select_model(model_tier)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._call_chat_completion_with_retry(
            model=model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )

        return response.choices[0].message.content or ""

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str = "",
        model_tier: str | ModelTier = ModelTier.FAST,
        **kwargs: Any,
    ) -> T:
        model_name = self._select_model(model_tier)
        schema_json = json.dumps(schema.model_json_schema(), indent=2)

        augmented_system_prompt = (
            (system_prompt + "\n\n" if system_prompt else "")
            + "You MUST output strictly valid JSON matching this schema:\n"
            + schema_json
        )

        messages = [
            {"role": "system", "content": augmented_system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self._call_chat_completion_with_retry(
            model=model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content or ""
        return parse_and_validate(raw_text, schema)

    def _call_chat_completion_with_retry(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            call_kwargs["response_format"] = response_format

        @with_retry(
            config=self.retry_config,
            retryable_exceptions=(RetryableError,),
            non_retryable_exceptions=(NonRetryableError,),
        )
        def _call() -> Any:
            try:
                return self.circuit_breaker.call(self.client.chat.completions.create, **call_kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if "rate limit" in err_str or "429" in err_str:
                    raise RetryableError(f"Groq rate limit: {e}", category=ErrorCategory.RATE_LIMIT, severity="low") from e
                elif "500" in err_str or "502" in err_str or "503" in err_str:
                    raise RetryableError(f"Groq server error: {e}", category=ErrorCategory.SERVER_ERROR, severity="medium") from e
                elif "auth" in err_str or "401" in err_str or "403" in err_str:
                    raise NonRetryableError(f"Groq auth error: {e}", category=ErrorCategory.CLIENT_ERROR, severity="high") from e
                raise RetryableError(f"Groq API error: {e}", category=ErrorCategory.UNKNOWN, severity="medium") from e

        return _call()

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
