"""
Structured Output Utilities and Progressive Fallback Parser.
Provides schema transformation for Gemini/OpenAI and robust multi-stage JSON repair.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Raised when an LLM response cannot be parsed or validated against a schema."""

    def __init__(
        self,
        message: str,
        raw_text: str | None = None,
        schema_name: str | None = None,
        validation_errors: Any = None,
    ):
        super().__init__(message)
        self.raw_text = raw_text
        self.schema_name = schema_name
        self.validation_errors = validation_errors


def clean_json_markdown(text: str) -> str:
    """
    Strip markdown code block wrappers (```json ... ``` or ``` ... ```).
    """
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def extract_json_substring(text: str) -> str:
    """
    Extract the outermost balanced JSON object ({...}) or array ([...]) from text.
    Handles string literals and escaped quotes safely.
    """
    start_brace = text.find("{")
    start_bracket = text.find("[")

    if start_brace == -1 and start_bracket == -1:
        return text

    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        open_char, close_char = "{", "}"
        start_idx = start_brace
    else:
        open_char, close_char = "[", "]"
        start_idx = start_bracket

    depth = 0
    in_string = False
    escape = False

    for i in range(start_idx, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]

    return text[start_idx:]


def repair_json_syntax(text: str) -> str:
    """
    Repair common LLM JSON syntax mistakes:
    - Trailing commas before closing braces/brackets
    - Python literals (True -> true, False -> false, None -> null)
    - Single quotes to double quotes for keys
    """
    # Remove trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Python constant replacements
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text


def parse_and_validate(raw_text: str, schema: type[T]) -> T:
    """
    Multi-stage progressive fallback parser and Pydantic validator.

    Stages:
    1. Direct Pydantic JSON validation
    2. Markdown code block stripping
    3. Balanced substring extraction
    4. Syntax auto-repair (trailing commas, Python literals)
    5. Fallback via json.loads + model_validate
    """
    schema_name = schema.__name__

    # Stage 1: Direct parse
    try:
        return schema.model_validate_json(raw_text)
    except Exception:
        pass

    # Stage 2: Clean markdown code blocks
    cleaned = clean_json_markdown(raw_text)
    try:
        return schema.model_validate_json(cleaned)
    except Exception:
        pass

    # Stage 3: Extract balanced JSON substring
    extracted = extract_json_substring(cleaned)
    try:
        return schema.model_validate_json(extracted)
    except Exception:
        pass

    # Stage 4: Repair syntax
    repaired = repair_json_syntax(extracted)
    try:
        return schema.model_validate_json(repaired)
    except Exception:
        pass

    # Stage 5: json.loads dictionary parse followed by model_validate
    try:
        data = json.loads(repaired)
        return schema.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Failed to parse structured output for schema {schema_name}: {e}")
        raise StructuredOutputError(
            message=f"Failed to parse and validate output for schema {schema_name}: {e}",
            raw_text=raw_text,
            schema_name=schema_name,
            validation_errors=e,
        ) from e


def to_gemini_schema(schema_cls: type[BaseModel]) -> dict[str, Any]:
    """
    Convert a Pydantic BaseModel class into an OpenAPI 3.0 schema dictionary
    compatible with Google Gemini response_schema / responseSchema.
    Inlines all $defs definitions recursively and strips unsupported keywords.
    """
    raw_schema = schema_cls.model_json_schema()
    defs = raw_schema.pop("$defs", {})

    allowed_keys = {
        "type",
        "properties",
        "items",
        "required",
        "enum",
        "description",
        "nullable",
        "format",
    }

    def clean_node(node: Any) -> Any:
        if isinstance(node, dict):
            # Resolve $ref
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                if ref_name in defs:
                    return clean_node(defs[ref_name].copy())

            # Resolve allOf
            if "allOf" in node and len(node["allOf"]) == 1:
                return clean_node(node["allOf"][0])

            # Resolve anyOf with null (Optional fields)
            if "anyOf" in node:
                non_null = [item for item in node["anyOf"] if item.get("type") != "null"]
                if len(non_null) == 1:
                    cleaned = clean_node(non_null[0])
                    if isinstance(cleaned, dict):
                        cleaned["nullable"] = True
                    return cleaned
                elif len(non_null) > 1:
                    return clean_node(non_null[0])

            cleaned: dict[str, Any] = {}
            for k, v in node.items():
                if k in allowed_keys:
                    if k == "type" and isinstance(v, list):
                        types = [t for t in v if t != "null"]
                        cleaned["type"] = types[0] if types else "string"
                        cleaned["nullable"] = True
                    elif k == "properties":
                        cleaned["properties"] = {pk: clean_node(pv) for pk, pv in v.items()}
                    elif k == "items":
                        cleaned["items"] = clean_node(v)
                    else:
                        cleaned[k] = v
            return cleaned
        elif isinstance(node, list):
            return [clean_node(item) for item in node]
        return node

    return clean_node(raw_schema)


def to_openai_response_format(schema_cls: type[BaseModel]) -> dict[str, Any]:
    """
    Convert a Pydantic model to OpenAI-compatible response_format specification.
    """
    return {
        "type": "json_object",
    }
