"""
Unit Tests for Structured Output Utilities and Fallback Parser.
"""

from typing import Literal
import pytest
from pydantic import BaseModel, Field

from agents.llm.structured_output import (
    StructuredOutputError,
    clean_json_markdown,
    extract_json_substring,
    parse_and_validate,
    repair_json_syntax,
    to_gemini_schema,
    to_openai_response_format,
)
from agents.schemas import EvidenceMatrixRow, ResearchGapItem, ThematicSection


class SampleItem(BaseModel):
    id: str
    name: str
    score: float
    active: bool = True
    tags: list[str] = Field(default_factory=list)


@pytest.mark.unit
class TestStructuredOutputUtilities:
    """Test suite for structured output transformations and repair."""

    def test_clean_json_markdown_variations(self):
        """Verify stripping markdown code fences."""
        raw_json_tag = "```json\n{\"id\": \"1\", \"name\": \"test\", \"score\": 9.5}\n```"
        assert clean_json_markdown(raw_json_tag) == '{"id": "1", "name": "test", "score": 9.5}'

        raw_no_tag = "```\n{\"id\": \"2\", \"name\": \"test2\", \"score\": 8.0}\n```"
        assert clean_json_markdown(raw_no_tag) == '{"id": "2", "name": "test2", "score": 8.0}'

        plain = '{"id": "3", "name": "test3", "score": 7.0}'
        assert clean_json_markdown(plain) == plain

    def test_extract_json_substring_nested_braces(self):
        """Verify extraction of outermost JSON object with nested structures and quotes."""
        preamble_text = "Here is your response: {\"id\": \"item_1\", \"meta\": {\"nested\": true, \"str\": \"value with } brace\"}} Thank you!"
        extracted = extract_json_substring(preamble_text)
        assert extracted == '{"id": "item_1", "meta": {"nested": true, "str": "value with } brace"}}'

    def test_repair_json_syntax_trailing_commas_and_literals(self):
        """Verify trailing commas and Python literals are repaired."""
        bad_json = '{"id": "item_1", "name": "Item", "score": 10.0, "active": True, "tags": ["a", "b",],}'
        repaired = repair_json_syntax(bad_json)
        assert "true" in repaired
        assert ",}" not in repaired
        assert ",]" not in repaired

    def test_parse_and_validate_success_across_all_stages(self):
        """Verify parse_and_validate succeeds on messy LLM output."""
        messy_response = """
        I have analyzed your request. Below is the structured object:
        ```json
        {
            "id": "p123",
            "name": "Transformer Architecture",
            "score": 95.5,
            "active": True,
            "tags": ["deep-learning", "attention",],
        }
        ```
        Let me know if you need more analysis.
        """
        result = parse_and_validate(messy_response, SampleItem)
        assert isinstance(result, SampleItem)
        assert result.id == "p123"
        assert result.score == 95.5
        assert result.active is True
        assert result.tags == ["deep-learning", "attention"]

    def test_parse_and_validate_raises_structured_output_error_on_invalid_data(self):
        """Verify StructuredOutputError is raised when schema validation cannot succeed."""
        invalid_response = "```json\n{\"id\": \"p123\", \"score\": \"not_a_number\"}\n```"
        with pytest.raises(StructuredOutputError) as exc_info:
            parse_and_validate(invalid_response, SampleItem)
        assert "SampleItem" in str(exc_info.value)
        assert exc_info.value.schema_name == "SampleItem"

    def test_to_gemini_schema_inlines_defs_recursively(self):
        """Verify to_gemini_schema creates OpenAPI 3.0 schema without unresolved $defs."""
        gemini_schema = to_gemini_schema(ResearchGapItem)
        assert "$defs" not in gemini_schema
        assert "properties" in gemini_schema
        assert "gap_id" in gemini_schema["properties"]
        assert "importance" in gemini_schema["properties"]

    def test_to_openai_response_format(self):
        """Verify OpenAI response_format payload generation."""
        rf = to_openai_response_format(EvidenceMatrixRow)
        assert rf == {"type": "json_object"}
