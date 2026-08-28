"""
Legacy Agent Tools Module for backwards compatibility with earlier agent stubs.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standardized result from a tool execution."""

    success: bool
    data: Any
    error: str | None = None


def extract_json_from_response(response: str, default: dict | None = None) -> dict[str, Any]:
    """
    Robustly extract JSON from LLM response, handling common issues.
    """
    if not response or not response.strip():
        return default or {}

    text = response.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    else:
        start_brace = text.find("{")
        end_brace = text.rfind("}")
        if start_brace != -1 and end_brace > start_brace:
            text = text[start_brace : end_brace + 1].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", text)
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"\bNone\b", "null", repaired)
        try:
            return json.loads(repaired)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return default or {}


# ============================================
# PLANNER TOOLS
# ============================================


def extract_keywords_from_question(llm_client: Any, research_question: str, title: str) -> ToolResult:
    """
    Tool: Extract search keywords from a research question.
    """
    prompt = f"""You are a research assistant AI. Your task is to generate a list of relevant keywords
    for searching academic databases.

    Research Question: "{research_question}"
    Title: "{title}"

    Generate 8-12 diverse and specific keywords that would help find relevant academic papers.
    Provide the output as a JSON object with a single key "keywords" containing a list of strings.
    
    CRITICAL: Output ONLY the JSON object. Do NOT include any preamble, explanations, or markdown. Start directly with {{ and end with }}.

    Example: {{"keywords": ["machine learning", "ML", "artificial intelligence", "predictive modeling"]}}
    """

    try:
        response = llm_client.chat(prompt)
        clean_response = re.sub(r"```json\s*|\s*```", "", response).strip()
        data = json.loads(clean_response)
        keywords = data.get("keywords", [])

        if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
            logger.info(f"Extracted {len(keywords)} keywords")
            return ToolResult(success=True, data=keywords)
        else:
            return ToolResult(success=False, data=[], error="Invalid keyword format from LLM")

    except Exception as e:
        logger.error(f"Failed to extract keywords: {e}")
        return ToolResult(success=False, data=[], error=str(e))


def identify_subtopics(llm_client: Any, research_question: str, title: str) -> ToolResult:
    """
    Tool: Identify subtopics for structuring the literature review.
    """
    prompt = f"""You are a research assistant AI. Your task is to identify key subtopics 
    that should be covered in a literature review.

    Research Question: "{research_question}"
    Title: "{title}"

    Generate 4-6 specific subtopics that would help structure a comprehensive literature review.
    Provide the output as a JSON object with a single key "subtopics" containing a list of strings.
    
    CRITICAL: Output ONLY the JSON object. Do NOT include any preamble, explanations, or markdown. Start directly with {{ and end with }}.

    Example: {{"subtopics": ["Historical Development", "Current Applications", "Challenges and Limitations", "Future Directions"]}}
    """

    try:
        response = llm_client.chat(prompt)
        clean_response = re.sub(r"```json\s*|\s*```", "", response).strip()
        data = json.loads(clean_response)
        subtopics = data.get("subtopics", [])

        if isinstance(subtopics, list) and all(isinstance(s, str) for s in subtopics):
            logger.info(f"Identified {len(subtopics)} subtopics")
            return ToolResult(success=True, data=subtopics)
        else:
            return ToolResult(success=False, data=[], error="Invalid subtopic format from LLM")

    except Exception as e:
        logger.error(f"Failed to identify subtopics: {e}")
        return ToolResult(success=False, data=[], error=str(e))


def refine_search_query(
    llm_client: Any, original_query: str, found_papers: int, target_papers: int
) -> ToolResult:
    """
    Tool: Refine search query if not enough papers were found.
    """
    prompt = f"""You are a research assistant AI. The current search query did not find enough papers.

    Original Search Terms: {original_query}
    Papers Found: {found_papers}
    Target Papers: {target_papers}

    Please suggest refined or alternative search terms that might find more relevant papers.
    Provide the output as a JSON object with a single key "refined_keywords" containing a list of strings.
    
    CRITICAL: Output ONLY the JSON object. Do NOT include any preamble, explanations, or markdown. Start directly with {{ and end with }}.
    """

    try:
        response = llm_client.chat(prompt)
        clean_response = re.sub(r"```json\s*|\s*```", "", response).strip()
        data = json.loads(clean_response)
        refined = data.get("refined_keywords", [])

        if isinstance(refined, list):
            logger.info(f"Generated {len(refined)} refined keywords")
            return ToolResult(success=True, data=refined)
        else:
            return ToolResult(success=False, data=[], error="Invalid format")

    except Exception as e:
        logger.error(f"Failed to refine query: {e}")
        return ToolResult(success=False, data=[], error=str(e))


# ============================================
# ANALYZER TOOLS
# ============================================


def score_paper_relevance(
    llm_client: Any, title: str, abstract: str, research_question: str
) -> ToolResult:
    """
    Tool: Score a paper's relevance to the research question.
    """
    prompt = f"""You are a Paper Analyzer agent. Score the relevance of this paper to the research question.

    Paper Title: "{title}"
    Abstract: "{abstract}"  
    Research Question: "{research_question}"

    Provide a relevance score from 0-100 where:
    - 0-30: Not relevant
    - 31-60: Somewhat relevant
    - 61-80: Relevant
    - 81-100: Highly relevant

    Output as JSON with "score" (integer) and "justification" (brief string, max 100 chars).
    
    CRITICAL: Output ONLY the JSON object. Do NOT include any preamble, explanations, or markdown. Start directly with {{ and end with }}.

    Example: {{"score": 75, "justification": "Directly addresses ML in education."}}
    """

    try:
        response = llm_client.chat(prompt, task_type="relevance_scoring")
        data = extract_json_from_response(
            response, {"score": 50, "justification": "Could not parse response"}
        )

        score = int(data.get("score", 50))
        score = max(0, min(100, score))
        justification = str(data.get("justification", ""))

        return ToolResult(success=True, data={"score": score, "justification": justification})

    except Exception as e:
        logger.error(f"Failed to score relevance: {e}")
        return ToolResult(
            success=True,
            data={"score": 50, "justification": f"Score estimation failed: {str(e)[:50]}"},
            error=str(e),
        )


def extract_paper_insights(
    llm_client: Any, title: str, abstract: str, research_question: str
) -> ToolResult:
    """
    Tool: Extract detailed insights from a paper.
    """
    abstract_text = abstract if abstract else ""

    prompt = f"""You are a Paper Analyzer agent. Extract key insights concisely.
    
    Paper Title: "{title}"
    Abstract: "{abstract_text}"
    Research Context: "{research_question}"
    
    Output JSON with:
    - "key_findings": List of 2-3 findings (strings)
    - "methodology": Brief method description (1 sentence)
    - "limitations": List of 1-2 limitations
    - "contribution": One sentence on contribution
    - "key_quotes": 1-2 brief quotes

    Output ONLY valid JSON, no explanation.
    """

    try:
        response = llm_client.chat(prompt, task_type="paper_analysis")

        default_data = {
            "key_findings": ["Analysis could not extract findings"],
            "methodology": "Not determined",
            "limitations": [],
            "contribution": "Paper contribution unclear",
            "key_quotes": [],
        }

        data = extract_json_from_response(response, default_data)

        if not isinstance(data.get("key_findings"), list):
            data["key_findings"] = (
                [str(data.get("key_findings", ""))] if data.get("key_findings") else []
            )
        if not isinstance(data.get("limitations"), list):
            data["limitations"] = (
                [str(data.get("limitations", ""))] if data.get("limitations") else []
            )
        if not isinstance(data.get("key_quotes"), list):
            data["key_quotes"] = [str(data.get("key_quotes", ""))] if data.get("key_quotes") else []

        data["methodology"] = str(data.get("methodology", ""))
        data["contribution"] = str(data.get("contribution", ""))

        return ToolResult(success=True, data=data)

    except Exception as e:
        logger.error(f"Failed to extract insights: {e}")
        return ToolResult(
            success=True,
            data={
                "key_findings": [f"Extraction failed for: {title[:50]}"],
                "methodology": "Could not determine",
                "limitations": [],
                "contribution": "Analysis incomplete",
                "key_quotes": [],
            },
            error=str(e),
        )


# ============================================
# SYNTHESIZER TOOLS
# ============================================


def synthesize_section(
    llm_client: Any,
    subtopic: str,
    paper_analyses: list[dict],
    academic_level: str,
    word_count: int,
    is_final_synthesis: bool = True,
) -> ToolResult:
    """
    Tool: Synthesize a literature review section from analyzed papers.
    """
    analyses_text = "\n\n---\n\n".join(
        [
            f"Paper: {pa.get('title', 'Unknown')}\n"
            f"Findings: {pa.get('key_findings', [])}\n"
            f"Methodology: {pa.get('methodology', '')}\n"
            f"Contribution: {pa.get('contribution', '')}"
            for pa in paper_analyses
        ]
    )

    prompt = f"""You are a Synthesis Executor agent specializing in academic writing.
    
    TASK: Create a literature review section synthesizing the following analyzed papers.
    
    Section Topic: {subtopic}
    Writing Style: Academic, formal, suitable for {academic_level} level
    Target Length: {word_count} words
    
    Analyzed Papers:
    {analyses_text}
    
    OUTPUT FORMAT INSTRUCTIONS:
    - Output the content as standard Markdown text.
    - Use ## for main section headers.
    - Do NOT output JSON, XML, or any other structured data format.
    - Do NOT include any preamble or "Here is the section" text.
    
    Write the section now in Markdown. Do NOT include a References section.
    """

    try:
        response = llm_client.chat(prompt, critical_priority=is_final_synthesis)
        if response and len(response) > 50:
            return ToolResult(success=True, data=response)
        else:
            return ToolResult(success=False, data="", error="Empty or too short response")

    except Exception as e:
        logger.error(f"Failed to synthesize section: {e}")
        return ToolResult(success=False, data="", error=str(e))


def identify_research_gaps(
    llm_client: Any, paper_analyses: list[dict], research_question: str
) -> ToolResult:
    """
    Tool: Identify research gaps from the analyzed papers.
    """
    all_limitations = []
    for pa in paper_analyses:
        limitations = pa.get("limitations", [])
        if isinstance(limitations, list):
            all_limitations.extend(limitations)

    prompt = f"""You are a Research Gap Identifier. Analyze the following information to identify significant research gaps.

    Research Question: "{research_question}"
    
    Limitations identified across papers:
    {all_limitations}

    Number of papers analyzed: {len(paper_analyses)}

    Identify 3-5 significant research gaps that future research should address.
    Output as JSON with key "research_gaps" containing a list of objects with keys "description", "importance", "directions".
    
    CRITICAL: Output ONLY the JSON object. Do NOT include any preamble, explanations, or markdown. Start directly with {{ and end with }}.
    """

    try:
        response = llm_client.chat(prompt)
        clean_response = re.sub(r"```json\s*|\s*```", "", response).strip()
        data = json.loads(clean_response)

        gaps = data.get("research_gaps", [])
        return ToolResult(success=True, data=gaps)

    except Exception as e:
        logger.error(f"Failed to identify research gaps: {e}")
        return ToolResult(success=False, data=[], error=str(e))


# ============================================
# QUALITY CHECKER TOOLS
# ============================================


def evaluate_synthesis_quality(
    llm_client: Any, synthesis: str, research_question: str, paper_count: int
) -> ToolResult:
    """
    Tool: Evaluate the quality of the synthesized literature review.
    """
    prompt = f"""You are a Quality Evaluator for academic literature reviews.

    Research Question: "{research_question}"
    Number of Papers Analyzed: {paper_count}
    
    Literature Review to Evaluate:
    {synthesis}

    Evaluate the quality on these criteria (score each 0-100):
    1. Coherence: Does it flow logically?
    2. Coverage: Does it adequately cover the topic?
    3. Critical Analysis: Does it compare/contrast findings?
    4. Academic Tone: Is the writing appropriately academic?
    5. Research Gaps: Does it identify gaps for future research?

    Output as JSON with:
    - "overall_score": Average of all criteria (0-100)
    - "criteria_scores": Object with each criterion's score
    - "feedback": Specific suggestions for improvement
    - "should_refine": Boolean, true if score < 70

    CRITICAL: Output ONLY the JSON object. Do NOT include any preamble, explanations, or markdown. Start directly with {{ and end with }}.
    """

    try:
        response = llm_client.chat(prompt)
        clean_response = re.sub(r"```json\s*|\s*```", "", response).strip()
        data = json.loads(clean_response)

        return ToolResult(
            success=True,
            data={
                "overall_score": data.get("overall_score", 0),
                "criteria_scores": data.get("criteria_scores", {}),
                "feedback": data.get("feedback", ""),
                "should_refine": data.get("should_refine", False),
            },
        )

    except Exception as e:
        logger.error(f"Failed to evaluate quality: {e}")
        return ToolResult(
            success=False,
            data={"overall_score": 0, "feedback": str(e), "should_refine": True},
            error=str(e),
        )
