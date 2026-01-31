# Agent Tools Module
# Defines tools that agents can use to perform specific actions

import json
import re
import logging
from typing import Optional, Any, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standardized result from a tool execution."""
    success: bool
    data: any
    error: Optional[str] = None


def extract_json_from_response(response: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Robustly extract JSON from LLM response, handling common issues.
    
    Handles:
    - Markdown code blocks (```json ... ```)
    - Multiple JSON objects (takes first complete one)
    - Extra text before/after JSON
    - Truncated responses
    
    Args:
        response: Raw LLM response string
        default: Default value if extraction fails
        
    Returns:
        Parsed JSON dict or default
    """
    if not response or not response.strip():
        return default or {}
    
    # Step 1: Clean markdown code blocks
    clean = re.sub(r'```json\s*', '', response)
    clean = re.sub(r'```\s*$', '', clean)
    clean = re.sub(r'```', '', clean)
    clean = clean.strip()
    
    # Step 2: Try direct parsing first
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    
    # Step 3: Find JSON object boundaries using brace matching
    start_idx = clean.find('{')
    if start_idx == -1:
        logger.warning(f"No JSON object found in response: {clean[:100]}...")
        return default or {}
    
    # Track brace depth to find matching closing brace
    depth = 0
    in_string = False
    escape_next = False
    end_idx = start_idx
    
    for i, char in enumerate(clean[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    
    if depth != 0:
        # Truncated JSON - try to fix common issues
        json_str = clean[start_idx:end_idx if end_idx > start_idx else None]
        
        # Try adding missing closing braces
        while depth > 0:
            json_str += '}'
            depth -= 1
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        logger.warning(f"Could not repair truncated JSON: {clean[start_idx:start_idx+100]}...")
        return default or {}
    
    # Extract the complete JSON object
    json_str = clean[start_idx:end_idx]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error after extraction: {e}. Content: {json_str[:100]}...")
        return default or {}


# ============================================
# PLANNER TOOLS
# ============================================

def extract_keywords_from_question(llm_client, research_question: str, title: str) -> ToolResult:
    """
    Tool: Extract search keywords from a research question.
    
    Uses LLM to generate diverse and specific keywords for academic database searches.
    
    Args:
        llm_client: The LLM client to use
        research_question: The research question to analyze
        title: The project title for context
        
    Returns:
        ToolResult with list of keywords
    """
    prompt = f"""You are a research assistant AI. Your task is to generate a list of relevant keywords
    for searching academic databases.

    Research Question: "{research_question}"
    Title: "{title}"

    Generate 8-12 diverse and specific keywords that would help find relevant academic papers.
    Consider:
    - Core concepts and their synonyms
    - Related technical terms
    - Broader and narrower terms
    - Common abbreviations in the field

    Provide the output as a JSON object with a single key "keywords" containing a list of strings.
    Output ONLY the JSON object, no other text.

    Example: {{"keywords": ["machine learning", "ML", "artificial intelligence", "predictive modeling"]}}
    """
    
    try:
        response = llm_client.chat(prompt)
        clean_response = re.sub(r'```json\s*|\s*```', '', response).strip()
        data = json.loads(clean_response)
        keywords = data.get("keywords", [])
        
        if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
            logger.info(f"Extracted {len(keywords)} keywords")
            return ToolResult(success=True, data=keywords)
        else:
            return ToolResult(success=False, data=[], error="Invalid keyword format from LLM")
            
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to extract keywords: {e}")
        return ToolResult(success=False, data=[], error=str(e))


def identify_subtopics(llm_client, research_question: str, title: str) -> ToolResult:
    """
    Tool: Identify subtopics for structuring the literature review.
    
    Args:
        llm_client: The LLM client to use
        research_question: The research question to analyze
        title: The project title for context
        
    Returns:
        ToolResult with list of subtopics
    """
    prompt = f"""You are a research assistant AI. Your task is to identify key subtopics 
    that should be covered in a literature review.

    Research Question: "{research_question}"
    Title: "{title}"

    Generate 4-6 specific subtopics that would help structure a comprehensive literature review.
    Each subtopic should represent a distinct aspect or theme of the research area.

    Provide the output as a JSON object with a single key "subtopics" containing a list of strings.
    Output ONLY the JSON object, no other text.

    Example: {{"subtopics": ["Historical Development", "Current Applications", "Challenges and Limitations", "Future Directions"]}}
    """
    
    try:
        response = llm_client.chat(prompt)
        clean_response = re.sub(r'```json\s*|\s*```', '', response).strip()
        data = json.loads(clean_response)
        subtopics = data.get("subtopics", [])
        
        if isinstance(subtopics, list) and all(isinstance(s, str) for s in subtopics):
            logger.info(f"Identified {len(subtopics)} subtopics")
            return ToolResult(success=True, data=subtopics)
        else:
            return ToolResult(success=False, data=[], error="Invalid subtopic format from LLM")
            
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to identify subtopics: {e}")
        return ToolResult(success=False, data=[], error=str(e))


def refine_search_query(llm_client, original_query: str, found_papers: int, target_papers: int) -> ToolResult:
    """
    Tool: Refine search query if not enough papers were found.
    
    Args:
        llm_client: The LLM client to use
        original_query: The original search keywords
        found_papers: Number of papers found with original query
        target_papers: Target number of papers needed
        
    Returns:
        ToolResult with refined keywords
    """
    prompt = f"""You are a research assistant AI. The current search query did not find enough papers.

    Original Search Terms: {original_query}
    Papers Found: {found_papers}
    Target Papers: {target_papers}

    Please suggest refined or alternative search terms that might find more relevant papers.
    Consider:
    - Broader terms that might capture more results
    - Alternative phrasings
    - Related concepts
    - Removing overly specific terms

    Provide the output as a JSON object with a single key "refined_keywords" containing a list of strings.
    Output ONLY the JSON object, no other text.
    """
    
    try:
        response = llm_client.chat(prompt)
        clean_response = re.sub(r'```json\s*|\s*```', '', response).strip()
        data = json.loads(clean_response)
        refined = data.get("refined_keywords", [])
        
        if isinstance(refined, list):
            logger.info(f"Generated {len(refined)} refined keywords")
            return ToolResult(success=True, data=refined)
        else:
            return ToolResult(success=False, data=[], error="Invalid format")
            
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to refine query: {e}")
        return ToolResult(success=False, data=[], error=str(e))


# ============================================
# ANALYZER TOOLS
# ============================================

def score_paper_relevance(llm_client, title: str, abstract: str, research_question: str) -> ToolResult:
    """
    Tool: Score a paper's relevance to the research question.
    
    Args:
        llm_client: The LLM client to use
        title: Paper title
        abstract: Paper abstract
        research_question: The research question
        
    Returns:
        ToolResult with relevance score (0-100) and justification
    """
    prompt = f"""You are a Paper Analyzer agent. Score the relevance of this paper to the research question.

    Paper Title: "{title}"
    Abstract: "{abstract[:1500]}"  
    Research Question: "{research_question}"

    Provide a relevance score from 0-100 where:
    - 0-30: Not relevant
    - 31-60: Somewhat relevant
    - 61-80: Relevant
    - 81-100: Highly relevant

    Output as JSON with "score" (integer) and "justification" (brief string, max 100 chars).
    Output ONLY the JSON object, no explanation.

    Example: {{"score": 75, "justification": "Directly addresses ML in education."}}
    """
    
    try:
        response = llm_client.chat(prompt, task_type="relevance_scoring")
        data = extract_json_from_response(response, {"score": 50, "justification": "Could not parse response"})
        
        score = int(data.get("score", 50))
        # Clamp score to valid range
        score = max(0, min(100, score))
        justification = str(data.get("justification", ""))[:200]
        
        return ToolResult(
            success=True,
            data={"score": score, "justification": justification}
        )
        
    except Exception as e:
        logger.error(f"Failed to score relevance: {e}")
        # Return a moderate score instead of failing completely
        return ToolResult(
            success=True,  # Don't fail the whole pipeline
            data={"score": 50, "justification": f"Score estimation failed: {str(e)[:50]}"},
            error=str(e)
        )


def extract_paper_insights(llm_client, title: str, abstract: str, research_question: str) -> ToolResult:
    """
    Tool: Extract detailed insights from a paper.
    
    Args:
        llm_client: The LLM client to use
        title: Paper title
        abstract: Paper abstract
        research_question: The research question for context
        
    Returns:
        ToolResult with structured paper analysis
    """
    # Truncate abstract to avoid token limits
    abstract_truncated = abstract[:1500] if abstract else ""
    
    prompt = f"""You are a Paper Analyzer agent. Extract key insights concisely.
    
    Paper Title: "{title}"
    Abstract: "{abstract_truncated}"
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
            "key_quotes": []
        }
        
        data = extract_json_from_response(response, default_data)
        
        # Ensure all required keys exist with proper types
        if not isinstance(data.get("key_findings"), list):
            data["key_findings"] = [str(data.get("key_findings", ""))] if data.get("key_findings") else []
        if not isinstance(data.get("limitations"), list):
            data["limitations"] = [str(data.get("limitations", ""))] if data.get("limitations") else []
        if not isinstance(data.get("key_quotes"), list):
            data["key_quotes"] = [str(data.get("key_quotes", ""))] if data.get("key_quotes") else []
        
        data["methodology"] = str(data.get("methodology", ""))[:300]
        data["contribution"] = str(data.get("contribution", ""))[:300]
        
        return ToolResult(success=True, data=data)
        
    except Exception as e:
        logger.error(f"Failed to extract insights: {e}")
        return ToolResult(
            success=True,  # Don't fail pipeline
            data={
                "key_findings": [f"Extraction failed for: {title[:50]}"],
                "methodology": "Could not determine",
                "limitations": [],
                "contribution": "Analysis incomplete",
                "key_quotes": []
            },
            error=str(e)
        )


# ============================================
# SYNTHESIZER TOOLS
# ============================================

def synthesize_section(
    llm_client,
    subtopic: str,
    paper_analyses: list[dict],
    academic_level: str,
    word_count: int
) -> ToolResult:
    """
    Tool: Synthesize a literature review section from analyzed papers.
    
    Args:
        llm_client: The LLM client to use
        subtopic: The section topic
        paper_analyses: List of paper analysis dictionaries
        academic_level: Writing level (e.g., "graduate")
        word_count: Target word count
        
    Returns:
        ToolResult with synthesized text
    """
    # Format paper analyses for the prompt
    analyses_text = "\n\n---\n\n".join([
        f"Paper: {pa.get('title', 'Unknown')}\n"
        f"Findings: {pa.get('key_findings', [])}\n"
        f"Methodology: {pa.get('methodology', '')}\n"
        f"Contribution: {pa.get('contribution', '')}"
        for pa in paper_analyses
    ])
    
    prompt = f"""You are a Synthesis Executor agent specializing in academic writing.
    
    TASK: Create a literature review section synthesizing the following analyzed papers.
    
    Section Topic: {subtopic}
    Writing Style: Academic, formal, suitable for {academic_level} level
    Target Length: {word_count} words
    
    Analyzed Papers:
    {analyses_text}
    
    OUTPUT FORMAT:
    1. Section Introduction: Overview of topic and scope
    2. Thematic Organization: Group findings by themes/approaches
    3. Critical Analysis: Compare/contrast findings, identify patterns
    4. Research Gaps: Highlight limitations and future directions
    5. Section Conclusion: Synthesize key takeaways
    
    Write the section now. Do NOT include a References section.
    """
    
    try:
        response = llm_client.chat(prompt)
        if response and len(response) > 50:
            return ToolResult(success=True, data=response)
        else:
            return ToolResult(success=False, data="", error="Empty or too short response")
            
    except Exception as e:
        logger.error(f"Failed to synthesize section: {e}")
        return ToolResult(success=False, data="", error=str(e))


def identify_research_gaps(llm_client, paper_analyses: list[dict], research_question: str) -> ToolResult:
    """
    Tool: Identify research gaps from the analyzed papers.
    
    Args:
        llm_client: The LLM client to use
        paper_analyses: List of paper analysis dictionaries
        research_question: The research question for context
        
    Returns:
        ToolResult with identified research gaps
    """
    # Summarize limitations from all papers
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
    For each gap, provide:
    - A clear description of the gap
    - Why it matters
    - Potential research directions

    Output as JSON with key "research_gaps" containing a list of objects with keys "description", "importance", "directions".
    Output ONLY the JSON object.
    """
    
    try:
        response = llm_client.chat(prompt)
        clean_response = re.sub(r'```json\s*|\s*```', '', response).strip()
        data = json.loads(clean_response)
        
        gaps = data.get("research_gaps", [])
        return ToolResult(success=True, data=gaps)
        
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to identify research gaps: {e}")
        return ToolResult(success=False, data=[], error=str(e))


# ============================================
# QUALITY CHECKER TOOLS
# ============================================

def evaluate_synthesis_quality(llm_client, synthesis: str, research_question: str, paper_count: int) -> ToolResult:
    """
    Tool: Evaluate the quality of the synthesized literature review.
    
    Args:
        llm_client: The LLM client to use
        synthesis: The synthesized text to evaluate
        research_question: The research question for context
        paper_count: Number of papers that were analyzed
        
    Returns:
        ToolResult with quality score and feedback
    """
    prompt = f"""You are a Quality Evaluator for academic literature reviews.

    Research Question: "{research_question}"
    Number of Papers Analyzed: {paper_count}
    
    Literature Review to Evaluate:
    {synthesis[:3000]}  # Truncate if too long

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

    Output ONLY the JSON object.
    """
    
    try:
        response = llm_client.chat(prompt)
        clean_response = re.sub(r'```json\s*|\s*```', '', response).strip()
        data = json.loads(clean_response)
        
        return ToolResult(
            success=True,
            data={
                "overall_score": data.get("overall_score", 0),
                "criteria_scores": data.get("criteria_scores", {}),
                "feedback": data.get("feedback", ""),
                "should_refine": data.get("should_refine", False)
            }
        )
        
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to evaluate quality: {e}")
        return ToolResult(
            success=False,
            data={"overall_score": 0, "feedback": str(e), "should_refine": True},
            error=str(e)
        )
