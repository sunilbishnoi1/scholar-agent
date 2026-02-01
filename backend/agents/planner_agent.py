# Research Planner Agent (LangGraph Compatible)
# Role: Analyzes research questions, creates search strategies, identifies key subtopics

import json
import logging
import re

from agents.base import ToolEnabledAgent
from agents.state import AgentResult, AgentState, AgentType
from agents.tools import extract_keywords_from_question, identify_subtopics, refine_search_query


class ResearchPlannerAgent(ToolEnabledAgent):
    """
    Planner Agent responsible for:
    - Extracting search keywords from research questions
    - Identifying subtopics for literature review structure
    - Refining search strategies when needed

    Uses the ReAct pattern: Reason about the task, then Act using tools.
    """

    def __init__(self, llm_client):
        super().__init__(llm_client, name="planner")
        self._register_tools()

    def _register_tools(self):
        """Register the tools this agent can use."""
        self.register_tool(
            "extract_keywords",
            lambda **kwargs: extract_keywords_from_question(self.llm_client, **kwargs),
            "Extract search keywords from a research question",
        )
        self.register_tool(
            "identify_subtopics",
            lambda **kwargs: identify_subtopics(self.llm_client, **kwargs),
            "Identify subtopics for structuring the literature review",
        )
        self.register_tool(
            "refine_search_query",
            lambda **kwargs: refine_search_query(self.llm_client, **kwargs),
            "Refine search query if not enough papers found",
        )

    async def run(self, state: AgentState) -> AgentState:
        """
        Execute the planner agent.

        Workflow:
        1. Extract keywords from research question
        2. Identify subtopics for literature review structure
        3. Create search strategy

        Args:
            state: Current pipeline state

        Returns:
            Updated state with keywords, subtopics, and search strategy
        """
        self._log_start(state)

        try:
            # Update current agent
            state["current_agent"] = AgentType.PLANNER

            # Tool 1: Extract keywords
            keywords_result = await self.invoke_tool(
                "extract_keywords",
                research_question=state["research_question"],
                title=state["title"],
            )

            if keywords_result.success:
                state["keywords"] = keywords_result.data
                state["messages"] = [
                    self._create_message(
                        "extract_keywords", f"Extracted {len(keywords_result.data)} keywords"
                    )
                ]
            else:
                self.logger.warning(f"Keyword extraction failed: {keywords_result.error}")
                # Fall back to basic keyword extraction
                state["keywords"] = self._fallback_keyword_extraction(
                    state["research_question"], state["title"]
                )

            # Tool 2: Identify subtopics
            subtopics_result = await self.invoke_tool(
                "identify_subtopics",
                research_question=state["research_question"],
                title=state["title"],
            )

            if subtopics_result.success:
                state["subtopics"] = subtopics_result.data
                state["messages"] = [
                    self._create_message(
                        "identify_subtopics", f"Identified {len(subtopics_result.data)} subtopics"
                    )
                ]
            else:
                self.logger.warning(f"Subtopic identification failed: {subtopics_result.error}")
                state["subtopics"] = ["Comprehensive Literature Review"]

            # Create search strategy
            state["search_strategy"] = {
                "primary_keywords": (
                    state["keywords"][:5] if len(state["keywords"]) > 5 else state["keywords"]
                ),
                "secondary_keywords": state["keywords"][5:] if len(state["keywords"]) > 5 else [],
                "max_papers_per_source": state["max_papers"] // 2,
                "sources": ["arXiv", "Semantic Scholar"],
            }

            # Log success
            result = AgentResult(
                success=True,
                data={"keywords": state["keywords"], "subtopics": state["subtopics"]},
                metadata={
                    "keyword_count": len(state["keywords"]),
                    "subtopic_count": len(state["subtopics"]),
                },
            )
            self._log_complete(state, result)

            return state

        except Exception as e:
            return self._handle_error(state, e)

    def _fallback_keyword_extraction(self, research_question: str, title: str) -> list[str]:
        """
        Fallback keyword extraction when LLM fails.
        Uses simple text processing.
        """
        # Combine title and question
        text = f"{title} {research_question}".lower()

        # Remove common words
        stopwords = {
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "how",
            "what",
            "why",
            "when",
            "does",
            "do",
            "can",
            "could",
            "would",
            "should",
        }

        # Extract words
        words = re.findall(r"\b[a-z]{3,}\b", text)
        keywords = [w for w in words if w not in stopwords]

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique_keywords.append(k)

        return unique_keywords[:10]

    # Legacy method for backward compatibility
    def generate_initial_plan(self, research_question: str, title: str) -> dict:
        """
        Legacy method for backward compatibility with existing code.

        Synchronously generates keywords and subtopics.
        """
        prompt = f"""
        You are a research assistant AI. Your task is to generate a list of relevant keywords
        and a list of research subtopics for a given research question and title.

        Research Question: "{research_question}"
        Title: "{title}"

        Please generate:
        1. A list of 8-12 diverse and specific keywords for searching academic databases.
        2. A list of 4-6 specific subtopics to structure the literature review.

        Provide the output as a single JSON object with two keys: "keywords" and "subtopics".
        Ensure the output is ONLY the JSON object, without any surrounding text or markdown.

        Example Output:
        {{
            "keywords": ["keyword1", "keyword2", "keyword3"],
            "subtopics": ["Subtopic A", "Subtopic B", "Subtopic C"]
        }}
        """
        response_str = self.llm_client.chat(prompt)
        try:
            clean_response = re.sub(r"```json\s*|\s*```", "", response_str).strip()
            data = json.loads(clean_response)

            keywords = data.get("keywords", [])
            subtopics = data.get("subtopics", [])

            if (
                isinstance(keywords, list)
                and all(isinstance(k, str) for k in keywords)
                and isinstance(subtopics, list)
                and all(isinstance(s, str) for s in subtopics)
            ):
                return data
            else:
                logging.warning(
                    f"LLM returned a malformed JSON object. Fallback required. Response: {response_str}"
                )
                return {"keywords": [], "subtopics": []}

        except (json.JSONDecodeError, AttributeError) as e:
            logging.error(
                f"Failed to parse JSON for initial plan, returning empty. Error: {e}. Raw response: {response_str}"
            )
            return {"keywords": [], "subtopics": []}
