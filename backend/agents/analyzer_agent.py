# Paper Analyzer Agent (LangGraph Compatible)
# Role: Analyzes papers for relevance and extracts key insights


from agents.base import ToolEnabledAgent
from agents.state import AgentResult, AgentState, AgentType
from agents.tools import extract_paper_insights, score_paper_relevance


class PaperAnalyzerAgent(ToolEnabledAgent):
    """
    Analyzer Agent responsible for:
    - Scoring paper relevance to research question
    - Extracting key findings, methodology, and limitations
    - Filtering high-quality papers for synthesis

    Uses the ReAct pattern for each paper analysis.
    """

    def __init__(self, llm_client):
        super().__init__(llm_client, name="analyzer")
        self._register_tools()

    def _register_tools(self):
        """Register the tools this agent can use."""
        self.register_tool(
            "score_relevance",
            lambda **kwargs: score_paper_relevance(self.llm_client, **kwargs),
            "Score a paper's relevance to the research question",
        )
        self.register_tool(
            "extract_insights",
            lambda **kwargs: extract_paper_insights(self.llm_client, **kwargs),
            "Extract detailed insights from a paper",
        )

    async def run(self, state: AgentState) -> AgentState:
        """
        Execute the analyzer agent.

        Workflow:
        1. For each paper in state:
           a. Score relevance
           b. If relevant, extract insights
        2. Filter high-quality papers
        3. Update state with analyses

        Args:
            state: Current pipeline state

        Returns:
            Updated state with analyzed papers
        """
        self._log_start(state)

        try:
            state["current_agent"] = AgentType.ANALYZER

            papers = state.get("papers", [])
            research_question = state["research_question"]
            relevance_threshold = state.get("relevance_threshold", 60.0)

            if not papers:
                self.logger.warning("No papers to analyze")
                state["errors"].append("No papers available for analysis")
                return state

            analyzed_papers = []
            high_quality_papers = []

            for i, paper in enumerate(papers):
                self.logger.info(f"Analyzing paper {i+1}/{len(papers)}: {paper['title'][:50]}...")

                # Skip papers without abstract
                if not paper.get("abstract"):
                    self.logger.warning(f"Skipping paper without abstract: {paper['title']}")
                    continue

                # Score relevance
                relevance_result = await self.invoke_tool(
                    "score_relevance",
                    title=paper["title"],
                    abstract=paper["abstract"],
                    research_question=research_question,
                )

                relevance_score = 0.0
                justification = ""

                if relevance_result.success:
                    relevance_score = relevance_result.data.get("score", 0)
                    justification = relevance_result.data.get("justification", "")

                # Update paper with relevance score
                paper["relevance_score"] = relevance_score

                # Only extract detailed insights for relevant papers
                if relevance_score >= relevance_threshold:
                    insights_result = await self.invoke_tool(
                        "extract_insights",
                        title=paper["title"],
                        abstract=paper["abstract"],
                        research_question=research_question,
                    )

                    if insights_result.success:
                        paper["analysis"] = {
                            "relevance_score": relevance_score,
                            "justification": justification,
                            **insights_result.data,
                        }
                        high_quality_papers.append(paper)
                    else:
                        paper["analysis"] = {
                            "relevance_score": relevance_score,
                            "justification": justification,
                            "error": insights_result.error,
                        }
                else:
                    paper["analysis"] = {
                        "relevance_score": relevance_score,
                        "justification": justification,
                        "skipped": True,
                        "reason": "Below relevance threshold",
                    }

                analyzed_papers.append(paper)

                # Rate limiting - avoid overwhelming the LLM
                import asyncio

                await asyncio.sleep(1.0)

            # Update state
            state["analyzed_papers"] = analyzed_papers
            state["high_quality_papers"] = high_quality_papers

            state["messages"] = [
                self._create_message(
                    "analyze_papers",
                    f"Analyzed {len(analyzed_papers)} papers, {len(high_quality_papers)} above threshold",
                )
            ]

            result = AgentResult(
                success=True,
                data={
                    "total_analyzed": len(analyzed_papers),
                    "high_quality": len(high_quality_papers),
                },
                metadata={
                    "relevance_threshold": relevance_threshold,
                    "avg_relevance": (
                        sum(p.get("relevance_score", 0) or 0 for p in analyzed_papers)
                        / len(analyzed_papers)
                        if analyzed_papers
                        else 0
                    ),
                },
            )
            self._log_complete(state, result)

            return state

        except Exception as e:
            return self._handle_error(state, e)

    # Legacy method for backward compatibility
    def analyze_paper(self, title: str, abstract: str, content: str, research_question: str) -> str:
        """
        Legacy method for backward compatibility with existing code.

        Returns raw LLM response string.
        """
        prompt = f"""You are a Paper Analyzer agent specializing in academic content extraction.
        TASK: Analyze the academic paper for its relevance to the research context and extract key insights.
        
        Paper Title: "{title}"
        Abstract: "{abstract}"
        Research Context: "{research_question}"
        
        OUTPUT: Provide your analysis in a structured JSON format. The JSON object must contain the following keys:
        - "relevance_score": An integer between 0 and 100.
        - "justification": A brief (1-2 sentence) justification for the score.
        - "key_findings": A list of 3-5 strings, where each string is a primary research finding.
        - "methodology": A brief description of the research methods and sample.
        - "limitations": A list of strings, where each string is an identified study limitation.
        - "contribution": A brief description of how this paper fits into the broader research landscape.
        - "key_quotes": A list of 2-3 strings of key quotable insights.

        Example JSON output format:
        {{
            "relevance_score": 85,
            "justification": "The paper directly addresses the use of machine learning to predict student performance, which is highly relevant to the research question.",
            "key_findings": ["Finding 1...", "Finding 2..."],
            "methodology": "The study used a quantitative approach with a sample of 500 university students.",
            "limitations": ["The study was conducted at a single institution.", "The dataset used was limited in scope."],
            "contribution": "This work provides a foundational model for predicting student success factors.",
            "key_quotes": ["Quote 1...", "Quote 2..."]
        }}
        
        CRITICAL: Output ONLY the JSON object. Do NOT include any preamble text (like "Here's the analysis" or "Here is the JSON"), explanations, or markdown code blocks. Start your response directly with {{ and end with }}.
        """
        return self.llm_client.chat(prompt)
