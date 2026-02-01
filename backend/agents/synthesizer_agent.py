# Synthesis Executor Agent (LangGraph Compatible)
# Role: Combines analyzed papers into coherent literature review sections


from agents.base import ToolEnabledAgent
from agents.state import AgentResult, AgentState, AgentType
from agents.tools import identify_research_gaps, synthesize_section


class SynthesisExecutorAgent(ToolEnabledAgent):
    """
    Synthesizer Agent responsible for:
    - Creating literature review sections from analyzed papers
    - Identifying research gaps
    - Maintaining academic writing style
    
    This is the final agent in the main pipeline before quality checking.
    """

    def __init__(self, llm_client):
        super().__init__(llm_client, name="synthesizer")
        self._register_tools()

    def _register_tools(self):
        """Register the tools this agent can use."""
        self.register_tool(
            "synthesize_section",
            lambda **kwargs: synthesize_section(self.llm_client, **kwargs),
            "Synthesize a literature review section from analyzed papers"
        )
        self.register_tool(
            "identify_research_gaps",
            lambda **kwargs: identify_research_gaps(self.llm_client, **kwargs),
            "Identify research gaps from the analyzed papers"
        )

    async def run(self, state: AgentState) -> AgentState:
        """
        Execute the synthesizer agent.
        
        Workflow:
        1. Get high-quality papers from state
        2. For each subtopic, synthesize a section
        3. Identify research gaps
        4. Combine into final synthesis
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated state with synthesis
        """
        self._log_start(state)

        try:
            state["current_agent"] = AgentType.SYNTHESIZER

            high_quality_papers = state.get("high_quality_papers", [])
            subtopics = state.get("subtopics", ["Comprehensive Literature Review"])
            academic_level = state.get("academic_level", "graduate")
            target_word_count = state.get("target_word_count", 500)

            if not high_quality_papers:
                self.logger.warning("No high-quality papers for synthesis")
                # Use all analyzed papers as fallback
                high_quality_papers = state.get("analyzed_papers", [])

                if not high_quality_papers:
                    state["errors"].append("No papers available for synthesis")
                    state["synthesis"] = "Unable to generate synthesis: No papers were found or analyzed."
                    return state

            # Prepare paper analyses for synthesis
            paper_analyses = []
            for paper in high_quality_papers:
                analysis = paper.get("analysis", {})
                paper_analyses.append({
                    "title": paper.get("title", "Unknown"),
                    "key_findings": analysis.get("key_findings", []),
                    "methodology": analysis.get("methodology", ""),
                    "limitations": analysis.get("limitations", []),
                    "contribution": analysis.get("contribution", ""),
                    "relevance_score": analysis.get("relevance_score", 0)
                })

            # Synthesize sections
            synthesis_sections = []

            # Calculate word count per section
            words_per_section = target_word_count // len(subtopics) if subtopics else target_word_count

            for subtopic in subtopics:
                self.logger.info(f"Synthesizing section: {subtopic}")

                section_result = await self.invoke_tool(
                    "synthesize_section",
                    subtopic=subtopic,
                    paper_analyses=paper_analyses,
                    academic_level=academic_level,
                    word_count=words_per_section
                )

                if section_result.success:
                    synthesis_sections.append({
                        "subtopic": subtopic,
                        "content": section_result.data
                    })
                else:
                    self.logger.error(f"Failed to synthesize section '{subtopic}': {section_result.error}")
                    synthesis_sections.append({
                        "subtopic": subtopic,
                        "content": f"[Section generation failed: {section_result.error}]"
                    })

            # Identify research gaps
            gaps_result = await self.invoke_tool(
                "identify_research_gaps",
                paper_analyses=paper_analyses,
                research_question=state["research_question"]
            )

            research_gaps = []
            if gaps_result.success:
                research_gaps = gaps_result.data

            # Combine all sections into final synthesis
            final_synthesis = self._combine_sections(
                synthesis_sections,
                research_gaps,
                state["title"],
                len(high_quality_papers)
            )

            # Update state
            state["synthesis"] = final_synthesis
            state["synthesis_sections"] = synthesis_sections

            state["messages"] = [self._create_message(
                "synthesize",
                f"Generated {len(synthesis_sections)} sections with {len(research_gaps)} identified research gaps"
            )]

            result = AgentResult(
                success=True,
                data={
                    "sections": len(synthesis_sections),
                    "research_gaps": len(research_gaps),
                    "word_count": len(final_synthesis.split())
                },
                metadata={
                    "papers_used": len(high_quality_papers),
                    "subtopics": subtopics
                }
            )
            self._log_complete(state, result)

            return state

        except Exception as e:
            return self._handle_error(state, e)

    def _combine_sections(
        self,
        sections: list[dict],
        research_gaps: list[dict],
        title: str,
        paper_count: int
    ) -> str:
        """
        Combine all synthesized sections into a final document.
        
        Args:
            sections: List of section dictionaries with subtopic and content
            research_gaps: List of identified research gaps
            title: Project title
            paper_count: Number of papers analyzed
            
        Returns:
            Combined synthesis as a string
        """
        parts = []

        # Introduction
        parts.append(f"# Literature Review: {title}\n")
        parts.append(f"*This review synthesizes findings from {paper_count} academic papers.*\n\n")

        # Add each section
        for section in sections:
            parts.append(f"## {section['subtopic']}\n\n")
            parts.append(section['content'])
            parts.append("\n\n")

        # Add research gaps section if available
        if research_gaps:
            parts.append("## Research Gaps and Future Directions\n\n")
            for i, gap in enumerate(research_gaps, 1):
                if isinstance(gap, dict):
                    parts.append(f"### Gap {i}: {gap.get('description', 'Unnamed gap')}\n")
                    parts.append(f"**Importance:** {gap.get('importance', 'N/A')}\n")
                    parts.append(f"**Potential Directions:** {gap.get('directions', 'N/A')}\n\n")
                else:
                    parts.append(f"{i}. {gap}\n")

        return "".join(parts)

    # Legacy method for backward compatibility
    def synthesize_section(self, subtopic: str, paper_analyses: str, academic_level: str, word_count: int) -> str:
        """
        Legacy method for backward compatibility with existing code.
        
        Returns raw LLM response string.
        """
        prompt = f"""You are a Synthesis Executor agent specializing in academic writing and literature review generation.
        TASK: Create a literature review section synthesizing the following analyzed papers.
        Section Topic: {subtopic}
        Analyzed Papers: {paper_analyses}
        Writing Style: Academic, formal, suitable for {academic_level} level
        Target Length: {word_count} words
        OUTPUT FORMAT:
        1. Section Introduction: Overview of topic and scope
        2. Thematic Organization: Group findings by themes/approaches
        3. Critical Analysis: Compare/contrast findings, identify patterns
        4. Research Gaps: Highlight limitations and future directions
        5. Section Conclusion: Synthesize key takeaways
        6. DON'T include References section in the end"""
        return self.llm_client.chat(prompt)
