# Research Orchestrator - LangGraph State Machine
# Manages the agent pipeline and state transitions

import asyncio
import logging
from collections.abc import Callable
from typing import Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyzer_agent import PaperAnalyzerAgent
from agents.planner_agent import ResearchPlannerAgent
from agents.quality_checker_agent import QualityCheckerAgent
from agents.retriever_agent import PaperRetrieverAgent
from agents.state import AgentState, create_initial_state
from agents.synthesizer_agent import SynthesisExecutorAgent

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """
    Orchestrates the research literature review pipeline using LangGraph.

    The pipeline follows this flow:

    ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌─────────────┐   ┌─────────────────┐
    │ Planner │──▶│ Retriever │──▶│ Analyzer │──▶│ Synthesizer │──▶│ Quality Checker │
    └─────────┘   └───────────┘   └──────────┘   └─────────────┘   └────────┬────────┘
                                       ▲                                     │
                                       │              ┌─────────────────────┘
                                       │              │
                                       │         ┌────▼────┐
                                       │         │ Refine? │
                                       │         └────┬────┘
                                       │              │ Yes (quality < threshold)
                                       └──────────────┘

    Key Features:
    - State machine-based orchestration
    - Conditional routing based on quality scores
    - Iteration limits to prevent infinite loops
    - Comprehensive error handling
    - Progress callbacks for real-time updates
    """

    def __init__(
        self, llm_client, progress_callback: Callable[[str, str, float], None] | None = None
    ):
        """
        Initialize the orchestrator.

        Args:
            llm_client: The LLM client (e.g., GeminiClient) for agent use
            progress_callback: Optional callback for progress updates
                              Signature: (agent_name, status_message, progress_percent)
        """
        self.llm_client = llm_client
        self.progress_callback = progress_callback

        # Initialize agents
        self.planner = ResearchPlannerAgent(llm_client)
        self.retriever = PaperRetrieverAgent(llm_client)
        self.analyzer = PaperAnalyzerAgent(llm_client)
        self.synthesizer = SynthesisExecutorAgent(llm_client)
        self.quality_checker = QualityCheckerAgent(llm_client)

        # Build the graph
        self.graph = self._build_graph()

        logger.info("ResearchOrchestrator initialized with LangGraph pipeline")

    def _build_graph(self) -> CompiledStateGraph:
        """
        Build the LangGraph state machine.

        Returns:
            Compiled StateGraph ready for execution
        """
        # Create the graph with our state type
        workflow = StateGraph(AgentState)

        # Add nodes (each agent becomes a node)
        workflow.add_node("planner", self._run_planner)
        workflow.add_node("retriever", self._run_retriever)
        workflow.add_node("analyzer", self._run_analyzer)
        workflow.add_node("synthesizer", self._run_synthesizer)
        workflow.add_node("quality_checker", self._run_quality_checker)

        # Define the edges (transitions between nodes)
        workflow.set_entry_point("planner")

        # Linear flow: planner -> retriever -> analyzer -> synthesizer -> quality_checker
        workflow.add_edge("planner", "retriever")
        workflow.add_edge("retriever", "analyzer")
        workflow.add_edge("analyzer", "synthesizer")
        workflow.add_edge("synthesizer", "quality_checker")

        # Conditional edge from quality_checker
        workflow.add_conditional_edges(
            "quality_checker",
            self._should_continue_or_end,
            {
                "refine": "planner",  # Loop back to planner if quality is low
                "complete": END,  # End if quality is acceptable
            },
        )

        return workflow.compile()

    async def _run_planner(self, state: AgentState) -> AgentState:
        """Run the planner agent node."""
        self._report_progress("planner", "Creating search strategy...", 10)
        return await self.planner.run(state)

    async def _run_retriever(self, state: AgentState) -> AgentState:
        """Run the retriever agent node."""
        self._report_progress("retriever", "Fetching papers from academic databases...", 25)
        return await self.retriever.run(state)

    async def _run_analyzer(self, state: AgentState) -> AgentState:
        """Run the analyzer agent node."""
        paper_count = len(state.get("papers", []))
        self._report_progress("analyzer", f"Analyzing {paper_count} papers...", 50)
        return await self.analyzer.run(state)

    async def _run_synthesizer(self, state: AgentState) -> AgentState:
        """Run the synthesizer agent node."""
        self._report_progress("synthesizer", "Synthesizing literature review...", 75)
        return await self.synthesizer.run(state)

    async def _run_quality_checker(self, state: AgentState) -> AgentState:
        """Run the quality checker agent node."""
        self._report_progress("quality_checker", "Evaluating synthesis quality...", 90)
        return await self.quality_checker.run(state)

    def _should_continue_or_end(self, state: AgentState) -> Literal["refine", "complete"]:
        """
        Decide whether to refine the synthesis or complete the pipeline.

        This is the key decision point that enables iterative improvement.

        Args:
            state: Current pipeline state

        Returns:
            "refine" to loop back to planner, "complete" to end
        """
        status = state.get("status", "completed")
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        if status == "needs_refinement" and iteration < max_iterations:
            logger.info(
                f"Quality check failed, refining (iteration {iteration + 1}/{max_iterations})"
            )
            # Increment iteration counter for next loop
            state["iteration"] = iteration + 1
            return "refine"

        logger.info(f"Pipeline complete. Final status: {status}")
        return "complete"

    def _report_progress(self, agent: str, message: str, percent: float):
        """Report progress through the callback if available."""
        logger.info(f"[{agent}] {message} ({percent}%)")
        if self.progress_callback:
            try:
                self.progress_callback(agent, message, percent)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    async def run(
        self,
        project_id: str,
        user_id: str,
        title: str,
        research_question: str,
        max_papers: int = 50,
        max_iterations: int = 3,
        relevance_threshold: float = 60.0,
        academic_level: str = "graduate",
        target_word_count: int = 500,
    ) -> AgentState:
        """
        Run the complete literature review pipeline.

        Args:
            project_id: Database ID of the research project
            user_id: Database ID of the user
            title: Title of the research project
            research_question: The main research question
            max_papers: Maximum number of papers to retrieve
            max_iterations: Maximum refinement iterations
            relevance_threshold: Minimum relevance score (0-100)
            academic_level: Academic level for writing style
            target_word_count: Target word count for synthesis

        Returns:
            Final AgentState with all results
        """
        logger.info(f"Starting literature review pipeline for project {project_id}")

        # Create initial state
        initial_state = create_initial_state(
            project_id=project_id,
            user_id=user_id,
            title=title,
            research_question=research_question,
            max_papers=max_papers,
            max_iterations=max_iterations,
            relevance_threshold=relevance_threshold,
            academic_level=academic_level,
            target_word_count=target_word_count,
        )

        self._report_progress("orchestrator", "Starting pipeline...", 0)

        try:
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)

            self._report_progress("orchestrator", "Pipeline complete!", 100)

            logger.info(
                f"Pipeline completed for project {project_id}. "
                f"Papers analyzed: {len(final_state.get('analyzed_papers', []))}. "
                f"Quality score: {final_state.get('quality_score', 0):.1f}"
            )

            return final_state

        except Exception as e:
            logger.error(f"Pipeline failed for project {project_id}: {e}", exc_info=True)
            initial_state["status"] = "error"
            initial_state["errors"].append(str(e))
            return initial_state

    def run_sync(
        self, project_id: str, user_id: str, title: str, research_question: str, **kwargs
    ) -> AgentState:
        """
        Synchronous wrapper for running the pipeline.

        Useful for Celery tasks and other sync contexts.
        """
        return asyncio.run(
            self.run(
                project_id=project_id,
                user_id=user_id,
                title=title,
                research_question=research_question,
                **kwargs,
            )
        )


def create_orchestrator(llm_client, progress_callback=None) -> ResearchOrchestrator:
    """
    Factory function to create a ResearchOrchestrator.

    Args:
        llm_client: The LLM client to use
        progress_callback: Optional progress callback

    Returns:
        Configured ResearchOrchestrator instance
    """
    return ResearchOrchestrator(llm_client, progress_callback)
