"""
Research Orchestrator - LangGraph StateGraph DAG Coordinator for Scholar Agent.

Manages multi-agent pipeline execution, telemetry streaming, working memory blackboard,
and database persistence.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session

try:
    from agents.base import ToolEnabledAgent
    from agents.blackboard import WorkingMemoryBlackboard
    from agents.core.auditor import DeterministicCitationAuditorAgent
    from agents.core.critic import AdversarialCriticAgent
    from agents.core.discovery import AutonomousLiteratureExplorer
    from agents.core.ingestion import FullTextIngestionSpecialist
    from agents.core.matrix_builder import EvidenceMatrixBuilder
    from agents.core.supervisor import AutonomousSupervisorAgent, build_scholar_agent_graph
    from agents.core.synthesizer import ThematicSynthesizerAgent
    from agents.llm.base import BaseLLMClient
    from agents.schemas import ResearchReport
    from agents.state import AgentMessage, AgentResult, AgentState, AgentType, create_initial_agent_state
    from agents.tools.academic_search import MultiSourceAcademicSearch
    from agents.tools.citation_graph import CitationGraphTraverser
    from agents.tools.oa_resolver import OAResolver
    from agents.tools.pdf_parser import PDFParser
    from rag.chunker import SectionAwareChunker
    from rag.vector_store import AcademicVectorStore
    from services.cancellation_manager import TaskCancelledException, cancellation_manager
except ImportError:
    from backend.agents.base import ToolEnabledAgent
    from backend.agents.blackboard import WorkingMemoryBlackboard
    from backend.agents.core.auditor import DeterministicCitationAuditorAgent
    from backend.agents.core.critic import AdversarialCriticAgent
    from backend.agents.core.discovery import AutonomousLiteratureExplorer
    from backend.agents.core.ingestion import FullTextIngestionSpecialist
    from backend.agents.core.matrix_builder import EvidenceMatrixBuilder
    from backend.agents.core.supervisor import AutonomousSupervisorAgent, build_scholar_agent_graph
    from backend.agents.core.synthesizer import ThematicSynthesizerAgent
    from backend.agents.llm.base import BaseLLMClient
    from backend.agents.schemas import ResearchReport
    from backend.agents.state import AgentMessage, AgentResult, AgentState, AgentType, create_initial_agent_state
    from backend.agents.tools.academic_search import MultiSourceAcademicSearch
    from backend.agents.tools.citation_graph import CitationGraphTraverser
    from backend.agents.tools.oa_resolver import OAResolver
    from backend.agents.tools.pdf_parser import PDFParser
    from backend.rag.chunker import SectionAwareChunker
    from backend.rag.vector_store import AcademicVectorStore
    try:
        from backend.services.cancellation_manager import TaskCancelledException, cancellation_manager
    except ImportError:
        cancellation_manager = None
        TaskCancelledException = Exception

logger = logging.getLogger(__name__)


class ScholarAgentOrchestrator:
    """
    Scholar Agent Autonomous Reasoning Pipeline Orchestrator.

    Pipeline DAG:
        Supervisor -> Discovery -> Ingestion -> Matrix Builder -> Synthesizer -> Critic
                                                                      ^            |
                                                                      |--(refine)--|
                                                                                   |
                                                                                Auditor -> Finalizer
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        db_session: Any | None = None,
        vector_store: Any | None = None,
        progress_callback: Callable[[str, str, float], None] | None = None,
    ) -> None:
        # If second positional arg is callable and not a DB Session, treat as progress_callback
        if callable(db_session) and not hasattr(db_session, "query") and not hasattr(db_session, "commit") and progress_callback is None:
            progress_callback = db_session
            db_session = None

        self.llm_client = llm_client
        self.db_session = db_session
        self.vector_store = vector_store
        self.progress_callback = progress_callback


        # Initialize domain specialists
        self.discovery_agent = AutonomousLiteratureExplorer(
            llm_client=llm_client,
            search_tool=MultiSourceAcademicSearch(),
            citation_tool=CitationGraphTraverser(),
        )
        self.ingestion_agent = FullTextIngestionSpecialist(
            llm_client=llm_client,
            oa_resolver=OAResolver(),
            pdf_parser=PDFParser(),
            chunker=SectionAwareChunker(),
            vector_store=vector_store,
            db_session=db_session,
        )
        self.matrix_builder_agent = EvidenceMatrixBuilder(
            llm_client=llm_client,
            db_session=db_session,
        )
        self.synthesizer_agent = ThematicSynthesizerAgent(
            llm_client=llm_client,
        )
        self.critic_agent = AdversarialCriticAgent(
            llm_client=llm_client,
        )
        self.auditor_agent = DeterministicCitationAuditorAgent(
            llm_client=llm_client,
        )
        self.supervisor_agent = AutonomousSupervisorAgent(
            llm_client=llm_client,
        )

        # Backward compatibility agent aliases
        self.planner = self.discovery_agent
        self.retriever = self.ingestion_agent
        self.analyzer = self.matrix_builder_agent
        self.synthesizer = self.synthesizer_agent
        self.quality_checker = self.critic_agent

        # Build StateGraph
        self.graph = self._build_compiled_graph()
        logger.info("ScholarAgentOrchestrator initialized with compiled LangGraph StateGraph DAG.")

    def _build_compiled_graph(self) -> CompiledStateGraph:
        """Construct and compile the LangGraph workflow DAG."""
        state_graph = build_scholar_agent_graph(
            discovery_agent=self.discovery_agent,
            ingestion_agent=self.ingestion_agent,
            matrix_builder_agent=self.matrix_builder_agent,
            synthesizer_agent=self.synthesizer_agent,
            critic_agent=self.critic_agent,
            auditor_agent=self.auditor_agent,
            supervisor_agent=self.supervisor_agent,
        )
        return state_graph.compile()

    def _should_continue_or_end(self, state: AgentState) -> str:
        """Legacy routing condition evaluator for backward compatibility."""
        status = state.get("status")
        if status == "completed":
            return "complete"
        if status == "error":
            return "error"
        iteration = max(state.get("iteration", 0), state.get("iteration_count", 0))
        max_iter = state.get("max_iterations", 2)
        if iteration >= max_iter:
            return "complete"
        if state.get("should_refine") or status == "needs_refinement":
            state["iteration"] = iteration + 1
            state["iteration_count"] = iteration + 1
            return "refine"
        return "complete"


    async def _run_planner_node(self, state: AgentState) -> AgentState:
        return await self.planner.run(state)

    async def _run_retriever_node(self, state: AgentState) -> AgentState:
        return await self.retriever.run(state)

    async def _run_analyzer_node(self, state: AgentState) -> AgentState:
        return await self.analyzer.run(state)

    async def _run_synthesizer_node(self, state: AgentState) -> AgentState:
        return await self.synthesizer.run(state)

    async def _run_quality_checker_node(self, state: AgentState) -> AgentState:
        return await self.quality_checker.run(state)

    # Legacy runner method aliases
    _run_planner = _run_planner_node
    _run_retriever = _run_retriever_node
    _run_analyzer = _run_analyzer_node
    _run_synthesizer = _run_synthesizer_node
    _run_quality_checker = _run_quality_checker_node


    def _report_progress(self, agent: str, message: str, percent: float) -> None:
        """Send realtime progress telemetry via observer callback."""
        logger.info(f"[{agent}] {message} ({percent:.0f}%)")
        if self.progress_callback:
            try:
                self.progress_callback(agent, message, percent)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")


    async def run(
        self,
        project_id: str,
        user_id: str = "default_user",
        title: str = "",
        research_question: str = "",
        max_papers: int = 25,
        max_iterations: int = 2,
        relevance_threshold: float = 60.0,
        academic_level: str = "graduate",
        target_word_count: int = 3500,
        sync_to_db: bool = True,
    ) -> AgentState:
        """
        Execute the autonomous multi-agent reasoning pipeline.
        """
        logger.info(f"Starting Scholar Agent autonomous reasoning pipeline for project: '{project_id}'")

        # Initialize in-flight working memory blackboard
        blackboard = WorkingMemoryBlackboard(
            project_id=project_id,
            user_id=user_id,
            title=title,
            research_question=research_question,
            max_papers=max_papers,
            max_iterations=max_iterations,
        )

        # Create initial state
        initial_state = create_initial_agent_state(
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

        self._report_progress("orchestrator", "Initializing reasoning pipeline...", 5)

        if cancellation_manager and cancellation_manager.is_cancelled(project_id):
            logger.info(f"ScholarAgentOrchestrator: project '{project_id}' cancelled before StateGraph execution.")
            initial_state["status"] = "stopped"
            return initial_state

        try:
            # Execute StateGraph DAG
            final_state = await self.graph.ainvoke(initial_state)

            if cancellation_manager and cancellation_manager.is_cancelled(project_id):
                logger.info(f"ScholarAgentOrchestrator: project '{project_id}' cancelled during StateGraph execution.")
                final_state["status"] = "stopped"
                return final_state

            # Ingest final state back into blackboard
            blackboard.update_from_agent_state(final_state)

            # Persist to relational models if requested and session available
            if sync_to_db and self.db_session:
                try:
                    blackboard.sync_to_database(self.db_session)
                except Exception as e:
                    logger.error(f"Failed to sync blackboard to database: {e}")

            self._report_progress("orchestrator", "Reasoning pipeline completed successfully!", 100)
            return final_state

        except TaskCancelledException as e:
            logger.info(f"ScholarAgentOrchestrator: execution cancelled for project '{project_id}': {e}")
            initial_state["status"] = "stopped"
            return initial_state

        except Exception as e:
            if cancellation_manager and cancellation_manager.is_cancelled(project_id):
                logger.info(
                    f"ScholarAgentOrchestrator: caught error during active cancellation for project '{project_id}': {e}"
                )
                initial_state["status"] = "stopped"
                return initial_state

            logger.error(f"Reasoning pipeline failed for project '{project_id}': {e}", exc_info=True)
            initial_state["status"] = "error"
            if "errors" not in initial_state or initial_state["errors"] is None:
                initial_state["errors"] = []
            initial_state["errors"].append(str(e))
            return initial_state

    def run_sync(
        self,
        project_id: str,
        user_id: str = "default_user",
        title: str = "",
        research_question: str = "",
        **kwargs: Any,
    ) -> AgentState:
        """Synchronous wrapper for Celery and blocking contexts."""
        return asyncio.run(
            self.run(
                project_id=project_id,
                user_id=user_id,
                title=title,
                research_question=research_question,
                **kwargs,
            )
        )


# Backward compatibility aliases
ResearchOrchestrator = ScholarAgentOrchestrator
QualityCheckerAgent = AdversarialCriticAgent


def create_orchestrator(
    llm_client: Optional[BaseLLMClient] = None,
    progress_callback: Optional[Callable[[str, str, float], None]] = None,
    db_session: Optional[Session] = None,
    vector_store: Optional[AcademicVectorStore] = None,
) -> ScholarAgentOrchestrator:
    """Factory function creating a ScholarAgentOrchestrator instance."""
    return ScholarAgentOrchestrator(
        llm_client=llm_client,
        db_session=db_session,
        vector_store=vector_store,
        progress_callback=progress_callback,
    )
