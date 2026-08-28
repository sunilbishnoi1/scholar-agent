"""
Autonomous Supervisor Agent and StateGraph DAG Coordinator for Scholar Agent.

Constructs and manages the hierarchical LangGraph StateGraph DAG:
Discovery -> Ingestion -> Matrix Builder -> Synthesizer -> Critic -> [Refinement Loop <= 2] -> Auditor -> Finalizer.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Literal, Optional

from langgraph.graph import END, START, StateGraph

try:
    from agents.base import BaseAgent
    from agents.blackboard import WorkingMemoryBlackboard
    from agents.llm.base import BaseLLMClient
    from agents.schemas import ResearchReport
    from agents.state import AgentMessage, AgentResult, AgentState, AgentType, GoalStatus
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.agents.blackboard import WorkingMemoryBlackboard
    from backend.agents.llm.base import BaseLLMClient
    from backend.agents.schemas import ResearchReport
    from backend.agents.state import AgentMessage, AgentResult, AgentState, AgentType, GoalStatus

logger = logging.getLogger(__name__)


def should_refine_or_finalize(state: AgentState) -> Literal["synthesizer", "auditor"]:
    """
    Conditional routing function for LangGraph refinement loop.
    Enforces hard bounded termination:
    - If overall_score >= 75.0 or iteration_count >= max_iterations (default 2), proceeds to 'auditor'.
    - Otherwise loops back to 'synthesizer' with incremented iteration_count.
    """
    iteration_count = state.get("iteration_count", state.get("iteration", 0))
    max_iterations = state.get("max_iterations", 2)
    current_score = state.get("current_critic_score", state.get("quality_score", 0.0))
    should_refine = state.get("should_refine", False)

    logger.info(
        f"Supervisor evaluating refinement condition: score={current_score:.1f}, "
        f"iteration={iteration_count}/{max_iterations}, should_refine={should_refine}"
    )

    # Termination guarantees: score meets standard OR max refinement iterations reached
    if iteration_count >= max_iterations:
        logger.info(f"Max refinement iterations ({max_iterations}) reached. Proceeding to citation audit.")
        return "auditor"

    if current_score >= 75.0 and not should_refine:
        logger.info(f"Quality score ({current_score:.1f} >= 75.0) passed. Proceeding to citation audit.")
        return "auditor"

    logger.info(f"Quality score ({current_score:.1f} < 75.0). Looping back to Synthesizer for revision.")
    state["iteration_count"] = iteration_count + 1
    state["iteration"] = state["iteration_count"]
    return "synthesizer"


class AutonomousSupervisorAgent(BaseAgent):
    """
    Hierarchical Supervisor Agent coordinating workflow goals, node telemetry,
    and the final Pydantic deliverable compilation.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        blackboard: Optional[WorkingMemoryBlackboard] = None,
        name: str = "supervisor",
    ) -> None:
        super().__init__(llm_client=llm_client, name=name)
        self.blackboard = blackboard

    async def run(self, state: AgentState) -> AgentState:
        """Initialize pipeline goal stack and prepare state for execution."""
        self._log_start(state)
        state["current_agent"] = AgentType.SUPERVISOR
        state["status"] = "running"

        # Initialize standard goal stack if empty
        if not state.get("goal_stack"):
            state["goal_stack"] = [
                {
                    "goal_id": "goal_1_discovery",
                    "name": "Literature Discovery",
                    "description": "Formulate search query plan and retrieve candidate papers.",
                    "target_agent": AgentType.DISCOVERY,
                    "status": GoalStatus.PENDING,
                    "priority": 1,
                    "parent_goal_id": None,
                    "created_at": None,
                    "completed_at": None,
                    "error_message": None,
                },
                {
                    "goal_id": "goal_2_ingestion",
                    "name": "Full-Text Ingestion",
                    "description": "Resolve open-access PDFs, parse markdown, and index chunks.",
                    "target_agent": AgentType.INGESTION,
                    "status": GoalStatus.PENDING,
                    "priority": 2,
                    "parent_goal_id": None,
                    "created_at": None,
                    "completed_at": None,
                    "error_message": None,
                },
                {
                    "goal_id": "goal_3_matrix",
                    "name": "Evidence Matrix Extraction",
                    "description": "Extract comparative technical matrix across all acquired papers.",
                    "target_agent": AgentType.MATRIX_BUILDER,
                    "status": GoalStatus.PENDING,
                    "priority": 3,
                    "parent_goal_id": None,
                    "created_at": None,
                    "completed_at": None,
                    "error_message": None,
                },
                {
                    "goal_id": "goal_4_synthesis",
                    "name": "Thematic Synthesis",
                    "description": "Draft comparative literature review, scientific debates, and research gaps.",
                    "target_agent": AgentType.SYNTHESIZER,
                    "status": GoalStatus.PENDING,
                    "priority": 4,
                    "parent_goal_id": None,
                    "created_at": None,
                    "completed_at": None,
                    "error_message": None,
                },
                {
                    "goal_id": "goal_5_critique",
                    "name": "Adversarial Critique",
                    "description": "Evaluate empirical rigor, statistical validity, and baseline coverage.",
                    "target_agent": AgentType.CRITIC,
                    "status": GoalStatus.PENDING,
                    "priority": 5,
                    "parent_goal_id": None,
                    "created_at": None,
                    "completed_at": None,
                    "error_message": None,
                },
                {
                    "goal_id": "goal_6_audit",
                    "name": "Citation Grounding Audit",
                    "description": "Verify factual claims against source chunks and compile bibliography.",
                    "target_agent": AgentType.AUDITOR,
                    "status": GoalStatus.PENDING,
                    "priority": 6,
                    "parent_goal_id": None,
                    "created_at": None,
                    "completed_at": None,
                    "error_message": None,
                },
            ]

        msg = self._create_message(
            action="pipeline_initialization",
            content={"status": "initialized", "goals_count": len(state["goal_stack"])},
        )
        if "messages" not in state or state["messages"] is None:
            state["messages"] = []
        state["messages"].append(msg)

        self._log_complete(state, AgentResult(success=True, data={"status": "initialized"}))
        return state


async def finalizer_node(state: AgentState) -> AgentState:
    """
    Final node assembling the complete ResearchReport Pydantic deliverable.
    """
    logger.info("Executing finalizer node: Assembling ResearchReport deliverable...")
    state["current_agent"] = AgentType.SUPERVISOR
    state["status"] = "completed"

    # Create temporary blackboard to leverage report assembly logic
    bb = WorkingMemoryBlackboard(
        project_id=state.get("project_id", "default_project"),
        user_id=state.get("user_id", "default_user"),
        title=state.get("title", ""),
        research_question=state.get("research_question", ""),
    )
    bb.update_from_agent_state(state)
    report = bb.assemble_research_report()
    state["final_report"] = report.model_dump()

    return state


def build_scholar_agent_graph(
    discovery_agent: BaseAgent,
    ingestion_agent: BaseAgent,
    matrix_builder_agent: BaseAgent,
    synthesizer_agent: BaseAgent,
    critic_agent: BaseAgent,
    auditor_agent: BaseAgent,
    supervisor_agent: Optional[BaseAgent] = None,
) -> StateGraph:
    """
    Build and compile the LangGraph StateGraph DAG for the Scholar Agent system.

    Topology:
        START -> supervisor -> discovery -> ingestion -> matrix_builder -> synthesizer -> critic
                                                                             ^            |
                                                                             |--(refine)--|
                                                                                          |
                                                                                       auditor -> finalizer -> END
    """
    graph = StateGraph(AgentState)

    # 1. Register Nodes
    sup_agent = supervisor_agent or AutonomousSupervisorAgent()
    graph.add_node("supervisor", sup_agent.run)
    graph.add_node("discovery", discovery_agent.run)
    graph.add_node("ingestion", ingestion_agent.run)
    graph.add_node("matrix_builder", matrix_builder_agent.run)
    graph.add_node("synthesizer", synthesizer_agent.run)
    graph.add_node("critic", critic_agent.run)
    graph.add_node("auditor", auditor_agent.run)
    graph.add_node("finalizer", finalizer_node)

    # 2. Register Deterministic Edges
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "discovery")
    graph.add_edge("discovery", "ingestion")
    graph.add_edge("ingestion", "matrix_builder")
    graph.add_edge("matrix_builder", "synthesizer")
    graph.add_edge("synthesizer", "critic")

    # 3. Register Conditional Refinement Edge
    graph.add_conditional_edges(
        "critic",
        should_refine_or_finalize,
        {
            "synthesizer": "synthesizer",
            "auditor": "auditor",
        },
    )

    # 4. Register Terminal Edges
    graph.add_edge("auditor", "finalizer")
    graph.add_edge("finalizer", END)

    return graph

