# Quality Checker Agent (LangGraph Compatible)
# Role: Evaluates synthesis quality and decides if refinement is needed


from agents.base import ToolEnabledAgent
from agents.state import AgentResult, AgentState, AgentType
from agents.tools import evaluate_synthesis_quality


class QualityCheckerAgent(ToolEnabledAgent):
    """
    Quality Checker Agent responsible for:
    - Evaluating the quality of synthesized literature review
    - Providing feedback for improvement
    - Deciding if the synthesis meets quality thresholds
    
    This agent enables the iterative refinement loop in the pipeline.
    """

    def __init__(self, llm_client):
        super().__init__(llm_client, name="quality_checker")
        self._register_tools()
        self.quality_threshold = 70.0  # Minimum acceptable quality score

    def _register_tools(self):
        """Register the tools this agent can use."""
        self.register_tool(
            "evaluate_quality",
            lambda **kwargs: evaluate_synthesis_quality(self.llm_client, **kwargs),
            "Evaluate the quality of the synthesized literature review"
        )

    async def run(self, state: AgentState) -> AgentState:
        """
        Execute the quality checker agent.
        
        Workflow:
        1. Evaluate synthesis quality
        2. Provide feedback
        3. Update state with quality metrics
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated state with quality assessment
        """
        self._log_start(state)

        try:
            state["current_agent"] = AgentType.QUALITY_CHECKER

            synthesis = state.get("synthesis", "")
            research_question = state["research_question"]
            paper_count = len(state.get("high_quality_papers", state.get("analyzed_papers", [])))

            if not synthesis:
                self.logger.warning("No synthesis to evaluate")
                state["quality_score"] = 0.0
                state["quality_feedback"] = "No synthesis available for evaluation"
                state["status"] = "error"
                return state

            # Evaluate quality
            quality_result = await self.invoke_tool(
                "evaluate_quality",
                synthesis=synthesis,
                research_question=research_question,
                paper_count=paper_count
            )

            if quality_result.success:
                quality_data = quality_result.data
                state["quality_score"] = quality_data.get("overall_score", 0)
                state["quality_feedback"] = quality_data.get("feedback", "")

                # Determine next action
                should_refine = quality_data.get("should_refine", False)
                can_refine = state["iteration"] < state["max_iterations"]

                if should_refine and can_refine:
                    state["status"] = "needs_refinement"
                    self.logger.info(
                        f"Quality score {state['quality_score']:.1f} below threshold, "
                        f"recommending refinement (iteration {state['iteration'] + 1}/{state['max_iterations']})"
                    )
                else:
                    state["status"] = "completed"
                    if should_refine:
                        self.logger.info(
                            f"Quality score {state['quality_score']:.1f} below threshold, "
                            f"but max iterations reached. Completing anyway."
                        )
                    else:
                        self.logger.info(f"Quality score {state['quality_score']:.1f} meets threshold")
            else:
                self.logger.error(f"Quality evaluation failed: {quality_result.error}")
                state["quality_score"] = 0.0
                state["quality_feedback"] = f"Evaluation failed: {quality_result.error}"
                state["status"] = "completed"  # Don't block completion on evaluation failure

            state["messages"] = [self._create_message(
                "evaluate_quality",
                f"Quality score: {state['quality_score']:.1f}, Status: {state['status']}"
            )]

            result = AgentResult(
                success=True,
                data={
                    "quality_score": state["quality_score"],
                    "status": state["status"]
                },
                metadata={
                    "iteration": state["iteration"],
                    "max_iterations": state["max_iterations"]
                }
            )
            self._log_complete(state, result)

            return state

        except Exception as e:
            return self._handle_error(state, e)

    def should_refine(self, state: AgentState) -> bool:
        """
        Determine if the synthesis should be refined.
        
        This method is used by the orchestrator for routing decisions.
        
        Args:
            state: Current pipeline state
            
        Returns:
            True if refinement is needed and possible
        """
        return (
            state.get("status") == "needs_refinement" and
            state.get("iteration", 0) < state.get("max_iterations", 3)
        )
