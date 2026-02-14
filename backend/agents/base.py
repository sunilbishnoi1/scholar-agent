# Base Agent Class
# Abstract base class defining the interface for all agents

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from agents.state import AgentMessage, AgentResult, AgentState


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the pipeline.

    Provides common functionality like logging, error handling,
    and state management. All agents must implement the `run` method.
    """

    def __init__(self, llm_client, name: str):
        """
        Initialize the base agent.

        Args:
            llm_client: The LLM client (e.g., GeminiClient) for API calls
            name: Human-readable name of the agent
        """
        self.llm_client = llm_client
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    async def run(self, state: AgentState) -> AgentState:
        """
        Execute the agent's main logic.

        Args:
            state: The current pipeline state

        Returns:
            AgentState: Updated state after agent execution
        """
        pass

    def _create_message(self, action: str, content: Any) -> AgentMessage:
        """Create a standardized message for the state history."""
        return AgentMessage(
            agent=self.name, action=action, content=content, timestamp=datetime.now(UTC).isoformat()
        )

    def _log_start(self, state: AgentState) -> None:
        """Log the start of agent execution."""
        self.logger.info(
            f"Starting {self.name} | Project: {state['project_id']} | "
            f"Iteration: {state['iteration']}"
        )

    def _log_complete(self, state: AgentState, result: AgentResult) -> None:
        """Log the completion of agent execution."""
        status = "SUCCESS" if result.success else "FAILED"
        self.logger.info(
            f"Completed {self.name} | Status: {status} | " f"Project: {state['project_id']}"
        )
        if result.metadata:
            self.logger.debug(f"Metadata: {result.metadata}")

    def _handle_error(self, state: AgentState, error: Exception) -> AgentState:
        """Handle errors during agent execution."""
        error_msg = f"{self.name} error: {error!s}"
        self.logger.error(error_msg, exc_info=True)

        # Update state with error information
        state["errors"].append(error_msg)
        state["messages"] = [self._create_message("error", error_msg)]

        return state

    def _call_llm(self, prompt: str) -> str:
        """
        Make a call to the LLM with error handling.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            str: The LLM response

        Raises:
            Exception: If the LLM call fails
        """
        self.logger.debug(f"LLM call | Prompt length: {len(prompt)} chars")
        response = self.llm_client.chat(prompt)
        self.logger.debug(f"LLM response | Length: {len(response)} chars")
        return response


from collections.abc import Callable

...

class ToolEnabledAgent(BaseAgent):
    """
    Extended base class for agents that use tools.

    Tools are specific capabilities that agents can invoke to perform
    actions like searching databases, analyzing papers, etc.
    """

    def __init__(self, llm_client, name: str):
        super().__init__(llm_client, name)
        self.tools: dict[str, dict[str, Any]] = {}

    def register_tool(self, name: str, func: Callable, description: str = "") -> None:
        """
        Register a tool that this agent can use.

        Args:
            name: Name of the tool
            func: The callable function to execute
            description: Human-readable description of the tool
        """
        self.tools[name] = {"function": func, "description": description}
        self.logger.debug(f"Registered tool: {name}")

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Get list of available tools with descriptions."""
        return [
            {"name": name, "description": tool["description"]} for name, tool in self.tools.items()
        ]

    async def invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Invoke a registered tool.

        Args:
            tool_name: Name of the tool to invoke
            **kwargs: Arguments to pass to the tool

        Returns:
            The result of the tool execution

        Raises:
            ValueError: If the tool is not registered
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not registered")

        self.logger.info(f"Invoking tool: {tool_name}")
        tool_func = self.tools[tool_name]["function"]

        # Handle both sync and async tools
        import asyncio

        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(**kwargs)
        else:
            result = tool_func(**kwargs)

        self.logger.debug(f"Tool {tool_name} completed")
        return result
