# Smart Model Router - Cost-Aware LLM Selection
# Routes tasks to appropriate Gemini models based on complexity, budget, and performance requirements

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class ModelTier(StrEnum):
    """Available Gemini model tiers with different cost/performance tradeoffs."""

    FAST_CHEAP = "gemini-2.0-flash-lite"  # Simple tasks, lowest cost
    BALANCED = "gemini-2.0-flash"  # Most tasks, balanced cost/performance
    POWERFUL = "gemini-1.5-pro"  # Complex reasoning, highest quality


@dataclass
class RoutingDecision:
    """Decision made by the router about which model to use."""

    model: ModelTier
    reason: str
    estimated_cost: float
    estimated_latency_ms: int
    budget_remaining: float


class SmartModelRouter:
    """
    Routes requests to appropriate models based on:
    - Task complexity (determined by task type and prompt analysis)
    - Token budget remaining (prevents overspending)
    - Performance requirements (latency vs quality tradeoffs)

    This demonstrates production-grade AI cost optimization and is a key
    differentiator in FAANG-level systems.

    Cost Savings Example:
    - Using flash-lite for simple tasks: $0.00001 / 1K tokens
    - Using pro for complex tasks: $0.001 / 1K tokens
    - Smart routing can reduce costs by 60-80% while maintaining quality
    """

    # Cost per 1K tokens for different models (Google Gemini pricing)
    COST_PER_1K_TOKENS = {
        ModelTier.FAST_CHEAP: 0.00001,  # $0.01 per 1M tokens
        ModelTier.BALANCED: 0.0001,  # $0.10 per 1M tokens
        ModelTier.POWERFUL: 0.001,  # $1.00 per 1M tokens
    }

    # Base latency estimates (milliseconds)
    BASE_LATENCY = {
        ModelTier.FAST_CHEAP: 200,
        ModelTier.BALANCED: 500,
        ModelTier.POWERFUL: 1500,
    }

    # Task type to default model tier mapping
    TASK_ROUTING_RULES = {
        # Simple extraction/classification tasks
        "extract_keywords": ModelTier.FAST_CHEAP,
        "classify_relevance": ModelTier.FAST_CHEAP,
        "simple_parsing": ModelTier.FAST_CHEAP,
        "format_output": ModelTier.FAST_CHEAP,
        # Medium complexity tasks
        "paper_analysis": ModelTier.BALANCED,
        "summarization": ModelTier.BALANCED,
        "search_planning": ModelTier.BALANCED,
        "relevance_scoring": ModelTier.BALANCED,
        # Complex reasoning tasks
        "synthesis": ModelTier.POWERFUL,
        "quality_evaluation": ModelTier.POWERFUL,
        "research_gap_identification": ModelTier.POWERFUL,
        "complex_reasoning": ModelTier.POWERFUL,
    }

    def __init__(self, user_budget: float = 1.0, user_id: str = "default"):
        """
        Initialize the router with a user budget.

        Args:
            user_budget: Maximum amount ($USD) user can spend
            user_id: User identifier for tracking per-user spending
        """
        self.user_budget = user_budget
        self.user_id = user_id
        self.spent = 0.0
        logger.info(f"ModelRouter initialized for user {user_id} with ${user_budget:.4f} budget")

    def route(
        self,
        task_type: str,
        prompt: str,
        complexity_hint: str | None = None,
        max_latency_ms: int | None = None,
    ) -> RoutingDecision:
        """
        Route request to appropriate model based on multiple factors.

        Args:
            task_type: Type of task (e.g., "synthesis", "extract_keywords")
            prompt: The actual prompt text (used for token estimation)
            complexity_hint: Optional hint about complexity ("low", "medium", "high")
            max_latency_ms: Optional maximum latency requirement

        Returns:
            RoutingDecision with selected model and metadata
        """
        # Estimate token count (rough approximation: 4 chars ≈ 1 token)
        token_count = self._estimate_tokens(prompt)

        # Start with task-based routing rule
        suggested_tier = self.TASK_ROUTING_RULES.get(
            task_type, ModelTier.BALANCED  # Default to balanced if task type unknown
        )

        # Budget-aware downgrade
        estimated_cost = self._estimate_cost(suggested_tier, token_count)
        budget_remaining = self.user_budget - self.spent
        budget_usage_ratio = self.spent / self.user_budget if self.user_budget > 0 else 0
        budget_constrained = False  # Track if we downgraded due to budget

        # Check if we've already used 80%+ of budget, or if this call would exceed 90%
        if budget_usage_ratio >= 0.8 or self.spent + estimated_cost > self.user_budget * 0.9:
            # Running low on budget, downgrade to cheapest model
            suggested_tier = ModelTier.FAST_CHEAP
            estimated_cost = self._estimate_cost(suggested_tier, token_count)
            budget_constrained = True  # Mark that we're budget constrained
            logger.warning(
                f"Budget constraint: Downgrading to {suggested_tier.value} "
                f"(${self.spent:.4f} / ${self.user_budget:.4f} spent, {budget_usage_ratio*100:.1f}% used)"
            )

        # Complexity-based upgrade (only if not budget constrained)
        if (
            not budget_constrained
            and complexity_hint == "high"
            and suggested_tier != ModelTier.POWERFUL
        ):
            upgrade_cost = self._estimate_cost(ModelTier.POWERFUL, token_count)
            if self.spent + upgrade_cost < self.user_budget:
                suggested_tier = ModelTier.POWERFUL
                estimated_cost = upgrade_cost
                logger.info(
                    f"Complexity upgrade: Using {suggested_tier.value} for high complexity task"
                )

        # Latency constraint downgrade
        if max_latency_ms:
            estimated_latency = self._estimate_latency(suggested_tier, token_count)
            if estimated_latency > max_latency_ms:
                # Try to use faster model
                if suggested_tier == ModelTier.POWERFUL:
                    suggested_tier = ModelTier.BALANCED
                elif suggested_tier == ModelTier.BALANCED:
                    suggested_tier = ModelTier.FAST_CHEAP
                logger.info(f"Latency optimization: Downgraded to {suggested_tier.value}")

        # Complexity hint from prompt analysis (only if not budget constrained)
        if not budget_constrained and not complexity_hint:
            complexity_hint = self._analyze_prompt_complexity(prompt)
            if complexity_hint == "high" and suggested_tier == ModelTier.FAST_CHEAP:
                # Upgrade from cheap to balanced for complex prompts
                if (
                    self.spent + self._estimate_cost(ModelTier.BALANCED, token_count)
                    < self.user_budget
                ):
                    suggested_tier = ModelTier.BALANCED
                    logger.info("Prompt analysis: Upgrading to BALANCED model")

        # Final cost calculation
        estimated_cost = self._estimate_cost(suggested_tier, token_count)
        estimated_latency = self._estimate_latency(suggested_tier, token_count)

        reason = self._generate_reason(task_type, suggested_tier, budget_remaining, complexity_hint)

        decision = RoutingDecision(
            model=suggested_tier,
            reason=reason,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            budget_remaining=budget_remaining - estimated_cost,
        )

        logger.info(
            f"Routing decision: {suggested_tier.value} | "
            f"Cost: ${estimated_cost:.6f} | "
            f"Latency: ~{estimated_latency}ms | "
            f"Budget remaining: ${decision.budget_remaining:.4f}"
        )

        return decision

    def record_usage(self, actual_cost: float):
        """
        Record actual cost after API call.

        Args:
            actual_cost: Actual cost incurred from the API call
        """
        self.spent += actual_cost
        logger.info(f"Recorded usage: ${actual_cost:.6f} | Total spent: ${self.spent:.4f}")

    def reset_budget(self, new_budget: float | None = None):
        """
        Reset spending tracker. Typically called at the start of a new billing period.

        Args:
            new_budget: Optional new budget amount (defaults to current budget)
        """
        if new_budget is not None:
            self.user_budget = new_budget
        self.spent = 0.0
        logger.info(f"Budget reset to ${self.user_budget:.4f}")

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough estimate of token count.
        Rule of thumb: ~4 characters per token for English text.

        Args:
            text: Input text to estimate

        Returns:
            Estimated token count
        """
        return max(len(text) // 4, 10)  # Minimum 10 tokens

    def _estimate_cost(self, tier: ModelTier, tokens: int) -> float:
        """
        Estimate cost for a request.
        Assumes output tokens ≈ input tokens for research tasks.

        Args:
            tier: Model tier to use
            tokens: Estimated input token count

        Returns:
            Estimated cost in USD
        """
        # Estimate total tokens (input + output, assume 2x for safety)
        total_tokens = tokens * 2
        return (total_tokens / 1000) * self.COST_PER_1K_TOKENS[tier]

    def _estimate_latency(self, tier: ModelTier, tokens: int) -> int:
        """
        Estimate latency for a request.

        Args:
            tier: Model tier to use
            tokens: Token count

        Returns:
            Estimated latency in milliseconds
        """
        base = self.BASE_LATENCY[tier]
        # Add ~50ms per 100 tokens
        token_overhead = (tokens // 100) * 50
        return base + token_overhead

    def _analyze_prompt_complexity(self, prompt: str) -> str:
        """
        Analyze prompt to infer complexity.

        Heuristics:
        - Long prompts (>2000 chars) = high complexity
        - Keywords like "analyze", "synthesize", "compare" = medium/high
        - Short prompts with simple verbs = low

        Args:
            prompt: The prompt text

        Returns:
            Complexity level: "low", "medium", or "high"
        """
        prompt_lower = prompt.lower()

        # Length-based heuristic
        if len(prompt) > 2000:
            return "high"
        if len(prompt) < 200:
            return "low"

        # Keyword-based heuristic
        complex_keywords = [
            "synthesize",
            "compare",
            "contrast",
            "evaluate",
            "research gap",
            "critical analysis",
            "methodology",
        ]
        medium_keywords = ["analyze", "summarize", "explain", "describe"]

        if any(keyword in prompt_lower for keyword in complex_keywords):
            return "high"
        if any(keyword in prompt_lower for keyword in medium_keywords):
            return "medium"

        return "low"

    def _generate_reason(
        self, task_type: str, tier: ModelTier, budget_remaining: float, complexity_hint: str | None
    ) -> str:
        """Generate human-readable reason for routing decision."""
        reasons = []

        reasons.append(f"Task '{task_type}' → {tier.value}")

        if complexity_hint:
            reasons.append(f"complexity={complexity_hint}")

        if budget_remaining < 0.1:
            reasons.append("low budget")

        return " | ".join(reasons)

    def get_stats(self) -> dict:
        """
        Get current usage statistics.

        Returns:
            Dictionary with spending stats
        """
        return {
            "user_id": self.user_id,
            "budget": self.user_budget,
            "spent": self.spent,
            "remaining": self.user_budget - self.spent,
            "usage_percent": (self.spent / self.user_budget * 100) if self.user_budget > 0 else 0,
        }


# Singleton instance for application-wide use
# In production, you'd want per-user routers with Redis/DB persistence
_default_router: SmartModelRouter | None = None


def get_router(user_budget: float = 1.0, user_id: str = "default") -> SmartModelRouter:
    """
    Get or create a model router instance.

    Args:
        user_budget: Budget for this user
        user_id: User identifier

    Returns:
        SmartModelRouter instance
    """
    global _default_router
    if _default_router is None or _default_router.user_id != user_id:
        _default_router = SmartModelRouter(user_budget=user_budget, user_id=user_id)
    return _default_router
