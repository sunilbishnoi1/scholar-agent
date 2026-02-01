# Tests for Smart Model Router
# Validates cost-aware LLM selection and budget tracking

import pytest

from agents.model_router import ModelTier, RoutingDecision, SmartModelRouter, get_router


class TestSmartModelRouter:
    """Tests for the SmartModelRouter class."""

    def test_router_initializes_with_budget(self):
        """Router should initialize with specified budget."""
        router = SmartModelRouter(user_budget=5.0, user_id="test")

        assert router.user_budget == 5.0
        assert router.user_id == "test"
        assert router.spent == 0.0

    def test_routes_simple_task_to_cheap_model(self):
        """Simple tasks should route to FAST_CHEAP model."""
        router = SmartModelRouter(user_budget=1.0)

        decision = router.route(
            task_type="extract_keywords", prompt="Extract keywords from: AI in education"
        )

        assert decision.model == ModelTier.FAST_CHEAP
        assert decision.estimated_cost < 0.001
        assert decision.budget_remaining > 0.999

    def test_routes_medium_task_to_balanced_model(self):
        """Medium complexity tasks should route to BALANCED model."""
        router = SmartModelRouter(user_budget=1.0)

        decision = router.route(
            task_type="paper_analysis", prompt="Analyze this paper: " + "X" * 500
        )

        assert decision.model == ModelTier.BALANCED

    def test_routes_complex_task_to_powerful_model(self):
        """Complex tasks should route to POWERFUL model."""
        router = SmartModelRouter(user_budget=1.0)

        decision = router.route(
            task_type="research_gap_identification",
            prompt="Identify research gaps in these 50 papers: " + "X" * 2000,
        )

        assert decision.model == ModelTier.POWERFUL

    def test_downgrades_when_budget_low(self):
        """Router should downgrade model when budget is exhausted."""
        router = SmartModelRouter(user_budget=0.01)
        router.spent = 0.009  # 90% of budget spent

        # Even complex task should be downgraded
        decision = router.route(task_type="synthesis", prompt="Synthesize findings: " + "X" * 1000)

        assert decision.model == ModelTier.FAST_CHEAP

    def test_upgrades_with_complexity_hint(self):
        """Router should upgrade model when complexity hint is high."""
        router = SmartModelRouter(user_budget=1.0)

        decision = router.route(
            task_type="paper_analysis",  # Normally BALANCED
            prompt="Analyze paper: " + "X" * 500,
            complexity_hint="high",
        )

        # Should upgrade to POWERFUL
        assert decision.model == ModelTier.POWERFUL

    def test_tracks_spending_correctly(self):
        """Router should track spending across multiple calls."""
        router = SmartModelRouter(user_budget=1.0)

        # Make multiple routing decisions
        total_estimated = 0.0
        for i in range(5):
            decision = router.route(task_type="paper_analysis", prompt="Analyze paper " + "X" * 500)
            router.record_usage(decision.estimated_cost)
            total_estimated += decision.estimated_cost

        assert router.spent > 0
        assert abs(router.spent - total_estimated) < 0.0001

    def test_reset_budget_clears_spending(self):
        """Reset should clear spending and optionally update budget."""
        router = SmartModelRouter(user_budget=1.0)
        router.spent = 0.5

        router.reset_budget(new_budget=2.0)

        assert router.spent == 0.0
        assert router.user_budget == 2.0

    def test_get_stats_returns_usage_info(self):
        """get_stats should return comprehensive usage information."""
        router = SmartModelRouter(user_budget=1.0, user_id="test123")
        router.spent = 0.3

        stats = router.get_stats()

        assert stats["user_id"] == "test123"
        assert stats["budget"] == 1.0
        assert stats["spent"] == 0.3
        assert stats["remaining"] == 0.7
        assert abs(stats["usage_percent"] - 30.0) < 0.01

    def test_unknown_task_type_uses_balanced(self):
        """Unknown task types should default to BALANCED model."""
        router = SmartModelRouter(user_budget=1.0)

        decision = router.route(task_type="unknown_task_type", prompt="Do something: " + "X" * 300)

        assert decision.model == ModelTier.BALANCED

    def test_prompt_complexity_analysis(self):
        """Router should analyze prompt complexity automatically."""
        router = SmartModelRouter(user_budget=1.0)

        # Long prompt with complex keywords should be high complexity
        long_complex_prompt = (
            "Synthesize and compare the methodologies across these papers, "
            "identifying critical research gaps and providing a comprehensive "
            "evaluation of the field: " + "X" * 3000
        )

        decision = router.route(
            task_type="extract_keywords", prompt=long_complex_prompt  # Normally cheap
        )

        # Should upgrade from FAST_CHEAP to at least BALANCED
        assert decision.model in (ModelTier.BALANCED, ModelTier.POWERFUL)

    def test_latency_constraint_downgrade(self):
        """Router should downgrade model when latency constraint is tight."""
        router = SmartModelRouter(user_budget=1.0)

        decision = router.route(
            task_type="synthesis",  # Normally POWERFUL
            prompt="Synthesize: " + "X" * 500,
            max_latency_ms=300,  # Very tight constraint
        )

        # Should downgrade to faster model
        assert decision.model != ModelTier.POWERFUL

    def test_routing_decision_includes_metadata(self):
        """Routing decision should include all required metadata."""
        router = SmartModelRouter(user_budget=1.0)

        decision = router.route(task_type="synthesis", prompt="Test prompt")

        assert hasattr(decision, "model")
        assert hasattr(decision, "reason")
        assert hasattr(decision, "estimated_cost")
        assert hasattr(decision, "estimated_latency_ms")
        assert hasattr(decision, "budget_remaining")

        assert isinstance(decision.reason, str)
        assert decision.estimated_cost >= 0
        assert decision.estimated_latency_ms > 0


class TestModelTierEnum:
    """Tests for ModelTier enum."""

    def test_model_tier_has_correct_values(self):
        """ModelTier should have correct model names."""
        assert ModelTier.FAST_CHEAP.value == "gemini-2.0-flash-lite"
        assert ModelTier.BALANCED.value == "gemini-2.0-flash"
        assert ModelTier.POWERFUL.value == "gemini-1.5-pro"

    def test_model_tiers_are_unique(self):
        """All model tiers should be unique."""
        tiers = [tier.value for tier in ModelTier]
        assert len(tiers) == len(set(tiers))


class TestGetRouter:
    """Tests for the get_router factory function."""

    def test_get_router_creates_router(self):
        """get_router should create a router instance."""
        router = get_router(user_budget=2.0, user_id="test")

        assert isinstance(router, SmartModelRouter)
        assert router.user_budget == 2.0
        assert router.user_id == "test"

    def test_get_router_returns_same_instance_for_same_user(self):
        """get_router should return cached instance for same user."""
        router1 = get_router(user_budget=1.0, user_id="user1")
        router2 = get_router(user_budget=1.0, user_id="user1")

        # Should be the same instance
        assert router1 is router2

    def test_get_router_creates_new_instance_for_different_user(self):
        """get_router should create new instance for different user."""
        router1 = get_router(user_budget=1.0, user_id="user1")
        router2 = get_router(user_budget=1.0, user_id="user2")

        # Should be different instances
        assert router1 is not router2


@pytest.mark.parametrize(
    "task_type,expected_tier",
    [
        ("extract_keywords", ModelTier.FAST_CHEAP),
        ("classify_relevance", ModelTier.FAST_CHEAP),
        ("paper_analysis", ModelTier.BALANCED),
        ("summarization", ModelTier.BALANCED),
        ("synthesis", ModelTier.POWERFUL),
        ("research_gap_identification", ModelTier.POWERFUL),
    ],
)
def test_task_routing_rules(task_type, expected_tier):
    """Test that different task types route to correct models."""
    router = SmartModelRouter(user_budget=10.0)  # High budget

    decision = router.route(
        task_type=task_type, prompt="Test prompt with sufficient length: " + "X" * 200
    )

    # With high budget, should use the expected tier
    assert decision.model == expected_tier


@pytest.mark.parametrize(
    "budget,spent,should_downgrade",
    [
        (1.0, 0.0, False),  # No spending, no downgrade
        (1.0, 0.5, False),  # 50% spent, no downgrade
        (1.0, 0.9, True),  # 90% spent, downgrade
        (1.0, 0.95, True),  # 95% spent, downgrade
    ],
)
def test_budget_based_downgrade(budget, spent, should_downgrade):
    """Test that budget exhaustion triggers downgrades."""
    router = SmartModelRouter(user_budget=budget)
    router.spent = spent

    decision = router.route(
        task_type="synthesis", prompt="Test: " + "X" * 500  # Normally uses POWERFUL
    )

    if should_downgrade:
        assert decision.model == ModelTier.FAST_CHEAP
    else:
        assert decision.model == ModelTier.POWERFUL
