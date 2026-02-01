# Tests for Usage Tracking (Phase 3: Production Features)
from datetime import date, datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class TestUsageTracker:
    """Tests for the UsageTracker service."""

    @pytest.fixture
    def db_session(self):
        """Create an in-memory SQLite database for testing."""
        from models.database import Base, LLMInteraction, User, UserUsage

        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        yield session

        session.close()

    @pytest.fixture
    def test_user(self, db_session):
        """Create a test user."""
        from models.database import User

        user = User(
            id="test-user-123",
            email="test@example.com",
            name="Test User",
            hashed_password="hashed",
            tier="free",
            monthly_budget_usd=1.0,
        )
        db_session.add(user)
        db_session.commit()
        return user

    @pytest.fixture
    def usage_tracker(self, db_session):
        """Create a UsageTracker instance."""
        from services.usage_tracker import UsageTracker

        return UsageTracker(db_session)

    def test_get_or_create_usage_creates_new_record(self, usage_tracker, test_user, db_session):
        """Test that get_or_create_usage creates a new record if none exists."""
        from models.database import UserUsage

        # Should have no usage records initially
        count_before = db_session.query(UserUsage).filter_by(user_id=test_user.id).count()
        assert count_before == 0

        # Get or create usage
        usage = usage_tracker.get_or_create_usage(test_user.id)

        # Should have one record now
        count_after = db_session.query(UserUsage).filter_by(user_id=test_user.id).count()
        assert count_after == 1
        assert usage.user_id == test_user.id
        assert usage.total_tokens == 0
        assert usage.total_cost_usd == 0.0

    def test_get_or_create_usage_returns_existing(self, usage_tracker, test_user, db_session):
        """Test that get_or_create_usage returns existing record."""
        from models.database import UserUsage

        # Create initial usage
        usage1 = usage_tracker.get_or_create_usage(test_user.id)
        usage1.total_tokens = 1000
        db_session.commit()

        # Get again - should return same record
        usage2 = usage_tracker.get_or_create_usage(test_user.id)
        assert usage2.total_tokens == 1000
        assert usage2.id == usage1.id

    def test_check_budget_allows_under_limit(self, usage_tracker, test_user):
        """Test that budget check allows operations under the limit."""
        result = usage_tracker.check_budget(test_user, estimated_cost=0.1)

        assert result["allowed"] is True
        assert result["remaining_budget"] == 1.0
        assert result["limit"] == 1.0

    def test_check_budget_denies_over_limit(self, usage_tracker, test_user, db_session):
        """Test that budget check denies operations over the limit."""
        # Use up most of the budget
        usage = usage_tracker.get_or_create_usage(test_user.id)
        usage.total_cost_usd = 0.95
        db_session.commit()

        # Try to use more than remaining
        result = usage_tracker.check_budget(test_user, estimated_cost=0.1)

        assert result["allowed"] is False
        assert "error" in result

    def test_check_budget_warning_near_limit(self, usage_tracker, test_user, db_session):
        """Test that budget check warns when near limit."""
        # Use 95% of budget
        usage = usage_tracker.get_or_create_usage(test_user.id)
        usage.total_cost_usd = 0.95
        db_session.commit()

        result = usage_tracker.check_budget(test_user)

        assert "warning" in result

    def test_record_llm_call_updates_usage(self, usage_tracker, test_user, db_session):
        """Test that recording an LLM call updates usage statistics."""
        from models.database import LLMInteraction

        usage_tracker.record_llm_call(
            user_id=test_user.id,
            agent_type="analyzer",
            model="gemini-2.0-flash-lite",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.001,
            latency_ms=500,
            project_id="project-123",
            task_type="paper_analysis",
            prompt_preview="Analyze this paper...",
            response_preview="The paper discusses...",
            success=True,
        )

        # Check that LLMInteraction was created
        interactions = db_session.query(LLMInteraction).filter_by(user_id=test_user.id).all()
        assert len(interactions) == 1
        assert interactions[0].total_tokens == 150

        # Check that usage was updated
        usage = usage_tracker.get_or_create_usage(test_user.id)
        assert usage.total_tokens == 150
        assert usage.llm_calls == 1
        assert usage.total_cost_usd == 0.001

    def test_record_project_created(self, usage_tracker, test_user):
        """Test that recording project creation updates count."""
        usage_tracker.record_project_created(test_user.id)

        usage = usage_tracker.get_or_create_usage(test_user.id)
        assert usage.projects_created == 1

        usage_tracker.record_project_created(test_user.id)
        usage = usage_tracker.get_or_create_usage(test_user.id)
        assert usage.projects_created == 2

    def test_record_paper_analyzed(self, usage_tracker, test_user):
        """Test that recording papers analyzed updates count."""
        usage_tracker.record_paper_analyzed(test_user.id, count=5)

        usage = usage_tracker.get_or_create_usage(test_user.id)
        assert usage.papers_analyzed == 5

    def test_get_usage_summary(self, usage_tracker, test_user, db_session):
        """Test that usage summary returns correct data."""
        # Add some usage data
        usage_tracker.record_llm_call(
            user_id=test_user.id,
            agent_type="planner",
            model="gemini-2.0-flash",
            prompt_tokens=200,
            completion_tokens=100,
            cost_usd=0.03,
            latency_ms=800,
            success=True,
        )
        usage_tracker.record_project_created(test_user.id)
        usage_tracker.record_paper_analyzed(test_user.id, count=10)

        summary = usage_tracker.get_usage_summary(test_user)

        assert summary["user_id"] == test_user.id
        assert summary["tier"] == "free"
        assert summary["budget"]["used_usd"] == 0.03
        assert summary["budget"]["limit_usd"] == 1.0
        assert summary["tokens"]["used"] == 300
        assert summary["activity"]["projects_created"] == 1
        assert summary["activity"]["papers_analyzed"] == 10
        assert summary["activity"]["llm_calls"] == 1

    def test_get_user_limits_by_tier(self, usage_tracker, test_user, db_session):
        """Test that user limits vary by tier."""
        from services.usage_tracker import TIER_LIMITS

        # Test free tier limits
        limits = usage_tracker.get_user_limits(test_user)
        assert limits["monthly_budget_usd"] == 1.0
        assert limits["max_projects"] == 5

        # Test pro tier limits
        test_user.tier = "pro"
        test_user.monthly_budget_usd = 10.0
        db_session.commit()

        limits = usage_tracker.get_user_limits(test_user)
        assert limits["monthly_budget_usd"] == 10.0
        assert limits["max_projects"] == 50


class TestTierLimits:
    """Tests for tier limit configuration."""

    def test_free_tier_limits(self):
        """Test that free tier has appropriate limits."""
        from services.usage_tracker import TIER_LIMITS

        free = TIER_LIMITS["free"]
        assert free["monthly_budget_usd"] == 1.0
        assert free["monthly_tokens"] == 100000
        assert free["max_projects"] == 5
        assert free["max_papers_per_project"] == 30

    def test_pro_tier_limits(self):
        """Test that pro tier has higher limits."""
        from services.usage_tracker import TIER_LIMITS

        pro = TIER_LIMITS["pro"]
        assert pro["monthly_budget_usd"] == 10.0
        assert pro["monthly_tokens"] == 1000000
        assert pro["max_projects"] == 50
        assert pro["max_papers_per_project"] == 100

    def test_enterprise_tier_limits(self):
        """Test that enterprise tier has highest limits."""
        from services.usage_tracker import TIER_LIMITS

        enterprise = TIER_LIMITS["enterprise"]
        assert enterprise["monthly_budget_usd"] == 100.0
        assert enterprise["max_projects"] is None  # Unlimited
        assert enterprise["max_papers_per_project"] is None  # Unlimited
