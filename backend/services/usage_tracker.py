# Usage Tracking Service
# Provides business logic for tracking and enforcing usage quotas

import logging
from datetime import date
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from models.database import LLMInteraction, User, UserUsage

logger = logging.getLogger(__name__)


# =========================================================================
# User Tier Limits
# =========================================================================

TIER_LIMITS = {
    "free": {
        "monthly_budget_usd": 1.0,
        "monthly_tokens": 100000,
        "max_projects": 5,
        "max_papers_per_project": 30,
    },
    "pro": {
        "monthly_budget_usd": 10.0,
        "monthly_tokens": 1000000,
        "max_projects": 50,
        "max_papers_per_project": 100,
    },
    "enterprise": {
        "monthly_budget_usd": 100.0,
        "monthly_tokens": 10000000,
        "max_projects": None,  # Unlimited
        "max_papers_per_project": None,
    },
}


class UsageTracker:
    """
    Tracks and enforces usage quotas for users.
    
    Features:
    - Monthly usage tracking
    - Budget enforcement
    - LLM interaction logging
    - Analytics and reporting
    """

    def __init__(self, db: Session):
        self.db = db

    def get_current_month(self) -> date:
        """Get the first day of the current month."""
        today = date.today()
        return today.replace(day=1)

    def get_or_create_usage(self, user_id: str) -> UserUsage:
        """Get or create the current month's usage record for a user."""
        current_month = self.get_current_month()

        usage = self.db.query(UserUsage).filter(
            UserUsage.user_id == user_id,
            UserUsage.month == current_month
        ).first()

        if not usage:
            usage = UserUsage(
                user_id=user_id,
                month=current_month,
                total_tokens=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_cost_usd=0.0,
                projects_created=0,
                papers_analyzed=0,
                llm_calls=0
            )
            self.db.add(usage)
            self.db.commit()
            self.db.refresh(usage)

        return usage

    def get_user_limits(self, user: User) -> dict[str, Any]:
        """Get the limits for a user based on their tier."""
        tier = user.tier or "free"
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"]).copy()

        # Override with user-specific budget if set
        if user.monthly_budget_usd:
            limits["monthly_budget_usd"] = user.monthly_budget_usd

        return limits

    def check_budget(self, user: User, estimated_cost: float = 0.0) -> dict[str, Any]:
        """
        Check if user has remaining budget.
        
        Returns:
            Dict with:
            - allowed: bool - whether the operation is allowed
            - remaining_budget: float - remaining budget
            - current_usage: float - current month's usage
            - limit: float - monthly limit
            - warning: str - optional warning message
        """
        usage = self.get_or_create_usage(user.id)
        limits = self.get_user_limits(user)

        current_cost = usage.total_cost_usd or 0.0
        budget_limit = limits["monthly_budget_usd"]
        remaining = budget_limit - current_cost

        result = {
            "allowed": True,
            "remaining_budget": remaining,
            "current_usage": current_cost,
            "limit": budget_limit,
            "usage_percent": (current_cost / budget_limit * 100) if budget_limit > 0 else 0,
        }

        # Check if estimated cost would exceed budget
        if estimated_cost > 0 and current_cost + estimated_cost > budget_limit:
            result["allowed"] = False
            result["error"] = f"Estimated cost ${estimated_cost:.4f} would exceed remaining budget ${remaining:.4f}"

        # Add warning if close to limit
        if remaining < budget_limit * 0.1:  # Less than 10% remaining
            result["warning"] = f"You have used {result['usage_percent']:.1f}% of your monthly budget"

        return result

    def check_token_limit(self, user: User, estimated_tokens: int = 0) -> dict[str, Any]:
        """Check if user has remaining token allowance."""
        usage = self.get_or_create_usage(user.id)
        limits = self.get_user_limits(user)

        current_tokens = usage.total_tokens or 0
        token_limit = limits["monthly_tokens"]
        remaining = token_limit - current_tokens

        result = {
            "allowed": True,
            "remaining_tokens": remaining,
            "current_usage": current_tokens,
            "limit": token_limit,
            "usage_percent": (current_tokens / token_limit * 100) if token_limit > 0 else 0,
        }

        if estimated_tokens > 0 and current_tokens + estimated_tokens > token_limit:
            result["allowed"] = False
            result["error"] = f"Estimated tokens {estimated_tokens} would exceed remaining allowance {remaining}"

        return result

    def record_llm_call(
        self,
        user_id: str,
        agent_type: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: int,
        project_id: str = None,
        task_type: str = None,
        prompt_preview: str = None,
        response_preview: str = None,
        success: bool = True,
        error_message: str = None,
        metadata: dict = None
    ):
        """
        Record an LLM interaction and update usage counters.
        
        This method:
        1. Creates an LLMInteraction record for detailed logging
        2. Updates the UserUsage aggregates
        """
        # Create detailed interaction record
        interaction = LLMInteraction(
            user_id=user_id,
            project_id=project_id,
            agent_type=agent_type,
            task_type=task_type,
            model=model,
            prompt_preview=prompt_preview[:500] if prompt_preview else None,
            response_preview=response_preview[:500] if response_preview else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            success=success,
            error_message=error_message,
            metadata=metadata
        )
        self.db.add(interaction)

        # Update usage aggregates
        usage = self.get_or_create_usage(user_id)
        usage.total_tokens = (usage.total_tokens or 0) + prompt_tokens + completion_tokens
        usage.prompt_tokens = (usage.prompt_tokens or 0) + prompt_tokens
        usage.completion_tokens = (usage.completion_tokens or 0) + completion_tokens
        usage.total_cost_usd = (usage.total_cost_usd or 0) + cost_usd
        usage.llm_calls = (usage.llm_calls or 0) + 1

        self.db.commit()

        logger.info(
            f"LLM call recorded: user={user_id}, agent={agent_type}, "
            f"tokens={prompt_tokens + completion_tokens}, cost=${cost_usd:.6f}"
        )

    def record_project_created(self, user_id: str):
        """Increment project created counter."""
        usage = self.get_or_create_usage(user_id)
        usage.projects_created = (usage.projects_created or 0) + 1
        self.db.commit()

    def record_paper_analyzed(self, user_id: str, count: int = 1):
        """Increment papers analyzed counter."""
        usage = self.get_or_create_usage(user_id)
        usage.papers_analyzed = (usage.papers_analyzed or 0) + count
        self.db.commit()

    def get_usage_summary(self, user: User) -> dict[str, Any]:
        """
        Get a comprehensive usage summary for a user.
        
        Returns:
            Dict with current usage, limits, and analytics
        """
        usage = self.get_or_create_usage(user.id)
        limits = self.get_user_limits(user)

        # Budget info
        budget_limit = limits["monthly_budget_usd"]
        current_cost = usage.total_cost_usd or 0.0
        budget_remaining = budget_limit - current_cost

        # Token info
        token_limit = limits["monthly_tokens"]
        current_tokens = usage.total_tokens or 0
        tokens_remaining = token_limit - current_tokens

        return {
            "user_id": user.id,
            "tier": user.tier or "free",
            "month": usage.month.isoformat(),

            # Budget
            "budget": {
                "used_usd": current_cost,
                "limit_usd": budget_limit,
                "remaining_usd": budget_remaining,
                "usage_percent": (current_cost / budget_limit * 100) if budget_limit > 0 else 0,
            },

            # Tokens
            "tokens": {
                "used": current_tokens,
                "limit": token_limit,
                "remaining": tokens_remaining,
                "usage_percent": (current_tokens / token_limit * 100) if token_limit > 0 else 0,
                "prompt_tokens": usage.prompt_tokens or 0,
                "completion_tokens": usage.completion_tokens or 0,
            },

            # Activity
            "activity": {
                "projects_created": usage.projects_created or 0,
                "papers_analyzed": usage.papers_analyzed or 0,
                "llm_calls": usage.llm_calls or 0,
            },

            # Limits
            "limits": {
                "max_projects": limits["max_projects"],
                "max_papers_per_project": limits["max_papers_per_project"],
            },

            "updated_at": usage.updated_at.isoformat() if usage.updated_at else None,
        }

    def get_usage_analytics(
        self,
        user_id: str,
        start_date: date = None,
        end_date: date = None
    ) -> dict[str, Any]:
        """
        Get detailed analytics for a user's usage.
        
        Args:
            user_id: The user's ID
            start_date: Start of analytics period (defaults to 30 days ago)
            end_date: End of analytics period (defaults to today)
        """
        if not end_date:
            end_date = date.today()
        if not start_date:
            start_date = end_date.replace(day=1)  # Start of current month

        # Get LLM interactions for the period
        interactions = self.db.query(LLMInteraction).filter(
            LLMInteraction.user_id == user_id,
            func.date(LLMInteraction.created_at) >= start_date,
            func.date(LLMInteraction.created_at) <= end_date
        ).all()

        # Aggregate by agent type
        by_agent = {}
        for interaction in interactions:
            agent = interaction.agent_type
            if agent not in by_agent:
                by_agent[agent] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "avg_latency_ms": 0,
                    "errors": 0
                }
            by_agent[agent]["calls"] += 1
            by_agent[agent]["tokens"] += interaction.total_tokens or 0
            by_agent[agent]["cost_usd"] += interaction.cost_usd or 0.0
            by_agent[agent]["avg_latency_ms"] += interaction.latency_ms or 0
            if not interaction.success:
                by_agent[agent]["errors"] += 1

        # Calculate averages
        for agent in by_agent:
            if by_agent[agent]["calls"] > 0:
                by_agent[agent]["avg_latency_ms"] //= by_agent[agent]["calls"]

        # Aggregate by model
        by_model = {}
        for interaction in interactions:
            model = interaction.model
            if model not in by_model:
                by_model[model] = {"calls": 0, "tokens": 0, "cost_usd": 0.0}
            by_model[model]["calls"] += 1
            by_model[model]["tokens"] += interaction.total_tokens or 0
            by_model[model]["cost_usd"] += interaction.cost_usd or 0.0

        # Daily breakdown
        daily = {}
        for interaction in interactions:
            day = interaction.created_at.date().isoformat()
            if day not in daily:
                daily[day] = {"calls": 0, "tokens": 0, "cost_usd": 0.0}
            daily[day]["calls"] += 1
            daily[day]["tokens"] += interaction.total_tokens or 0
            daily[day]["cost_usd"] += interaction.cost_usd or 0.0

        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "totals": {
                "calls": len(interactions),
                "tokens": sum(i.total_tokens or 0 for i in interactions),
                "cost_usd": sum(i.cost_usd or 0 for i in interactions),
                "errors": sum(1 for i in interactions if not i.success),
            },
            "by_agent": by_agent,
            "by_model": by_model,
            "daily": daily,
        }


def get_usage_tracker(db: Session) -> UsageTracker:
    """Factory function to create a UsageTracker instance."""
    return UsageTracker(db)
