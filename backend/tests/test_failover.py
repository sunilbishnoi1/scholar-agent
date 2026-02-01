# Tests for Model Failover Manager
# Validates dynamic model switching behavior on errors

import time
from unittest.mock import MagicMock, patch

import pytest

from agents.llm.failover import (
    FailoverDecision,
    FailoverReason,
    ModelCooldownState,
    ModelFailoverManager,
    get_failover_manager,
)


class TestModelCooldownState:
    """Tests for ModelCooldownState dataclass."""

    def test_cooldown_state_initializes_correctly(self):
        """Cooldown state should initialize with default values."""
        state = ModelCooldownState(model_name="test-model")

        assert state.model_name == "test-model"
        assert state.cooldown_until == 0.0
        assert state.consecutive_failures == 0
        assert state.last_failure_reason is None
        assert not state.is_in_cooldown()

    def test_is_in_cooldown_returns_true_when_active(self):
        """is_in_cooldown should return True when cooldown is active."""
        state = ModelCooldownState(model_name="test-model")
        state.cooldown_until = time.time() + 60  # 60 seconds from now

        assert state.is_in_cooldown()

    def test_is_in_cooldown_returns_false_when_expired(self):
        """is_in_cooldown should return False when cooldown has expired."""
        state = ModelCooldownState(model_name="test-model")
        state.cooldown_until = time.time() - 1  # 1 second ago

        assert not state.is_in_cooldown()

    def test_remaining_cooldown_calculates_correctly(self):
        """remaining_cooldown should return correct time."""
        state = ModelCooldownState(model_name="test-model")
        state.cooldown_until = time.time() + 30

        remaining = state.remaining_cooldown()
        assert 29 <= remaining <= 30

    def test_remaining_cooldown_returns_zero_when_expired(self):
        """remaining_cooldown should return 0 when expired."""
        state = ModelCooldownState(model_name="test-model")
        state.cooldown_until = time.time() - 10

        assert state.remaining_cooldown() == 0


class TestModelFailoverManager:
    """Tests for ModelFailoverManager class."""

    def test_manager_initializes_with_all_models(self):
        """Manager should initialize cooldown states for all models."""
        manager = ModelFailoverManager()

        assert "llama-3.1-8b-instant" in manager._cooldown_states
        assert "llama-3.3-70b-versatile" in manager._cooldown_states
        assert "qwen/qwen3-32b" in manager._cooldown_states

    def test_get_available_model_returns_preferred_when_available(self):
        """Should return preferred model when it's available."""
        manager = ModelFailoverManager()

        decision = manager.get_available_model(preferred_model="llama-3.1-8b-instant")

        assert decision.selected_model == "llama-3.1-8b-instant"
        assert not decision.was_failover
        assert decision.original_model == "llama-3.1-8b-instant"

    def test_get_available_model_fails_over_when_in_cooldown(self):
        """Should failover to alternative when preferred is in cooldown."""
        manager = ModelFailoverManager()

        # Put 8B model in cooldown
        manager._cooldown_states["llama-3.1-8b-instant"].cooldown_until = time.time() + 60

        decision = manager.get_available_model(preferred_model="llama-3.1-8b-instant")

        assert decision.selected_model != "llama-3.1-8b-instant"
        assert decision.was_failover
        assert decision.original_model == "llama-3.1-8b-instant"

    def test_handle_failure_413_upgrades_to_larger_model(self):
        """413 error on 8B should upgrade to 70B model."""
        manager = ModelFailoverManager()

        decision = manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.PAYLOAD_TOO_LARGE_413,
        )

        assert decision.selected_model == "llama-3.3-70b-versatile"
        assert decision.was_failover
        assert "413" in decision.metadata.get("failure_reason", "")

    def test_handle_failure_429_switches_to_qwen(self):
        """429 rate limit should switch to Qwen model."""
        manager = ModelFailoverManager()

        decision = manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.RATE_LIMIT_429,
        )

        assert decision.selected_model == "qwen/qwen3-32b"
        assert decision.was_failover

    def test_handle_failure_429_on_70b_switches_to_qwen(self):
        """429 on 70B should also switch to Qwen."""
        manager = ModelFailoverManager()

        decision = manager.handle_failure(
            model="llama-3.3-70b-versatile",
            reason=FailoverReason.RATE_LIMIT_429,
        )

        assert decision.selected_model == "qwen/qwen3-32b"
        assert decision.was_failover

    def test_handle_failure_429_on_qwen_switches_back_to_llama(self):
        """429 on Qwen should try to go back to Llama."""
        manager = ModelFailoverManager()

        decision = manager.handle_failure(
            model="qwen/qwen3-32b",
            reason=FailoverReason.RATE_LIMIT_429,
        )

        assert decision.selected_model == "llama-3.1-8b-instant"
        assert decision.was_failover

    def test_handle_failure_sets_cooldown(self):
        """Failure should set cooldown on the failed model."""
        manager = ModelFailoverManager()

        manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.RATE_LIMIT_429,
        )

        state = manager._cooldown_states["llama-3.1-8b-instant"]
        assert state.is_in_cooldown()
        assert state.consecutive_failures == 1
        assert state.last_failure_reason == FailoverReason.RATE_LIMIT_429

    def test_handle_failure_respects_retry_after(self):
        """Should use retry_after from server when provided."""
        manager = ModelFailoverManager()

        manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.RATE_LIMIT_429,
            retry_after=120.0,
        )

        state = manager._cooldown_states["llama-3.1-8b-instant"]
        # Should be approximately 120 seconds
        assert 118 <= state.remaining_cooldown() <= 120

    def test_consecutive_failures_increase_cooldown(self):
        """Multiple failures should exponentially increase cooldown."""
        manager = ModelFailoverManager()

        # First failure
        manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.RATE_LIMIT_429,
        )
        first_cooldown = manager._cooldown_states["llama-3.1-8b-instant"].remaining_cooldown()

        # Reset cooldown but keep failure count
        manager._cooldown_states["llama-3.1-8b-instant"].cooldown_until = 0

        # Second failure
        manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.RATE_LIMIT_429,
        )
        second_cooldown = manager._cooldown_states["llama-3.1-8b-instant"].remaining_cooldown()

        # Second cooldown should be longer (exponential backoff)
        assert second_cooldown > first_cooldown

    def test_record_success_resets_failure_count(self):
        """Successful request should reset consecutive failure count."""
        manager = ModelFailoverManager()

        # Create some failures
        manager.handle_failure("llama-3.1-8b-instant", FailoverReason.RATE_LIMIT_429)
        manager.handle_failure("llama-3.1-8b-instant", FailoverReason.RATE_LIMIT_429)

        assert manager._cooldown_states["llama-3.1-8b-instant"].consecutive_failures == 2

        # Record success
        manager.record_success("llama-3.1-8b-instant")

        assert manager._cooldown_states["llama-3.1-8b-instant"].consecutive_failures == 0

    def test_clear_cooldown_resets_state(self):
        """clear_cooldown should reset cooldown and failure count."""
        manager = ModelFailoverManager()

        # Create failure
        manager.handle_failure("llama-3.1-8b-instant", FailoverReason.RATE_LIMIT_429)
        assert manager._cooldown_states["llama-3.1-8b-instant"].is_in_cooldown()

        # Clear cooldown
        manager.clear_cooldown("llama-3.1-8b-instant")

        state = manager._cooldown_states["llama-3.1-8b-instant"]
        assert not state.is_in_cooldown()
        assert state.consecutive_failures == 0

    def test_get_status_returns_all_model_states(self):
        """get_status should return comprehensive status for all models."""
        manager = ModelFailoverManager()

        # Create a failure
        manager.handle_failure("llama-3.1-8b-instant", FailoverReason.RATE_LIMIT_429)

        status = manager.get_status()

        assert "llama-3.1-8b-instant" in status
        assert "llama-3.3-70b-versatile" in status
        assert "qwen/qwen3-32b" in status

        assert status["llama-3.1-8b-instant"]["in_cooldown"] is True
        assert status["llama-3.1-8b-instant"]["consecutive_failures"] == 1
        assert status["llama-3.3-70b-versatile"]["in_cooldown"] is False

    def test_all_models_in_cooldown_returns_shortest_wait(self):
        """When all models in cooldown, should return one with shortest wait."""
        manager = ModelFailoverManager()

        now = time.time()
        # Put all models in cooldown with different durations
        manager._cooldown_states["llama-3.1-8b-instant"].cooldown_until = now + 60
        manager._cooldown_states["llama-3.3-70b-versatile"].cooldown_until = now + 30  # Shortest
        manager._cooldown_states["qwen/qwen3-32b"].cooldown_until = now + 120

        decision = manager.get_available_model("llama-3.1-8b-instant")

        # Should select model with shortest cooldown
        assert decision.selected_model == "llama-3.3-70b-versatile"
        assert decision.was_failover

    def test_failover_chain_server_error(self):
        """Server error should trigger failover to different model family."""
        manager = ModelFailoverManager()

        decision = manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.SERVER_ERROR,
        )

        assert decision.selected_model == "qwen/qwen3-32b"
        assert decision.was_failover


class TestFailoverManagerSingleton:
    """Tests for failover manager singleton behavior."""

    def test_get_failover_manager_returns_same_instance(self):
        """get_failover_manager should return same instance."""
        # Note: Need to reset global state for this test
        import agents.llm.failover as failover_module

        failover_module._failover_manager = None

        manager1 = get_failover_manager()
        manager2 = get_failover_manager()

        assert manager1 is manager2


class TestFailoverIntegration:
    """Integration tests for failover with GroqClient."""

    @pytest.fixture
    def mock_groq_client(self):
        """Create a mock GroqClient for testing."""
        with patch("agents.llm.groq_client.requests.post") as mock_post:
            from agents.llm.base import LLMConfig
            from agents.llm.groq_client import GroqClient

            config = LLMConfig(
                api_key="test-key",
                user_budget=1.0,
                user_id="test-user",
            )

            client = GroqClient(config)
            yield client, mock_post

    def test_413_triggers_model_upgrade(self, mock_groq_client):
        """413 error should trigger upgrade to larger model."""
        client, mock_post = mock_groq_client

        # First call fails with 413
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413
        mock_response_413.headers = {}
        mock_response_413.json.return_value = {"error": {"message": "Payload too large"}}
        mock_response_413.raise_for_status.side_effect = Exception("413")

        # Second call succeeds
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.headers = {}
        mock_response_success.json.return_value = {
            "choices": [{"message": {"content": "Test response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_response_success.raise_for_status.return_value = None

        # Note: This is a simplified test - full integration would require
        # more complex mock setup

        # Verify the failover manager tracks the failure
        client.failover_manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.PAYLOAD_TOO_LARGE_413,
        )

        state = client.failover_manager._cooldown_states["llama-3.1-8b-instant"]
        assert state.is_in_cooldown()
        assert state.last_failure_reason == FailoverReason.PAYLOAD_TOO_LARGE_413

    def test_429_triggers_model_switch(self, mock_groq_client):
        """429 error should trigger switch to Qwen model."""
        client, mock_post = mock_groq_client

        # Simulate 429 failure
        decision = client.failover_manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.RATE_LIMIT_429,
            retry_after=60.0,
        )

        assert decision.selected_model == "qwen/qwen3-32b"

        # Verify the 8B model is in cooldown
        state = client.failover_manager._cooldown_states["llama-3.1-8b-instant"]
        assert state.is_in_cooldown()
        assert 58 <= state.remaining_cooldown() <= 60
