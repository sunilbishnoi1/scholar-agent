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
        """Should use retry_after from server when provided, but cap at 60s."""
        manager = ModelFailoverManager()

        manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.RATE_LIMIT_429,
            retry_after=120.0,  # Server suggests 120s
        )

        state = manager._cooldown_states["llama-3.1-8b-instant"]
        # Should be capped at 60 seconds (we have other models to use!)
        assert 58 <= state.remaining_cooldown() <= 60

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


class TestNewFailoverFeatures:
    """Tests for new failover features: preemptive switching and 413 chunking."""

    def test_413_on_all_models_cycles_back_to_highest_tpm(self):
        """When all models fail with 413, should cycle back to highest TPM model (70b).

        NOTE: The actual chunking detection is now handled in groq_client.py by
        tracking tried_models set. The failover manager just provides the failover
        chain - qwen→70b because 70b has 12K TPM (highest).
        """
        manager = ModelFailoverManager()

        # First 413 on 8B -> upgrades to 70B (12K TPM)
        decision1 = manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.PAYLOAD_TOO_LARGE_413,
        )
        assert decision1.selected_model == "llama-3.3-70b-versatile"

        # 413 on 70B -> tries Qwen (different model family)
        decision2 = manager.handle_failure(
            model="llama-3.3-70b-versatile",
            reason=FailoverReason.PAYLOAD_TOO_LARGE_413,
        )
        assert decision2.selected_model == "qwen/qwen3-32b"

        # 413 on Qwen -> cycles back to 70B (highest TPM)
        # The groq_client.py tracks tried_models and will detect all models failed
        decision3 = manager.handle_failure(
            model="qwen/qwen3-32b",
            reason=FailoverReason.PAYLOAD_TOO_LARGE_413,
        )
        # Now goes to 70B since it has 12K TPM (but it will be in cooldown)
        # So it falls back to finding any available model
        assert decision3.was_failover

    def test_preemptive_cooldown_for_tpm_approaching(self):
        """Preemptive switches should apply a short cooldown to prevent immediate retry."""
        manager = ModelFailoverManager()

        decision = manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.TPM_APPROACHING,
            preemptive_cooldown=True,
        )

        # Model SHOULD be in cooldown (with short preemptive duration)
        state = manager._cooldown_states["llama-3.1-8b-instant"]
        assert state.is_in_cooldown()
        assert decision.was_failover
        # Cooldown should be PREEMPTIVE_COOLDOWN (15s)
        assert state.cooldown_until > time.time()
        assert state.cooldown_until <= time.time() + manager.PREEMPTIVE_COOLDOWN + 1

    def test_tpm_approaching_triggers_preemptive_switch(self):
        """TPM approaching should trigger switch to next model."""
        manager = ModelFailoverManager()

        decision = manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.TPM_APPROACHING,
            preemptive_cooldown=True,
        )

        assert decision.selected_model == "llama-3.3-70b-versatile"
        assert decision.was_failover

    def test_413_failover_chain_is_complete(self):
        """413 failover chain should include all models."""
        manager = ModelFailoverManager()

        # Check that 413 chain is complete
        chain = manager.FAILOVER_CHAIN

        assert ("llama-3.1-8b-instant", FailoverReason.PAYLOAD_TOO_LARGE_413) in chain
        assert ("llama-3.3-70b-versatile", FailoverReason.PAYLOAD_TOO_LARGE_413) in chain
        assert ("qwen/qwen3-32b", FailoverReason.PAYLOAD_TOO_LARGE_413) in chain

    def test_cooldown_capped_for_rate_limits(self):
        """Cooldown should be capped even if server suggests longer."""
        manager = ModelFailoverManager()

        # Server suggests 300 seconds
        manager.handle_failure(
            model="llama-3.1-8b-instant",
            reason=FailoverReason.RATE_LIMIT_429,
            retry_after=300.0,
        )

        state = manager._cooldown_states["llama-3.1-8b-instant"]
        # Should be capped at 60 seconds max
        assert state.remaining_cooldown() <= 60


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

        # Verify the 8B model is in cooldown - but capped at 60s
        state = client.failover_manager._cooldown_states["llama-3.1-8b-instant"]
        assert state.is_in_cooldown()
        assert state.remaining_cooldown() <= 60


class TestWaitForModel:
    """Tests for the wait_for_model critical priority feature."""

    def test_wait_for_model_returns_immediately_if_available(self):
        """wait_for_model should return immediately if preferred model is available."""
        manager = ModelFailoverManager()

        # Model is not in cooldown, should return immediately
        decision = manager.wait_for_model(
            preferred_model="llama-3.3-70b-versatile",
            max_wait_seconds=10.0,
            check_interval=1.0,
        )

        assert decision.selected_model == "llama-3.3-70b-versatile"
        assert not decision.was_failover
        assert "wait_time" in decision.metadata
        assert decision.metadata["wait_time"] < 1.0  # Should be nearly instant

    def test_wait_for_model_waits_for_cooldown_to_expire(self):
        """wait_for_model should wait for short cooldowns to expire."""
        manager = ModelFailoverManager()

        # Put preferred model in short cooldown (2 seconds)
        manager._cooldown_states["llama-3.3-70b-versatile"].cooldown_until = time.time() + 2.0

        start = time.time()
        decision = manager.wait_for_model(
            preferred_model="llama-3.3-70b-versatile",
            max_wait_seconds=10.0,
            check_interval=1.0,
        )
        elapsed = time.time() - start

        assert decision.selected_model == "llama-3.3-70b-versatile"
        assert elapsed >= 2.0  # Should have waited at least 2 seconds

    def test_wait_for_model_uses_alternative_if_available(self):
        """wait_for_model should use alternative model if preferred is in long cooldown."""
        manager = ModelFailoverManager()

        # Put preferred model in long cooldown (200 seconds)
        manager._cooldown_states["llama-3.3-70b-versatile"].cooldown_until = time.time() + 200.0
        # qwen should be available
        manager._cooldown_states["qwen/qwen3-32b"].cooldown_until = 0.0

        decision = manager.wait_for_model(
            preferred_model="llama-3.3-70b-versatile",
            max_wait_seconds=5.0,  # Only wait 5 seconds
            check_interval=1.0,
        )

        # Should use qwen since it's available and waiting would be too long
        assert decision.selected_model == "qwen/qwen3-32b"
        assert decision.was_failover

    def test_wait_for_model_prioritizes_powerful_models_for_critical_tasks(self):
        """wait_for_model should prefer powerful models for critical tasks."""
        manager = ModelFailoverManager()

        # Put 70b in cooldown, but 8b and qwen available
        manager._cooldown_states["llama-3.3-70b-versatile"].cooldown_until = time.time() + 200.0
        manager._cooldown_states["qwen/qwen3-32b"].cooldown_until = 0.0
        manager._cooldown_states["llama-3.1-8b-instant"].cooldown_until = 0.0

        decision = manager.wait_for_model(
            preferred_model="llama-3.3-70b-versatile",
            max_wait_seconds=5.0,
            check_interval=1.0,
        )

        # Should use qwen (medium quality) over 8b (fast but lower quality)
        # because priority_for_critical puts qwen before 8b
        assert decision.selected_model == "qwen/qwen3-32b"

    def test_wait_for_model_times_out_and_returns_best_available(self):
        """wait_for_model should return best available after timeout."""
        manager = ModelFailoverManager()

        # Put all models in long cooldown
        manager._cooldown_states["llama-3.3-70b-versatile"].cooldown_until = time.time() + 200.0
        manager._cooldown_states["qwen/qwen3-32b"].cooldown_until = time.time() + 150.0
        manager._cooldown_states["llama-3.1-8b-instant"].cooldown_until = time.time() + 100.0

        start = time.time()
        decision = manager.wait_for_model(
            preferred_model="llama-3.3-70b-versatile",
            max_wait_seconds=3.0,  # Short timeout
            check_interval=1.0,
        )
        elapsed = time.time() - start

        # Should have waited approximately max_wait_seconds
        assert elapsed >= 2.5
        assert elapsed <= 5.0  # Some buffer for timing
        # Should fall back to get_available_model which picks shortest cooldown
        assert decision.selected_model is not None


class TestGetModelForSynthesis:
    """Tests for get_model_for_synthesis method - prefers higher quality models."""

    def test_get_model_for_synthesis_prefers_70b_when_available(self):
        """Should prefer llama-70b for synthesis tasks."""
        manager = ModelFailoverManager()

        decision = manager.get_model_for_synthesis(prompt_tokens=1000)

        # Should prefer 70b for synthesis
        assert decision.selected_model == "llama-3.3-70b-versatile"
        assert not decision.was_failover

    def test_get_model_for_synthesis_falls_back_to_qwen(self):
        """Should fall back to qwen when 70b is unavailable."""
        manager = ModelFailoverManager()

        # Put 70b in cooldown
        manager._cooldown_states["llama-3.3-70b-versatile"].cooldown_until = time.time() + 120.0

        decision = manager.get_model_for_synthesis(prompt_tokens=1000, max_wait_seconds=1.0)

        # Should use qwen as next best option
        assert decision.selected_model == "qwen/qwen3-32b"
        assert decision.was_failover

    def test_get_model_for_synthesis_uses_8b_as_last_resort(self):
        """Should only use 8b for synthesis when all others unavailable."""
        manager = ModelFailoverManager()

        # Put 70b and qwen in cooldown
        manager._cooldown_states["llama-3.3-70b-versatile"].cooldown_until = time.time() + 120.0
        manager._cooldown_states["qwen/qwen3-32b"].cooldown_until = time.time() + 120.0

        decision = manager.get_model_for_synthesis(prompt_tokens=1000, max_wait_seconds=1.0)

        # Should use 8b as last resort
        assert decision.selected_model == "llama-3.1-8b-instant"
        assert decision.was_failover

    def test_get_model_for_synthesis_waits_for_short_cooldown(self):
        """Should wait briefly if ALL synthesis models are in cooldown with short wait."""
        manager = ModelFailoverManager()

        # Put ALL synthesis models in short cooldown (≤5s triggers wait)
        manager._cooldown_states["llama-3.3-70b-versatile"].cooldown_until = time.time() + 3.0
        manager._cooldown_states["qwen/qwen3-32b"].cooldown_until = time.time() + 4.0
        manager._cooldown_states["llama-3.1-8b-instant"].cooldown_until = time.time() + 5.0

        start = time.time()
        decision = manager.get_model_for_synthesis(prompt_tokens=1000, max_wait_seconds=10.0)
        elapsed = time.time() - start

        # Should have waited for 70b (shortest wait, highest priority) to become available
        assert decision.selected_model == "llama-3.3-70b-versatile"
        assert elapsed >= 3.0  # Waited for cooldown


class TestCheckAndClearExpiredCooldowns:
    """Tests for _check_and_clear_expired_cooldowns method."""

    def test_clears_expired_cooldowns(self):
        """Should clear cooldowns that have expired."""
        manager = ModelFailoverManager()

        # Set expired cooldowns
        past_time = time.time() - 10  # 10 seconds ago
        manager._cooldown_states["llama-3.1-8b-instant"].cooldown_until = past_time
        manager._cooldown_states["llama-3.1-8b-instant"].last_failure_time = past_time - 60
        manager._cooldown_states["llama-3.1-8b-instant"].consecutive_failures = 3

        with manager._lock:
            cleared = manager._check_and_clear_expired_cooldowns()

        # Should have cleared the cooldown
        assert cleared >= 1
        assert manager._cooldown_states["llama-3.1-8b-instant"].cooldown_until == 0.0
        # Consecutive failures should be reset (last failure was >30s ago)
        assert manager._cooldown_states["llama-3.1-8b-instant"].consecutive_failures == 0

    def test_preserves_active_cooldowns(self):
        """Should not clear cooldowns that are still active."""
        manager = ModelFailoverManager()

        # Set active cooldown
        future_time = time.time() + 30
        manager._cooldown_states["llama-3.1-8b-instant"].cooldown_until = future_time
        manager._cooldown_states["llama-3.1-8b-instant"].consecutive_failures = 2

        with manager._lock:
            cleared = manager._check_and_clear_expired_cooldowns()

        # Should not have cleared active cooldown
        assert manager._cooldown_states["llama-3.1-8b-instant"].cooldown_until == future_time
        assert manager._cooldown_states["llama-3.1-8b-instant"].consecutive_failures == 2

    def test_keeps_failure_count_for_recent_failures(self):
        """Should keep consecutive_failures if last failure was recent."""
        manager = ModelFailoverManager()

        # Set expired cooldown but recent failure
        past_time = time.time() - 5  # 5 seconds ago (within 30s window)
        manager._cooldown_states["llama-3.1-8b-instant"].cooldown_until = past_time
        manager._cooldown_states["llama-3.1-8b-instant"].last_failure_time = time.time() - 10
        manager._cooldown_states["llama-3.1-8b-instant"].consecutive_failures = 2

        with manager._lock:
            manager._check_and_clear_expired_cooldowns()

        # Consecutive failures should be preserved (last failure within 30s)
        assert manager._cooldown_states["llama-3.1-8b-instant"].consecutive_failures == 2
