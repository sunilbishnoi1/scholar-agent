"""
Empirical Resilience and Adversarial Stress Test Suite for Milestone 4.
Challenger 1: Backend & Multi-Agent Resilience Challenger.

Covers:
1. Rate Limiter sliding window and burst smoothing under high concurrency.
2. Structured output 5-stage progressive fallback parser under malformed JSON, markdown fences, and syntax repairs.
3. Cancellation Manager state propagation, thread safety, and Celery task revoking.
4. Blackboard state transitions, LangGraph state conversions, database sync, and agent handoff routing.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import time
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from agents.blackboard import WorkingMemoryBlackboard
from agents.core.supervisor import should_refine_or_finalize
from agents.llm.rate_limiter import (
    ProviderRateLimiter,
    get_provider_limiter,
    get_rate_limiter,
)
from agents.llm.structured_output import (
    StructuredOutputError,
    clean_json_markdown,
    extract_json_substring,
    parse_and_validate,
    repair_json_syntax,
    to_gemini_schema,
    to_openai_response_format,
)
from agents.schemas import (
    CitationAuditReport,
    ConflictingDebate,
    CriticEvaluation,
    EvidenceMatrixRow,
    MethodologyDistribution,
    ResearchGapItem,
    ResearchReport,
    ThematicSection,
)
from agents.state import AgentState, AgentType, GoalStatus
from models.database import (
    Base,
    EvidenceMatrixEntry,
    PaperCache,
    ResearchGapModel,
    ResearchProject,
    ResearchReportModel,
)
from services.cancellation_manager import (
    CancellationManager,
    TaskCancelledException,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# =============================================================================
# 1. RATE LIMITER STRESS & CONCURRENCY TESTS
# =============================================================================

class TestRateLimiterResilience:
    """Empirical tests for ProviderRateLimiter sliding window and burst smoothing."""

    def test_rate_limiter_initialization_defaults(self):
        limiter = ProviderRateLimiter(key="test_init", max_rpm=15)
        assert limiter.max_rpm == 15
        assert limiter.min_interval == pytest.approx(60.0 / 15, rel=1e-3)
        assert limiter._timestamps == []

    def test_rate_limiter_zero_or_negative_rpm(self):
        limiter = ProviderRateLimiter(key="test_zero", max_rpm=0)
        assert limiter.acquire() == 0.0
        assert limiter.min_interval == 0.0

    def test_provider_hash_key_isolation(self):
        lim1 = get_provider_limiter("gemini", api_key="key_alpha", max_rpm=14)
        lim2 = get_provider_limiter("gemini", api_key="key_alpha", max_rpm=14)
        lim3 = get_provider_limiter("gemini", api_key="key_beta", max_rpm=14)
        lim4 = get_provider_limiter("deepseek", api_key="key_alpha", max_rpm=14)

        assert lim1 is lim2, "Same provider and API key must share singleton limiter instance"
        assert lim1 is not lim3, "Different API keys must have isolated limiter instances"
        assert lim1 is not lim4, "Different providers must have isolated limiter instances"

    def test_in_memory_burst_smoothing_spacing(self):
        # Configure limiter with short min_interval to test spacing
        limiter = ProviderRateLimiter(key="burst_test", max_rpm=600, min_interval=0.05)
        
        t0 = time.time()
        wait1 = limiter.acquire()
        t1 = time.time()
        wait2 = limiter.acquire()
        t2 = time.time()

        assert wait1 == 0.0
        assert (t2 - t1) >= 0.045, f"Burst smoothing should enforce spacing near 0.05s, got {t2 - t1:.4f}s"
        assert wait2 >= 0.04

    @pytest.mark.asyncio
    async def test_async_concurrent_acquire_respects_sliding_window(self):
        # 10 concurrent async tasks hitting limiter with min_interval = 0.02s
        limiter = ProviderRateLimiter(key="async_burst", max_rpm=500, min_interval=0.02)

        start_time = time.time()
        
        async def call_limiter(idx: int):
            waited = await limiter.acquire_async()
            timestamp = time.time()
            return idx, waited, timestamp

        results = await asyncio.gather(*[call_limiter(i) for i in range(5)])
        end_time = time.time()

        # Check that timestamps are ordered and spaced
        timestamps = sorted([r[2] for r in results])
        for i in range(1, len(timestamps)):
            diff = timestamps[i] - timestamps[i - 1]
            assert diff >= 0.015, f"Requests {i-1} and {i} were spaced only {diff:.4f}s apart"

        total_span = end_time - start_time
        assert total_span >= 0.07, f"Total execution time {total_span:.4f}s should span at least 4 intervals"

    def test_threaded_concurrency_stress(self):
        limiter = ProviderRateLimiter(key="threaded_stress", max_rpm=1000, min_interval=0.01)
        timestamps = []
        lock = concurrent.futures.thread.threading.Lock()

        def worker():
            limiter.acquire()
            now = time.time()
            with lock:
                timestamps.append(now)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker) for _ in range(8)]
            concurrent.futures.wait(futures)

        sorted_ts = sorted(timestamps)
        assert len(sorted_ts) == 8
        for i in range(1, len(sorted_ts)):
            diff = sorted_ts[i] - sorted_ts[i - 1]
            assert diff >= 0.008, f"Threaded requests spaced only {diff:.4f}s apart"


# =============================================================================
# 2. STRUCTURED OUTPUT 5-STAGE PROGRESSIVE FALLBACK PARSER
# =============================================================================

class SampleModel(BaseModel):
    title: str
    confidence: float
    tags: List[str]
    is_valid: bool
    optional_note: Optional[str] = None


class TestStructuredOutputProgressiveParser:
    """Empirical tests for 5-stage progressive JSON parsing and repair."""

    def test_stage1_direct_valid_json(self):
        raw = '{"title": "Clean Title", "confidence": 0.95, "tags": ["AI", "NLP"], "is_valid": true}'
        res = parse_and_validate(raw, SampleModel)
        assert res.title == "Clean Title"
        assert res.confidence == 0.95
        assert res.tags == ["AI", "NLP"]
        assert res.is_valid is True

    def test_stage2_markdown_code_fence_cleaning(self):
        raw = """```json
        {
            "title": "Markdown Title",
            "confidence": 0.88,
            "tags": ["agents"],
            "is_valid": true
        }
        ```"""
        res = parse_and_validate(raw, SampleModel)
        assert res.title == "Markdown Title"
        assert res.confidence == 0.88

    def test_stage2_markdown_fence_without_json_tag(self):
        raw = """```
        {
            "title": "Plain Fence Title",
            "confidence": 0.80,
            "tags": ["rag"],
            "is_valid": false
        }
        ```"""
        res = parse_and_validate(raw, SampleModel)
        assert res.title == "Plain Fence Title"
        assert res.is_valid is False

    def test_stage3_conversational_preamble_and_trailer_extraction(self):
        raw = """Here is the extracted analysis you requested:
        
        {
            "title": "Surrounded JSON",
            "confidence": 0.92,
            "tags": ["benchmark"],
            "is_valid": true,
            "optional_note": "Note containing {nested braces} and \\"quotes\\""
        }
        
        I hope this helps your research workflow! Please let me know if you need more."""
        res = parse_and_validate(raw, SampleModel)
        assert res.title == "Surrounded JSON"
        assert res.confidence == 0.92
        assert "nested braces" in (res.optional_note or "")

    def test_stage4_syntax_auto_repair_trailing_commas_and_python_literals(self):
        raw = """{
            "title": "Repaired JSON",
            "confidence": 0.77,
            "tags": ["test1", "test2", ],
            "is_valid": True,
            "optional_note": None,
        }"""
        res = parse_and_validate(raw, SampleModel)
        assert res.title == "Repaired JSON"
        assert res.is_valid is True
        assert res.optional_note is None
        assert res.tags == ["test1", "test2"]

    def test_unrecoverable_malformed_json_raises_structured_output_error(self):
        raw = "This is completely unparseable text without any JSON structure at all."
        with pytest.raises(StructuredOutputError) as exc_info:
            parse_and_validate(raw, SampleModel)
        assert "SampleModel" in str(exc_info.value)
        assert exc_info.value.raw_text == raw

    def test_schema_validation_error_raises_structured_output_error(self):
        # confidence is expected to be float, tags expected to be list
        raw = '{"title": "Bad Types", "confidence": "not_a_number", "tags": 12345, "is_valid": true}'
        with pytest.raises(StructuredOutputError) as exc_info:
            parse_and_validate(raw, SampleModel)
        assert exc_info.value.validation_errors is not None

    def test_to_gemini_schema_resolves_defs_and_nullables(self):
        schema = to_gemini_schema(SampleModel)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "title" in schema["properties"]
        assert "confidence" in schema["properties"]
        assert "$defs" not in schema
        assert schema["properties"]["title"]["type"] == "string"
        assert schema["properties"]["optional_note"].get("nullable") is True

    def test_to_openai_response_format(self):
        fmt = to_openai_response_format(SampleModel)
        assert fmt == {"type": "json_object"}


# =============================================================================
# 3. CANCELLATION MANAGER DISTRIBUTED & CONCURRENT TESTS
# =============================================================================

class TestCancellationManagerResilience:
    """Empirical tests for CancellationManager state propagation and Celery task revoking."""

    def test_cancellation_lifecycle(self):
        cm = CancellationManager()
        cm.clear_cancellation("proj_001")
        assert not cm.is_cancelled("proj_001")

        cm.cancel_project("proj_001")
        assert cm.is_cancelled("proj_001")

        with pytest.raises(TaskCancelledException) as exc_info:
            cm.check_and_raise_if_cancelled("proj_001")
        assert exc_info.value.project_id == "proj_001"

        cm.clear_cancellation("proj_001")
        assert not cm.is_cancelled("proj_001")
        # Should not raise now
        cm.check_and_raise_if_cancelled("proj_001")

    def test_empty_or_none_project_id_safety(self):
        cm = CancellationManager()
        assert not cm.is_cancelled("")
        assert not cm.is_cancelled(None)  # type: ignore

    def test_task_registration_and_revocation(self):
        cm = CancellationManager()
        project_id = "proj_celery_test"
        task_id = "celery_task_abc_123"

        cm.register_task(project_id, task_id)
        assert cm.get_task_id(project_id) == task_id

        # Mock Celery app
        mock_celery = MagicMock()
        success = cm.revoke_task(project_id, mock_celery)
        assert success is True
        mock_celery.control.revoke.assert_called_once_with(task_id, terminate=True, signal="SIGTERM")

        cm.unregister_task(project_id)
        assert cm.get_task_id(project_id) is None

    def test_multi_threaded_cancellation_race(self):
        cm = CancellationManager()
        project_id = "proj_race_test"
        cm.clear_cancellation(project_id)

        stop_event = concurrent.futures.thread.threading.Event()
        cancelled_detected = []

        def worker_loop():
            while not stop_event.is_set():
                if cm.is_cancelled(project_id):
                    cancelled_detected.append(True)
                    break
                time.sleep(0.005)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker_loop) for _ in range(4)]
            time.sleep(0.02)
            # Signal cancel
            cm.cancel_project(project_id)
            time.sleep(0.05)
            stop_event.set()
            concurrent.futures.wait(futures)

        assert len(cancelled_detected) == 4, "All worker threads must observe the cancellation signal"


# =============================================================================
# 4. BLACKBOARD STATE TRANSITIONS, HANDOFFS & DATABASE SYNC
# =============================================================================

class TestBlackboardAndAgentHandoffs:
    """Empirical tests for Blackboard working memory, StateGraph sync, and DB persistence."""

    def test_blackboard_goal_stack_lifecycle(self):
        bb = WorkingMemoryBlackboard(project_id="p_goal_test", title="Goal Test")
        
        goal = bb.push_goal(
            goal_id="g1",
            name="Discovery Goal",
            description="Discover papers",
            target_agent=AgentType.DISCOVERY,
            priority=1,
        )
        assert len(bb.goal_stack) == 1
        assert bb.goal_stack[0]["status"] == GoalStatus.PENDING

        # Update to running
        res = bb.update_goal_status("g1", GoalStatus.IN_PROGRESS)
        assert res is True
        assert bb.goal_stack[0]["status"] == GoalStatus.IN_PROGRESS
        assert bb.goal_stack[0]["completed_at"] is None

        # Complete goal
        res = bb.update_goal_status("g1", GoalStatus.COMPLETED)
        assert res is True
        assert bb.goal_stack[0]["status"] == GoalStatus.COMPLETED
        assert bb.goal_stack[0]["completed_at"] is not None

    def test_blackboard_type_coercion_and_artifact_setting(self):
        bb = WorkingMemoryBlackboard(project_id="p_artifact_test")

        # 1. Parsed Papers
        bb.add_parsed_paper({
            "paper_id": "p1",
            "title": "Quantum Agent Routing",
            "authors": ["Alice", "Bob"],
            "year": 2025,
            "is_full_text": True,
        })
        assert "p1" in bb.parsed_papers
        assert bb.parsed_papers["p1"]["title"] == "Quantum Agent Routing"

        # 2. Evidence Matrix
        bb.set_evidence_matrix([{
            "paper_id": "p1",
            "title": "Quantum Agent Routing",
            "authors": ["Alice"],
            "year": 2025,
            "methodology": "Quantum RL",
            "benchmark_dataset": "AgentBench",
            "primary_metric": "Accuracy 94%",
            "primary_limitation": "Hardware requirements",
            "is_full_text": True,
        }])
        assert len(bb.evidence_matrix) == 1
        assert isinstance(bb.evidence_matrix[0], EvidenceMatrixRow)

        # 3. Thematic Synthesis & Debates & Gaps
        bb.set_thematic_synthesis(
            executive_summary="Summary text...",
            sections=[{
                "theme_id": "sec_1",
                "title": "Section 1",
                "synthesis_prose": "Content 1",
                "key_takeaways": ["Finding A"],
                "cited_paper_ids": ["p1"],
            }],
            debates=[{
                "topic": "Centralized vs Decentralized",
                "perspective_a": "Centralized Coordination",
                "perspective_b": "Decentralized Scalability",
                "critical_evaluation": "Trade-offs depend on network conditions.",
            }],
            gaps=[{
                "gap_id": "gap_1",
                "description": "Lack of real-time multi-agent benchmarks",
                "importance": "high",
                "recommended_methodology": "Empirical evaluation",
                "grounding_paper_ids": ["p1"],
            }],
        )
        assert len(bb.draft_thematic_sections) == 1
        assert len(bb.debates) == 1
        assert len(bb.research_gaps) == 1

        # 4. Critic Evaluation
        bb.add_critic_evaluation({
            "overall_score": 82.5,
            "dimension_scores": [
                {"dimension": "rigor", "score": 85.0, "justification": "High rigor"},
                {"dimension": "breadth", "score": 80.0, "justification": "Good coverage"},
            ],
            "strengths": ["Strong evidence matrix"],
            "weaknesses": ["None"],
            "refinement_guidance": [],
            "should_refine": False,
        })
        assert len(bb.critic_feedback) == 1
        assert bb.critic_feedback[0].overall_score == 82.5

        # 5. Citation Audit
        bb.set_audit_report({
            "audit_passed": True,
            "precision_score": 95.0,
            "total_citations_checked": 10,
            "entailed_count": 9,
            "contradiction_count": 0,
            "unverifiable_count": 1,
            "hallucination_rate": 0.0,
            "claim_verifications": [],
        })
        assert bb.audit_report is not None
        assert bb.audit_report.precision_score == 95.0

    def test_state_roundtrip_langgraph_to_blackboard(self):
        bb = WorkingMemoryBlackboard(project_id="p_roundtrip", title="Roundtrip Test")
        bb.add_parsed_paper({"paper_id": "p_rt", "title": "RT Paper", "is_full_text": True})
        bb.set_thematic_synthesis(
            executive_summary="Exec summary",
            sections=[ThematicSection(
                theme_id="s1",
                title="T1",
                synthesis_prose="C1",
                key_takeaways=[],
                cited_paper_ids=["p_rt"],
            )],
            debates=[],
            gaps=[],
        )

        agent_state = bb.to_agent_state()
        assert agent_state["project_id"] == "p_roundtrip"
        assert len(agent_state["parsed_papers"]) == 1
        assert len(agent_state["draft_thematic_sections"]) == 1

        # Ingest state into a fresh blackboard
        bb_new = WorkingMemoryBlackboard(project_id="p_roundtrip_2")
        bb_new.update_from_agent_state(agent_state)

        assert "p_rt" in bb_new.parsed_papers
        assert bb_new.executive_summary == "Exec summary"
        assert len(bb_new.draft_thematic_sections) == 1
        assert bb_new.draft_thematic_sections[0].title == "T1"

    def test_assemble_research_report_pydantic_deliverable(self):
        bb = WorkingMemoryBlackboard(
            project_id="p_deliverable",
            title="Comprehensive Literature Review",
            research_question="How to build robust multi-agent systems?",
        )
        bb.add_parsed_paper({
            "paper_id": "p_del_1",
            "title": "Agent Architectures",
            "authors": ["Dr. Smith"],
            "year": 2024,
            "is_full_text": True,
        })
        bb.set_thematic_synthesis(
            executive_summary="Key insights on autonomous agents.",
            sections=[ThematicSection(
                theme_id="s1",
                title="Section One",
                synthesis_prose="Details...",
                key_takeaways=["Agent scaling works"],
                cited_paper_ids=["p_del_1"],
            )],
            debates=[],
            gaps=[],
        )

        report = bb.assemble_research_report()
        assert isinstance(report, ResearchReport)
        assert report.metadata.project_id == "p_deliverable"
        assert report.executive_summary == "Key insights on autonomous agents."
        assert len(report.thematic_sections) == 1
        assert report.thematic_sections[0].title == "Section One"
        assert len(report.bibliography) == 1
        assert report.bibliography[0].paper_id == "p_del_1"

    def test_database_sync_and_load_sqlite(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()

        try:
            # Create project record
            proj = ResearchProject(
                id="proj_db_test",
                user_id="user_test",
                title="DB Sync Test",
                research_question="Testing DB persistence",
                status="running",
            )
            db.add(proj)
            db.commit()

            bb = WorkingMemoryBlackboard(
                project_id="proj_db_test",
                title="DB Sync Test",
                research_question="Testing DB persistence",
            )
            bb.add_parsed_paper({
                "paper_id": "p_db_1",
                "arxiv_id": "2401.00001",
                "title": "DB Paper 1",
                "authors": ["Alice"],
                "year": 2024,
                "is_full_text": True,
            })
            bb.set_evidence_matrix([EvidenceMatrixRow(
                paper_id="p_db_1",
                title="DB Paper 1",
                authors=["Alice"],
                year=2024,
                methodology="Empirical",
                benchmark_dataset="DataX",
                primary_metric="F1: 0.91",
                primary_limitation="Memory",
                is_full_text=True,
            )])
            bb.set_thematic_synthesis(
                executive_summary="DB Summary",
                sections=[ThematicSection(
                    theme_id="s1",
                    title="Section 1",
                    synthesis_prose="Content 1",
                    key_takeaways=[],
                    cited_paper_ids=["p_db_1"],
                )],
                debates=[],
                gaps=[ResearchGapItem(
                    gap_id="gap_db_1",
                    description="Open gap",
                    importance="high",
                    recommended_methodology="Survey",
                    grounding_paper_ids=["p_db_1"],
                )],
            )

            # Sync to SQLite DB
            bb.sync_to_database(db)

            # Load into fresh blackboard
            bb_loaded = WorkingMemoryBlackboard(project_id="proj_db_test")
            loaded = bb_loaded.load_from_database(db, "proj_db_test")
            assert loaded is True
            assert bb_loaded.title == "DB Sync Test"
            assert bb_loaded.executive_summary == "DB Summary"
            assert len(bb_loaded.evidence_matrix) == 1
            assert bb_loaded.evidence_matrix[0].paper_id == "p_db_1"
            assert len(bb_loaded.research_gaps) == 1
            assert bb_loaded.research_gaps[0].gap_id == "gap_db_1"

        finally:
            db.close()

    def test_supervisor_refinement_loop_bounded_routing(self):
        # Case 1: High quality score (>= 75.0) and should_refine=False -> proceed to auditor
        state_pass = {
            "current_critic_score": 85.0,
            "should_refine": False,
            "iteration_count": 0,
            "max_iterations": 2,
        }
        assert should_refine_or_finalize(state_pass) == "auditor"

        # Case 2: Low quality score (< 75.0) -> loops back to synthesizer and increments iteration
        state_loop1 = {
            "current_critic_score": 60.0,
            "should_refine": True,
            "iteration_count": 0,
            "max_iterations": 2,
        }
        dest1 = should_refine_or_finalize(state_loop1)
        assert dest1 == "synthesizer"
        assert state_loop1["iteration_count"] == 1

        # Case 3: Still low score on iteration 1 -> loops back to synthesizer
        state_loop2 = {
            "current_critic_score": 65.0,
            "should_refine": True,
            "iteration_count": 1,
            "max_iterations": 2,
        }
        dest2 = should_refine_or_finalize(state_loop2)
        assert dest2 == "synthesizer"
        assert state_loop2["iteration_count"] == 2

        # Case 4: Max iterations reached (2 >= 2) -> hard terminates and routes to auditor
        state_max = {
            "current_critic_score": 65.0,
            "should_refine": True,
            "iteration_count": 2,
            "max_iterations": 2,
        }
        dest3 = should_refine_or_finalize(state_max)
        assert dest3 == "auditor", "Refinement loop must enforce hard bounded termination at max_iterations"
