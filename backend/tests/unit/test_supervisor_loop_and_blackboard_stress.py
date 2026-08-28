"""
Supervisor Refinement Loop, State Routing, and Working Memory Blackboard Stress Test Suite.
Validates Bounded LangGraph Supervisor StateGraph, conditional refinement loop dynamics (score < 75.0, max 2 iterations),
Working Memory Blackboard synchronization, DB persistence, and specialist agent edge cases.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.agents.blackboard import WorkingMemoryBlackboard
from backend.agents.core.auditor import DeterministicCitationAuditorAgent
from backend.agents.core.critic import AdversarialCriticAgent
from backend.agents.core.discovery import AutonomousLiteratureExplorer
from backend.agents.core.ingestion import FullTextIngestionSpecialist
from backend.agents.core.matrix_builder import EvidenceMatrixBuilder
from backend.agents.core.supervisor import (
    AutonomousSupervisorAgent,
    build_scholar_agent_graph,
    finalizer_node,
    should_refine_or_finalize,
)
from backend.agents.core.synthesizer import SectionAwareContextPacker, ThematicSynthesizerAgent
from backend.agents.orchestrator import ScholarAgentOrchestrator
from backend.agents.schemas import (
    AcademicPaperCandidate,
    BibliographyItem,
    CitationAuditReport,
    ConflictingDebate,
    CriticDimensionScore,
    CriticEvaluation,
    EvidenceMatrixRow,
    MethodologyDistribution,
    NLIVerdict,
    PropositionVerification,
    ReportMetadata,
    ReportStatus,
    ResearchGapItem,
    ResearchReport,
    SearchQueryPlan,
    ThematicSection,
    ThematicSynthesisDraft,
)
from backend.agents.state import (
    AgentMessage,
    AgentState,
    AgentType,
    GoalItem,
    GoalStatus,
    ParsedPaperData,
    TelemetryEvent,
    create_initial_agent_state,
)
from backend.agents.tools.fact_checker import FactCheckerEngine
from backend.agents.tools.oa_resolver import OAResolutionResult
from backend.models.database import (
    Base,
    EvidenceMatrixEntry,
    PaperCache,
    ResearchGapModel,
    ResearchProject,
    ResearchReportModel,
)


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


# =============================================================================
# Category 1: Conditional Refinement Loop & Bounded Termination Dynamics
# =============================================================================

class TestRefinementLoopBoundariesAndTermination:
    """Stress tests for conditional routing, score thresholds, and iteration hard-caps."""

    def test_multi_loop_refinement_cycle_and_eventual_pass(self):
        """
        Verify multi-step refinement:
        Iter 0 (score 60.0 < 75) -> Synthesizer, iter becomes 1.
        Iter 1 (score 70.0 < 75) -> Synthesizer, iter becomes 2.
        Iter 2 (score 82.0 >= 75) -> Auditor (passed).
        """
        state = create_initial_agent_state(project_id="test_proj", max_iterations=2)

        # Iteration 0: Low score
        state["current_critic_score"] = 60.0
        state["should_refine"] = True
        state["iteration_count"] = 0
        route_0 = should_refine_or_finalize(state)
        assert route_0 == "synthesizer"
        assert state["iteration_count"] == 1
        assert state["iteration"] == 1

        # Iteration 1: Still below threshold
        state["current_critic_score"] = 70.0
        state["should_refine"] = True
        route_1 = should_refine_or_finalize(state)
        assert route_1 == "synthesizer"
        assert state["iteration_count"] == 2
        assert state["iteration"] == 2

        # Iteration 2: Passes quality threshold
        state["current_critic_score"] = 82.0
        state["should_refine"] = False
        route_2 = should_refine_or_finalize(state)
        assert route_2 == "auditor"
        assert state["iteration_count"] == 2

    def test_hard_cap_termination_on_persistent_poor_quality(self):
        """
        Verify that when critic score NEVER reaches 75.0, the loop MUST terminate
        deterministically at max_iterations=2 and proceed to auditor without infinite looping.
        """
        state = create_initial_agent_state(project_id="test_proj", max_iterations=2)

        # Iteration 0 (score 45.0) -> loop back (iter -> 1)
        state["current_critic_score"] = 45.0
        state["should_refine"] = True
        state["iteration_count"] = 0
        assert should_refine_or_finalize(state) == "synthesizer"
        assert state["iteration_count"] == 1

        # Iteration 1 (score 50.0) -> loop back (iter -> 2)
        state["current_critic_score"] = 50.0
        state["should_refine"] = True
        assert should_refine_or_finalize(state) == "synthesizer"
        assert state["iteration_count"] == 2

        # Iteration 2 (score 55.0 < 75.0) -> MUST ROUTE TO AUDITOR due to hard cap max_iterations=2
        state["current_critic_score"] = 55.0
        state["should_refine"] = True
        route = should_refine_or_finalize(state)
        assert route == "auditor", "Pipeline must terminate at max_iterations even if score < 75"

    def test_exact_boundary_threshold_74_9_vs_75_0(self):
        """Verify strict precision at 75.0 boundary."""
        # 74.9 -> refine
        state_fail = create_initial_agent_state(project_id="p1", max_iterations=2)
        state_fail["current_critic_score"] = 74.9
        state_fail["should_refine"] = True
        state_fail["iteration_count"] = 0
        assert should_refine_or_finalize(state_fail) == "synthesizer"

        # 75.0 -> pass
        state_pass = create_initial_agent_state(project_id="p2", max_iterations=2)
        state_pass["current_critic_score"] = 75.0
        state_pass["should_refine"] = False
        state_pass["iteration_count"] = 0
        assert should_refine_or_finalize(state_pass) == "auditor"

        # 75.1 -> pass
        state_pass2 = create_initial_agent_state(project_id="p3", max_iterations=2)
        state_pass2["current_critic_score"] = 75.1
        state_pass2["should_refine"] = False
        state_pass2["iteration_count"] = 0
        assert should_refine_or_finalize(state_pass2) == "auditor"

    def test_max_iterations_bounded_range_clamp(self):
        """Verify max_iterations is clamped to [1, 2] in factory methods and blackboard."""
        s0 = create_initial_agent_state(project_id="p0", max_iterations=0)
        assert s0["max_iterations"] == 1

        s5 = create_initial_agent_state(project_id="p5", max_iterations=5)
        assert s5["max_iterations"] == 2

        bb0 = WorkingMemoryBlackboard(project_id="bb0", max_iterations=0)
        assert bb0.max_iterations == 1

        bb10 = WorkingMemoryBlackboard(project_id="bb10", max_iterations=10)
        assert bb10.max_iterations == 2

    def test_critic_agent_scoring_and_refinement_flags(self):
        """Verify CriticAgent properly scores and sets should_refine."""
        critic = AdversarialCriticAgent(llm_client=None)

        # 1. Fallback with poor sections -> score 65.0 < 75.0, should_refine=True
        eval_poor = critic._fallback_evaluation(
            thematic_sections=[{"title": "Weak Section", "synthesis_prose": "No citations here."}],
            iteration=0,
        )
        assert eval_poor.overall_score == 65.0
        assert eval_poor.should_refine is True
        assert len(eval_poor.weaknesses) > 0
        assert len(eval_poor.refinement_guidance) > 0

        # 2. Fallback with anchors and multiple sections on iter 1 -> score 78.0 >= 75.0, should_refine=False
        eval_good = critic._fallback_evaluation(
            thematic_sections=[
                {"title": "Sec 1", "synthesis_prose": "Methods in [ref_1#sec_1]."},
                {"title": "Sec 2", "synthesis_prose": "Results in [ref_2#sec_2]."},
            ],
            iteration=1,
        )
        assert eval_good.overall_score == 78.0
        assert eval_good.should_refine is False
        assert eval_poor.should_refine is True


# =============================================================================
# Category 2: Working Memory Blackboard Synchronization & Concurrency
# =============================================================================

class TestWorkingMemoryBlackboardSynchronization:
    """Stress tests for in-flight working memory blackboard, state conversion, and DB persistence."""

    def test_blackboard_full_state_roundtrip_fidelity(self):
        """Verify 100% roundtrip fidelity between WorkingMemoryBlackboard and AgentState TypedDict."""
        bb = WorkingMemoryBlackboard(
            project_id="proj_sync_test",
            user_id="user_123",
            title="Transformer Scaling Laws",
            research_question="What are the empirical compute scaling bounds?",
            max_papers=10,
            max_iterations=2,
        )

        # 1. Populate Goals
        bb.push_goal(
            goal_id="g1",
            name="Discovery",
            description="Find papers",
            target_agent=AgentType.DISCOVERY,
            priority=1,
        )
        bb.update_goal_status("g1", GoalStatus.COMPLETED)

        # 2. Populate Parsed Papers
        bb.add_parsed_paper({
            "paper_id": "ref_1",
            "doi": "10.1000/182",
            "title": "Scaling Laws for Neural Language Models",
            "authors": ["Kaplan et al."],
            "year": 2020,
            "abstract": "We observe power-law scaling relationships.",
            "is_full_text": True,
            "sections": [{"heading": "Methods", "content": "Loss fits power law.", "section_type": "methodology"}],
            "tables": [{"col1": "val1"}],
            "equations": ["L(N) = (N_c/N)^alpha_N"],
        })

        # 3. Populate Matrix
        bb.set_evidence_matrix([
            EvidenceMatrixRow(
                paper_id="ref_1",
                title="Scaling Laws for Neural Language Models",
                authors=["Kaplan et al."],
                year=2020,
                methodology="Empirical parameter and dataset power-law curve fitting",
                benchmark_dataset="WebText, OpenWebText",
                primary_metric="Cross-Entropy Loss (nats/char)",
                primary_limitation="Compute-optimal frontier later revised by Chinchilla",
                is_full_text=True,
            )
        ])

        # 4. Populate Synthesis
        bb.set_thematic_synthesis(
            executive_summary="Executive synthesis on transformer scaling laws.",
            sections=[
                ThematicSection(
                    theme_id="theme_power_law",
                    title="Empirical Power-Law Scaling",
                    synthesis_prose="Kaplan et al. established cross-entropy loss scales smoothly [ref_1#sec_1].",
                    key_takeaways=["Power law scaling holds across 6 orders of magnitude."],
                    cited_paper_ids=["ref_1"],
                )
            ],
            debates=[
                ConflictingDebate(
                    topic="Optimal Token vs Parameter Allocation",
                    perspective_a="Kaplan argues for scaling model parameters faster than tokens.",
                    perspective_b="Hoffmann (Chinchilla) demonstrates equal scaling of tokens and parameters.",
                    critical_evaluation="Kaplan used a constant learning rate schedule which under-trained larger models.",
                )
            ],
            gaps=[
                ResearchGapItem(
                    gap_id="GAP-SCALE-01",
                    description="Sub-quadratic attention scaling bounds under infinite sequence lengths.",
                    importance="high",
                    recommended_methodology="Empirically evaluate linear RNN vs FlashAttention-3 on 1M token contexts.",
                    grounding_paper_ids=["ref_1"],
                )
            ],
            methodology_overview=MethodologyDistribution(
                distribution={"Empirical Analysis": 1},
                dominant_approach="Empirical Curve Fitting",
                trend_description="Progressive focus on compute-optimal data scaling.",
            ),
        )

        # 5. Populate Critic & Audit
        bb.add_critic_evaluation(
            CriticEvaluation(
                overall_score=88.5,
                dimension_scores=[
                    CriticDimensionScore(dimension="Empirical Rigor", score=90.0, justification="Strong grounding"),
                ],
                strengths=["Excellent debate analysis"],
                weaknesses=[],
                refinement_guidance=[],
                should_refine=False,
            )
        )
        bb.set_audit_report(
            CitationAuditReport(
                precision_score=95.0,
                total_propositions=2,
                entailed_count=2,
                neutral_count=0,
                contradiction_count=0,
                hallucinated_anchors=[],
                verifications=[],
                audit_passed=True,
            )
        )

        # 6. Record Telemetry
        bb.record_telemetry(agent="critic", action="review", duration_ms=450.0, input_tokens=1000, output_tokens=300)

        # Export to AgentState
        state = bb.to_agent_state()

        # Ingest into a fresh blackboard
        bb2 = WorkingMemoryBlackboard(project_id="proj_sync_test")
        bb2.update_from_agent_state(state)

        # Verify Round-trip
        assert bb2.project_id == "proj_sync_test"
        assert len(bb2.goal_stack) == 1
        assert bb2.goal_stack[0]["status"] == GoalStatus.COMPLETED
        assert len(bb2.parsed_papers) == 1
        assert "ref_1" in bb2.parsed_papers
        assert len(bb2.evidence_matrix) == 1
        assert bb2.evidence_matrix[0].paper_id == "ref_1"
        assert bb2.evidence_matrix[0].benchmark_dataset == "WebText, OpenWebText"
        assert len(bb2.draft_thematic_sections) == 1
        assert bb2.draft_thematic_sections[0].theme_id == "theme_power_law"
        assert len(bb2.debates) == 1
        assert bb2.debates[0].topic == "Optimal Token vs Parameter Allocation"
        assert len(bb2.research_gaps) == 1
        assert bb2.research_gaps[0].gap_id == "GAP-SCALE-01"
        assert len(bb2.critic_feedback) == 1
        assert bb2.critic_feedback[0].overall_score == 88.5
        assert bb2.audit_report is not None
        assert bb2.audit_report.precision_score == 95.0
        assert len(bb2.telemetry_events) == 1

    def test_blackboard_relational_database_sync_and_load(self, in_memory_db):
        """Verify blackboard sync_to_database and load_from_database operations."""
        project_id = "proj_db_test_001"
        # Seed parent project
        proj = ResearchProject(
            id=project_id,
            user_id="user_db",
            title="Database Sync Test",
            research_question="Does DB sync preserve all fields?",
        )
        in_memory_db.add(proj)
        in_memory_db.commit()

        bb = WorkingMemoryBlackboard(
            project_id=project_id,
            user_id="user_db",
            title="Database Sync Test",
            research_question="Does DB sync preserve all fields?",
        )

        bb.add_parsed_paper({
            "paper_id": "ref_db_1",
            "doi": "10.1145/test_doi",
            "title": "Database Grounding Paper",
            "authors": ["Author A", "Author B"],
            "year": 2023,
            "abstract": "Abstract text here.",
            "is_full_text": True,
            "full_text_markdown": "# Title\n## Methods\nMethods text.",
            "sections": [{"heading": "Methods", "content": "Methods text.", "section_type": "methodology"}],
        })

        bb.set_evidence_matrix([
            EvidenceMatrixRow(
                paper_id="ref_db_1",
                title="Database Grounding Paper",
                authors=["Author A"],
                year=2023,
                methodology="Relational ORM Mapping",
                benchmark_dataset="TPC-H",
                primary_metric="QPS 15000",
                primary_limitation="Lock contention under high write volume",
                is_full_text=True,
            )
        ])

        bb.set_thematic_synthesis(
            executive_summary="DB test executive summary.",
            sections=[
                ThematicSection(
                    theme_id="sec_1",
                    title="Relational Mapping",
                    synthesis_prose="Prose with [ref_db_1#sec_1].",
                    key_takeaways=["Point 1"],
                    cited_paper_ids=["ref_db_1"],
                )
            ],
            debates=[
                ConflictingDebate(
                    topic="ORM vs Raw SQL",
                    perspective_a="ORM provides type safety.",
                    perspective_b="Raw SQL gives query control.",
                    critical_evaluation="Hybrid approaches optimize both.",
                )
            ],
            gaps=[
                ResearchGapItem(
                    gap_id="GAP-DB-01",
                    description="Zero-copy serialization for high-throughput pipelines.",
                    importance="high",
                    recommended_methodology="Implement Apache Arrow integration.",
                    grounding_paper_ids=["ref_db_1"],
                )
            ],
            methodology_overview=MethodologyDistribution(
                distribution={"Benchmark": 1},
                dominant_approach="Empirical Benchmarking",
                trend_description="Arrow-based acceleration.",
            ),
        )

        bb.add_critic_evaluation(
            CriticEvaluation(
                overall_score=80.0,
                dimension_scores=[],
                strengths=["Good DB coverage"],
                weaknesses=[],
                refinement_guidance=[],
                should_refine=False,
            )
        )

        # Sync to DB
        bb.sync_to_database(in_memory_db)

        # Verify raw tables in DB
        report_row = in_memory_db.query(ResearchReportModel).filter(ResearchReportModel.project_id == project_id).first()
        assert report_row is not None
        assert report_row.title == "Database Sync Test"
        assert report_row.quality_score == 80.0
        assert len(report_row.thematic_sections) == 1

        matrix_rows = in_memory_db.query(EvidenceMatrixEntry).filter(EvidenceMatrixEntry.project_id == project_id).all()
        assert len(matrix_rows) == 1
        assert matrix_rows[0].paper_id == "ref_db_1"
        assert matrix_rows[0].primary_metric == "QPS 15000"

        gap_rows = in_memory_db.query(ResearchGapModel).filter(ResearchGapModel.project_id == project_id).all()
        assert len(gap_rows) == 1
        assert gap_rows[0].gap_id == "GAP-DB-01"

        cache_row = in_memory_db.query(PaperCache).filter(PaperCache.doi == "10.1145/test_doi").first()
        assert cache_row is not None
        assert cache_row.is_full_text is True

        # Load into new blackboard
        bb_loaded = WorkingMemoryBlackboard(project_id=project_id)
        load_success = bb_loaded.load_from_database(in_memory_db, project_id)
        assert load_success is True
        assert bb_loaded.title == "Database Sync Test"
        assert len(bb_loaded.evidence_matrix) == 1
        assert len(bb_loaded.research_gaps) == 1
        assert bb_loaded.research_gaps[0].gap_id == "GAP-DB-01"

    @pytest.mark.asyncio
    async def test_blackboard_concurrent_observer_notifications(self):
        """Verify thread-safety and observer notifications under concurrent state mutations."""
        bb = WorkingMemoryBlackboard(project_id="proj_concurrent")
        events_received: list[dict[str, Any]] = []

        def observer(event_type: str, payload: dict[str, Any]):
            events_received.append({"event": event_type, "payload": payload})

        bb.subscribe(observer)

        async def worker_1():
            for i in range(10):
                bb.push_goal(f"g_{i}", f"Goal {i}", "Desc", AgentType.DISCOVERY)
                await asyncio.sleep(0.001)

        async def worker_2():
            for i in range(10):
                bb.add_parsed_paper({"paper_id": f"p_{i}", "title": f"Paper {i}", "is_full_text": True})
                await asyncio.sleep(0.001)

        await asyncio.gather(worker_1(), worker_2())

        assert len(bb.goal_stack) == 10
        assert len(bb.parsed_papers) == 10
        assert len(events_received) == 20


# =============================================================================
# Category 3: Autonomous Supervisor StateGraph & Specialist Agents Edge Cases
# =============================================================================

class TestSpecialistAgentsAndSupervisorEdgeCases:
    """Stress tests for specialist agents, failure modes, and DAG edge handling."""

    @pytest.mark.asyncio
    async def test_empty_candidate_pool_graceful_handling(self):
        """Verify that zero candidates from discovery degrades gracefully in synthesizer and matrix builder."""
        state = create_initial_agent_state(project_id="proj_empty", research_question="Unknown topic")
        state["parsed_papers"] = {}
        state["papers"] = []
        state["evidence_matrix"] = []

        # Matrix builder on empty
        mb = EvidenceMatrixBuilder(llm_client=None)
        state_mb = await mb.run(state)
        assert len(state_mb.get("evidence_matrix", [])) == 0

        # Synthesizer on empty
        synth = ThematicSynthesizerAgent(llm_client=None)
        state_synth = await synth.run(state_mb)
        assert "Cannot synthesize" in str(state_synth.get("errors", []))

    def test_section_aware_context_packer_priority_and_truncation(self):
        """Verify context packing prioritizes results/methods over introduction and respects token limit."""
        papers = [
            {
                "paper_id": "ref_1",
                "title": "Attention Paper",
                "year": 2017,
                "sections": [
                    {"section_type": "introduction", "heading": "Intro", "content": "Introduction text."},
                    {"section_type": "results", "heading": "Main Results", "content": "BLEU score 28.4 (+2.0 over baseline)."},
                    {"section_type": "methodology", "heading": "Multi-Head Attention", "content": "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V."},
                    {"section_type": "limitations", "heading": "Limitations", "content": "O(N^2) memory footprint."},
                ],
            }
        ]

        packed = SectionAwareContextPacker.pack_corpus(papers, max_chars=10000)
        assert "TYPE=RESULTS" in packed
        assert "TYPE=METHODOLOGY" in packed
        assert "TYPE=LIMITATIONS" in packed

        # Results should appear before Introduction in packed string due to priority sorting
        idx_results = packed.find("TYPE=RESULTS")
        idx_intro = packed.find("TYPE=INTRODUCTION")
        assert idx_results < idx_intro, "Results section must appear before Introduction"

    @pytest.mark.asyncio
    async def test_auditor_strips_hallucinated_anchors_and_verifies_grounding(self):
        """Verify auditor flags fake anchors, cleans prose, and compiles valid bibliography."""
        auditor = DeterministicCitationAuditorAgent(llm_client=None)

        state = create_initial_agent_state(project_id="proj_auditor_adv")
        state["parsed_papers"] = {
            "ref_1": {
                "paper_id": "ref_1",
                "title": "True Reference",
                "sections": [{"heading": "Methods", "content": "Real methods.", "anchor_tag": "[ref_1#sec_methods]"}],
            }
        }

        # Draft with 1 real anchor and 1 hallucinated anchor
        state["synthesis_draft"] = {
            "executive_summary": "Summary with [ref_1#sec_methods].",
            "thematic_sections": [
                {
                    "theme_id": "t1",
                    "title": "Theme 1",
                    "synthesis_prose": "Real claim [ref_1#sec_methods]. Fake claim [ref_ghost#sec_99].",
                    "key_takeaways": [],
                    "cited_paper_ids": ["ref_1", "ref_ghost"],
                }
            ],
            "conflicting_findings_and_debates": [],
            "actionable_research_gaps": [],
            "methodology_overview": {
                "distribution": {"Empirical": 1},
                "dominant_approach": "Empirical",
                "trend_description": "Scaling",
            },
        }

        new_state = await auditor.run(state)
        assert new_state["audit_report"] is not None
        assert "ref_ghost#sec_99" in new_state["audit_report"].hallucinated_anchors

        # Check prose cleaning
        cleaned_prose = new_state["thematic_sections"][0]["synthesis_prose"]
        assert "[ref_ghost#sec_99]" not in cleaned_prose
        assert "[ref_1#sec_methods]" in cleaned_prose

        # Bibliography should only include ref_1
        bib = new_state["bibliography"]
        assert len(bib) == 1
        assert bib[0]["paper_id"] == "ref_1"
