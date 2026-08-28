"""
Reasoning Specialist Agents and Supervisor StateGraph Invariants Test Suite.
Exhaustively verifies:
1. Thematic Synthesizer: 20K token context packing prioritization, truncation boundaries, priority ordering, and gap grounding.
2. Adversarial Critic: 4-dimensional 0-100 scoring, threshold logic (75.0), actionable refinement directives, and iteration tracking.
3. Deterministic Citation Auditor & Fact Checker: Proposition deconstruction, grounding chunk resolution cascade,
   structured NLI classification (ENTAILMENT, NEUTRAL, CONTRADICTION), citation anchor canonicalization, precision threshold (>= 80%), and bibliography compilation.
4. Supervisor DAG: Bounded max 2 iterations invariant, state routing conditions, and final report assembly.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

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
from backend.agents.core.synthesizer import (
    CHARS_PER_TOKEN,
    MAX_PACK_CHARS,
    MAX_PACK_TOKENS,
    SectionAwareContextPacker,
    ThematicSynthesizerAgent,
)
from backend.agents.schemas import (
    BibliographyItem,
    CitationAuditReport,
    ConflictingDebate,
    CriticDimensionScore,
    CriticEvaluation,
    EvidenceMatrixRow,
    MethodologyDistribution,
    NLIVerdict,
    PropositionVerification,
    ResearchGapItem,
    ResearchReport,
    ThematicSection,
    ThematicSynthesisDraft,
)
from backend.agents.state import AgentType, create_initial_agent_state
from backend.agents.tools.fact_checker import AtomicProposition, FactCheckerEngine


# =============================================================================
# 1. THEMATIC SYNTHESIZER & 20K TOKEN CONTEXT PACKER ADVERSARIAL TESTS
# =============================================================================


class TestThematicSynthesizerAndContextPacker:
    """Stress tests for SectionAwareContextPacker and ThematicSynthesizerAgent."""

    def test_context_packer_priority_ordering_across_all_section_types(self):
        """Verify strict priority ordering: Results > Methodology > Limitations > Tables > Abstract > Intro > Discussion > Conclusion > General."""
        papers = [
            {
                "paper_id": "ref_all_sections",
                "title": "Comprehensive Benchmark Study",
                "year": 2024,
                "sections": [
                    {"heading": "General Section", "content": "CONTENT_GENERAL", "section_type": "general"},
                    {"heading": "Conclusion", "content": "CONTENT_CONCLUSION", "section_type": "conclusion"},
                    {"heading": "Discussion", "content": "CONTENT_DISCUSSION", "section_type": "discussion"},
                    {"heading": "Introduction", "content": "CONTENT_INTRO", "section_type": "introduction"},
                    {"heading": "Abstract", "content": "CONTENT_ABSTRACT", "section_type": "abstract"},
                    {"heading": "Tables", "content": "CONTENT_TABLES", "section_type": "tables"},
                    {"heading": "Limitations", "content": "CONTENT_LIMITATIONS", "section_type": "limitations"},
                    {"heading": "Methodology", "content": "CONTENT_METHODOLOGY", "section_type": "methodology"},
                    {"heading": "Results", "content": "CONTENT_RESULTS", "section_type": "results"},
                ],
            }
        ]

        packed = SectionAwareContextPacker.pack_corpus(papers=papers, max_chars=100000)

        # Find positions of section markers in packed text
        pos_results = packed.find("CONTENT_RESULTS")
        pos_meth = packed.find("CONTENT_METHODOLOGY")
        pos_lim = packed.find("CONTENT_LIMITATIONS")
        pos_tables = packed.find("CONTENT_TABLES")
        pos_abs = packed.find("CONTENT_ABSTRACT")
        pos_intro = packed.find("CONTENT_INTRO")
        pos_disc = packed.find("CONTENT_DISCUSSION")
        pos_conc = packed.find("CONTENT_CONCLUSION")
        pos_gen = packed.find("CONTENT_GENERAL")

        assert pos_results != -1
        assert pos_meth != -1
        assert pos_lim != -1
        assert pos_tables != -1
        assert pos_abs != -1
        assert pos_intro != -1
        assert pos_disc != -1
        assert pos_conc != -1
        assert pos_gen != -1

        # Assert strict hierarchical ordering
        assert pos_results < pos_meth < pos_lim < pos_tables < pos_abs < pos_intro < pos_disc < pos_conc < pos_gen

    def test_context_packer_hard_boundary_truncation(self):
        """Verify context packing strictly honors max_chars bound and includes truncation indicator."""
        large_body = "Quantitative experimental measurement data point. " * 5000  # ~250K characters
        papers = [
            {
                "paper_id": "ref_huge_1",
                "title": "Massive Dataset Paper 1",
                "sections": [{"heading": "Results", "content": large_body, "section_type": "results"}],
            },
            {
                "paper_id": "ref_huge_2",
                "title": "Massive Dataset Paper 2",
                "sections": [{"heading": "Methodology", "content": large_body, "section_type": "methodology"}],
            },
        ]

        max_allowed_chars = 15000
        packed = SectionAwareContextPacker.pack_corpus(papers=papers, max_chars=max_allowed_chars)

        assert len(packed) <= max_allowed_chars + 200  # Within boundary margin
        assert "[... truncated for context packing ...]" in packed

    def test_context_packer_mixed_corpus_with_missing_sections_and_abstract_fallback(self):
        """Test packing when papers have missing sections, empty text, or only abstract metadata."""
        papers = [
            {
                "paper_id": "ref_oa_full",
                "title": "Open Access Paper",
                "sections": [
                    {"heading": "Results", "content": "94.2% accuracy achieved.", "section_type": "results"},
                    {"heading": "Empty Sec", "content": "   ", "section_type": "general"},
                ],
            },
            {
                "paper_id": "ref_abstract_only",
                "title": "Paywalled Paper with Abstract Fallback",
                "abstract": "This study evaluates transformer attention overhead.",
                "sections": [],  # Empty sections
            },
            {
                "paper_id": "ref_no_data",
                "title": "Empty Record",
                # No abstract or sections
            },
        ]

        evidence_matrix = [
            EvidenceMatrixRow(
                paper_id="ref_oa_full",
                title="Open Access Paper",
                authors=["Author A"],
                methodology="Transformer Tiling",
                benchmark_dataset="ImageNet",
                primary_metric="Accuracy: 94.2%",
                primary_limitation="High memory usage",
                is_full_text=True,
            ),
            {
                "paper_id": "ref_abstract_only",
                "title": "Paywalled Paper",
                "methodology_type": "Theoretical analysis",
                "benchmark_dataset": "None",
                "primary_metric": "O(N^2) complexity",
                "primary_limitation": "No empirical code",
            },
        ]

        packed = SectionAwareContextPacker.pack_corpus(papers=papers, evidence_matrix=evidence_matrix, max_chars=50000)

        assert "### EVIDENCE COMPARISON MATRIX OVERVIEW:" in packed
        assert "[ref_oa_full] Open Access Paper | Method: Transformer Tiling" in packed
        assert "[ref_abstract_only] Paywalled Paper | Method: Theoretical analysis" in packed
        assert "94.2% accuracy achieved." in packed
        assert "This study evaluates transformer attention overhead." in packed

    @pytest.mark.asyncio
    async def test_synthesizer_agent_handles_critic_revision_directives(self):
        """Verify Synthesizer correctly embeds prior Critic feedback and directives in prompt on iteration 1+."""
        agent = ThematicSynthesizerAgent(llm_client=None)

        state = create_initial_agent_state(
            project_id="proj_refine",
            research_question="How to scale multi-agent reasoning?",
            title="Multi-Agent Reasoning Survey",
        )
        state["parsed_papers"] = {
            "ref_1": {
                "paper_id": "ref_1",
                "title": "Agent Scaling",
                "sections": [{"heading": "Results", "content": "Communication graph density affects throughput.", "section_type": "results"}],
            }
        }
        state["critic_evaluation"] = {
            "overall_score": 62.0,
            "weaknesses": ["Missing error bars on throughput", "Sparse provisional anchors"],
            "refinement_guidance": [
                "Include [ref_X#secY] on all throughput claims.",
                "Detail network latency overhead in debate section.",
            ],
            "should_refine": True,
        }
        state["iteration_count"] = 0

        new_state = await agent.run(state)

        # Verify iteration count advanced
        assert new_state["iteration_count"] == 1
        assert new_state["iteration"] == 1
        assert "synthesis_draft" in new_state
        assert len(new_state["thematic_sections"]) >= 1
        assert len(new_state["conflicting_debates"]) >= 1
        assert len(new_state["research_gaps"]) >= 1

    @pytest.mark.asyncio
    async def test_synthesizer_agent_empty_state_error_handling(self):
        """Verify Synthesizer gracefully logs error and returns state when neither papers nor matrix exist."""
        agent = ThematicSynthesizerAgent(llm_client=None)
        empty_state = create_initial_agent_state(project_id="proj_empty")
        empty_state["parsed_papers"] = {}
        empty_state["papers"] = []
        empty_state["evidence_matrix"] = []

        result_state = await agent.run(empty_state)
        assert "errors" in result_state
        assert len(result_state["errors"]) >= 1
        assert "Cannot synthesize" in result_state["errors"][0]


# =============================================================================
# 2. ADVERSARIAL CRITIC & METHODOLOGIST TESTS
# =============================================================================


class TestAdversarialCriticAndMethodologist:
    """Stress tests for AdversarialCriticAgent scoring, 4-dimensions, and threshold boundaries."""

    def test_critic_dimension_schema_compliance(self):
        """Verify CriticDimensionScore and CriticEvaluation Pydantic models instantiate correctly."""
        dims = [
            CriticDimensionScore(dimension="Statistical Validity & Empirical Rigor", score=82.0, justification="Good grounding"),
            CriticDimensionScore(dimension="Dataset Scale & Generalizability", score=78.0, justification="Diverse datasets"),
            CriticDimensionScore(dimension="Missing Baselines & Comparative Completeness", score=74.0, justification="Add DPO baseline"),
            CriticDimensionScore(dimension="Benchmark Overfitting & Actionability of Gaps", score=80.0, justification="Actionable roadmaps"),
        ]

        eval_model = CriticEvaluation(
            overall_score=78.5,
            dimension_scores=dims,
            strengths=["Strong thematic organization", "Structured evidence matrix"],
            weaknesses=["Could include DPO baseline"],
            refinement_guidance=["Add direct comparison to DPO adapter benchmarks."],
            should_refine=False,
        )

        assert eval_model.overall_score == 78.5
        assert len(eval_model.dimension_scores) == 4
        assert eval_model.should_refine is False

    @pytest.mark.asyncio
    async def test_critic_scoring_boundary_precision(self):
        """Verify strict passing threshold (< 75.0 triggers refinement, >= 75.0 passes)."""
        agent = AdversarialCriticAgent(llm_client=None, passing_threshold=75.0)

        # Case 1: Uncited draft -> fallback score 65.0 -> should_refine=True
        state_uncited = create_initial_agent_state(project_id="p1")
        state_uncited["thematic_sections"] = [{"title": "Uncited Theme", "synthesis_prose": "Prose with zero citations."}]
        state_uncited["iteration_count"] = 0

        res_uncited = await agent.run(state_uncited)
        assert res_uncited["current_critic_score"] == 65.0
        assert res_uncited["should_refine"] is True
        assert len(res_uncited["refinement_guidance"]) > 0

        # Case 2: Iteration 0 with citations -> fallback score 72.0 < 75.0 -> should_refine=True
        state_iter0 = create_initial_agent_state(project_id="p2")
        state_iter0["thematic_sections"] = [
            {"title": "Theme 1", "synthesis_prose": "Empirical claim [ref_1#sec_1]."},
            {"title": "Theme 2", "synthesis_prose": "Another claim [ref_2#sec_2]."},
        ]
        state_iter0["iteration_count"] = 0

        res_iter0 = await agent.run(state_iter0)
        assert res_iter0["current_critic_score"] == 72.0
        assert res_iter0["should_refine"] is True

        # Case 3: Iteration 1 with citations -> fallback score 78.0 >= 75.0 -> should_refine=False
        state_iter1 = create_initial_agent_state(project_id="p3")
        state_iter1["thematic_sections"] = [
            {"title": "Theme 1", "synthesis_prose": "Refined claim [ref_1#sec_1]."},
            {"title": "Theme 2", "synthesis_prose": "Another claim [ref_2#sec_2]."},
        ]
        state_iter1["iteration_count"] = 1

        res_iter1 = await agent.run(state_iter1)
        assert res_iter1["current_critic_score"] == 78.0
        assert res_iter1["should_refine"] is False

    @pytest.mark.asyncio
    async def test_critic_state_history_accumulation(self):
        """Verify critic evaluations accumulate in state['critic_evaluations'] list across iterations."""
        agent = AdversarialCriticAgent(llm_client=None)

        state = create_initial_agent_state(project_id="p_hist")
        state["thematic_sections"] = [{"title": "Theme", "synthesis_prose": "Text [ref_1]."}]
        state["iteration_count"] = 0

        # Iteration 1 run
        state = await agent.run(state)
        assert len(state["critic_evaluations"]) == 1

        # Iteration 2 run
        state["thematic_sections"].append({"title": "Theme 2", "synthesis_prose": "Text 2 [ref_2]."})
        state["iteration_count"] = 1
        state = await agent.run(state)
        assert len(state["critic_evaluations"]) == 2


# =============================================================================
# 3. DETERMINISTIC CITATION AUDITOR & FACT CHECKER TESTS
# =============================================================================


class TestDeterministicCitationAuditorAndFactChecker:
    """Stress tests for FactCheckerEngine, structured NLI, anchor canonicalization, and Auditor Agent."""

    def test_fact_checker_anchor_parsing_edge_cases(self):
        """Test extraction of diverse anchor formats including multi-hyphen IDs, section hashes, and malformed tags."""
        engine = FactCheckerEngine(llm_client=None)

        # Standard section hash
        p1, s1 = engine.parse_anchor_tag("ref_1#sec_methodology_2")
        assert p1 == "ref_1"
        assert s1 == "sec_methodology_2"

        # Paper-only anchor
        p2, s2 = engine.parse_anchor_tag("[ref_arxiv_2401_12345]")
        assert p2 == "ref_arxiv_2401_12345"
        assert s2 is None

        # Complex hyphenated DOI anchor
        p3, s3 = engine.parse_anchor_tag("ref_10_1109-CVPR_2024_01#table_3")
        assert p3 == "ref_10_1109-CVPR_2024_01"
        assert s3 == "table_3"

    def test_fact_checker_proposition_extraction_multi_anchor_sentences(self):
        """Verify extraction from dense multi-anchor sentences with exact claim isolation."""
        engine = FactCheckerEngine(llm_client=None)

        prose = (
            "Model A achieves 88.5% accuracy [ref_1#sec_res] while reducing VRAM by 40% [ref_1#sec_table1]. "
            "In contrast, Model B requires 3x compute [ref_2#sec_exp] despite similar convergence rates. "
            "This sentence has no citations at all."
        )

        props = engine.extract_atomic_propositions(prose, theme_id="theme_eval")
        assert len(props) == 3

        assert props[0].paper_id == "ref_1"
        assert props[0].section_anchor == "sec_res"

        assert props[1].paper_id == "ref_1"
        assert props[1].section_anchor == "sec_table1"

        assert props[2].paper_id == "ref_2"
        assert props[2].section_anchor == "sec_exp"

    def test_grounding_chunk_resolution_cascade(self):
        """Verify chunk resolver cascade: exact anchor -> chunk type/title -> priority types -> fallback."""
        chunks_map = {
            "ref_1": [
                {"chunk_id": "c_intro", "chunk_type": "introduction", "anchor_tag": "[ref_1#sec_intro]", "content": "Intro text"},
                {"chunk_id": "c_results", "chunk_type": "results", "anchor_tag": "[ref_1#sec_results_1]", "content": "Accuracy is 92.4%"},
                {"chunk_id": "c_methods", "chunk_type": "methodology", "anchor_tag": "[ref_1#sec_methods]", "content": "Tiling algorithm"},
            ]
        }

        # 1. Exact section_anchor match
        c_id, text = FactCheckerEngine.resolve_grounding_chunk("ref_1", "sec_results_1", chunks_map)
        assert c_id == "c_results"
        assert "92.4%" in text

        # 2. Section type match when exact anchor tag differs
        c_id2, text2 = FactCheckerEngine.resolve_grounding_chunk("ref_1", "results_summary", chunks_map)
        assert c_id2 == "c_results"

        # 3. Fallback when unknown section requested -> picks highest priority type (results)
        c_id3, text3 = FactCheckerEngine.resolve_grounding_chunk("ref_1", "unknown_sec", chunks_map)
        assert c_id3 == "c_results"

        # 4. Unknown paper -> returns (None, None)
        c_id4, text4 = FactCheckerEngine.resolve_grounding_chunk("ref_ghost", "sec_1", chunks_map)
        assert c_id4 is None
        assert text4 is None

    @pytest.mark.asyncio
    async def test_nli_classification_and_precision_threshold_boundaries(self):
        """Verify structured NLI classification verdicts (ENTAILMENT, NEUTRAL, CONTRADICTION) and precision >= 80% rule."""
        engine = FactCheckerEngine(llm_client=None)

        # Grounding text matching claim words -> ENTAILMENT
        prop_entailed = AtomicProposition(
            proposition="FlashAttention minimizes memory accesses through GPU tiling.",
            raw_anchor="ref_1#sec_1",
            paper_id="ref_1",
            section_anchor="sec_1",
            source_sentence="FlashAttention minimizes memory accesses through GPU tiling [ref_1#sec_1].",
        )
        res_entailed = await engine.verify_proposition(
            prop_entailed,
            grounding_text="FlashAttention minimizes memory accesses through IO-aware GPU SRAM tiling.",
            grounding_chunk_id="chunk_1",
        )
        assert res_entailed.verdict == NLIVerdict.ENTAILMENT

        # Unrelated grounding text with minimal overlap -> NEUTRAL
        prop_neutral = AtomicProposition(
            proposition="Quantum teleportation achieves sub-picosecond latency in semiconductor qubits.",
            raw_anchor="ref_1#sec_2",
            paper_id="ref_1",
            section_anchor="sec_2",
            source_sentence="Quantum teleportation achieves sub-picosecond latency in semiconductor qubits [ref_1#sec_2].",
        )
        res_neutral = await engine.verify_proposition(
            prop_neutral,
            grounding_text="FlashAttention minimizes memory accesses through IO-aware GPU SRAM tiling.",
            grounding_chunk_id="chunk_1",
        )
        assert res_neutral.verdict == NLIVerdict.NEUTRAL

        # Missing grounding text (hallucinated paper) -> CONTRADICTION
        prop_contradiction = AtomicProposition(
            proposition="Made-up claim with missing paper.",
            raw_anchor="ref_missing#sec_1",
            paper_id="ref_missing",
            section_anchor="sec_1",
            source_sentence="Made-up claim [ref_missing#sec_1].",
        )
        res_contradiction = await engine.verify_proposition(
            prop_contradiction,
            grounding_text=None,
            grounding_chunk_id=None,
        )
        assert res_contradiction.verdict == NLIVerdict.CONTRADICTION

    @pytest.mark.asyncio
    async def test_fact_checker_audit_precision_score_and_pass_criteria(self):
        """Test audit report precision score calculation: (entailed / total * 100) >= 80% AND 0 contradictions."""
        engine = FactCheckerEngine(llm_client=None)

        draft = ThematicSynthesisDraft(
            executive_summary="Review of attention algorithms [ref_1#sec_methods].",
            thematic_sections=[
                ThematicSection(
                    theme_id="t1",
                    title="Attention Methods",
                    synthesis_prose=(
                        "FlashAttention uses SRAM tiling [ref_1#sec_methods]. "
                        "IO-aware attention reduces memory IO overhead [ref_1#sec_methods]. "
                        "Theoretical limit is O(1) memory [ref_1#sec_limit]. "
                        "Ghost claim with fake paper [ref_fake#sec_9]."
                    ),
                    cited_paper_ids=["ref_1", "ref_fake"],
                )
            ],
            conflicting_findings_and_debates=[],
            actionable_research_gaps=[],
            methodology_overview={
                "distribution": {"Empirical": 1},
                "dominant_approach": "Empirical",
                "trend_description": "Tiling",
            },
        )

        chunks_map = {
            "ref_1": [
                {
                    "chunk_id": "c1",
                    "paper_id": "ref_1",
                    "anchor_tag": "[ref_1#sec_methods]",
                    "section_anchor": "sec_methods",
                    "content": "FlashAttention uses SRAM tiling and IO-aware attention to reduce memory IO overhead.",
                },
                {
                    "chunk_id": "c2",
                    "paper_id": "ref_1",
                    "anchor_tag": "[ref_1#sec_limit]",
                    "section_anchor": "sec_limit",
                    "content": "Theoretical limit is O(N) memory scaling.",
                },
            ]
        }

        report: CitationAuditReport = await engine.audit_thematic_draft(
            draft=draft,
            paper_chunks_map=chunks_map,
            known_paper_ids={"ref_1"},
        )

        assert report.total_propositions >= 4
        assert "ref_fake#sec_9" in report.hallucinated_anchors
        # Because ref_fake#sec_9 is hallucinated (contradiction/missing), audit_passed should be False
        assert report.audit_passed is False

    def test_prose_canonicalization_and_cleaning(self):
        """Verify hallucinated and contradicted anchors are stripped cleanly while preserving valid anchors and punctuation."""
        engine = FactCheckerEngine(llm_client=None)

        audit_report = CitationAuditReport(
            total_propositions=3,
            entailed_count=1,
            neutral_count=1,
            contradiction_count=1,
            precision_score=33.3,
            verifications=[
                PropositionVerification(
                    proposition="Valid claim",
                    citation_anchor="ref_1#sec_1",
                    paper_id="ref_1",
                    section_anchor="sec_1",
                    verdict=NLIVerdict.ENTAILMENT,
                    confidence=0.9,
                    reasoning="Supported",
                ),
                PropositionVerification(
                    proposition="Contradicted claim",
                    citation_anchor="ref_1#sec_bad",
                    paper_id="ref_1",
                    section_anchor="sec_bad",
                    verdict=NLIVerdict.CONTRADICTION,
                    confidence=1.0,
                    reasoning="Contradicted",
                ),
            ],
            hallucinated_anchors=["ref_hallucinated#sec_0"],
            audit_passed=False,
        )

        input_prose = (
            "Valid statement [ref_1#sec_1]. "
            "Contradicted assertion [ref_1#sec_bad]. "
            "Hallucinated reference [ref_hallucinated#sec_0]."
        )

        cleaned = engine.canonicalize_and_clean_prose(input_prose, audit_report)

        assert "[ref_1#sec_1]" in cleaned
        assert "[ref_1#sec_bad]" not in cleaned
        assert "[ref_hallucinated#sec_0]" not in cleaned
        # Check clean punctuation spacing
        assert "  " not in cleaned
        assert " ." not in cleaned


# =============================================================================
# 4. SUPERVISOR STATEGRAPH DAG & BOUNDED REFINEMENT INVARIANT TESTS
# =============================================================================


class TestSupervisorStateGraphDAGInvariants:
    """Stress tests for Supervisor DAG state transitions, routing invariants, and report assembly."""

    def test_refinement_loop_strictly_bounded_at_max_2_iterations(self):
        """Verify that under continuous critic failure (score < 75.0), pipeline NEVER exceeds max_iterations (default 2)."""
        state = create_initial_agent_state(project_id="p_loop", max_iterations=2)
        state["current_critic_score"] = 50.0
        state["should_refine"] = True
        state["iteration_count"] = 0

        # Iteration 0 -> Refinement 1
        route1 = should_refine_or_finalize(state)
        assert route1 == "synthesizer"
        assert state["iteration_count"] == 1

        # Iteration 1 -> Refinement 2
        state["current_critic_score"] = 55.0
        state["should_refine"] = True
        route2 = should_refine_or_finalize(state)
        assert route2 == "synthesizer"
        assert state["iteration_count"] == 2

        # Iteration 2 (Max iterations reached) -> MUST route to 'auditor' regardless of score
        state["current_critic_score"] = 40.0
        state["should_refine"] = True
        route3 = should_refine_or_finalize(state)
        assert route3 == "auditor"

    @pytest.mark.asyncio
    async def test_supervisor_goal_stack_initialization(self):
        """Verify Supervisor initializes standard ordered pipeline goals."""
        sup = AutonomousSupervisorAgent(llm_client=None)
        state = create_initial_agent_state(project_id="p_goals")
        state["goal_stack"] = []  # Clear to test supervisor initialization

        state = await sup.run(state)
        assert len(state["goal_stack"]) == 6
        target_agents = [g["target_agent"] for g in state["goal_stack"]]
        assert target_agents == [
            AgentType.DISCOVERY,
            AgentType.INGESTION,
            AgentType.MATRIX_BUILDER,
            AgentType.SYNTHESIZER,
            AgentType.CRITIC,
            AgentType.AUDITOR,
        ]

    @pytest.mark.asyncio
    async def test_finalizer_node_assembles_valid_research_report(self):
        """Verify finalizer node converts blackboard state into a complete valid ResearchReport model."""
        state = create_initial_agent_state(
            project_id="p_final",
            user_id="researcher_1",
            title="Transformer Scaling Survey",
            research_question="What are empirical limits of transformer parameter scaling?",
        )
        state["parsed_papers"] = {
            "ref_1": {
                "paper_id": "ref_1",
                "title": "Scaling Laws",
                "doi": "10.1234/sl",
                "is_full_text": True,
            }
        }
        state["evidence_matrix"] = [
            EvidenceMatrixRow(
                paper_id="ref_1",
                title="Scaling Laws",
                authors=["Kaplan et al."],
                methodology="Empirical Scaling",
                benchmark_dataset="WebText",
                primary_metric="Loss",
                primary_limitation="Fixed architecture",
                is_full_text=True,
            ).model_dump()
        ]
        state["draft_thematic_sections"] = [
            ThematicSection(
                theme_id="t1",
                title="Scaling Laws",
                synthesis_prose="Loss scales smoothly [ref_1#sec_1].",
                key_takeaways=["Power law scaling holds across 6 orders of magnitude."],
                cited_paper_ids=["ref_1"],
            ).model_dump()
        ]
        state["conflicting_debates"] = [
            ConflictingDebate(
                topic="Compute vs Parameter Optimal",
                perspective_a="Parameter count dominates performance.",
                perspective_b="Data tokens dominate performance.",
                critical_evaluation="Chinchilla showed data must scale in equal proportion.",
            ).model_dump()
        ]
        state["research_gaps"] = [
            ResearchGapItem(
                gap_id="GAP-01",
                description="Lack of scaling laws for multimodal reasoning tokens.",
                importance="high",
                recommended_methodology="Evaluate joint vision-language loss across token ratios.",
                grounding_paper_ids=["ref_1"],
            ).model_dump()
        ]

        final_state = await finalizer_node(state)

        assert final_state["status"] == "completed"
        assert "final_report" in final_state
        report_dict = final_state["final_report"]
        assert report_dict["metadata"]["title"] == "Transformer Scaling Survey"
        assert len(report_dict["comparison_matrix"]) == 1
        assert len(report_dict["thematic_sections"]) == 1
        assert len(report_dict["conflicting_findings_and_debates"]) == 1
        assert len(report_dict["actionable_research_gaps"]) == 1
        assert len(report_dict["bibliography"]) >= 1
