"""
LLM Client Contracts, Context Boundaries, and Schema Validation Test Suite.
Stress-tests:
1. Legacy Purge (no legacy files, no legacy router/failover modules, no slicing caps in tools.py).
2. Unbounded context in tools.py and BaseLLMClient (100k-1M char strings, no slicing).
3. BaseLLMClient, MockLLMClient, GeminiProvider, DeepSeekProvider, GroqClient contracts.
4. Robustness of structured output parser (5 fallback stages, schema extraction, syntax repair).
5. Strictness and resilience of Pydantic v2 schemas across edge cases and adversarial inputs.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from agents.llm import (
    DEEPSEEK_MODELS,
    GEMINI_MODELS,
    GROQ_MODELS,
    BaseLLMClient,
    DeepSeekProvider,
    GeminiClient,
    GeminiProvider,
    GroqClient,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    MockLLMClient,
    ModelTier,
    StructuredOutputError,
    clean_json_markdown,
    clear_client_cache,
    extract_json_substring,
    get_available_providers,
    get_best_available_provider,
    get_default_provider,
    get_llm_client,
    get_model_config,
    parse_and_validate,
    repair_json_syntax,
    set_default_provider,
    to_gemini_schema,
    to_openai_response_format,
)
from agents.schemas import (
    AcademicPaperCandidate,
    BibliographyItem,
    Citation,
    CitationAuditReport,
    ConflictingDebate,
    CriticDimensionScore,
    CriticEvaluation,
    EvidenceMatrixExtraction,
    EvidenceMatrixRow,
    GapImportance,
    MethodologyDistribution,
    NLIVerdict,
    PropositionVerification,
    ReportMetadata,
    ReportStatus,
    ResearchGapItem,
    ResearchReport,
    SearchQueryPlan,
    SectionType,
    ThematicSection,
    ThematicSynthesisDraft,
)
from agents.tools import (
    ToolResult,
    evaluate_synthesis_quality,
    extract_json_from_response,
    extract_keywords_from_question,
    extract_paper_insights,
    identify_research_gaps,
    identify_subtopics,
    refine_search_query,
    score_paper_relevance,
    synthesize_section,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# 1. LEGACY PURGE EMPIRICAL TESTS
# ============================================================================


class TestLegacyPurgeVerification:
    """Empirically verify all obsolete files, modules, and slicing hacks are eliminated."""

    DELETED_FILES = [
        "agents/llm/failover.py",
        "agents/model_router.py",
        "agents/model_scheduler.py",
        "agents/local_nlp.py",
        "agents/gemini_client.py",
        "agents/quality_checker_agent.py",
        "agents/analyzer.py",
        "agents/planner.py",
        "agents/synthesizer.py",
    ]

    def test_legacy_files_do_not_exist_on_disk(self):
        """Verify none of the deprecated legacy files exist anywhere on disk."""
        for rel_path in self.DELETED_FILES:
            full_path = BACKEND_DIR / rel_path
            assert not full_path.exists(), f"Legacy file must be deleted: {full_path}"

    def test_legacy_module_imports_fail(self):
        """Verify attempting to import legacy modules fails with ModuleNotFoundError."""
        legacy_modules = [
            "agents.llm.failover",
            "agents.model_router",
            "agents.model_scheduler",
            "agents.local_nlp",
            "agents.gemini_client",
            "agents.quality_checker_agent",
            "agents.analyzer",
            "agents.planner",
            "agents.synthesizer",
        ]
        for mod in legacy_modules:
            with pytest.raises(ModuleNotFoundError):
                __import__(mod)

    def test_agents_init_exports_no_legacy_symbols(self):
        """Verify agents/__init__.py contains no references to deprecated router or failover classes."""
        import agents

        assert not hasattr(agents, "LegacyAnalyzerAgent")
        assert not hasattr(agents, "LegacyPlannerAgent")
        assert not hasattr(agents, "LegacySynthesizerAgent")
        assert not hasattr(agents, "SmartModelRouter")
        assert not hasattr(agents, "ModelFailoverManager")

    def test_llm_init_exports_no_failover_symbols(self):
        """Verify agents/llm/__init__.py contains no failover exports."""
        import agents.llm

        assert not hasattr(agents.llm, "ModelFailoverManager")
        assert not hasattr(agents.llm, "FailoverRouter")
        assert not hasattr(agents.llm, "SmartModelRouter")

    def test_tools_py_has_no_defensive_slicing_caps(self):
        """Empirically inspect tools.py source to ensure no slicing caps remain on prompt inputs."""
        tools_path = BACKEND_DIR / "agents" / "tools.py"
        source = tools_path.read_text(encoding="utf-8")

        # Check for forbidden truncation patterns in tools.py
        forbidden_patterns = [
            r"abstract\[:\d+\]",
            r"synthesis\[:\d+\]",
            r"justification\[:\d+\]",
            r"analyses_text\[:\d+\]",
            r"findings\[:\d+\]",
            r"methodology\[:\d+\]",
            r"contribution\[:\d+\]",
        ]
        for pat in forbidden_patterns:
            matches = re.findall(pat, source)
            assert (
                len(matches) == 0
            ), f"Found forbidden slicing pattern '{pat}' in tools.py: {matches}"


# ============================================================================
# 2. UNBOUNDED CONTEXT STRESS TESTS (tools.py and BaseLLMClient)
# ============================================================================


class TestUnboundedContextStress:
    """Empirically verify that tools.py and BaseLLMClient process massive texts without truncating."""

    def test_score_paper_relevance_with_huge_abstract(self):
        """Pass a 200,000 character abstract to score_paper_relevance."""
        mock_llm = MockLLMClient()
        mock_llm.mock_text_responses = [
            '{"score": 92, "justification": "Highly relevant deep study."}'
        ]

        huge_abstract = "Scientific finding about quantum LLMs. " * 5000  # ~200,000 chars
        title = "Quantum Multi-Agent Systems for Scalable Literature Synthesis"
        rq = "How do quantum multi-agent systems improve scientific literature reasoning?"

        result = score_paper_relevance(mock_llm, title, huge_abstract, rq)
        assert result.success is True
        assert result.data["score"] == 92
        assert result.data["justification"] == "Highly relevant deep study."

        # Verify MockLLM received the entire unabridged prompt
        assert len(mock_llm.call_history) == 1
        recorded_prompt = mock_llm.call_history[0]["prompt"]
        assert len(recorded_prompt) > 190000
        assert huge_abstract in recorded_prompt

    def test_extract_paper_insights_with_huge_abstract(self):
        """Pass a 200,000 character abstract to extract_paper_insights."""
        mock_llm = MockLLMClient()
        mock_llm.mock_text_responses = [
            json.dumps({
                "key_findings": ["Discovered non-linear scaling laws", "Zero hallucination bounds"],
                "methodology": "Empirical benchmarking on 10M arXiv papers",
                "limitations": ["Requires GPU cluster"],
                "contribution": "State-of-the-art literature engine",
                "key_quotes": ["Scaling achieves bounded reasoning"],
            })
        ]

        huge_abstract = "Massive paper abstract text with detailed methodology. " * 4000
        result = extract_paper_insights(
            mock_llm, "Massive Scaling Paper", huge_abstract, "Scaling laws"
        )
        assert result.success is True
        assert len(result.data["key_findings"]) == 2
        assert result.data["methodology"] == "Empirical benchmarking on 10M arXiv papers"

        # Verify full prompt was preserved
        recorded_prompt = mock_llm.call_history[0]["prompt"]
        assert len(recorded_prompt) > 150000

    def test_synthesize_section_with_massive_paper_analyses(self):
        """Pass 50 paper analyses amounting to ~500,000 chars to synthesize_section."""
        mock_llm = MockLLMClient()
        mock_llm.mock_text_responses = [
            "## Architectural Advances\n\n### Thematic Organization\nEmpirical synthesis prose."
        ]

        huge_analyses = [
            {
                "title": f"Paper {i}: Scalable Reasoning in Multi-Agent Systems",
                "key_findings": [f"Finding {i}.1: High-throughput reasoning", f"Finding {i}.2: Zero-drift memory"],
                "methodology": f"Empirical evaluation of {i * 1000} agents with full-text ingestion.",
                "contribution": f"Demonstrated O(1) memory lookup for paper {i}.",
            }
            for i in range(100)
        ]

        result = synthesize_section(
            mock_llm,
            subtopic="Architectural Advances",
            paper_analyses=huge_analyses,
            academic_level="postgraduate",
            word_count=2000,
        )
        assert result.success is True
        assert "Architectural Advances" in result.data

        recorded_prompt = mock_llm.call_history[0]["prompt"]
        assert "Paper 99:" in recorded_prompt
        assert len(recorded_prompt) > 10000

    def test_evaluate_synthesis_quality_with_huge_text(self):
        """Pass a 100,000 character synthesis document to evaluate_synthesis_quality."""
        mock_llm = MockLLMClient()
        mock_llm.mock_text_responses = [
            json.dumps({
                "overall_score": 88.5,
                "criteria_scores": {
                    "coherence": 90,
                    "coverage": 85,
                    "critical_analysis": 90,
                    "academic_tone": 90,
                    "research_gaps": 85,
                },
                "feedback": "Rigorous and well-grounded review.",
                "should_refine": False,
            })
        ]

        huge_synthesis = "## Literature Review\n\nExtensive thematic synthesis section. " * 3000
        result = evaluate_synthesis_quality(
            mock_llm, huge_synthesis, "Multi-agent scientific systems", 50
        )
        assert result.success is True
        assert result.data["overall_score"] == 88.5
        assert result.data["should_refine"] is False

    def test_extract_json_from_response_adversarial_cases(self):
        """Stress-test extract_json_from_response against messy, noisy, and malformed inputs."""
        # 1. Standard markdown block
        r1 = "Here is the result:\n```json\n{\"score\": 95, \"label\": \"optimal\"}\n```\nHope that helps!"
        d1 = extract_json_from_response(r1)
        assert d1 == {"score": 95, "label": "optimal"}

        # 2. Generic markdown code block (no 'json')
        r2 = "```\n{\"valid\": true, \"count\": 10}\n```"
        d2 = extract_json_from_response(r2)
        assert d2 == {"valid": True, "count": 10}

        # 3. Trailing commas & Python booleans/None
        r3 = "Output: {\"active\": True, \"items\": [1, 2, 3,], \"missing\": None,}"
        d3 = extract_json_from_response(r3)
        assert d3 == {"active": True, "items": [1, 2, 3], "missing": None}

        # 4. Outermost balanced JSON with conversational text
        r4 = "Analysis complete. {\"papers\": [\"p1\", \"p2\"], \"confidence\": 0.99} Let me know if you need more."
        d4 = extract_json_from_response(r4)
        assert d4 == {"papers": ["p1", "p2"], "confidence": 0.99}

        # 5. Empty or whitespace string returns default
        assert extract_json_from_response("", {"fallback": 1}) == {"fallback": 1}
        assert extract_json_from_response("   ", {"fallback": 2}) == {"fallback": 2}
        assert extract_json_from_response(None, {"fallback": 3}) == {"fallback": 3}

        # 6. Completely unparseable garbage returns default gracefully
        assert extract_json_from_response("This is not JSON at all!", {"safe": True}) == {"safe": True}


# ============================================================================
# 3. LLM CLIENT & PROVIDER CONTRACT EMPIRICAL TESTS
# ============================================================================


class SampleTestSchema(BaseModel):
    name: str
    score: float
    tags: list[str] = Field(default_factory=list)


class TestLLMClientContracts:
    """Verify BaseLLMClient interface, MockLLMClient, GeminiProvider, DeepSeekProvider, and GroqClient."""

    def test_mock_llm_client_contract(self):
        """Verify MockLLMClient meets all BaseLLMClient specifications."""
        mock = MockLLMClient()
        assert mock.get_provider_name() == "mock"
        assert mock.is_available() is True

        # Text generation default
        text = mock.generate_text("Test prompt")
        assert "Synthetic Scientific Response" in text

        # Queue-based text generation
        mock.mock_text_responses = ["Response 1", "Response 2"]
        assert mock.generate_text("Prompt 1") == "Response 1"
        assert mock.generate_text("Prompt 2") == "Response 2"

        # Structured output default (model_construct)
        item = mock.generate_structured("Generate item", SampleTestSchema)
        assert isinstance(item, SampleTestSchema)

        # Structured output mock override
        custom_item = SampleTestSchema(name="EmpiricalTest", score=99.9, tags=["verified"])
        mock.mock_structured_responses[SampleTestSchema] = custom_item
        assert mock.generate_structured("Generate item", SampleTestSchema) == custom_item

        # Usage stats
        stats = mock.get_usage_stats()
        assert stats["provider"] == "mock"
        assert stats["calls"] == 5

    def test_gemini_provider_contract_and_payload_building(self):
        """Verify GeminiProvider schema conversion, payload construction, and tier mapping."""
        config = LLMConfig(api_key="test-api-key", temperature=0.3, max_tokens=2048)
        provider = GeminiProvider(config)

        assert provider.get_provider_name() == "gemini"
        assert provider.is_available() is True
        assert provider._select_model(ModelTier.FAST) == "gemini-3.5-flash-lite"
        assert provider._select_model(ModelTier.REASONING) == "gemini-3.5-flash-lite"
        assert provider._select_model(ModelTier.STRUCTURED_NLI) == "gemini-3.5-flash-lite"

        # Payload building for text
        text_payload = provider._build_payload(
            prompt="Summarize quantum computing",
            system_prompt="You are a scientific AI.",
            temperature=0.5,
            max_tokens=1000,
        )
        assert text_payload["contents"][0]["parts"][0]["text"] == "Summarize quantum computing"
        assert text_payload["systemInstruction"]["parts"][0]["text"] == "You are a scientific AI."
        assert text_payload["generationConfig"]["temperature"] == 0.5
        assert text_payload["generationConfig"]["maxOutputTokens"] == 1000

        # Payload building for structured output with response_schema
        gemini_schema = to_gemini_schema(SampleTestSchema)
        struct_payload = provider._build_payload(
            prompt="Extract sample",
            response_schema=gemini_schema,
            response_mime_type="application/json",
        )
        assert struct_payload["generationConfig"]["responseMimeType"] == "application/json"
        assert "properties" in struct_payload["generationConfig"]["responseSchema"]

    def test_deepseek_provider_contract_and_tier_mapping(self):
        """Verify DeepSeekProvider model tier selection and OpenAI-compatible configuration."""
        config = LLMConfig(api_key="test-deepseek-key", base_url="https://api.deepseek.com")
        provider = DeepSeekProvider(config)

        assert provider.get_provider_name() == "deepseek"
        assert provider.is_available() is True
        assert provider._select_model(ModelTier.FAST) == "deepseek-chat"
        assert provider._select_model(ModelTier.REASONING) == "deepseek-reasoner"
        assert provider._select_model(ModelTier.STRUCTURED_NLI) == "deepseek-chat"

    def test_groq_client_contract_and_tier_mapping(self):
        """Verify GroqClient model tier selection and configuration."""
        config = LLMConfig(api_key="test-groq-key")
        provider = GroqClient(config)

        assert provider.get_provider_name() == "groq"
        assert provider.is_available() is True
        assert provider._select_model(ModelTier.FAST) == "llama-3.1-8b-instant"
        assert provider._select_model(ModelTier.REASONING) == "llama-3.3-70b-versatile"
        assert provider._select_model(ModelTier.STRUCTURED_NLI) == "llama-3.1-8b-instant"

    def test_factory_and_cache_management(self):
        """Verify centralized get_llm_client factory and cache resolution."""
        clear_client_cache()

        mock_client1 = get_llm_client(provider=LLMProvider.MOCK)
        mock_client2 = get_llm_client(provider="mock")
        assert mock_client1 is mock_client2

        gemini_client = get_llm_client(provider="gemini", config=LLMConfig(api_key="gem-key"))
        assert isinstance(gemini_client, GeminiProvider)

        deepseek_client = get_llm_client(provider="deepseek", config=LLMConfig(api_key="ds-key"))
        assert isinstance(deepseek_client, DeepSeekProvider)

        groq_client = get_llm_client(provider="groq", config=LLMConfig(api_key="groq-key"))
        assert isinstance(groq_client, GroqClient)

        clear_client_cache()


# ============================================================================
# 4. STRUCTURED OUTPUT PARSER EMPIRICAL TESTS
# ============================================================================


class TestStructuredOutputParser:
    """Stress-test parse_and_validate against 5 progressive fallback stages and complex schemas."""

    def test_stage_1_direct_json(self):
        raw = '{"name": "Alpha", "score": 98.5, "tags": ["fast", "accurate"]}'
        result = parse_and_validate(raw, SampleTestSchema)
        assert result.name == "Alpha"
        assert result.score == 98.5
        assert result.tags == ["fast", "accurate"]

    def test_stage_2_markdown_fences(self):
        raw = '```json\n{"name": "Beta", "score": 88.0, "tags": ["rag"]}\n```'
        result = parse_and_validate(raw, SampleTestSchema)
        assert result.name == "Beta"
        assert result.score == 88.0

    def test_stage_3_balanced_substring_with_chatter(self):
        raw = (
            "Here is your generated schema instance:\n\n"
            '{"name": "Gamma", "score": 75.2, "tags": ["benchmark"]}\n\n'
            "Hope this meets your requirements."
        )
        result = parse_and_validate(raw, SampleTestSchema)
        assert result.name == "Gamma"
        assert result.score == 75.2

    def test_stage_4_syntax_auto_repair(self):
        # Trailing commas + Python True/False/None
        raw = '{"name": "Delta", "score": 90.0, "tags": ["a", "b",],}'
        result = parse_and_validate(raw, SampleTestSchema)
        assert result.name == "Delta"
        assert result.tags == ["a", "b"]

    def test_structured_output_error_on_unrecoverable_failure(self):
        raw = "Completely unparseable garbage text that has no JSON whatsoever."
        with pytest.raises(StructuredOutputError) as exc_info:
            parse_and_validate(raw, SampleTestSchema)

        err = exc_info.value
        assert err.schema_name == "SampleTestSchema"
        assert err.raw_text == raw
        assert err.validation_errors is not None

    def test_to_gemini_schema_recursive_defs_resolution(self):
        """Verify to_gemini_schema converts nested Pydantic models into valid OpenAPI 3.0 without $defs."""
        schema = to_gemini_schema(ResearchReport)
        assert "$defs" not in schema
        assert "properties" in schema
        assert "metadata" in schema["properties"]
        assert "thematic_sections" in schema["properties"]


# ============================================================================
# 5. PYDANTIC V2 CONTRACT SCHEMAS ADVERSARIAL STRESS TESTS
# ============================================================================


class TestPydanticV2Schemas:
    """Stress-test all contract schemas with adversarial inputs, type coercions, and serialization."""

    def test_evidence_matrix_row_coercion_and_clean(self):
        # Test string-delimited authors (comma or semicolon), string year extraction, whitespace cleanup
        row = EvidenceMatrixRow(
            paper_id="  ref_2024_01  ",
            title=" Scalable Agent Reasoning ",
            authors="Alice Smith, Bob Jones, Charlie Brown",
            year="Published in December 2024 (NeurIPS)",
            methodology="Hierarchical LangGraph StateGraph",
            benchmark_dataset="HumanEval, ScienceQA",
            primary_metric="Accuracy: 94.2%",
            primary_limitation="Requires high-capacity LLM context",
            is_full_text=True,
        )
        assert row.paper_id == "ref_2024_01"
        assert row.title == "Scalable Agent Reasoning"
        assert row.authors == ["Alice Smith", "Bob Jones", "Charlie Brown"]
        assert row.year == 2024
        assert row.is_full_text is True

        # Test semicolon-delimited authors
        row_semi = EvidenceMatrixRow(
            paper_id="ref_2",
            title="Title",
            authors="David Miller; Eve Wilson",
            methodology="M",
            benchmark_dataset="B",
            primary_metric="P",
            primary_limitation="L",
        )
        assert row_semi.authors == ["David Miller", "Eve Wilson"]

    def test_thematic_section_citation_anchors_extraction(self):
        section = ThematicSection(
            theme_id="theme_1",
            title="Transformer Scalability",
            synthesis_prose=(
                "Prior work showed that dense retrieval scales quadratically [ref_1#sec_intro], "
                "whereas hybrid sparse-dense architectures achieve sub-linear scaling [ref_2#sec_methods_3]. "
                "Furthermore, empirical evaluations confirm these findings [ref_3]."
            ),
            key_takeaways="- Scaling is sub-linear\n- Hybrid retrieval is optimal",
            cited_paper_ids=["ref_1", "ref_2", "ref_3"],
        )
        anchors = section.extract_citation_anchors()
        assert anchors == ["[ref_1#sec_intro]", "[ref_2#sec_methods_3]", "[ref_3]"]
        assert len(section.key_takeaways) == 2
        assert section.key_takeaways[0] == "Scaling is sub-linear"
        assert section.key_takeaways[1] == "Hybrid retrieval is optimal"

    def test_research_gap_item_and_importance_literals(self):
        gap = ResearchGapItem(
            gap_id="gap_1",
            description="Lack of long-term episodic memory evaluation in multi-agent scientific discovery",
            importance="high",
            recommended_methodology="Construct continuous benchmark tracking 30-day agent iterations",
            grounding_paper_ids="ref_1, ref_4, ref_7",
        )
        assert gap.importance == "high"
        assert gap.grounding_paper_ids == ["ref_1", "ref_4", "ref_7"]

    def test_methodology_distribution_coercion(self):
        dist = MethodologyDistribution(
            distribution={"Empirical": "15", "Theoretical": 5, "Survey": "2"},
            dominant_approach="Empirical",
            trend_description="Clear shift towards empirical benchmarking.",
        )
        assert dist.distribution == {"Empirical": 15, "Theoretical": 5, "Survey": 2}
        assert dist.dominant_approach == "Empirical"

    def test_report_metadata_quality_score_clamping(self):
        # Clamps quality score between 0 and 100
        meta1 = ReportMetadata(
            project_id="proj_1",
            user_id="user_1",
            title="Test",
            research_question="RQ?",
            quality_score=150.0,
            status="COMPLETE",
        )
        assert meta1.quality_score == 100.0
        assert meta1.status == ReportStatus.COMPLETE

        meta2 = ReportMetadata(
            project_id="proj_2",
            user_id="user_2",
            title="Test",
            research_question="RQ?",
            quality_score=-25.0,
            status="partial",
        )
        assert meta2.quality_score == 0.0
        assert meta2.status == ReportStatus.PARTIAL

    def test_research_report_to_markdown_formatting(self):
        """Verify full ResearchReport model instantiates, serializes, and renders complete Markdown."""
        report = ResearchReport(
            metadata=ReportMetadata(
                project_id="p-100",
                user_id="u-200",
                title="Autonomous Literature Review on Multi-Agent Reasoning",
                research_question="What are the state-of-the-art architectures for multi-agent scientific reasoning?",
                quality_score=89.5,
                papers_analyzed_full_text=5,
                total_citations=8,
            ),
            executive_summary="This report synthesizes contemporary architectures for autonomous scientific discovery.",
            comparison_matrix=[
                EvidenceMatrixRow(
                    paper_id="ref_1",
                    title="LangGraph: Stateful Multi-Agent Applications",
                    authors=["Harrison Chase", "Rotem Weiss"],
                    year=2024,
                    methodology="StateGraph cyclic execution",
                    benchmark_dataset="SWE-bench",
                    primary_metric="Accuracy: 88.5%",
                    primary_limitation="State serialization overhead",
                    is_full_text=True,
                )
            ],
            thematic_sections=[
                ThematicSection(
                    theme_id="theme_1",
                    title="Cyclic State Graphs vs Static Pipelines",
                    synthesis_prose="Cyclic graphs provide self-correcting feedback loops [ref_1#sec_methodology].",
                    key_takeaways=["Feedback loops reduce hallucination."],
                    cited_paper_ids=["ref_1"],
                )
            ],
            conflicting_findings_and_debates=[
                ConflictingDebate(
                    topic="Centralized Supervisor vs Peer Choreography",
                    perspective_a="Supervisor DAG provides deterministic bounds and auditability.",
                    perspective_b="Choreography allows higher emergence and flexible exploration.",
                    critical_evaluation="For scientific literature review, supervisor architecture guarantees termination.",
                )
            ],
            actionable_research_gaps=[
                ResearchGapItem(
                    gap_id="gap_1",
                    description="Standardized evaluation harness for multi-agent citation auditing",
                    importance="high",
                    recommended_methodology="Create synthetic NLI benchmark with known counterfactual mutations.",
                    grounding_paper_ids=["ref_1"],
                )
            ],
            methodology_overview=MethodologyDistribution(
                distribution={"Empirical": 6, "Theoretical": 2},
                dominant_approach="Empirical",
                trend_description="Shift from heuristic agents to structured LangGraph DAGs.",
            ),
            bibliography=[
                BibliographyItem(
                    paper_id="ref_1",
                    title="LangGraph: Stateful Multi-Agent Applications",
                    authors=["Harrison Chase", "Rotem Weiss"],
                    year=2024,
                    venue="arXiv",
                    doi="10.48550/arXiv.2401.12345",
                    pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
                    is_full_text_analyzed=True,
                )
            ],
        )

        md = report.to_markdown()
        assert "# Autonomous Literature Review on Multi-Agent Reasoning" in md
        assert "## Executive Summary" in md
        assert "## Evidence Comparison Matrix" in md
        assert "| ref_1 | LangGraph: Stateful Multi-Agent Applications |" in md
        assert "## Thematic Synthesis" in md
        assert "### Cyclic State Graphs vs Static Pipelines" in md
        assert "## Conflicting Findings & Scientific Debates" in md
        assert "## Actionable Research Gaps & Future Directions" in md
        assert "## Bibliography" in md
        assert "DOI: [10.48550/arXiv.2401.12345]" in md

    def test_citation_auditor_and_critic_schemas(self):
        """Verify PropositionVerification, CitationAuditReport, and CriticEvaluation schemas."""
        prop = PropositionVerification(
            proposition="Hybrid RAG with RRF k=60 achieves superior retrieval precision.",
            citation_anchor="ref_1#sec_results_2",
            paper_id="ref_1",
            section_anchor="sec_results_2",
            grounding_text="We found RRF k=60 yields 14% higher NDCG@10 than single-vector search.",
            verdict=NLIVerdict.ENTAILMENT,
            confidence=0.98,
            reasoning="Source text explicitly confirms superior retrieval metrics.",
        )
        assert prop.verdict == NLIVerdict.ENTAILMENT

        audit = CitationAuditReport(
            total_propositions=10,
            entailed_count=9,
            neutral_count=1,
            contradiction_count=0,
            precision_score=90.0,
            verifications=[prop],
            audit_passed=True,
        )
        assert audit.audit_passed is True

        critic = CriticEvaluation(
            overall_score=82.0,
            dimension_scores=[
                CriticDimensionScore(
                    dimension="Grounding",
                    score=88.0,
                    justification="High precision citation grounding.",
                )
            ],
            strengths=["Thorough evidence matrix."],
            weaknesses=["Could expand on computational cost."],
            refinement_guidance=["Add discussion on FLOPs."],
            should_refine=False,
        )
        assert critic.overall_score == 82.0
        assert critic.should_refine is False
