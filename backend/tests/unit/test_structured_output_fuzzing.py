"""
Structured Output Parsing, Fuzzing, and Schema Resilience Stress Test Suite.
Covers:
- Malformed JSON outputs & multi-stage parsing resilience
- Extreme data stress testing (massive payloads, 500k char strings, 1k items)
- Unicode edge cases (CJK, RTL, emoji, zero-width spaces)
- LaTeX equations & mathematical notation serialization
- Scientific notation & numeric boundary conditions
- Pydantic v2 serialization round-tripping & Markdown conversion
- Gemini / OpenAI schema transformations
- LLM Provider error handling & circuit breaker behaviors
"""

import json
import re
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from agents.error_handling import NonRetryableError, RetryableError
from agents.llm.base import BaseLLMClient, LLMConfig, LLMResponse, ModelTier
from agents.llm.factory import LLMProvider, MockLLMClient, get_llm_client, set_default_provider
from agents.llm.gemini_provider import GeminiProvider
from agents.llm.deepseek_provider import DeepSeekProvider
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
    AcademicPaperCandidate,
    BibliographyItem,
    CitationAuditReport,
    ConflictingDebate,
    CriticDimensionScore,
    CriticEvaluation,
    EvidenceMatrixExtraction,
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


# ============================================================================
# 1. Malformed JSON Outputs & Multi-Stage Parser Fuzzing
# ============================================================================

@pytest.mark.unit
class TestMalformedJSONAndParserFuzzing:
    """Adversarial challenge for JSON extraction, syntax repair, and fallback parsing."""

    def test_clean_json_markdown_adversarial_fences(self):
        """Test markdown code block stripping with various fence permutations."""
        # Triple backticks with json language tag
        raw_1 = "```json\n{\"paper_id\": \"ref_1\", \"title\": \"Test\"}\n```"
        assert clean_json_markdown(raw_1) == '{"paper_id": "ref_1", "title": "Test"}'

        # Uppercase JSON tag
        raw_2 = "```JSON\n{\"paper_id\": \"ref_2\", \"title\": \"Test\"}\n```"
        assert clean_json_markdown(raw_2) == '{"paper_id": "ref_2", "title": "Test"}'

        # Plain backticks without language tag
        raw_3 = "```\n{\"paper_id\": \"ref_3\", \"title\": \"Test\"}\n```"
        assert clean_json_markdown(raw_3) == '{"paper_id": "ref_3", "title": "Test"}'

        # Code block surrounded by conversational chatter and markdown headers
        raw_4 = (
            "Here is the synthesized data you requested:\n\n"
            "```json\n"
            "{\n  \"paper_id\": \"ref_4\",\n  \"title\": \"Complex Paper\"\n}\n"
            "```\n\n"
            "Let me know if you need any adjustments!"
        )
        assert json.loads(clean_json_markdown(raw_4))["paper_id"] == "ref_4"

    def test_extract_json_substring_nested_structures_and_escapes(self):
        """Test balanced substring extraction with nested braces and escaped quotes."""
        # String with embedded braces and escaped quotes
        text_1 = 'Preamble: {"paper_id": "ref_1", "title": "Title with {braces} and \\"escaped quotes\\""} postscript'
        extracted_1 = extract_json_substring(text_1)
        assert extracted_1 == '{"paper_id": "ref_1", "title": "Title with {braces} and \\"escaped quotes\\""}'
        assert json.loads(extracted_1)["title"] == 'Title with {braces} and "escaped quotes"'

        # Top-level array extraction
        text_2 = 'Notes: [{"id": 1}, {"id": 2}] end.'
        extracted_2 = extract_json_substring(text_2)
        assert extracted_2 == '[{"id": 1}, {"id": 2}]'
        assert json.loads(extracted_2)[1]["id"] == 2

        # Windows file paths with escaped backslashes inside JSON
        text_3 = 'Prefix {"path": "C:\\\\Users\\\\ScholarAgent\\\\file.pdf"} Suffix'
        extracted_3 = extract_json_substring(text_3)
        assert json.loads(extracted_3)["path"] == "C:\\Users\\ScholarAgent\\file.pdf"

    def test_repair_json_syntax_trailing_commas_and_python_literals(self):
        """Test syntax repair on common LLM formatting flaws."""
        # Trailing commas in arrays and dicts
        broken_json = '{"items": [1, 2, 3, ], "config": {"debug": True, "cache": False, "timeout": None, }, }'
        repaired = repair_json_syntax(broken_json)
        parsed = json.loads(repaired)
        assert parsed["items"] == [1, 2, 3]
        assert parsed["config"]["debug"] is True
        assert parsed["config"]["cache"] is False
        assert parsed["config"]["timeout"] is None

        # Verify word boundaries so variable names like NoneOfTheAbove are not broken
        text_with_names = '{"name": "NoneOfTheAbove", "flag": True, "value": "TrueValues"}'
        repaired_names = repair_json_syntax(text_with_names)
        parsed_names = json.loads(repaired_names)
        assert parsed_names["name"] == "NoneOfTheAbove"
        assert parsed_names["flag"] is True
        assert parsed_names["value"] == "TrueValues"

    def test_parse_and_validate_all_stages(self):
        """Test that parse_and_validate correctly recovers through all 5 progressive fallback stages."""
        # Stage 1: Clean raw JSON
        raw_1 = '{"gap_id": "gap_1", "description": "Desc 1", "importance": "high", "recommended_methodology": "Meth 1"}'
        res_1 = parse_and_validate(raw_1, ResearchGapItem)
        assert res_1.gap_id == "gap_1"

        # Stage 2: Markdown fence wrapped
        raw_2 = '```json\n{"gap_id": "gap_2", "description": "Desc 2", "importance": "medium", "recommended_methodology": "Meth 2"}\n```'
        res_2 = parse_and_validate(raw_2, ResearchGapItem)
        assert res_2.gap_id == "gap_2"

        # Stage 3: Chatty wrapper
        raw_3 = 'Sure! Here is the gap item:\n\n```json\n{"gap_id": "gap_3", "description": "Desc 3", "importance": "low", "recommended_methodology": "Meth 3"}\n```\nHope that helps!'
        res_3 = parse_and_validate(raw_3, ResearchGapItem)
        assert res_3.gap_id == "gap_3"

        # Stage 4: Trailing commas and Python True/False/None
        raw_4 = '```json\n{"gap_id": "gap_4", "description": "Desc 4", "importance": "high", "recommended_methodology": "Meth 4", "grounding_paper_ids": ["ref_1", "ref_2", ], }\n```'
        res_4 = parse_and_validate(raw_4, ResearchGapItem)
        assert res_4.gap_id == "gap_4"
        assert res_4.grounding_paper_ids == ["ref_1", "ref_2"]

    def test_parse_and_validate_unrecoverable_malformed_raises_structured_output_error(self):
        """Test that truly invalid or unrecoverable payloads raise StructuredOutputError with rich debugging metadata."""
        # Completely unparseable garbage
        garbage = "Not JSON at all, just random text."
        with pytest.raises(StructuredOutputError) as exc_info:
            parse_and_validate(garbage, EvidenceMatrixRow)
        assert "EvidenceMatrixRow" in str(exc_info.value)
        assert exc_info.value.schema_name == "EvidenceMatrixRow"
        assert exc_info.value.raw_text == garbage

        # Missing required field in JSON
        invalid_schema = '{"paper_id": "ref_1"}'  # missing title, methodology, etc.
        with pytest.raises(StructuredOutputError) as exc_info2:
            parse_and_validate(invalid_schema, EvidenceMatrixRow)
        assert exc_info2.value.schema_name == "EvidenceMatrixRow"


# ============================================================================
# 2. Extreme Data & Schema Stress Testing
# ============================================================================

@pytest.mark.unit
class TestSchemaStressAndEdgeCases:
    """Fuzzing and stress-testing schemas against massive datasets, Unicode, LaTeX, and numeric extremes."""

    def test_massive_payload_thousand_evidence_matrix_rows(self):
        """Stress-test processing and validation of 1,000 EvidenceMatrixRows in EvidenceMatrixExtraction."""
        rows = [
            EvidenceMatrixRow(
                paper_id=f"paper_{i:04d}",
                title=f"Scalable Scientific Reasoning at Scale #{i}",
                authors=[f"Author A{i}", f"Author B{i}"],
                year=2020 + (i % 5),
                methodology=f"Deep Reinforcement Learning framework with policy optimization #{i}",
                benchmark_dataset=f"SciBench-v{i % 10}",
                primary_metric=f"Accuracy: {80.0 + (i % 20):.1f}%",
                primary_limitation=f"High GPU memory requirement for step {i}",
                is_full_text=(i % 2 == 0),
            )
            for i in range(1000)
        ]

        extraction = EvidenceMatrixExtraction(rows=rows)
        assert len(extraction.rows) == 1000

        # Serialize to JSON and parse back
        json_str = extraction.model_dump_json()
        assert len(json_str) > 200_000  # Substantial payload

        loaded = EvidenceMatrixExtraction.model_validate_json(json_str)
        assert len(loaded.rows) == 1000
        assert loaded.rows[500].paper_id == "paper_0500"
        assert loaded.rows[500].is_full_text is True

    def test_giant_string_payload_500k_chars(self):
        """Stress-test handling of extremely long scientific text (500k characters) without truncation."""
        giant_prose = "The empirical analysis of multi-agent transformer dynamics reveals emergent behaviors. " * 5500
        assert len(giant_prose) > 450_000

        section = ThematicSection(
            theme_id="theme_giant",
            title="Large Scale Literature Synthesis",
            synthesis_prose=giant_prose,
            key_takeaways=["Takeaway 1", "Takeaway 2"],
            cited_paper_ids=["ref_1", "ref_2"],
        )

        assert len(section.synthesis_prose) > 450_000
        # Check round-trip serialization
        json_str = section.model_dump_json()
        loaded = ThematicSection.model_validate_json(json_str)
        assert len(loaded.synthesis_prose) == len(giant_prose)

    def test_unicode_cjk_arabic_hebrew_and_emojis(self):
        """Verify seamless handling of international scientific text and non-ASCII character sets."""
        unicode_prose = (
            "This section covers global multilingual NLP research: "
            "中文 (大规模语言模型), 日本語 (自然言語処理の進化), "
            "العربية (معالجة اللغات الطبيعية), עברית (עיבוד שפה טבעית), "
            "Emojis: 🔬 🧠 📊 🚀 💡 [ref_cjk#sec1]."
        )

        section = ThematicSection(
            theme_id="theme_intl",
            title="Multilingual Reasoning & Global NLP 🌐",
            synthesis_prose=unicode_prose,
            key_takeaways=["Supports UTF-8 fully ✅", "No mojibake 🚀"],
            cited_paper_ids=["ref_cjk"],
        )

        anchors = section.extract_citation_anchors()
        assert anchors == ["[ref_cjk#sec1]"]

        json_str = section.model_dump_json()
        loaded = ThematicSection.model_validate_json(json_str)
        assert "大规模语言模型" in loaded.synthesis_prose
        assert "معالجة اللغات الطبيعية" in loaded.synthesis_prose
        assert "🔬" in loaded.synthesis_prose

    def test_complex_latex_mathematical_equations(self):
        """Verify that LaTeX math formulas, escape backslashes, sub/superscripts serialize without corruption."""
        latex_formula = (
            "We define the evidence loss as: "
            r"$$\mathcal{L}_{\text{total}} = \sum_{i=1}^{N} \left( \alpha \cdot \mathbb{E}_{x \sim \mathcal{D}} [\log D(x)] + \beta \cdot \|\nabla_{\theta} f(x)\|_2^2 \right) + \frac{\sqrt{\lambda}}{\sigma^2}$$ "
            "as substantiated by [ref_math#sec_eq4]."
        )

        debate = ConflictingDebate(
            topic=r"Optimal Convergence Bounds for $\mathcal{O}(N \log N)$ Optimization",
            perspective_a=r"Proves sub-quadratic bound $\mathcal{O}(N \sqrt{\log N})$ using Kantorovich metric",
            perspective_b=r"Demonstrates lower bound $\Omega(N \log N)$ under non-convex Lipschitz continuous manifolds",
            critical_evaluation=latex_formula,
        )

        json_str = debate.model_dump_json()
        # Verify JSON validity and roundtrip
        loaded = ConflictingDebate.model_validate_json(json_str)
        assert r"\mathcal{L}_{\text{total}}" in loaded.critical_evaluation
        assert r"\sum_{i=1}^{N}" in loaded.critical_evaluation
        assert r"\frac{\sqrt{\lambda}}{\sigma^2}" in loaded.critical_evaluation

    def test_scientific_notation_and_numeric_edge_cases(self):
        """Test scientific notation in metrics, float year conversions, and quality score bounds."""
        # EvidenceMatrixRow with scientific notation in metric string
        row = EvidenceMatrixRow(
            paper_id="ref_sci_not",
            title="Quantum Monte Carlo Estimation",
            authors=["R. Feynman", "E. Fermi"],
            year=2024,
            methodology="QMC Simulation",
            benchmark_dataset="Hamiltonian-128",
            primary_metric="Energy variance: 1.42e-6 eV, p-value: 3.18e-12, FLOPs: 4.5e+18",
            primary_limitation="Exponential state space scaling O(2^N)",
        )
        assert "1.42e-6" in row.primary_metric
        assert "4.5e+18" in row.primary_metric

        # Float year conversion
        row_float_year = EvidenceMatrixRow(
            paper_id="ref_float_yr",
            title="Title",
            authors=["Author"],
            year=2023.0,  # float
            methodology="Method",
            benchmark_dataset="Data",
            primary_metric="Metric",
            primary_limitation="Limitation",
        )
        assert row_float_year.year == 2023

        # Year boundary enforcement (1800 to 2100)
        row_ancient = EvidenceMatrixRow(
            paper_id="ref_ancient",
            title="Ancient",
            authors=["Aristotle"],
            year=350,  # Before 1800 -> should coerce to None
            methodology="Philosophy",
            benchmark_dataset="Nature",
            primary_metric="N/A",
            primary_limitation="None",
        )
        assert row_ancient.year is None

        row_future = EvidenceMatrixRow(
            paper_id="ref_future",
            title="Sci-Fi",
            authors=["Spock"],
            year=3000,  # After 2100 -> should coerce to None
            methodology="Warp Theory",
            benchmark_dataset="Galaxy",
            primary_metric="Warp 9",
            primary_limitation="Dilithium shortage",
        )
        assert row_future.year is None

        # Quality score clamping
        meta_negative = ReportMetadata(
            project_id="proj_01",
            user_id="user_01",
            title="Test",
            research_question="Question",
            quality_score=-15.0,  # Below 0 -> clamped to 0.0
        )
        assert meta_negative.quality_score == 0.0

        meta_overflow = ReportMetadata(
            project_id="proj_02",
            user_id="user_02",
            title="Test",
            research_question="Question",
            quality_score=150.0,  # Above 100 -> clamped to 100.0
        )
        assert meta_overflow.quality_score == 100.0

    def test_flexible_coercion_for_authors_and_lists(self):
        """Verify field validators handle string-encoded lists from LLM outputs."""
        # Comma-separated authors
        row_comma = EvidenceMatrixRow(
            paper_id="ref_1",
            title="Title",
            authors="John Doe, Jane Smith, Alan Turing",  # string with commas
            year="Published in 2022",
            methodology="Method",
            benchmark_dataset="Data",
            primary_metric="95%",
            primary_limitation="None",
        )
        assert row_comma.authors == ["John Doe", "Jane Smith", "Alan Turing"]
        assert row_comma.year == 2022

        # Semicolon-separated authors
        row_semi = EvidenceMatrixRow(
            paper_id="ref_2",
            title="Title",
            authors="Alice Bob; Charlie Brown",
            year=2021,
            methodology="Method",
            benchmark_dataset="Data",
            primary_metric="88%",
            primary_limitation="None",
        )
        assert row_semi.authors == ["Alice Bob", "Charlie Brown"]

        # Multiline string key takeaways
        section = ThematicSection(
            theme_id="t1",
            title="Theme 1",
            synthesis_prose="Prose",
            key_takeaways="- Transformer scaling improves generalization\n* Sparse attention reduces memory footprint\n• Hybrid search outperforms dense alone",
        )
        assert len(section.key_takeaways) == 3
        assert "Transformer scaling improves generalization" in section.key_takeaways[0]
        assert "Sparse attention reduces memory footprint" in section.key_takeaways[1]
        assert "Hybrid search outperforms dense alone" in section.key_takeaways[2]


# ============================================================================
# 3. Master Research Report & Serialization Round-Tripping
# ============================================================================

@pytest.mark.unit
class TestResearchReportFullRoundtripAndMarkdown:
    """Verify end-to-end Pydantic v2 serialization round-tripping and Markdown export."""

    def test_complete_research_report_roundtrip(self):
        """Test full nested ResearchReport serialization to JSON and back."""
        metadata = ReportMetadata(
            project_id="proj_deep_001",
            user_id="researcher_42",
            title="Next-Generation Multi-Agent Systems for Autonomous Scientific Discovery",
            research_question="How do autonomous multi-agent reasoning architectures enhance literature synthesis?",
            generated_at=datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc),
            pipeline_duration_seconds=42.5,
            status=ReportStatus.COMPLETE,
            quality_score=94.5,
            papers_analyzed_full_text=15,
            total_citations=20,
            llm_calls_made=28,
            tokens_consumed=145000,
            models_used=["gemini-2.0-flash", "deepseek-reasoner"],
        )

        comparison_matrix = [
            EvidenceMatrixRow(
                paper_id="ref_1",
                title="Scaling Laws for Neural Language Models",
                authors=["J. Kaplan", "S. McCandlish", "T. Henighan"],
                year=2020,
                methodology="Empirical power-law scaling analysis across parameter count and dataset size",
                benchmark_dataset="WebText2, CommonCrawl",
                primary_metric="Loss: L(N) = (N_c / N)^alpha_N with alpha_N = 0.076",
                primary_limitation="Evaluated primarily on autoregressive cross-entropy, not reasoning tasks",
                is_full_text=True,
            ),
            EvidenceMatrixRow(
                paper_id="ref_2",
                title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
                authors=["J. Wei", "X. Wang", "D. Schuurmans"],
                year=2022,
                methodology="Step-by-step rationalization prompting mechanism",
                benchmark_dataset="GSM8K, SVAMP, MultiArith",
                primary_metric="GSM8K Accuracy: 58.1% (PaLM 540B)",
                primary_limitation="Requires sufficient model scale (>100B params) to emerge",
                is_full_text=True,
            ),
        ]

        thematic_sections = [
            ThematicSection(
                theme_id="theme_scaling",
                title="Empirical Foundations of Model Scaling",
                synthesis_prose="Early discoveries established predictable power-law scaling [ref_1#sec_results_2]. Furthermore, step-by-step reasoning emerged as parameter scale expanded [ref_2#sec_methodology_1].",
                key_takeaways=[
                    "Power laws govern performance predictable over 8 orders of magnitude",
                    "Reasoning capabilities require emergent reasoning scaffolds",
                ],
                cited_paper_ids=["ref_1", "ref_2"],
            )
        ]

        conflicting_debates = [
            ConflictingDebate(
                topic="Emergence vs. Continuous Metric Nonlinearity",
                perspective_a="Abrupt emergent abilities exist as distinct phase transitions in large models",
                perspective_b="Emergence is a mirage caused by discontinuous non-linear evaluation metrics",
                critical_evaluation="Continuous metrics demonstrate smooth progress, while task-level accuracy exhibits step-function thresholds.",
            )
        ]

        actionable_gaps = [
            ResearchGapItem(
                gap_id="gap_auditing",
                description="Lack of deterministic NLI-grounded citation fact-checking in autonomous review systems",
                importance="high",
                recommended_methodology="Integrate section-anchored NLI verification verifying claims against parsed full-text chunks",
                grounding_paper_ids=["ref_1", "ref_2"],
            )
        ]

        methodology_overview = MethodologyDistribution(
            distribution={"Empirical Scaling": 12, "Prompt Engineering": 8},
            dominant_approach="Empirical Scaling",
            trend_description="A clear transition from heuristic prompting toward architectural supervisor graphs.",
        )

        bibliography = [
            BibliographyItem(
                paper_id="ref_1",
                title="Scaling Laws for Neural Language Models",
                authors=["J. Kaplan", "S. McCandlish", "T. Henighan"],
                year=2020,
                venue="arXiv:2001.08361",
                doi="10.48550/arXiv.2001.08361",
                pdf_url="https://arxiv.org/pdf/2001.08361.pdf",
                citation_count=3500,
                is_full_text_analyzed=True,
            ),
            BibliographyItem(
                paper_id="ref_2",
                title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
                authors=["J. Wei", "X. Wang", "D. Schuurmans"],
                year=2022,
                venue="NeurIPS 2022",
                doi="10.48550/arXiv.2201.11903",
                pdf_url="https://arxiv.org/pdf/2201.11903.pdf",
                citation_count=4200,
                is_full_text_analyzed=True,
            ),
        ]

        report = ResearchReport(
            metadata=metadata,
            executive_summary="This report provides a comprehensive scientific review of autonomous reasoning systems.",
            comparison_matrix=comparison_matrix,
            thematic_sections=thematic_sections,
            conflicting_findings_and_debates=conflicting_debates,
            actionable_research_gaps=actionable_gaps,
            methodology_overview=methodology_overview,
            bibliography=bibliography,
        )

        # Serialize to JSON and deserialize back
        report_json = report.model_dump_json()
        loaded_report = ResearchReport.model_validate_json(report_json)

        assert loaded_report.metadata.project_id == "proj_deep_001"
        assert loaded_report.metadata.quality_score == 94.5
        assert len(loaded_report.comparison_matrix) == 2
        assert len(loaded_report.thematic_sections) == 1
        assert len(loaded_report.conflicting_findings_and_debates) == 1
        assert len(loaded_report.actionable_research_gaps) == 1
        assert len(loaded_report.bibliography) == 2

        # Test Markdown generation
        md = loaded_report.to_markdown()
        assert "# Next-Generation Multi-Agent Systems" in md
        assert "## Evidence Comparison Matrix" in md
        assert "| ref_1 | Scaling Laws" in md
        assert "## Conflicting Findings & Scientific Debates" in md
        assert "## Actionable Research Gaps & Future Directions" in md
        assert "## Bibliography" in md


# ============================================================================
# 4. Citation Auditor & Critic Evaluation Models
# ============================================================================

@pytest.mark.unit
class TestReasoningAndAuditingSchemas:
    """Verify NLI Proposition Verification, Citation Audit Report, and Critic Evaluation models."""

    def test_proposition_verification_and_nli_verdicts(self):
        """Test PropositionVerification model with different NLIVerdict enums."""
        verif_entailed = PropositionVerification(
            proposition="Scaling model parameters reduces cross-entropy loss predictably.",
            citation_anchor="ref_1#sec_results_2",
            paper_id="ref_1",
            section_anchor="sec_results_2",
            grounding_chunk_id="chunk_001_abc",
            grounding_text="We find that cross-entropy loss scales as a power law with model size.",
            verdict=NLIVerdict.ENTAILMENT,
            confidence=0.98,
            reasoning="The excerpt explicitly substantiates the power-law reduction in loss.",
        )
        assert verif_entailed.verdict == NLIVerdict.ENTAILMENT
        assert verif_entailed.confidence == 0.98

        verif_contradicted = PropositionVerification(
            proposition="Transformers cannot generalize beyond 10k tokens.",
            citation_anchor="ref_2#sec_3",
            paper_id="ref_2",
            verdict=NLIVerdict.CONTRADICTION,
            confidence=0.95,
            reasoning="The source paper demonstrates successful context extrapolation up to 100k tokens.",
            suggested_correction="Transformers can extrapolate to longer contexts using position interpolation.",
        )
        assert verif_contradicted.verdict == NLIVerdict.CONTRADICTION
        assert verif_contradicted.suggested_correction is not None

        # Build full CitationAuditReport
        audit_report = CitationAuditReport(
            total_propositions=2,
            entailed_count=1,
            neutral_count=0,
            contradiction_count=1,
            precision_score=50.0,
            verifications=[verif_entailed, verif_contradicted],
            hallucinated_anchors=[],
            audit_passed=False,
        )
        assert audit_report.audit_passed is False
        assert audit_report.precision_score == 50.0

        # Roundtrip JSON
        loaded_audit = CitationAuditReport.model_validate_json(audit_report.model_dump_json())
        assert len(loaded_audit.verifications) == 2
        assert loaded_audit.verifications[1].verdict == NLIVerdict.CONTRADICTION

    def test_critic_evaluation_and_refinement_trigger(self):
        """Test CriticEvaluation model and refinement loop condition."""
        dimension_scores = [
            CriticDimensionScore(
                dimension="Factual Grounding",
                score=65.0,
                justification="Several claims in Section 2 lack direct anchor citations.",
            ),
            CriticDimensionScore(
                dimension="Methodological Depth",
                score=70.0,
                justification="Comparative trade-offs between dense and sparse attention are not analyzed.",
            ),
            CriticDimensionScore(
                dimension="Research Gap Grounding",
                score=72.0,
                justification="Gaps are not directly linked to paper limitations.",
            ),
        ]

        critic_eval = CriticEvaluation(
            overall_score=69.0,
            dimension_scores=dimension_scores,
            strengths=["Clear narrative flow", "Good coverage of foundational papers"],
            weaknesses=["Insufficient anchor density", "Superficial gap descriptions"],
            refinement_guidance=[
                "Add [ref_X#secY] anchors to all empirical claims in Section 2",
                "Explicitly link Gap 1 to limitations in ref_2",
            ],
            should_refine=True,
        )

        assert critic_eval.overall_score == 69.0
        assert critic_eval.should_refine is True
        assert len(critic_eval.dimension_scores) == 3

        loaded_eval = CriticEvaluation.model_validate_json(critic_eval.model_dump_json())
        assert loaded_eval.overall_score < 75.0
        assert loaded_eval.should_refine is True


# ============================================================================
# 5. Gemini OpenAPI Schema & Provider Verification
# ============================================================================

@pytest.mark.unit
class TestOpenAPISchemaAndProviderIntegration:
    """Verify OpenAPI 3.0 schema conversion for Gemini and LLM provider interfaces."""

    def test_to_gemini_schema_recursive_defs_resolution(self):
        """Verify that to_gemini_schema completely inlines all $defs for complex nested schemas."""
        schema_dict = to_gemini_schema(ResearchReport)

        # Ensure no un-inlined $ref remains
        schema_json_str = json.dumps(schema_dict)
        assert "$ref" not in schema_json_str
        assert "$defs" not in schema_dict

        # Verify key properties exist at top level
        props = schema_dict.get("properties", {})
        assert "metadata" in props
        assert "executive_summary" in props
        assert "comparison_matrix" in props
        assert "thematic_sections" in props
        assert "conflicting_findings_and_debates" in props
        assert "actionable_research_gaps" in props
        assert "methodology_overview" in props
        assert "bibliography" in props

        # Check nested property structure inside comparison_matrix items
        matrix_items = props["comparison_matrix"].get("items", {})
        assert "properties" in matrix_items
        assert "paper_id" in matrix_items["properties"]
        assert "methodology" in matrix_items["properties"]

    def test_mock_llm_client_fifo_queue_and_history(self):
        """Test MockLLMClient FIFO text responses and invocation call history tracking."""
        mock_client = MockLLMClient()
        mock_client.mock_text_responses = [
            "Response 1 for testing",
            "Response 2 for testing",
        ]

        resp1 = mock_client.generate_text("Prompt 1", system_prompt="Sys 1", model_tier=ModelTier.FAST)
        assert resp1 == "Response 1 for testing"

        resp2 = mock_client.generate_text("Prompt 2", system_prompt="Sys 2", model_tier=ModelTier.REASONING)
        assert resp2 == "Response 2 for testing"

        # Third call falls back to synthetic scientific response
        resp3 = mock_client.generate_text("Prompt 3")
        assert "Synthetic Scientific Response" in resp3

        # Verify call history
        assert len(mock_client.call_history) == 3
        assert mock_client.call_history[0]["prompt"] == "Prompt 1"
        assert mock_client.call_history[1]["tier"] == ModelTier.REASONING

    def test_mock_llm_client_structured_output_custom_override(self):
        """Test MockLLMClient generate_structured with custom pre-configured response."""
        mock_client = MockLLMClient()
        custom_gap = ResearchGapItem(
            gap_id="gap_custom",
            description="Custom mock gap",
            importance="high",
            recommended_methodology="Custom methodology",
            grounding_paper_ids=["ref_test"],
        )
        mock_client.mock_structured_responses[ResearchGapItem] = custom_gap

        result = mock_client.generate_structured("Identify gaps", ResearchGapItem)
        assert result.gap_id == "gap_custom"
        assert result.grounding_paper_ids == ["ref_test"]
