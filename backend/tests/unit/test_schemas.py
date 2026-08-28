# Unit Tests for Pydantic v2 Contract Schemas
# Covers Tier 1 & Tier 2 schema validation, serialization, edge cases, and constraints

from datetime import datetime
import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from agents import schemas
from agents.schemas import (
    AnalyzerInput,
    AnalyzerOutput,
    Citation,
    MethodologyOverview,
    PaperAnalysis,
    PaperInsight,
    PlannerInput,
    PlannerOutput,
    QualityIndicators,
    RankedPaper,
    ReportMetadata,
    ReportSection,
    ReportStatistics,
    ReportStatus,
    ResearchGap,
    ResearchReport,
    RetrieverInput,
    RetrieverOutput,
    SearchStrategy,
    SynthesizerInput,
    SynthesizerOutput,
    Theme,
    YearDistribution,
)

# Optional imports for new target schemas (Phase 1 / v3.2 architecture)
EvidenceMatrixRow = getattr(schemas, "EvidenceMatrixRow", None)
ThematicSection = getattr(schemas, "ThematicSection", None)
ConflictingDebate = getattr(schemas, "ConflictingDebate", None)
ResearchGapItem = getattr(schemas, "ResearchGapItem", None)
MethodologyDistribution = getattr(schemas, "MethodologyDistribution", None)
BibliographyItem = getattr(schemas, "BibliographyItem", None)


@pytest.mark.unit
class TestLegacyAndAgentSchemas:
    """Test suite for existing agent schemas and backwards compatibility."""

    def test_report_status_enum(self):
        """Verify ReportStatus enum values."""
        assert ReportStatus.COMPLETE.value == "complete"
        assert ReportStatus.PARTIAL.value == "partial"
        assert ReportStatus.ANALYSIS_ONLY.value == "analysis_only"
        assert ReportStatus.ERROR.value == "error"

    def test_citation_instantiation_and_serialization(self):
        """Verify Citation schema serialization and field types."""
        citation = Citation(
            paper_id="paper_001",
            title="Transformer Scaling Laws in Scientific Discovery",
            authors=["A. Turing", "C. Shannon"],
            year=2024,
            url="https://arxiv.org/abs/2401.00001",
            source="arXiv",
            relevance_score=95,
            citation_count=42,
            abstract_snippet="We explore the scaling behavior of transformers on scientific data...",
        )
        assert citation.paper_id == "paper_001"
        assert citation.year == 2024
        assert len(citation.authors) == 2

        # Test JSON roundtrip
        json_str = citation.model_dump_json()
        loaded = Citation.model_validate_json(json_str)
        assert loaded.paper_id == citation.paper_id
        assert loaded.relevance_score == 95

    def test_citation_missing_required_field_raises(self):
        """Verify ValidationError when required fields are missing."""
        with pytest.raises(ValidationError):
            Citation(
                paper_id="paper_001",
                # missing title
                authors=["A. Turing"],
                url="https://arxiv.org/abs/2401.00001",
                source="arXiv",
                relevance_score=90,
                abstract_snippet="Snippet...",
            )

    def test_research_gap_coerce_to_str_list(self):
        """Verify BeforeValidator coerces integer paper IDs to strings."""
        gap = ResearchGap(
            description="Lack of benchmark datasets for multi-hop scientific reasoning",
            importance="high",
            potential_directions=["Construct novel graph reasoning benchmarks"],
            related_paper_ids=[101, 102, "paper_003"],  # type: ignore
        )
        assert gap.related_paper_ids == ["101", "102", "paper_003"]

    def test_paper_analysis_relevance_score_bounds(self):
        """Verify PaperAnalysis relevance score constraint [0, 100]."""
        valid_analysis = PaperAnalysis(
            paper_id="p1",
            title="Valid Title",
            relevance_score=85,
            key_findings=["Finding 1", "Finding 2"],
            methodology="Quantitative empirical study",
            limitations=["Sample size limited to 100"],
            contribution="Novel benchmark",
            themes=["Architecture", "Evaluation"],
        )
        assert valid_analysis.relevance_score == 85

        # Score > 100 must fail
        with pytest.raises(ValidationError):
            PaperAnalysis(
                paper_id="p1",
                title="Invalid",
                relevance_score=105,
                key_findings=["F1"],
                methodology="M1",
                limitations=["L1"],
                contribution="C1",
                themes=["T1"],
            )

        # Score < 0 must fail
        with pytest.raises(ValidationError):
            PaperAnalysis(
                paper_id="p1",
                title="Invalid",
                relevance_score=-5,
                key_findings=["F1"],
                methodology="M1",
                limitations=["L1"],
                contribution="C1",
                themes=["T1"],
            )

    def test_planner_output_structure(self):
        """Verify PlannerOutput with nested SearchStrategy."""
        strategy = SearchStrategy(
            primary_keywords=["transformers", "graph reasoning"],
            secondary_keywords=["multi-agent", "citations"],
            sources=["arXiv", "Semantic Scholar"],
            max_papers_per_source=10,
        )
        planner_out = PlannerOutput(
            keywords=["transformers", "agents", "reasoning"],
            subtopics=["Model Architecture", "Benchmark Results", "Open Challenges"],
            search_strategy=strategy,
        )
        assert len(planner_out.keywords) == 3
        assert planner_out.search_strategy.max_papers_per_source == 10
        data = planner_out.model_dump()
        assert data["search_strategy"]["sources"] == ["arXiv", "Semantic Scholar"]

    def test_analyzer_output_roundtrip(self):
        """Verify AnalyzerOutput serialization."""
        analysis = PaperAnalysis(
            paper_id="p1",
            title="Analysis Paper",
            relevance_score=90,
            key_findings=["Finding A"],
            methodology="Empirical",
            limitations=["None"],
            contribution="Breakthrough",
            themes=["Theme A"],
        )
        theme = Theme(
            name="Theme A",
            description="Theme description",
            paper_count=1,
            paper_ids=["p1"],
            strength="strong",
        )
        analyzer_out = AnalyzerOutput(
            paper_analyses=[analysis],
            cross_cutting_themes=[theme],
            methodology_distribution={"empirical": 1},
            high_quality_count=1,
            total_analyzed=1,
        )
        assert analyzer_out.total_analyzed == 1
        json_str = analyzer_out.model_dump_json()
        restored = AnalyzerOutput.model_validate_json(json_str)
        assert restored.high_quality_count == 1
        assert restored.paper_analyses[0].paper_id == "p1"


@pytest.mark.unit
class TestTargetV32ContractSchemas:
    """Test suite for target v3.2 architecture contracts defined in PROJECT.md."""

    def test_evidence_matrix_row_contract(self):
        """Verify EvidenceMatrixRow instantiation, types, defaults, and edge cases."""
        if EvidenceMatrixRow is None:
            pytest.skip("EvidenceMatrixRow not yet implemented in backend/agents/schemas.py")

        row = EvidenceMatrixRow(
            paper_id="doi:10.1000/182",
            title="Scaling Laws for Autonomous Scientific Synthesis",
            authors=["A. Turing", "J. von Neumann"],
            year=2024,
            methodology="Dual-track supervisor StateGraph with section-aware RAG",
            benchmark_dataset="PubMed-QA & SciFact",
            primary_metric="94.6% Accuracy on Claim Verification",
            primary_limitation="Requires OCR pre-processing on scanned PDFs",
            is_full_text=True,
        )
        assert row.paper_id == "doi:10.1000/182"
        assert row.is_full_text is True
        assert row.year == 2024

        # Test JSON roundtrip
        json_bytes = row.model_dump_json()
        loaded = EvidenceMatrixRow.model_validate_json(json_bytes)
        assert loaded.paper_id == row.paper_id
        assert loaded.primary_metric == row.primary_metric

    def test_evidence_matrix_row_invalid_types_raise(self):
        """Verify EvidenceMatrixRow fails when required fields are missing."""
        if EvidenceMatrixRow is None:
            pytest.skip("EvidenceMatrixRow not yet implemented in backend/agents/schemas.py")

        with pytest.raises(ValidationError):
            EvidenceMatrixRow(
                paper_id="doi:10.1000/182",
                # missing title, authors, methodology, etc.
            )

    def test_thematic_section_with_citation_anchors(self):
        """Verify ThematicSection handles dense citation anchors [ref_X#secY]."""
        if ThematicSection is None:
            pytest.skip("ThematicSection not yet implemented in backend/agents/schemas.py")

        section = ThematicSection(
            theme_id="theme_01",
            title="Multi-Agent Supervisor Architectures",
            synthesis_prose=(
                "Autonomous multi-agent architectures achieve superior reasoning [ref_1#sec2] "
                "by separating retrieval from adversarial verification [ref_2#sec4]."
            ),
            key_takeaways=[
                "Separation of concerns reduces context dilution.",
                "Adversarial critique improves statistical rigor.",
            ],
            cited_paper_ids=["ref_1", "ref_2"],
        )
        assert "[ref_1#sec2]" in section.synthesis_prose
        assert len(section.key_takeaways) == 2
        assert section.cited_paper_ids == ["ref_1", "ref_2"]

        # Serialization
        dumped = section.model_dump()
        assert dumped["theme_id"] == "theme_01"

    def test_conflicting_debate_contract(self):
        """Verify ConflictingDebate model instantiation and validation."""
        if ConflictingDebate is None:
            pytest.skip("ConflictingDebate not yet implemented in backend/agents/schemas.py")

        debate = ConflictingDebate(
            topic="Dense Vector Retrieval vs Hybrid BM25+Dense for Long-Context Scientific Synthesis",
            perspective_a="Dense vector embeddings capture semantic nuance across interdisciplinary terminology.",
            perspective_b="BM25 keyword indices prevent catastrophic false positives on specific gene/chemical nomenclature.",
            critical_evaluation="Empirical consensus indicates Reciprocal Rank Fusion (RRF k=60) reconciles both paradigms.",
        )
        assert debate.topic.startswith("Dense Vector Retrieval")
        assert len(debate.critical_evaluation) > 20

        # Roundtrip
        restored = ConflictingDebate.model_validate_json(debate.model_dump_json())
        assert restored.topic == debate.topic

    def test_research_gap_item_importance_literal(self):
        """Verify ResearchGapItem importance allows Literal['high', 'medium', 'low'] and rejects others."""
        if ResearchGapItem is None:
            pytest.skip("ResearchGapItem not yet implemented in backend/agents/schemas.py")

        # Valid importances
        for imp in ["high", "medium", "low"]:
            gap = ResearchGapItem(
                gap_id="gap_01",
                description="Evaluation on non-English scientific corpora is missing.",
                importance=imp,  # type: ignore
                recommended_methodology="Construct cross-lingual scientific benchmark using multilingual sentence transformers.",
                grounding_paper_ids=["paper_01", "paper_02"],
            )
            assert gap.importance == imp

        # Invalid importance must raise ValidationError
        with pytest.raises(ValidationError):
            ResearchGapItem(
                gap_id="gap_02",
                description="Invalid importance value test.",
                importance="critical",  # type: ignore
                recommended_methodology="Roadmap...",
                grounding_paper_ids=["paper_01"],
            )

    def test_methodology_distribution_contract(self):
        """Verify MethodologyDistribution model."""
        if MethodologyDistribution is None:
            pytest.skip("MethodologyDistribution not yet implemented in backend/agents/schemas.py")

        dist = MethodologyDistribution(
            distribution={"quantitative_empirical": 12, "theoretical_proofs": 4, "meta_review": 2},
            dominant_approach="quantitative_empirical",
            trend_description="A 40% shift towards empirical transformer evaluations since 2023.",
        )
        assert dist.distribution["quantitative_empirical"] == 12
        assert dist.dominant_approach == "quantitative_empirical"

    def test_bibliography_item_contract(self):
        """Verify BibliographyItem model defaults and serialization."""
        if BibliographyItem is None:
            pytest.skip("BibliographyItem not yet implemented in backend/agents/schemas.py")

        item = BibliographyItem(
            paper_id="paper_001",
            title="Attention Is All You Need",
            authors=["A. Vaswani", "N. Shazeer", "N. Parmar"],
            year=2017,
            venue="NeurIPS",
            doi="10.48550/arXiv.1706.03762",
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
            is_full_text_analyzed=True,
        )
        assert item.doi is not None
        assert item.is_full_text_analyzed is True

        # Optional fields default handling
        minimal_item = BibliographyItem(
            paper_id="paper_002",
            title="Minimal Paper",
            authors=["Author A"],
        )
        assert minimal_item.year is None
        assert minimal_item.doi is None
        assert minimal_item.is_full_text_analyzed is True

    def test_report_metadata_validation(self):
        """Verify ReportMetadata validation, datetime handling, and status enum."""
        now = datetime.utcnow()
        meta = ReportMetadata(
            project_id="proj_123",
            user_id="user_456",
            title="Autonomous Literature Synthesis",
            research_question="How do multi-agent DAGs improve literature review quality?",
            generated_at=now,
            pipeline_duration_seconds=34.5,
            status=ReportStatus.COMPLETE,
            llm_calls_made=12,
            tokens_consumed=45000,
            models_used=["gemini-2.0-flash", "deepseek-r1"],
        )
        assert meta.project_id == "proj_123"
        assert meta.pipeline_duration_seconds == 34.5
        assert meta.status == ReportStatus.COMPLETE

        # JSON serialization preserves ISO format
        json_data = json.loads(meta.model_dump_json())
        assert json_data["status"] == "complete"
        assert json_data["pipeline_duration_seconds"] == 34.5

    def test_complete_research_report_nested_roundtrip(self):
        """Verify complete ResearchReport instantiation and deep JSON serialization."""
        now = datetime.utcnow()
        metadata = ReportMetadata(
            project_id="proj_001",
            user_id="user_001",
            title="Comprehensive Survey on LLM Reasoning",
            research_question="What are the prevailing methodologies for LLM reasoning evaluation?",
            generated_at=now,
            pipeline_duration_seconds=45.2,
            status=ReportStatus.COMPLETE,
            llm_calls_made=8,
            tokens_consumed=32000,
            models_used=["gemini-2.0-flash"],
        )

        section = ReportSection(
            title="Chain-of-Thought Reasoning",
            content="Chain-of-thought prompting enables step-by-step reasoning across complex benchmarks.",
            key_insight="Step-by-step reasoning significantly enhances multi-step problem solving.",
            paper_ids=["paper_01"],
            word_count=15,
        )

        theme = Theme(
            name="Prompting Paradigms",
            description="Explores prompting techniques vs fine-tuning.",
            paper_count=3,
            paper_ids=["paper_01", "paper_02", "paper_03"],
            strength="strong",
        )

        gap = ResearchGap(
            description="Lack of standardized multi-hop reasoning datasets.",
            importance="high",
            potential_directions=["Develop synthetic reasoning benchmarks."],
            related_paper_ids=["paper_01"],
        )

        methodology = MethodologyOverview(
            distribution={"empirical": 5, "theoretical": 2},
            dominant_approach="empirical",
            trend_description="Rapid growth in benchmark-driven evaluations.",
        )

        insight = PaperInsight(
            paper_id="paper_01",
            title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            relevance_score=95,
            key_findings=["CoT improves math word problem performance."],
            methodology="Empirical prompting on GSM8K and SVAMP.",
            limitations=["CoT only benefits models with >100B parameters."],
            contribution="Introduced Chain-of-Thought prompting.",
            themes=["Prompting Paradigms"],
            url="https://arxiv.org/abs/2201.11903",
        )

        stats = ReportStatistics(
            total_papers_found=20,
            total_after_dedup=15,
            papers_analyzed=10,
            high_relevance_count=8,
            avg_relevance_score=86.4,
            year_distribution=[YearDistribution(year="2022", count=5), YearDistribution(year="2023", count=5)],
            source_distribution={"arXiv": 10},
            methodology_distribution={"empirical": 8, "theoretical": 2},
            top_keywords=["reasoning", "prompting", "transformers"],
        )

        citation = Citation(
            paper_id="paper_01",
            title="Chain-of-Thought Prompting",
            authors=["J. Wei", "X. Wang"],
            year=2022,
            url="https://arxiv.org/abs/2201.11903",
            source="arXiv",
            relevance_score=95,
            abstract_snippet="We explore how generating a chain of thought...",
        )

        quality = QualityIndicators(
            has_executive_summary=True,
            has_all_sections=True,
            section_count=1,
            papers_with_full_analysis=1,
            papers_with_partial_analysis=0,
            budget_exhausted=False,
            synthesis_model_used="gemini-2.0-flash",
        )

        report = ResearchReport(
            metadata=metadata,
            executive_summary="This report provides a comprehensive overview of LLM reasoning paradigms.",
            sections=[section],
            themes=[theme],
            research_gaps=[gap],
            methodology_overview=methodology,
            paper_insights=[insight],
            statistics=stats,
            bibliography=[citation],
            quality_indicators=quality,
        )

        assert report.metadata.project_id == "proj_001"
        assert len(report.sections) == 1
        assert len(report.themes) == 1
        assert report.statistics.avg_relevance_score == 86.4

        # Full JSON roundtrip
        json_report = report.model_dump_json()
        restored = ResearchReport.model_validate_json(json_report)
        assert restored.metadata.project_id == "proj_001"
        assert restored.quality_indicators.has_executive_summary is True
        assert restored.paper_insights[0].relevance_score == 95


@pytest.mark.unit
class TestSchemaAdversarialEdgeCases:
    """Adversarial and boundary stress tests for schemas."""

    def test_unicode_and_latex_math_in_prose(self):
        """Verify schemas accept Unicode, emoji, and LaTeX mathematical expressions in text fields."""
        latex_text = r"The loss $\mathcal{L}_{total} = \sum_{i=1}^N \alpha_i \cdot \text{KL}(p_i \parallel q_i)$ is optimized."
        unicode_title = "Quantum Entanglement in ℂⁿ: A Study of Schrödinger's Operators 🔬"

        section = ReportSection(
            title=unicode_title,
            content=latex_text,
            key_insight="Key LaTeX formula: $E=mc^2$",
            paper_ids=["p_unicode_01"],
            word_count=20,
        )
        assert "ℂⁿ" in section.title
        assert r"\mathcal{L}" in section.content
        assert "🔬" in section.title

        dumped = section.model_dump_json()
        restored = ReportSection.model_validate_json(dumped)
        assert restored.title == unicode_title
        assert restored.content == latex_text

    def test_empty_lists_and_minimal_strings(self):
        """Verify schemas handle empty lists and empty strings where type-valid."""
        stats = ReportStatistics(
            total_papers_found=0,
            total_after_dedup=0,
            papers_analyzed=0,
            high_relevance_count=0,
            avg_relevance_score=0.0,
            year_distribution=[],
            source_distribution={},
            methodology_distribution={},
            top_keywords=[],
        )
        assert stats.total_papers_found == 0
        assert stats.year_distribution == []

    def test_extreme_size_strings(self):
        """Verify schema handles large multi-paragraph text payloads without crash."""
        large_prose = "Scientific analysis paragraph. " * 500  # ~15,000 chars
        section = ReportSection(
            title="Large Text Section",
            content=large_prose,
            key_insight="Summary insight.",
            paper_ids=["p1", "p2"],
            word_count=1500,
        )
        assert len(section.content) > 10000
        json_str = section.model_dump_json()
        assert len(json_str) > 10000
        restored = ReportSection.model_validate_json(json_str)
        assert restored.word_count == 1500
