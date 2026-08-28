"""
Pydantic v2 Contract Schemas for Scholar Agent.
Defines strict data models for evidence extraction, thematic synthesis,
conflicting debates, research gaps, citation auditing, critic review, and research reports.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator


# ============================================================================
# Core Enums
# ============================================================================


class ReportStatus(StrEnum):
    """Execution outcome status for research reports."""

    COMPLETE = "complete"  # Full pipeline completed with high quality synthesis
    COMPLETED = "complete"  # Alias for complete
    PARTIAL = "partial"  # Pipeline completed with partial synthesis
    ANALYSIS_ONLY = "analysis_only"  # Extraction/matrix complete, synthesis omitted
    ERROR = "error"  # Unrecoverable pipeline error
    IN_PROGRESS = "in_progress"  # Currently executing in pipeline
    EMPTY = "empty"  # Initial state before execution



class GapImportance(StrEnum):
    """Significance / priority level for identified research gaps."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NLIVerdict(StrEnum):
    """Natural Language Inference verdict for citation auditing."""

    ENTAILMENT = "ENTAILMENT"
    NEUTRAL = "NEUTRAL"
    CONTRADICTION = "CONTRADICTION"


class SectionType(StrEnum):
    """Section classification for hierarchical document chunking and anchoring."""

    ABSTRACT = "ABSTRACT"
    INTRODUCTION = "INTRODUCTION"
    METHODOLOGY = "METHODOLOGY"
    RESULTS = "RESULTS"
    LIMITATIONS = "LIMITATIONS"
    TABLES = "TABLES"
    GENERAL = "GENERAL"


# ============================================================================
# Helper Coercion Functions
# ============================================================================


def _coerce_to_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        if "," in v:
            return [a.strip() for a in v.split(",") if a.strip()]
        if ";" in v:
            return [a.strip() for a in v.split(";") if a.strip()]
        return [v.strip()] if v.strip() else []
    if isinstance(v, (list, tuple, set)):
        return [str(item).strip() for item in v if str(item).strip()]
    return [str(v)]


# ============================================================================
# Target v3.2 / Milestone 1 Component Models
# ============================================================================


class EvidenceMatrixRow(BaseModel):
    """
    Uniform comparative extraction row for an individual paper in the Evidence Matrix.
    Captures problem formulation, architecture/method, benchmarks, quantitative metrics, and key limitations.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    paper_id: str = Field(
        ...,
        description="Unique paper identifier (e.g. arXiv ID, DOI, or canonical ref key like 'ref_1')",
    )
    title: str = Field(..., description="Full title of the scientific paper")
    authors: list[str] = Field(
        default_factory=list,
        description="List of author full names",
    )
    year: int | None = Field(
        default=None,
        description="Publication year (e.g. 2024)",
    )
    methodology: str = Field(
        ...,
        description="Concise summary of the core methodology, architecture, algorithm, or theoretical framework",
    )
    benchmark_dataset: str = Field(
        ...,
        description="Primary benchmark datasets, environments, or corpora used for empirical evaluation",
    )
    primary_metric: str = Field(
        ...,
        description="Key quantitative metric and score achieved (e.g., 'Accuracy: 88.4%', 'BLEU: 34.2')",
    )
    primary_limitation: str = Field(
        ...,
        description="Primary stated or identified limitation, bottleneck, computational constraint, or failure mode",
    )
    is_full_text: bool = Field(
        default=False,
        description="True if analysis was extracted from parsed full-text PDF; False if extracted from abstract metadata",
    )
    doi: str | None = Field(default=None, description="DOI identifier of paper")
    arxiv_id: str | None = Field(default=None, description="arXiv identifier of paper")
    url: str | None = Field(default=None, description="Origin URL or landing page of paper")

    @field_validator("authors", mode="before")
    @classmethod
    def coerce_authors(cls, v: Any) -> list[str]:
        return _coerce_to_str_list(v)

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, v: Any) -> int | None:
        if v is None or v == "":
            return None
        if isinstance(v, int):
            return v if 1800 <= v <= 2100 else None
        if isinstance(v, float):
            v_int = int(v)
            return v_int if 1800 <= v_int <= 2100 else None
        if isinstance(v, str):
            match = re.search(r"\b(19\d\d|20\d\d)\b", v)
            if match:
                return int(match.group(1))
            try:
                v_int = int(float(v.strip()))
                return v_int if 1800 <= v_int <= 2100 else None
            except ValueError:
                return None
        return None

    @field_validator(
        "paper_id",
        "title",
        "methodology",
        "benchmark_dataset",
        "primary_metric",
        "primary_limitation",
        mode="before",
    )
    @classmethod
    def clean_strings(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()


class ThematicSection(BaseModel):
    """
    Synthesized review section organized around a core subtopic or conceptual theme.
    Contains rigorous narrative prose grounded in citations using anchor format [ref_X#secY].
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    theme_id: str = Field(
        ...,
        description="Unique theme / section identifier (e.g. 'theme_1', 'sec_architectures')",
    )
    title: str = Field(..., description="Section title or theme heading")
    synthesis_prose: str = Field(
        ...,
        description="Dense narrative synthesis citing evidence using anchor format [ref_X#secY]",
    )
    key_takeaways: list[str] = Field(
        default_factory=list,
        description="Key bulleted takeaways or actionable findings for this theme",
    )
    cited_paper_ids: list[str] = Field(
        default_factory=list,
        description="List of paper_ids cited and referenced in this section",
    )

    @field_validator("cited_paper_ids", mode="before")
    @classmethod
    def coerce_cited_paper_ids(cls, v: Any) -> list[str]:
        return _coerce_to_str_list(v)

    @field_validator("key_takeaways", mode="before")
    @classmethod
    def coerce_key_takeaways(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            lines = [
                line.strip().lstrip("-*•123456789.) ")
                for line in v.splitlines()
                if line.strip()
            ]
            return lines if lines else [v.strip()]
        if isinstance(v, (list, tuple, set)):
            return [str(item).strip() for item in v if str(item).strip()]
        return [str(v)]

    def extract_citation_anchors(self) -> list[str]:
        """Extract all [ref_X#secY] or [ref_X] anchors embedded in synthesis_prose."""
        return re.findall(r"\[ref_[^\]]+\]", self.synthesis_prose)


class ConflictingDebate(BaseModel):
    """
    Structured representation of scientific controversies, competing hypotheses, or conflicting empirical results.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    topic: str = Field(
        ...,
        description="Specific area of scientific controversy, trade-off, or competing paradigm",
    )
    perspective_a: str = Field(
        ...,
        description="First viewpoint/hypothesis, including supporting methodologies and evidence",
    )
    perspective_b: str = Field(
        ...,
        description="Opposing viewpoint/hypothesis, including contrasting methodologies and evidence",
    )
    critical_evaluation: str = Field(
        ...,
        description="Critical comparative analysis resolving or evaluating the underlying tension and empirical conditions",
    )


class ResearchGapItem(BaseModel):
    """
    Actionable and grounded research gap identifying open challenges and proposed solutions.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    gap_id: str = Field(
        ...,
        description="Unique gap identifier (e.g. 'gap_1', 'gap_scalability')",
    )
    description: str = Field(
        ...,
        description="Detailed description of the unaddressed research challenge or missing exploration",
    )
    importance: Literal["high", "medium", "low"] = Field(
        default="high",
        description="Significance/impact rating of the research gap ('high', 'medium', 'low')",
    )
    recommended_methodology: str = Field(
        ...,
        description="Concrete suggested methodological approach or experimental framework to address this gap",
    )
    grounding_paper_ids: list[str] = Field(
        default_factory=list,
        description="List of paper_ids whose limitations or findings substantiate this gap",
    )

    @field_validator("grounding_paper_ids", mode="before")
    @classmethod
    def coerce_grounding_paper_ids(cls, v: Any) -> list[str]:
        return _coerce_to_str_list(v)


class MethodologyDistribution(BaseModel):
    """
    Quantitative breakdown and narrative synthesis of methodologies across the literature corpus.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of methodology category to frequency count",
    )
    dominant_approach: str = Field(
        ...,
        description="Name of the most prevalent methodology in the corpus",
    )
    trend_description: str = Field(
        ...,
        description="Narrative synthesis describing the overarching methodological shift or trend across time",
    )

    @field_validator("distribution", mode="before")
    @classmethod
    def coerce_distribution(cls, v: Any) -> dict[str, int]:
        if not isinstance(v, dict):
            return {}
        result = {}
        for k, val in v.items():
            try:
                result[str(k).strip()] = int(val)
            except (ValueError, TypeError):
                result[str(k).strip()] = 1
        return result


class BibliographyItem(BaseModel):
    """
    Complete bibliographic reference entry for a paper analyzed or cited in the report.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    paper_id: str = Field(
        ...,
        description="Unique paper identifier matching EvidenceMatrixRow.paper_id and citation anchors",
    )
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(
        default_factory=list,
        description="List of author full names",
    )
    year: int | None = Field(default=None, description="Publication year")
    venue: str | None = Field(
        default=None,
        description="Conference, journal, or preprint repository (e.g. 'NeurIPS 2024', 'arXiv')",
    )
    doi: str | None = Field(
        default=None,
        description="Digital Object Identifier (DOI) if available",
    )
    pdf_url: str | None = Field(
        default=None,
        description="Direct URL to open-access PDF or landing page",
    )
    citation_count: int | None = Field(
        default=None,
        description="Citation count from academic APIs",
    )
    is_full_text_analyzed: bool = Field(
        default=True,
        description="True if full-text PDF was acquired and parsed; False if abstract-only",
    )

    @field_validator("authors", mode="before")
    @classmethod
    def coerce_authors(cls, v: Any) -> list[str]:
        return _coerce_to_str_list(v)

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, v: Any) -> int | None:
        if v is None or v == "":
            return None
        if isinstance(v, int):
            return v if 1800 <= v <= 2100 else None
        if isinstance(v, float):
            return int(v) if 1800 <= int(v) <= 2100 else None
        if isinstance(v, str):
            match = re.search(r"\b(19\d\d|20\d\d)\b", v)
            if match:
                return int(match.group(1))
        return None


class ReportMetadata(BaseModel):
    """
    Top-level execution and project metadata for the generated research report.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    project_id: str = Field(
        ...,
        description="UUID or identifier of the research project",
    )
    user_id: str = Field(
        default="default_user",
        description="Identifier of the user who initiated the report",
    )

    title: str = Field(..., description="Report title")
    research_question: str = Field(
        ...,
        description="Primary research question addressed by the review",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when report generation finalized",
    )
    pipeline_duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock duration of the execution pipeline in seconds",
    )
    status: ReportStatus = Field(
        default=ReportStatus.COMPLETE,
        description="Execution outcome status",
    )
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Adversarial critic quality score (0-100)",
    )
    papers_analyzed_full_text: int = Field(
        default=0,
        ge=0,
        description="Number of papers where full-text PDF was successfully parsed",
    )
    total_citations: int = Field(
        default=0,
        ge=0,
        description="Total count of unique papers in the bibliography",
    )
    llm_calls_made: int = Field(
        default=0,
        description="Number of LLM calls made during generation",
    )
    tokens_consumed: int = Field(
        default=0,
        description="Total tokens consumed across all LLM calls",
    )
    models_used: list[str] = Field(
        default_factory=list,
        description="List of model names used during generation",
    )

    @field_validator("quality_score", mode="before")
    @classmethod
    def clamp_quality_score(cls, v: Any) -> float:
        try:
            val = float(v)
            return max(0.0, min(100.0, val))
        except (ValueError, TypeError):
            return 0.0

    @field_validator("status", mode="before")
    @classmethod
    def coerce_status(cls, v: Any) -> ReportStatus:
        if isinstance(v, ReportStatus):
            return v
        if isinstance(v, str):
            try:
                return ReportStatus(v.lower().strip())
            except ValueError:
                return ReportStatus.COMPLETE
        return ReportStatus.COMPLETE


# ============================================================================
# Legacy Agent Schemas (for Backward Compatibility)
# ============================================================================


class Citation(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    url: str = ""
    source: str = "arXiv"
    relevance_score: int = 0
    citation_count: int | None = None
    abstract_snippet: str = Field(default="", description="First 200 chars of abstract")


class ReportSection(BaseModel):
    title: str
    content: str
    key_insight: str = ""
    paper_ids: list[str] = Field(default_factory=list)
    word_count: int = 0


class ResearchGap(BaseModel):
    description: str
    importance: str = "high"
    potential_directions: list[str] = Field(default_factory=list)
    related_paper_ids: Annotated[list[str], BeforeValidator(_coerce_to_str_list)] = Field(
        default_factory=list
    )


class MethodologyOverview(BaseModel):
    distribution: dict[str, int] = Field(default_factory=dict)
    dominant_approach: str = "quantitative"
    trend_description: str = ""


class PaperInsight(BaseModel):
    paper_id: str
    title: str
    relevance_score: int = 0
    key_findings: list[str] = Field(default_factory=list)
    methodology: str = ""
    limitations: list[str] = Field(default_factory=list)
    contribution: str = ""
    themes: list[str] = Field(default_factory=list)
    url: str = ""


class YearDistribution(BaseModel):
    year: str
    count: int


class ReportStatistics(BaseModel):
    total_papers_found: int = 0
    total_after_dedup: int = 0
    papers_analyzed: int = 0
    high_relevance_count: int = 0
    avg_relevance_score: float = 0.0
    year_distribution: list[YearDistribution] = Field(default_factory=list)
    source_distribution: dict[str, int] = Field(default_factory=dict)
    methodology_distribution: dict[str, int] = Field(default_factory=dict)
    top_keywords: list[str] = Field(default_factory=list)


class QualityIndicators(BaseModel):
    has_executive_summary: bool = True
    has_all_sections: bool = True
    section_count: int = 0
    papers_with_full_analysis: int = 0
    papers_with_partial_analysis: int = 0
    budget_exhausted: bool = False
    synthesis_model_used: str = "gemini-2.0-flash"


class Theme(BaseModel):
    name: str
    description: str
    paper_count: int = 0
    paper_ids: list[str] = Field(default_factory=list)
    strength: str = "strong"


# ============================================================================
# Master Research Report Container
# ============================================================================


class ResearchReport(BaseModel):
    """
    The comprehensive autonomous scientific literature review report.
    Final structured deliverable containing executive summary, comparative matrix,
    thematic synthesis with citation anchors, conflicting debates, actionable gaps,
    methodological distribution, and full bibliography.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    metadata: ReportMetadata = Field(
        ...,
        description="Top-level execution and project metadata",
    )
    executive_summary: str = Field(
        ...,
        description="Comprehensive executive summary synthesizing findings and research landscape",
    )
    comparison_matrix: list[EvidenceMatrixRow] = Field(
        default_factory=list,
        description="Structured comparative matrix across all reviewed papers",
    )
    thematic_sections: list[ThematicSection] = Field(
        default_factory=list,
        description="Detailed synthesis broken down by core subtopics/themes with citation anchors",
    )
    conflicting_findings_and_debates: list[ConflictingDebate] = Field(
        default_factory=list,
        description="Structured analysis of controversies, opposing views, and empirical discrepancies",
    )
    actionable_research_gaps: list[ResearchGapItem] = Field(
        default_factory=list,
        description="Grounded open problems, missing experiments, and recommended future directions",
    )
    methodology_overview: MethodologyDistribution | MethodologyOverview = Field(
        ...,
        description="Quantitative breakdown and narrative of methodology distributions",
    )
    bibliography: list[BibliographyItem] | list[Citation] = Field(
        default_factory=list,
        description="Complete indexed list of papers referenced throughout the report",
    )

    # Legacy fields for backward compatibility with existing tests/endpoints
    sections: list[ReportSection] = Field(default_factory=list)
    themes: list[Theme] = Field(default_factory=list)
    research_gaps: list[ResearchGap] = Field(default_factory=list)
    paper_insights: list[PaperInsight] = Field(default_factory=list)
    statistics: ReportStatistics | None = None
    quality_indicators: QualityIndicators | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_report_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if not d.get("conflicting_findings_and_debates"):
            for k in ("conflicting_debates", "debates", "scientific_debates", "conflicts_and_debates"):
                if d.get(k):
                    raw_debates = d[k]
                    if isinstance(raw_debates, list) and raw_debates and (
                        (isinstance(raw_debates[0], dict) and "perspective_a" in raw_debates[0])
                        or hasattr(raw_debates[0], "perspective_a")
                    ):
                        d["conflicting_findings_and_debates"] = raw_debates
                        break

        if not d.get("actionable_research_gaps"):
            for k in ("actionable_gaps", "gaps", "research_gaps"):
                if d.get(k):
                    raw_gaps = d[k]
                    if isinstance(raw_gaps, list) and raw_gaps and (
                        (isinstance(raw_gaps[0], dict) and ("recommended_methodology" in raw_gaps[0] or "gap_id" in raw_gaps[0]))
                        or hasattr(raw_gaps[0], "recommended_methodology")
                    ):
                        d["actionable_research_gaps"] = raw_gaps
                        break

        if not d.get("thematic_sections"):
            if d.get("sections"):
                raw_secs = d["sections"]
                if isinstance(raw_secs, list) and raw_secs and (
                    (isinstance(raw_secs[0], dict) and ("synthesis_prose" in raw_secs[0] or "theme_id" in raw_secs[0]))
                    or hasattr(raw_secs[0], "synthesis_prose")
                ):
                    d["thematic_sections"] = raw_secs

        if not d.get("comparison_matrix"):
            if d.get("comparative_matrix"):
                d["comparison_matrix"] = d["comparative_matrix"]

        return d

    def to_markdown(self) -> str:
        """Convert the structured research report into a fully formatted Markdown document."""
        md_lines = [
            f"# {self.metadata.title}",
            "",
            f"**Research Question:** {self.metadata.research_question}  ",
            f"**Generated:** {self.metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC') if hasattr(self.metadata.generated_at, 'strftime') else str(self.metadata.generated_at)}  ",
            f"**Quality Score:** {self.metadata.quality_score:.1f}/100 | **Status:** {self.metadata.status.value.upper()} | **Full-Text Analyzed:** {self.metadata.papers_analyzed_full_text}/{self.metadata.total_citations}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            self.executive_summary,
            "",
            "## Evidence Comparison Matrix",
            "",
            "| Paper ID | Title | Authors | Year | Methodology | Benchmark Dataset | Primary Metric | Primary Limitation | Full-Text |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        def esc(s: str) -> str:
            return str(s).replace("|", "\\|").replace("\n", " ")

        for row in self.comparison_matrix:
            authors_str = ", ".join(row.authors[:2]) + (" et al." if len(row.authors) > 2 else "")
            year_str = str(row.year) if row.year else "N/A"
            full_text_str = "Yes" if row.is_full_text else "No (Abstract)"
            md_lines.append(
                f"| {esc(row.paper_id)} | {esc(row.title)} | {esc(authors_str)} | {year_str} | {esc(row.methodology)} | {esc(row.benchmark_dataset)} | {esc(row.primary_metric)} | {esc(row.primary_limitation)} | {full_text_str} |"
            )

        md_lines.extend([
            "",
            "## Methodology Overview",
            "",
            f"**Dominant Approach:** {self.methodology_overview.dominant_approach}",
            "",
            self.methodology_overview.trend_description,
            "",
        ])
        if self.methodology_overview.distribution:
            md_lines.append("### Distribution Breakdown")
            for meth, count in self.methodology_overview.distribution.items():
                md_lines.append(f"- **{meth}**: {count} papers")
            md_lines.append("")

        md_lines.extend(["## Thematic Synthesis", ""])
        for section in self.thematic_sections:
            md_lines.extend([
                f"### {section.title}",
                "",
                section.synthesis_prose,
                "",
            ])
            if section.key_takeaways:
                md_lines.append("**Key Takeaways:**")
                for takeaway in section.key_takeaways:
                    md_lines.append(f"- {takeaway}")
                md_lines.append("")

        if self.conflicting_findings_and_debates:
            md_lines.extend(["## Conflicting Findings & Scientific Debates", ""])
            for debate in self.conflicting_findings_and_debates:
                md_lines.extend([
                    f"### Debate: {debate.topic}",
                    "",
                    f"**Perspective A:** {debate.perspective_a}",
                    "",
                    f"**Perspective B:** {debate.perspective_b}",
                    "",
                    f"**Critical Evaluation:** {debate.critical_evaluation}",
                    "",
                ])

        if self.actionable_research_gaps:
            md_lines.extend(["## Actionable Research Gaps & Future Directions", ""])
            for gap in self.actionable_research_gaps:
                grounding = ", ".join(gap.grounding_paper_ids) if gap.grounding_paper_ids else "Literature corpus"
                md_lines.extend([
                    f"### [{gap.importance.upper()} PRIORITY] {gap.gap_id}: {gap.description}",
                    "",
                    f"- **Grounding Papers:** {grounding}",
                    f"- **Recommended Methodology:** {gap.recommended_methodology}",
                    "",
                ])

        if self.bibliography:
            md_lines.extend(["## Bibliography", ""])
            for item in self.bibliography:
                authors_str = ", ".join(item.authors) if item.authors else "Unknown Authors"
                year_str = f"({item.year})" if item.year else ""
                venue = getattr(item, "venue", getattr(item, "source", None))
                venue_str = f"*{venue}*." if venue else ""
                doi = getattr(item, "doi", None)
                doi_str = f"DOI: [{doi}](https://doi.org/{doi})" if doi else ""
                pdf_url = getattr(item, "pdf_url", getattr(item, "url", None))
                pdf_str = f"[Link]({pdf_url})" if pdf_url else ""
                is_ft = getattr(item, "is_full_text_analyzed", True)
                ft_str = "[Full-Text Analyzed]" if is_ft else "[Abstract Only]"
                links = " | ".join([s for s in (doi_str, pdf_str, ft_str) if s])
                md_lines.append(f"- **[{item.paper_id}]** {authors_str} {year_str}. **{item.title}**. {venue_str} {links}")

        return "\n".join(md_lines)


# ============================================
# Reasoning & Verification Agent Schemas
# ============================================


class PropositionVerification(BaseModel):
    """Structured audit verification of an atomic factual proposition against a paper section chunk."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    proposition: str = Field(
        ...,
        description="Atomic factual proposition extracted from synthesis prose",
    )
    citation_anchor: str = Field(
        ...,
        description="Citation anchor tag (e.g. 'ref_1#sec_methodology_2' or 'ref_1')",
    )
    paper_id: str = Field(..., description="Referenced paper ID")
    section_anchor: str | None = Field(
        default=None,
        description="Referenced section anchor identifier if specified",
    )
    grounding_chunk_id: str | None = Field(
        default=None,
        description="UUID or anchor of retrieved chunk used as grounding evidence",
    )
    grounding_text: str | None = Field(
        default=None,
        description="Exact evidence excerpt from source paper chunk",
    )
    verdict: NLIVerdict = Field(
        ...,
        description="NLI classification: ENTAILMENT, NEUTRAL, or CONTRADICTION",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the NLI evaluation (0.0 to 1.0)",
    )
    reasoning: str = Field(
        ...,
        description="Justification explaining why the chunk entails, contradicts, or fails to support the claim",
    )
    suggested_correction: str | None = Field(
        default=None,
        description="Recommended factual correction if verdict is NEUTRAL or CONTRADICTION",
    )


class CitationAuditReport(BaseModel):
    """Comprehensive fact-checking and citation grounding audit report."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    total_propositions: int = Field(default=0, ge=0)
    entailed_count: int = Field(default=0, ge=0)
    neutral_count: int = Field(default=0, ge=0)
    contradiction_count: int = Field(default=0, ge=0)
    precision_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Percentage of propositions verified as ENTAILMENT (0-100)",
    )
    verifications: list[PropositionVerification] = Field(default_factory=list)
    hallucinated_anchors: list[str] = Field(
        default_factory=list,
        description="Citation anchors with no corresponding paper or chunk in cache",
    )
    audit_passed: bool = Field(
        default=True,
        description="True if precision_score >= 80.0 and 0 contradictions",
    )


class CriticDimensionScore(BaseModel):
    """Evaluation score for a specific quality dimension."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    dimension: str = Field(..., description="Name of the evaluated dimension")
    score: float = Field(
        ge=0.0,
        le=100.0,
        description="Dimension score from 0.0 to 100.0",
    )
    justification: str = Field(
        ...,
        description="Specific qualitative justification and observed gaps",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_aliases(cls, v: Any) -> Any:
        if isinstance(v, dict):
            data = dict(v)
            if "dimension" not in data and "dimension_name" in data:
                data["dimension"] = data["dimension_name"]
            if "justification" not in data and "feedback" in data:
                data["justification"] = data["feedback"]
            return data
        return v


class CriticEvaluation(BaseModel):
    """
    Adversarial Critic evaluation report scoring synthesis quality and dictating refinement loops.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    overall_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Overall synthesis quality score (0-100)",
    )
    dimension_scores: list[CriticDimensionScore] = Field(default_factory=list)
    strengths: list[str] = Field(
        default_factory=list,
        description="Observed strengths in the synthesis",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Specific factual, thematic, or methodological weaknesses",
    )
    refinement_guidance: list[str] = Field(
        default_factory=list,
        description="Concrete instructions for the Synthesizer during refinement iteration",
    )
    should_refine: bool = Field(
        default=False,
        description="True if overall_score < 75.0, triggering refinement loop",
    )


class AcademicPaperCandidate(BaseModel):
    """Candidate paper discovered via multi-source academic search APIs."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    paper_id: str = Field(
        ...,
        description="Canonical identifier (e.g. 'arxiv:2401.12345' or 'doi:10.1145/...')",
    )
    title: str = Field(..., description="Title of the paper")
    authors: list[str] = Field(default_factory=list)
    abstract: str = Field(default="")
    year: int | None = Field(default=None)
    venue: str | None = Field(default=None)
    doi: str | None = Field(default=None)
    arxiv_id: str | None = Field(default=None)
    s2_id: str | None = Field(default=None)
    url: str | None = Field(default=None)
    citation_count: int | None = Field(default=None)
    source: str = Field(
        default="openalex",
        description="Source API: openalex, semanticscholar, arxiv, pubmed",
    )
    relevance_score: float | None = Field(
        default=None,
        description="Ranking score from 0.0 to 1.0",
    )


class SearchQueryPlan(BaseModel):
    """Search queries formulated by the Autonomous Literature Explorer."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    primary_queries: list[str] = Field(
        description="Direct search queries for main research concepts",
    )
    expanded_queries: list[str] = Field(
        default_factory=list,
        description="Synonym and facet expanded search queries",
    )
    target_domains: list[str] = Field(
        default_factory=list,
        description="Subject domains (e.g., 'cs.AI', 'stat.ML')",
    )
    subtopic_facets: list[str] = Field(
        default_factory=list,
        description="Key thematic subtopics to guide paper exploration",
    )


class EvidenceMatrixExtraction(BaseModel):
    """Structured extraction output from Evidence Matrix Builder."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    rows: list[EvidenceMatrixRow] = Field(default_factory=list)


class ThematicSynthesisDraft(BaseModel):
    """Structured output produced by Thematic Synthesis & Gap Specialist."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    executive_summary: str = Field(...)
    thematic_sections: list[ThematicSection] = Field(default_factory=list)
    conflicting_findings_and_debates: list[ConflictingDebate] = Field(
        default_factory=list,
    )
    actionable_research_gaps: list[ResearchGapItem] = Field(
        default_factory=list,
    )
    methodology_overview: MethodologyDistribution = Field(...)

    @model_validator(mode="before")
    @classmethod
    def normalize_draft_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        # Normalize sections
        if not d.get("thematic_sections"):
            if d.get("sections"):
                d["thematic_sections"] = d["sections"]
            elif d.get("thematic_synthesis") and isinstance(d["thematic_synthesis"], list):
                d["thematic_sections"] = d["thematic_synthesis"]

        # Normalize debates
        if not d.get("conflicting_findings_and_debates"):
            for k in ("conflicting_debates", "debates", "scientific_debates", "conflicts_and_debates", "controversies"):
                if d.get(k):
                    d["conflicting_findings_and_debates"] = d[k]
                    break

        # Normalize gaps
        if not d.get("actionable_research_gaps"):
            for k in ("research_gaps", "gaps", "actionable_gaps", "open_gaps"):
                if d.get(k):
                    d["actionable_research_gaps"] = d[k]
                    break

        # Normalize methodology_overview
        if not d.get("methodology_overview"):
            if isinstance(d.get("methodology"), dict):
                d["methodology_overview"] = d["methodology"]
            elif isinstance(d.get("methodology_distribution"), dict):
                d["methodology_overview"] = d["methodology_distribution"]
            else:
                d["methodology_overview"] = {
                    "distribution": {"Empirical Analysis": 1},
                    "dominant_approach": "Empirical Analysis",
                    "trend_description": "Empirical and comparative evaluation.",
                }
        return d


# ============================================
# Agent Communication Models (Pipeline flow)
# ============================================


class PlannerInput(BaseModel):
    research_question: str
    title: str
    max_papers: int = 20


class SearchStrategy(BaseModel):
    primary_keywords: list[str] = Field(description="Top 5 keywords for search")
    secondary_keywords: list[str] = Field(description="Broader/alternative terms")
    sources: list[str] = Field(default=["arXiv", "Semantic Scholar"])
    max_papers_per_source: int = 15


class PlannerOutput(BaseModel):
    keywords: list[str] = Field(description="8-12 search keywords")
    subtopics: list[str] = Field(description="4-6 sections for the review")
    search_strategy: SearchStrategy


class RetrieverInput(BaseModel):
    planner_output: PlannerOutput
    research_question: str
    project_id: str


class RankedPaper(BaseModel):
    id: str
    title: str
    abstract: str
    authors: list[str]
    url: str
    source: str  # "arXiv" | "Semantic Scholar"
    year: int | None = None
    citation_count: int | None = None
    embedding_similarity: float  # cosine similarity to research question


class RetrieverOutput(BaseModel):
    papers: list[RankedPaper] = Field(
        description="Papers sorted by embedding similarity, top-K only"
    )
    total_found: int
    total_after_dedup: int
    top_k_selected: int
    sources_searched: list[str]
    rag_ingestion_stats: dict[str, Any] = Field(default_factory=dict)


class AnalyzerInput(BaseModel):
    papers: list[RankedPaper]
    research_question: str
    subtopics: list[str]


class PaperAnalysis(BaseModel):
    paper_id: str
    title: str
    relevance_score: int = Field(ge=0, le=100, description="0-100 relevance to research question")
    key_findings: list[str] = Field(description="2-3 key findings")
    methodology: str = Field(description="Brief methodology description")
    limitations: list[str] = Field(description="1-2 limitations")
    contribution: str = Field(description="One-line contribution summary")
    themes: list[str] = Field(description="Which subtopics this paper relates to")


class AnalyzerOutput(BaseModel):
    paper_analyses: list[PaperAnalysis]
    cross_cutting_themes: list[Theme]
    methodology_distribution: dict[str, int]  # e.g., {"qualitative": 3, "quantitative": 8, ...}
    high_quality_count: int
    total_analyzed: int


class SynthesizerInput(BaseModel):
    analyzer_output: AnalyzerOutput
    subtopics: list[str]
    research_question: str
    academic_level: str = "graduate"
    bibliography: list[RankedPaper]


class SynthesizerOutput(BaseModel):
    """Output from the Synthesizer agent — the complete ResearchReport."""

    report: ResearchReport
