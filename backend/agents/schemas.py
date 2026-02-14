from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field


class ReportStatus(StrEnum):
    COMPLETE = "complete"  # Full pipeline completed
    PARTIAL = "partial"  # Budget exhaustion — partial synthesis available
    ANALYSIS_ONLY = "analysis_only"  # Only analysis data, no synthesis
    ERROR = "error"  # Pipeline failed but may have partial data


class ReportMetadata(BaseModel):
    project_id: str
    user_id: str
    title: str
    research_question: str
    generated_at: datetime
    pipeline_duration_seconds: float
    status: ReportStatus
    llm_calls_made: int
    tokens_consumed: int
    models_used: list[str]


class Citation(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: int | None = None
    url: str
    source: str  # "arXiv" | "Semantic Scholar"
    relevance_score: int  # 0-100
    citation_count: int | None = None
    abstract_snippet: str = Field(description="First 200 chars of abstract")


class ReportSection(BaseModel):
    title: str  # Subtopic name
    content: str  # Well-written 200-400 word prose
    key_insight: str  # One-sentence highlight for UI cards
    paper_ids: list[str]  # Papers cited in this section
    word_count: int


def _coerce_to_str_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(item) for item in v]
    return v


class ResearchGap(BaseModel):
    description: str
    importance: str  # "high" | "medium" | "low"
    potential_directions: list[str]
    related_paper_ids: Annotated[list[str], BeforeValidator(_coerce_to_str_list)]


class MethodologyOverview(BaseModel):
    distribution: dict[str, int]  # {"quantitative": 8, "qualitative": 3, ...}
    dominant_approach: str  # Most common methodology
    trend_description: str  # One sentence about methodology trends


class PaperInsight(BaseModel):
    """Per-paper structured analysis — drillable in UI."""

    paper_id: str
    title: str
    relevance_score: int
    key_findings: list[str]
    methodology: str
    limitations: list[str]
    contribution: str
    themes: list[str]  # Which subtopics this paper maps to
    url: str


class YearDistribution(BaseModel):
    year: str
    count: int


class ReportStatistics(BaseModel):
    total_papers_found: int
    total_after_dedup: int
    papers_analyzed: int
    high_relevance_count: int  # Papers with relevance >= 70
    avg_relevance_score: float
    year_distribution: list[YearDistribution]
    source_distribution: dict[str, int]  # {"arXiv": 12, "Semantic Scholar": 8}
    methodology_distribution: dict[str, int]
    top_keywords: list[str]


class QualityIndicators(BaseModel):
    """
    Signals to the frontend about report completeness.
    Frontend uses these to show badges/warnings.
    """

    has_executive_summary: bool
    has_all_sections: bool
    section_count: int
    papers_with_full_analysis: int
    papers_with_partial_analysis: int
    budget_exhausted: bool
    synthesis_model_used: str  # Which model produced synthesis


class Theme(BaseModel):
    """
    Cross-cutting theme across papers.
    Frontend renders as insight cards.
    """

    name: str
    description: str
    paper_count: int
    paper_ids: list[str]
    strength: str  # "strong" | "moderate" | "emerging"


class ResearchReport(BaseModel):
    """
    The complete structured report — the final deliverable.
    This is what gets stored in the database and sent to the frontend.
    """

    metadata: ReportMetadata
    executive_summary: str
    sections: list[ReportSection]
    themes: list[Theme]
    research_gaps: list[ResearchGap]
    methodology_overview: MethodologyOverview
    paper_insights: list[PaperInsight]
    statistics: ReportStatistics
    bibliography: list[Citation]
    quality_indicators: QualityIndicators


# --- Agent Communication Models (from 02-AGENTS.md) ---


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
    rag_ingestion_stats: dict[str, Any]


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
