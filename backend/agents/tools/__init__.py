"""
Scholar Agent Tools Package.

Exports:
- Multi-tier Open-Access Resolution Cascade (OAResolver, OAResolutionResult)
- Full-text scientific PDF parser (PDFParser, ParsedDocument)
- Federated multi-source academic search (MultiSourceAcademicSearch)
- 1-hop bidirectional citation graph traversal (CitationGraphTraverser)
- Backwards-compatible legacy tool helpers (ToolResult, extract_paper_insights, etc.)
"""

from .academic_search import (
    AcademicPaperCandidate,
    MultiSourceAcademicSearch,
    deduplicate_and_merge_candidates,
    merge_candidate_into,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
    reconstruct_openalex_abstract,
    titles_match,
)
from .citation_graph import CitationGraphTraverser
from .fact_checker import (
    CITATION_ANCHOR_REGEX,
    AtomicProposition,
    FactCheckerEngine,
)
from .legacy import (
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
from .oa_resolver import (
    AbstractFallbackMetadata,
    OAResolutionResult,
    OAResolver,
    extract_openalex_concepts,
    extract_openalex_mesh_terms,
    is_valid_pdf_bytes,
)
from .pdf_parser import (
    DISPLAY_MATH_PATTERN,
    INLINE_MATH_PATTERN,
    PDFParser,
    ParsedDocument,
)

__all__ = [
    # OA Resolver
    "OAResolver",
    "OAResolutionResult",
    "AbstractFallbackMetadata",
    "normalize_doi",
    "normalize_arxiv_id",
    "is_valid_pdf_bytes",
    "reconstruct_openalex_abstract",
    "extract_openalex_mesh_terms",
    "extract_openalex_concepts",
    # PDF Parser
    "PDFParser",
    "ParsedDocument",
    "DISPLAY_MATH_PATTERN",
    "INLINE_MATH_PATTERN",
    # Academic Search
    "MultiSourceAcademicSearch",
    "AcademicPaperCandidate",
    "normalize_title",
    "titles_match",
    "merge_candidate_into",
    "deduplicate_and_merge_candidates",
    # Citation Graph
    "CitationGraphTraverser",
    # Fact Checker
    "FactCheckerEngine",
    "AtomicProposition",
    "CITATION_ANCHOR_REGEX",
    # Legacy Tools
    "ToolResult",
    "extract_json_from_response",
    "extract_keywords_from_question",
    "identify_subtopics",
    "refine_search_query",
    "score_paper_relevance",
    "extract_paper_insights",
    "synthesize_section",
    "identify_research_gaps",
    "evaluate_synthesis_quality",
]

