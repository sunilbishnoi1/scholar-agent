"""
Academic Retrieval Tools Adversarial Robustness and Boundary Test Suite.
Stress tests edge cases, malicious payloads, corrupted inputs, network error simulation,
and boundary conditions for OA Resolver, PDF Parser, Academic Search, and Citation Graph.
"""

import io
import math
from unittest.mock import MagicMock, patch
import pytest
import pymupdf
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.agents.schemas import AcademicPaperCandidate
from backend.models.database import Base, PaperCache
from backend.rag.chunker import ChunkType, ParsedSection
from backend.agents.tools.oa_resolver import (
    AbstractFallbackMetadata,
    OAResolutionResult,
    OAResolver,
    extract_openalex_concepts,
    extract_openalex_mesh_terms,
    is_valid_pdf_bytes,
    normalize_arxiv_id,
    normalize_doi,
    reconstruct_openalex_abstract,
)
from backend.agents.tools.pdf_parser import (
    DISPLAY_MATH_PATTERN,
    INLINE_MATH_PATTERN,
    PDFParser,
    ParsedDocument,
)
from backend.agents.tools.academic_search import (
    MultiSourceAcademicSearch,
    deduplicate_and_merge_candidates,
    merge_candidate_into,
    normalize_title,
    titles_match,
)
from backend.agents.tools.citation_graph import CitationGraphTraverser


# ============================================================================
# 1. OA Resolver Adversarial Stress Tests
# ============================================================================


class TestOAResolverAdversarial:
    """Adversarial testing of OA Resolver cascade under extreme failure modes."""

    def test_normalize_doi_adversarial_inputs(self):
        # Complex DOIs with prefixes and punctuation
        assert normalize_doi("https://doi.org/10.1038/s41586-020-2649-2") == "10.1038/s41586-020-2649-2"
        assert normalize_doi("doi: 10.1038/s41586-020-2649-2\n") == "10.1038/s41586-020-2649-2"
        assert normalize_doi("10.5555/12345678/extra/slashes") == "10.5555/12345678/extra/slashes"
        assert normalize_doi("https://dx.doi.org/10.1000/182/") == "10.1000/182"
        assert normalize_doi("10.1002/(sici)1097-0142(19980801)83:3") == "10.1002/(sici)1097-0142(19980801)83:3"
        # Invalid inputs
        assert normalize_doi("http://notadoi.org/123") is None
        assert normalize_doi("   ") is None
        assert normalize_doi(None) is None
        assert normalize_doi("10.") is None
        assert normalize_doi(12345) is None  # type: ignore

    def test_normalize_arxiv_id_adversarial_inputs(self):
        assert normalize_arxiv_id("https://arxiv.org/abs/2401.01234v99") == "2401.01234v99"
        assert normalize_arxiv_id("https://arxiv.org/pdf/2401.01234v1.pdf") == "2401.01234v1"
        assert normalize_arxiv_id("arxiv:math.PR/0501001v2") == "math.PR/0501001v2"
        assert normalize_arxiv_id(None, doi="10.48550/arXiv.math-ph/0102034") == "math-ph/0102034"
        assert normalize_arxiv_id("") is None
        assert normalize_arxiv_id(None) is None
        assert normalize_arxiv_id(1234) is None  # type: ignore

    def test_is_valid_pdf_bytes_boundary_and_malicious(self):
        # Exact length boundaries
        assert is_valid_pdf_bytes(b"%PDF-" + b"A" * 1018) is False  # 1023 bytes -> False
        assert is_valid_pdf_bytes(b"%PDF-" + b"A" * 1019) is True   # 1024 bytes -> True
        # HTML masquerading
        assert is_valid_pdf_bytes(b"%PDF-1.4\n<HTML><BODY>Access Denied Cloudflare</BODY></HTML>" + b" " * 1000) is False
        assert is_valid_pdf_bytes(b"%PDF-1.7\n<!DOCTYPE html><html><body>Error 403</body></html>" + b" " * 1000) is False
        # Null bytes and binary junk without PDF magic
        assert is_valid_pdf_bytes(b"\x00" * 2048) is False
        assert is_valid_pdf_bytes(b"PK\x03\x04" + b"A" * 2000) is False  # Zip file

    def test_reconstruct_openalex_abstract_malformed_structures(self):
        # Empty and non-standard indices
        assert reconstruct_openalex_abstract({}) == ""
        assert reconstruct_openalex_abstract({"word": []}) == ""
        assert reconstruct_openalex_abstract({"word": [-1, 0, 1]}) == "word word"
        assert reconstruct_openalex_abstract({"B": [1], "A": [0], "C": [2]}) == "A B C"
        assert reconstruct_openalex_abstract("not a dict") == ""  # type: ignore
        assert reconstruct_openalex_abstract(None) == ""

    @patch("requests.Session.get")
    def test_cascade_all_http_error_codes(self, mock_get):
        resolver = OAResolver()
        error_statuses = [400, 401, 403, 404, 429, 500, 502, 503, 504]

        for status in error_statuses:
            mock_resp = MagicMock()
            mock_resp.status_code = status
            mock_get.return_value = mock_resp

            result = resolver.resolve_paper(
                doi=f"10.1000/error.{status}",
                arxiv_id="2401.00000",
                title=f"Paper with HTTP {status}",
            )
            # Must NEVER raise exception and gracefully return Tier 3 fallback
            assert result.is_full_text is False
            assert result.resolution_tier == 3
            assert result.source == "abstract_fallback"
            assert result.title == f"Paper with HTTP {status}"
            assert result.doi == f"10.1000/error.{status}"

    @patch("requests.Session.get")
    def test_cascade_all_network_exceptions(self, mock_get):
        resolver = OAResolver()
        exceptions = [
            requests.exceptions.ConnectTimeout("Timeout"),
            requests.exceptions.ReadTimeout("Read Timeout"),
            requests.exceptions.SSLError("SSL Certificate verify failed"),
            requests.exceptions.ConnectionError("Connection refused"),
            requests.exceptions.ChunkedEncodingError("Incomplete read"),
            requests.exceptions.ContentDecodingError("Bad gzip"),
        ]

        for exc in exceptions:
            mock_get.side_effect = exc
            result = resolver.resolve_paper(
                doi="10.1000/network.fail",
                title="Network Resilience Test",
            )
            assert result.is_full_text is False
            assert result.resolution_tier == 3
            assert result.source == "abstract_fallback"
            assert result.title == "Network Resilience Test"


# ============================================================================
# 2. PDF Parser Adversarial Stress Tests
# ============================================================================


class TestPDFParserAdversarial:
    """Adversarial stress testing for PDF Parser."""

    def test_parse_pdf_malformed_and_truncated_streams(self):
        parser = PDFParser()
        # Truncated PDF header
        res1 = parser.parse_pdf(b"%PDF-1.4\n%%EOF", doi="10.1000/truncated")
        assert res1.is_full_text is False
        assert "error" in res1.metadata

        # Random garbage bytes
        res2 = parser.parse_pdf(b"\xde\xad\xbe\xef" * 50, doi="10.1000/garbage")
        assert res2.is_full_text is False
        assert "error" in res2.metadata

        # Empty document with 0 pages
        with patch("pymupdf.open") as mock_open:
            mock_doc = MagicMock()
            mock_doc.__len__.return_value = 0
            mock_open.return_value = mock_doc
            res3 = parser.parse_pdf(b"%PDF-1.4\n1234567890" * 10, doi="10.1000/zero_pages")
            assert res3.is_full_text is False
            assert "0 pages" in res3.metadata.get("error", "")

    def test_parse_multi_page_pdf_max_pages_limit(self):
        doc = pymupdf.open()
        for i in range(10):
            p = doc.new_page(width=612, height=792)
            p.insert_text((54, 70), f"Section {i+1}. Heading", fontsize=14)
            p.insert_text((54, 100), f"Body paragraph content for page {i+1}.", fontsize=10)
        pdf_bytes = doc.tobytes()
        doc.close()

        # Parse with max_pages=3
        parser = PDFParser(max_pages=3)
        parsed = parser.parse_pdf(pdf_bytes, doi="10.1000/ten_pages")
        assert parsed.page_count == 10
        # Only sections from first 3 pages should be extracted
        section_titles = [s.title for s in parsed.sections]
        assert len(section_titles) <= 4

    def test_latex_equation_extraction_complex_math(self):
        text = r"""
        Here is an inline equation $x_{i,j}^{(t)} \in \mathbb{R}^d$ and another $\alpha + \beta = \gamma$.
        Here is display math:
        $$
        \\min_{\\theta} \\mathbb{E}_{x \sim \mathcal{D}} [\\mathcal{L}(f_\\theta(x), y)] + \\lambda \\|\\theta\\|_2^2
        $$
        And an equation environment:
        \\begin{equation}
        \\nabla \\times \\mathbf{B} = \\mu_0 \\left( \\mathbf{J} + \\varepsilon_0 \\frac{\\partial \\mathbf{E}}{\\partial t} \\right)
        \\end{equation}
        """
        eqs = PDFParser.extract_latex_equations(text)
        assert len(eqs) >= 3
        combined = " ".join(eqs)
        assert "\\min_{\\theta}" in combined
        assert "\\nabla \\times \\mathbf{B}" in combined
        assert "$x_{i,j}^{(t)} \\in \\mathbb{R}^d$" in combined

    def test_table_markdown_cleaner_escaping_and_irregular_cells(self):
        parser = PDFParser()
        matrix = [
            ["Metric | Unit", "Score", "Notes"],
            ["Accuracy (0-1)", "0.95 | High", "Best result\n(new state-of-the-art)"],
            ["Latency", "12ms"],  # Missing 3rd column
        ]
        md_table = parser._clean_table_to_markdown(matrix)
        assert "| Metric \\| Unit | Score | Notes |" in md_table
        assert "| --- | --- | --- |" in md_table
        assert "0.95 \\| High" in md_table
        assert "Best result (new state-of-the-art)" in md_table


# ============================================================================
# 3. Academic Search & Deduplication Adversarial Tests
# ============================================================================


class TestAcademicSearchAdversarial:
    """Adversarial testing of Academic Search deduplication and bounds."""

    def test_extreme_query_and_bounds_clamping(self):
        search_tool = MultiSourceAcademicSearch(
            enable_openalex=False,
            enable_semanticscholar=False,
            enable_arxiv=False,
            enable_pubmed=False,
        )
        assert search_tool.search("") == []
        assert search_tool.search("   ") == []

    def test_deduplication_with_multiple_colliding_identifiers(self):
        c1 = AcademicPaperCandidate(
            paper_id="doi:10.1000/dup",
            title="Deep Residual Learning for Image Recognition",
            authors=["Kaiming He"],
            doi="10.1000/dup",
            source="openalex",
        )
        c2 = AcademicPaperCandidate(
            paper_id="arxiv:1512.03385",
            title="Deep Residual Learning for Image Recognition",
            authors=["Kaiming He", "Xiangyu Zhang"],
            arxiv_id="1512.03385",
            source="arxiv",
        )
        c3 = AcademicPaperCandidate(
            paper_id="s2:resnet",
            title="Deep Residual Learning for Image Recognition",
            doi="10.1000/dup",
            arxiv_id="1512.03385",
            s2_id="resnet",
            citation_count=100000,
            source="semanticscholar",
        )
        c4 = AcademicPaperCandidate(
            paper_id="pubmed:99999",
            title="Deep Residual Learning for Image Recognition",
            authors=["Kaiming He"],
            source="pubmed",
        )

        unique = deduplicate_and_merge_candidates([c1, c2, c3, c4])
        assert len(unique) == 1
        merged = unique[0]
        assert merged.doi == "10.1000/dup"
        assert merged.arxiv_id == "1512.03385"
        assert merged.s2_id == "resnet"
        assert merged.citation_count == 100000
        sources = merged.source.split(",")
        assert len(sources) == 4
        assert set(sources) == {"openalex", "arxiv", "semanticscholar", "pubmed"}


# ============================================================================
# 4. Citation Graph Traversal Adversarial Tests
# ============================================================================


class TestCitationGraphAdversarial:
    """Adversarial testing of 1-hop Citation Graph Traversal."""

    def test_empty_seeds_and_none_handling(self):
        traverser = CitationGraphTraverser()
        assert traverser.traverse_1hop([]) == []
        assert traverser.traverse_1hop(["   ", ""]) == []

    @patch("requests.get")
    def test_cycle_and_self_citation_filtering(self, mock_get):
        # Mock seed paper returning itself and duplicates
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "citingPaper": {
                        "paperId": "seed_001",
                        "title": "Autonomous Multi-Agent Scientific Reasoning",
                        "externalIds": {"DOI": "10.1000/seed.001", "ArXiv": "2401.00001"},
                        "citationCount": 50,
                    }
                },
                {
                    "citingPaper": {
                        "paperId": "citing_002",
                        "title": "Autonomous Multi-Agent Scientific Reasoning",  # Duplicate title
                        "externalIds": {"DOI": "10.1000/different.002"},
                        "citationCount": 20,
                    }
                },
                {
                    "citingPaper": {
                        "paperId": "citing_003",
                        "title": "Legitimate Followup Study",
                        "externalIds": {"DOI": "10.1000/legit.003"},
                        "citationCount": 150,
                    }
                },
            ]
        }
        mock_get.return_value = mock_resp

        traverser = CitationGraphTraverser()
        results = traverser.traverse_1hop(
            seed_paper_ids=["10.1000/seed.001"],
            include_forward=True,
            include_backward=False,
        )

        # Seed DOI, seed arXiv, and duplicate seed title must all be excluded
        assert len(results) == 1
        assert results[0].doi == "10.1000/legit.003"
        assert results[0].citation_count == 150
