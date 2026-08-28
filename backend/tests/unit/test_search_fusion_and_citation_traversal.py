"""
Multi-Source Search Fusion, Candidate Deduplication, and Citation Graph Traversal Test Suite.
Empirically tests and documents:
1. Multi-Source Search & Deduplication Across 4 Providers (OpenAlex, Semantic Scholar, arXiv, PubMed)
2. Strict Candidate Clamping Bounds [1, 40] and Default (25)
3. 1-Hop Forward/Backward Citation Graph Traversal & Snowballing
4. Strict Seed Exclusion Across Varied Identifier Formats (DOI, arXiv, S2 ID, Candidate Objects, Titles)
5. Adversarial Input Robustness, Error Handling, and Metadata Merging Correctness
6. Edge case coverage for identifier normalization and string matching
"""

import math
from unittest.mock import MagicMock, patch
import pytest

from backend.agents.schemas import AcademicPaperCandidate
from backend.agents.tools.academic_search import (
    DEFAULT_SEARCH_LIMIT,
    MAX_CANDIDATE_CAP,
    MIN_CANDIDATE_CAP,
    MultiSourceAcademicSearch,
    deduplicate_and_merge_candidates,
    merge_candidate_into,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
    reconstruct_openalex_abstract,
    titles_match,
)
from backend.agents.tools.citation_graph import (
    MAX_GRAPH_LIMIT,
    CitationGraphTraverser,
)


class TestIdentifierNormalizationAndMatching:
    """Adversarial testing of identifier normalization and string matching."""

    @pytest.mark.parametrize(
        "input_doi,expected",
        [
            ("10.1000/182", "10.1000/182"),
            ("https://doi.org/10.1000/182", "10.1000/182"),
            ("http://doi.org/10.1000/182", "10.1000/182"),
            ("https://dx.doi.org/10.1145/3318464.3389700", "10.1145/3318464.3389700"),
            ("doi:10.1016/j.artint.2023.103980", "10.1016/j.artint.2023.103980"),
            ("DOI:10.1016/J.ARTINT.2023.103980", "10.1016/j.artint.2023.103980"),
            ("doi.org/10.1000/182", "10.1000/182"),
            ("https://doi.org/doi:10.1000/182", "10.1000/182"),
            ("  10.1000/182  ", "10.1000/182"),
            ("10.1000/182/", "10.1000/182"),
            ("10.1002/(sici)1097-0142(19980801)83:3<497::aid-cncr19>3.0.co;2-z", "10.1002/(sici)1097-0142(19980801)83:3<497::aid-cncr19>3.0.co;2-z"),
            (None, None),
            ("", None),
            ("   ", None),
            ("invalid_doi_without_prefix", None),
            ("10.", None),
            ("10.1000", None),
        ],
    )
    def test_normalize_doi_cases(self, input_doi, expected):
        assert normalize_doi(input_doi) == expected

    @pytest.mark.parametrize(
        "input_arxiv,expected",
        [
            ("2401.00001", "2401.00001"),
            ("2401.00001v1", "2401.00001"),
            ("2401.00001v99", "2401.00001"),
            ("https://arxiv.org/abs/2401.00001", "2401.00001"),
            ("https://arxiv.org/abs/2401.00001v3", "2401.00001"),
            ("http://arxiv.org/pdf/2401.00001.pdf", "2401.00001"),
            ("arxiv:2401.00001", "2401.00001"),
            ("ARXIV: 2401.00001v2", "2401.00001"),
            ("arxiv:2401.00001v1.pdf", "2401.00001"),
            ("hep-th/9901001", "hep-th/9901001"),
            ("https://arxiv.org/abs/hep-th/9901001v2", "hep-th/9901001"),
            (None, None),
            ("", None),
            ("   ", None),
        ],
    )
    def test_normalize_arxiv_id_cases(self, input_arxiv, expected):
        assert normalize_arxiv_id(input_arxiv) == expected

    def test_normalize_title_and_matching(self):
        # Exact match under varying punctuation
        assert titles_match(
            "Attention Is All You Need!",
            "attention is all you need",
        ) is True

        # Non-matching distinct titles
        assert titles_match(
            "Quantum Computing for Chemical Simulation",
            "Deep Reinforcement Learning in Robotics",
        ) is False

        # Edge cases: empty/None
        assert titles_match("", "Some title") is False
        assert titles_match(None, "Some title") is False
        assert titles_match("Some title", None) is False

    def test_reconstruct_openalex_abstract_cases(self):
        # Valid inverted index
        inverted = {
            "Scholar": [0],
            "Agent": [1],
            "enables": [2],
            "autonomous": [3],
            "reasoning.": [4],
        }
        assert reconstruct_openalex_abstract(inverted) == "Scholar Agent enables autonomous reasoning."

        # Out-of-order inverted index
        inverted_mixed = {
            "reasoning.": [4],
            "Scholar": [0],
            "enables": [2],
            "Agent": [1],
            "autonomous": [3],
        }
        assert reconstruct_openalex_abstract(inverted_mixed) == "Scholar Agent enables autonomous reasoning."

        # Empty / malformed
        assert reconstruct_openalex_abstract({}) == ""
        assert reconstruct_openalex_abstract(None) == ""
        assert reconstruct_openalex_abstract("invalid") == ""


class TestCandidateDeduplicationAndFusion:
    """Adversarial stress testing of cross-provider candidate merging."""

    def test_4_provider_fusion_aggregates_ids_and_sources(self):
        """Verify merging candidate data across OpenAlex, S2, arXiv, and PubMed."""
        c_openalex = AcademicPaperCandidate(
            paper_id="openalex:W12345",
            title="Next-Generation Multi-Agent Scientific Reasoning",
            authors=["Dr. Alice"],
            abstract="Brief abstract from OpenAlex.",
            year=2024,
            venue="NeurIPS 2024",
            doi="10.1000/scireason.2024",
            arxiv_id=None,
            s2_id=None,
            citation_count=45,
            source="openalex",
        )
        c_s2 = AcademicPaperCandidate(
            paper_id="s2:abcdef",
            title="Next-Generation Multi-Agent Scientific Reasoning",
            authors=["Dr. Alice", "Dr. Bob"],
            abstract="A comprehensive, highly detailed abstract provided by Semantic Scholar with complete experimental results.",
            year=2024,
            venue=None,
            doi="10.1000/scireason.2024",
            arxiv_id="2401.99999",
            s2_id="abcdef",
            citation_count=45,
            source="semanticscholar",
        )
        c_arxiv = AcademicPaperCandidate(
            paper_id="arxiv:2401.99999",
            title="Next-Generation Multi-Agent Scientific Reasoning",
            authors=["Dr. Alice", "Dr. Bob", "Dr. Charlie"],
            abstract="ArXiv version abstract.",
            year=2024,
            venue="arXiv preprint",
            doi=None,
            arxiv_id="2401.99999",
            source="arxiv",
        )
        c_pubmed = AcademicPaperCandidate(
            paper_id="pubmed:77777777",
            title="Next-Generation Multi-Agent Scientific Reasoning",
            authors=["Dr. Alice"],
            abstract="",
            year=2024,
            venue="Nat Sci",
            doi="10.1000/scireason.2024",
            source="pubmed",
        )

        merged_list = deduplicate_and_merge_candidates([c_openalex, c_s2, c_arxiv, c_pubmed])
        assert len(merged_list) == 1
        final = merged_list[0]

        # External IDs merged
        assert final.doi == "10.1000/scireason.2024"
        assert final.arxiv_id == "2401.99999"
        assert final.s2_id == "abcdef"

        # Longest abstract preserved
        assert "A comprehensive, highly detailed abstract" in final.abstract

        # All 4 sources aggregated
        sources = [s.strip() for s in final.source.split(",")]
        assert len(sources) == 4
        assert set(sources) == {"openalex", "semanticscholar", "arxiv", "pubmed"}

    def test_deduplication_preserves_unique_papers(self):
        papers = [
            AcademicPaperCandidate(
                paper_id=f"doi:10.1000/unique.{i}",
                title=f"Unique Scientific Paper Number {i} On Distinct Domain",
                doi=f"10.1000/unique.{i}",
                arxiv_id=f"2401.{20000+i}",
                source="openalex",
            )
            for i in range(25)
        ]
        unique = deduplicate_and_merge_candidates(papers)
        assert len(unique) == 25


class TestMultiSourceAcademicSearchExecution:
    """Adversarial testing of federated multi-source search."""

    @patch("requests.get")
    def test_search_clamping_and_bounds_enforcement(self, mock_get):
        """Verify strict clamping to [1, 40] with default 25."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {
                    "display_name": f"Deep Search Paper {i:03d}",
                    "doi": f"https://doi.org/10.1000/dsp.{i:03d}",
                    "id": f"https://openalex.org/W{i}",
                    "publication_year": 2023,
                    "cited_by_count": i * 10,
                    "authorships": [{"author": {"display_name": f"Author {i}"}}],
                    "abstract_inverted_index": {"Study": [0]},
                }
                for i in range(60)
            ]
        }
        mock_get.return_value = mock_resp

        search_tool = MultiSourceAcademicSearch(
            enable_openalex=True,
            enable_semanticscholar=False,
            enable_arxiv=False,
            enable_pubmed=False,
        )

        # 1. High limit (100) -> clamped to MAX_CANDIDATE_CAP (40)
        res_high = search_tool.search("machine learning", limit=100)
        assert len(res_high) == 40
        assert len(res_high) <= MAX_CANDIDATE_CAP

        # 2. Default limit (25) -> returns 25
        res_default = search_tool.search("machine learning")
        assert len(res_default) == 25

        # 3. Explicit limit 40 -> returns 40
        res_40 = search_tool.search("machine learning", limit=40)
        assert len(res_40) == 40

        # 4. Low limit (5) -> returns 5
        res_5 = search_tool.search("machine learning", limit=5)
        assert len(res_5) == 5

        # 5. Non-positive limit (0 or negative) -> clamped to MIN_CANDIDATE_CAP (1)
        res_0 = search_tool.search("machine learning", limit=0)
        assert len(res_0) == 1

        res_neg = search_tool.search("machine learning", limit=-10)
        assert len(res_neg) == 1

    def test_search_empty_queries(self):
        search_tool = MultiSourceAcademicSearch()
        assert search_tool.search("") == []
        assert search_tool.search("   ") == []
        assert search_tool.search("\n\t") == []

    @patch("requests.get")
    def test_search_scoring_and_ranking_integrity(self, mock_get):
        """Verify relevance scores are in [0, 1] and output is strictly sorted descending."""
        def mock_openalex(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "results": [
                    {
                        "display_name": "Highly Cited Landmark Paper",
                        "doi": "https://doi.org/10.1000/landmark",
                        "id": "https://openalex.org/W_LANDMARK",
                        "publication_year": 2024,
                        "cited_by_count": 50000,
                        "authorships": [{"author": {"display_name": "Landmark Author"}}],
                    },
                    {
                        "display_name": "Uncited Recent Preprint",
                        "doi": "https://doi.org/10.1000/uncited",
                        "id": "https://openalex.org/W_UNCITED",
                        "publication_year": 2024,
                        "cited_by_count": 0,
                        "authorships": [{"author": {"display_name": "Preprint Author"}}],
                    },
                    {
                        "display_name": "Older Mid-Impact Paper",
                        "doi": "https://doi.org/10.1000/older",
                        "id": "https://openalex.org/W_OLDER",
                        "publication_year": 2015,
                        "cited_by_count": 500,
                        "authorships": [{"author": {"display_name": "Older Author"}}],
                    },
                ]
            }
            return resp

        mock_get.side_effect = mock_openalex

        search_tool = MultiSourceAcademicSearch(
            enable_openalex=True,
            enable_semanticscholar=False,
            enable_arxiv=False,
            enable_pubmed=False,
        )

        results = search_tool.search("benchmark", limit=10)
        assert len(results) == 3

        # Invariant 1: All scores bounded in [0.0, 1.0]
        for r in results:
            assert r.relevance_score is not None
            assert 0.0 <= r.relevance_score <= 1.0

        # Invariant 2: Strictly sorted descending
        for i in range(len(results) - 1):
            assert results[i].relevance_score >= results[i + 1].relevance_score

    @patch("requests.get")
    def test_search_resilience_to_http_errors(self, mock_get):
        """Verify resilience when multiple providers fail with various HTTP error codes."""
        def mock_failing(url, **kwargs):
            resp = MagicMock()
            if "semanticscholar" in url:
                resp.status_code = 429
            elif "openalex" in url:
                resp.status_code = 503
            elif "arxiv" in url:
                resp.status_code = 500
            elif "ncbi" in url:
                resp.status_code = 200
                resp.json.return_value = {"esearchresult": {"idlist": []}}
            return resp

        mock_get.side_effect = mock_failing

        search_tool = MultiSourceAcademicSearch()
        # Must return empty list gracefully without throwing
        results = search_tool.search("failing test", limit=10)
        assert results == []


class TestCitationGraphTraverserExecution:
    """Adversarial testing of 1-hop forward/backward citation graph traversal."""

    @patch("requests.get")
    def test_strict_seed_exclusion_and_deduplication_via_string_id(self, mock_get):
        """
        Verify that seeds passed as DOI string are strictly excluded from citation results.
        """
        seed_doi = "10.1000/seed.primary"
        seed_arxiv = "2401.55555"
        seed_title = "Primary Seed Study on Foundation Models"

        def mock_s2_graph(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            if "citations" in url:
                resp.json.return_value = {
                    "data": [
                        # Seed paper returned by API (must be filtered out)
                        {
                            "citingPaper": {
                                "paperId": "seed_s2_id",
                                "title": seed_title,
                                "externalIds": {"DOI": seed_doi, "ArXiv": seed_arxiv},
                                "citationCount": 1000,
                            }
                        },
                        # Genuine new citing paper
                        {
                            "citingPaper": {
                                "paperId": "citing_001",
                                "title": "Follow-Up Expansion on Foundation Models",
                                "externalIds": {"DOI": "10.1000/citing.001"},
                                "citationCount": 50,
                            }
                        },
                    ]
                }
            elif "references" in url:
                resp.json.return_value = {
                    "data": [
                        # Baseline paper
                        {
                            "citedPaper": {
                                "paperId": "ref_001",
                                "title": "Foundational Predecessor Architecture",
                                "externalIds": {"DOI": "10.1000/ref.001"},
                                "citationCount": 5000,
                            }
                        }
                    ]
                }
            elif "openalex" in url:
                resp.json.return_value = {"results": []}
            return resp

        mock_get.side_effect = mock_s2_graph

        traverser = CitationGraphTraverser()

        results = traverser.traverse_1hop(
            seed_paper_ids=[seed_doi],
            include_forward=True,
            include_backward=True,
            total_limit=10,
        )

        assert len(results) == 2
        dois = [r.doi for r in results]
        assert seed_doi not in dois
        assert "10.1000/citing.001" in dois
        assert "10.1000/ref.001" in dois

        # Verify sorted descending by citation count (5000 > 50)
        assert results[0].doi == "10.1000/ref.001"
        assert results[0].citation_count == 5000
        assert results[1].doi == "10.1000/citing.001"
        assert results[1].citation_count == 50

    @patch("requests.get")
    def test_citation_graph_hard_limits_clamping(self, mock_get):
        """Verify total_limit clamping to MAX_GRAPH_LIMIT (40)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "citingPaper": {
                        "paperId": f"s2_{i}",
                        "title": f"Citing Paper {i}",
                        "externalIds": {"DOI": f"10.1000/cit.{i}"},
                        "citationCount": i * 5,
                    }
                }
                for i in range(60)
            ]
        }
        mock_get.return_value = mock_resp

        traverser = CitationGraphTraverser()

        # Requesting limit 100 should clamp to 40
        results = traverser.traverse_1hop(
            seed_paper_ids=["10.1000/seed.test"],
            include_forward=True,
            include_backward=False,
            total_limit=100,
        )
        assert len(results) == 40
        assert len(results) <= MAX_GRAPH_LIMIT

    def test_citation_graph_empty_and_whitespace_seeds(self):
        traverser = CitationGraphTraverser()
        assert traverser.traverse_1hop([]) == []
        assert traverser.traverse_1hop(["", "   "]) == []

    @patch("requests.get")
    def test_citation_graph_network_error_resilience(self, mock_get):
        mock_get.side_effect = Exception("Fatal connection reset by peer")

        traverser = CitationGraphTraverser()
        results = traverser.traverse_1hop(
            seed_paper_ids=["10.1000/seed.test"],
            include_forward=True,
            include_backward=True,
        )
        # Should gracefully return empty list without raising
        assert results == []
