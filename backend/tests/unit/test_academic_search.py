"""
Unit tests for Multi-Source Academic Search Tool (MultiSourceAcademicSearch).
Validates federated search across OpenAlex, Semantic Scholar, arXiv, and PubMed,
strict DOI deduplication, normalized title matching, and bounded candidate limits.
"""

from unittest.mock import MagicMock, patch
import pytest

try:
    from agents.schemas import AcademicPaperCandidate
    from agents.tools.academic_search import (
        MultiSourceAcademicSearch,
        deduplicate_and_merge_candidates,
        merge_candidate_into,
        normalize_arxiv_id,
        normalize_doi,
        normalize_title,
        reconstruct_openalex_abstract,
        titles_match,
    )
except ImportError:
    from backend.agents.schemas import AcademicPaperCandidate
    from backend.agents.tools.academic_search import (
        MultiSourceAcademicSearch,
        deduplicate_and_merge_candidates,
        merge_candidate_into,
        normalize_arxiv_id,
        normalize_doi,
        normalize_title,
        reconstruct_openalex_abstract,
        titles_match,
    )


class TestAcademicSearchNormalization:
    """Test suite for identifier and title normalization functions."""

    def test_normalize_doi_valid_and_prefixed(self):
        assert normalize_doi("https://doi.org/10.1000/182") == "10.1000/182"
        assert normalize_doi("http://dx.doi.org/10.1145/3318464.3389700") == "10.1145/3318464.3389700"
        assert normalize_doi("doi:10.1016/j.artint.2023.103980") == "10.1016/j.artint.2023.103980"
        assert normalize_doi("10.1000/182") == "10.1000/182"
        assert normalize_doi(None) is None
        assert normalize_doi("") is None
        assert normalize_doi("invalid_doi_string") is None

    def test_normalize_arxiv_id_versions_and_urls(self):
        assert normalize_arxiv_id("https://arxiv.org/abs/2401.00001v2") == "2401.00001"
        assert normalize_arxiv_id("arxiv:2401.00001v1.pdf") == "2401.00001"
        assert normalize_arxiv_id("2401.00001") == "2401.00001"
        assert normalize_arxiv_id(None) is None
        assert normalize_arxiv_id("") is None

    def test_normalize_title_and_fuzzy_matching(self):
        t1 = "Attention Is All You Need!"
        t2 = "Attention is all you need"
        t3 = "Attention Is All You Need: Architecture & Benchmarks"
        assert normalize_title(t1) == "attention is all you need"
        assert titles_match(t1, t2) is True
        assert titles_match(t1, t3) is True
        assert titles_match("Random Paper on Graph Theory", "Attention Is All You Need") is False
        assert titles_match("", "Some title") is False

    def test_reconstruct_openalex_abstract(self):
        inverted = {"We": [0], "propose": [1], "a": [2], "method": [3]}
        abstract = reconstruct_openalex_abstract(inverted)
        assert abstract == "We propose a method"
        assert reconstruct_openalex_abstract(None) == ""
        assert reconstruct_openalex_abstract({}) == ""


class TestCandidateDeduplicationAndMerging:
    """Test suite for candidate merging and deduplication across providers."""

    def test_deduplicate_by_doi(self):
        cand1 = AcademicPaperCandidate(
            paper_id="doi:10.1000/1",
            title="Transformer Scaling Laws",
            authors=["A. Researcher"],
            abstract="Short abstract",
            doi="10.1000/1",
            source="openalex",
        )
        cand2 = AcademicPaperCandidate(
            paper_id="arxiv:2401.12345",
            title="Transformer Scaling Laws",
            authors=["A. Researcher", "B. Scientist"],
            abstract="Much longer and more detailed abstract about scaling laws.",
            doi="10.1000/1",
            arxiv_id="2401.12345",
            citation_count=50,
            source="arxiv",
        )
        unique = deduplicate_and_merge_candidates([cand1, cand2])
        assert len(unique) == 1
        assert unique[0].doi == "10.1000/1"
        assert unique[0].arxiv_id == "2401.12345"
        assert unique[0].citation_count == 50
        assert "Much longer" in unique[0].abstract
        assert "openalex" in unique[0].source and "arxiv" in unique[0].source

    def test_deduplicate_by_arxiv_id(self):
        cand1 = AcademicPaperCandidate(
            paper_id="arxiv:2401.99999",
            title="Preprint on Quantum Algorithms",
            arxiv_id="2401.99999",
            source="arxiv",
        )
        cand2 = AcademicPaperCandidate(
            paper_id="s2:123456",
            title="Preprint on Quantum Algorithms",
            arxiv_id="2401.99999",
            s2_id="123456",
            source="semanticscholar",
        )
        unique = deduplicate_and_merge_candidates([cand1, cand2])
        assert len(unique) == 1
        assert unique[0].arxiv_id == "2401.99999"
        assert unique[0].s2_id == "123456"

    def test_deduplicate_by_normalized_title(self):
        cand1 = AcademicPaperCandidate(
            paper_id="paper_1",
            title="Emergent Abilities of Large Language Models: A Survey",
            authors=["Jason Wei"],
            source="openalex",
        )
        cand2 = AcademicPaperCandidate(
            paper_id="paper_2",
            title="Emergent Abilities of Large Language Models",
            authors=["Jason Wei", "Yi Tay"],
            source="semanticscholar",
        )
        unique = deduplicate_and_merge_candidates([cand1, cand2])
        assert len(unique) == 1


class TestMultiSourceAcademicSearchExecution:
    """Test suite for federated multi-source search execution."""

    @patch("requests.get")
    def test_search_bounded_limit_and_ranking(self, mock_get):
        # Mock OpenAlex response with 15 papers
        mock_resp_openalex = MagicMock()
        mock_resp_openalex.status_code = 200
        mock_resp_openalex.json.return_value = {
            "results": [
                {
                    "display_name": f"Paper Title {i}",
                    "doi": f"https://doi.org/10.1000/p.{i}",
                    "id": f"https://openalex.org/W{i}",
                    "publication_year": 2023,
                    "cited_by_count": 10 * i,
                    "authorships": [{"author": {"display_name": "Author 1"}}],
                    "abstract_inverted_index": {"Study": [0], "results": [1]},
                }
                for i in range(15)
            ]
        }

        mock_get.return_value = mock_resp_openalex

        search_tool = MultiSourceAcademicSearch(
            enable_openalex=True,
            enable_semanticscholar=False,
            enable_arxiv=False,
            enable_pubmed=False,
        )

        results = search_tool.search("machine learning", limit=5)
        assert len(results) == 5
        assert results[0].relevance_score is not None
        assert results[0].relevance_score >= results[-1].relevance_score

    @patch("requests.get")
    def test_arxiv_atom_xml_parsing(self, mock_get):
        atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
            <entry>
                <id>http://arxiv.org/abs/2401.01234v1</id>
                <title>Multi-Agent Reasoning with High Precision</title>
                <summary>We present a novel multi-agent reasoning framework.</summary>
                <author><name>Claude Shannon</name></author>
                <published>2024-01-15T00:00:00Z</published>
                <arxiv:doi>10.48550/arXiv.2401.01234</arxiv:doi>
            </entry>
        </feed>"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = atom_xml
        mock_get.return_value = mock_resp

        search_tool = MultiSourceAcademicSearch(
            enable_openalex=False,
            enable_semanticscholar=False,
            enable_arxiv=True,
            enable_pubmed=False,
        )
        candidates = search_tool.search_arxiv("reasoning", limit=5)
        assert len(candidates) == 1
        assert candidates[0].arxiv_id == "2401.01234"
        assert candidates[0].title == "Multi-Agent Reasoning with High Precision"
        assert candidates[0].authors == ["Claude Shannon"]
        assert candidates[0].year == 2024

    @patch("requests.get")
    def test_pubmed_esearch_and_esummary_parsing(self, mock_get):
        # 1. ESearch response
        mock_esearch = MagicMock()
        mock_esearch.status_code = 200
        mock_esearch.json.return_value = {
            "esearchresult": {"idlist": ["12345678"]}
        }

        # 2. ESummary response
        mock_esummary = MagicMock()
        mock_esummary.status_code = 200
        mock_esummary.json.return_value = {
            "result": {
                "12345678": {
                    "title": "Clinical Applications of AI in Healthcare.",
                    "authors": [{"name": "Doctor A"}],
                    "pubdate": "2023 Sep",
                    "source": "Lancet Digital Health",
                    "articleids": [{"idtype": "doi", "value": "10.1016/s2589-7500(23)00001-1"}],
                }
            }
        }

        mock_get.side_effect = [mock_esearch, mock_esummary]

        search_tool = MultiSourceAcademicSearch(
            enable_openalex=False,
            enable_semanticscholar=False,
            enable_arxiv=False,
            enable_pubmed=True,
        )
        candidates = search_tool.search_pubmed("clinical AI", limit=5)
        assert len(candidates) == 1
        assert candidates[0].doi == "10.1016/s2589-7500(23)00001-1"
        assert candidates[0].title == "Clinical Applications of AI in Healthcare"
        assert candidates[0].venue == "Lancet Digital Health"
        assert candidates[0].year == 2023

    @patch("requests.get")
    def test_resilience_on_provider_error(self, mock_get):
        # Simulate network failure
        mock_get.side_effect = Exception("Connection timed out")

        search_tool = MultiSourceAcademicSearch()
        results = search_tool.search("quantum computing", limit=10)
        # Should gracefully return empty list without raising
        assert results == []
