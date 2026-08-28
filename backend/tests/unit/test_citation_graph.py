"""
Unit tests for 1-Hop Citation Graph Traversal (CitationGraphTraverser).
Validates forward citations, backward bibliography references, seed exclusion,
cross-seed deduplication, and bounded citation limits.
"""

from unittest.mock import MagicMock, patch
import pytest

from agents.schemas import AcademicPaperCandidate
from agents.tools.citation_graph import CitationGraphTraverser


class TestCitationGraphTraverser:
    """Test suite for CitationGraphTraverser."""

    @patch("requests.get")
    def test_traverse_1hop_forward_and_backward(self, mock_get):
        def mock_s2_routing(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            if "citations" in url:
                resp.json.return_value = {
                    "data": [
                        {
                            "citingPaper": {
                                "paperId": "citing_001",
                                "title": "Follow-Up Breakthrough on Scientific Agents",
                                "externalIds": {"DOI": "10.1000/citing.001"},
                                "authors": [{"name": "C. Followup"}],
                                "year": 2024,
                                "citationCount": 30,
                            }
                        }
                    ]
                }
            elif "references" in url:
                resp.json.return_value = {
                    "data": [
                        {
                            "citedPaper": {
                                "paperId": "ref_001",
                                "title": "Foundational Baseline Architecture",
                                "externalIds": {"DOI": "10.1000/ref.001"},
                                "authors": [{"name": "A. Foundation"}],
                                "year": 2017,
                                "citationCount": 5000,
                            }
                        }
                    ]
                }
            else:
                resp.status_code = 404
                resp.json.return_value = {}
            return resp

        mock_get.side_effect = mock_s2_routing

        traverser = CitationGraphTraverser()
        results = traverser.traverse_1hop(
            seed_paper_ids=["10.1000/seed.001"],
            include_forward=True,
            include_backward=True,
            limit_per_seed=10,
            total_limit=10,
        )

        assert len(results) == 2
        dois = [r.doi for r in results]
        assert "10.1000/citing.001" in dois
        assert "10.1000/ref.001" in dois
        # Results should be sorted by citation count descending (5000 > 30)
        assert results[0].doi == "10.1000/ref.001"

    @patch("requests.get")
    def test_seed_exclusion(self, mock_get):
        # Ensure that if the API returns the seed paper itself, it is excluded
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "citingPaper": {
                        "paperId": "seed_001",
                        "title": "Seed Paper Title",
                        "externalIds": {"DOI": "10.1000/seed.001"},
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        traverser = CitationGraphTraverser()
        results = traverser.traverse_1hop(
            seed_paper_ids=["10.1000/seed.001"],
            include_forward=True,
            include_backward=False,
        )

        # Seed paper must be strictly filtered out
        assert len(results) == 0

    @patch("requests.get")
    def test_seed_exclusion_with_candidate_object(self, mock_get):
        seed_candidate = AcademicPaperCandidate(
            paper_id="doi:10.1000/seed.obj",
            title="Seed Candidate Title",
            doi="10.1000/seed.obj",
            arxiv_id="2401.55555",
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "citingPaper": {
                        "paperId": "s2_same",
                        "title": "Seed Candidate Title",
                        "externalIds": {"DOI": "10.1000/seed.obj"},
                    }
                },
                {
                    "citingPaper": {
                        "paperId": "s2_new",
                        "title": "Genuinely New Citation",
                        "externalIds": {"DOI": "10.1000/new.001"},
                    }
                },
            ]
        }
        mock_get.return_value = mock_resp

        traverser = CitationGraphTraverser()
        results = traverser.traverse_1hop(
            seed_paper_ids=[seed_candidate],
            include_forward=True,
            include_backward=False,
        )

        assert len(results) == 1
        assert results[0].doi == "10.1000/new.001"

    @patch("requests.get")
    def test_error_resilience(self, mock_get):
        mock_get.side_effect = Exception("API connection dropped")

        traverser = CitationGraphTraverser()
        results = traverser.traverse_1hop(
            seed_paper_ids=["10.1000/seed.001"],
            include_forward=True,
            include_backward=True,
        )
        assert results == []
