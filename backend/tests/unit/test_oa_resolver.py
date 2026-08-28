"""
Unit tests for Multi-Tier Open-Access Resolution Cascade (OAResolver).
Validates Tier 1 (Unpaywall/OpenAlex), Tier 2 (arXiv/S2), and Tier 3 (Structured Abstract Fallback),
along with anti-spoofing binary checks, normalization, and zero unhandled exception guarantees.
"""

from unittest.mock import MagicMock, patch
import pytest
import requests

try:
    from agents.tools.oa_resolver import (
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
except ImportError:
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


class TestIdentifierNormalizationAndValidation:
    """Test suite for DOI, arXiv ID normalization and PDF binary validation."""

    def test_normalize_doi_formats(self):
        assert normalize_doi("https://doi.org/10.1038/s41586-020-2649-2") == "10.1038/s41586-020-2649-2"
        assert normalize_doi("http://dx.doi.org/10.1145/3318464.3389700") == "10.1145/3318464.3389700"
        assert normalize_doi("doi:10.1016/j.artint.2023.103980") == "10.1016/j.artint.2023.103980"
        assert normalize_doi("10.1000/182/") == "10.1000/182"
        assert normalize_doi("10.1000/182.") == "10.1000/182"
        assert normalize_doi("10.1000/182") == "10.1000/182"
        assert normalize_doi(None) is None
        assert normalize_doi("") is None
        assert normalize_doi("not-a-doi") is None

    def test_normalize_arxiv_id_formats(self):
        assert normalize_arxiv_id("https://arxiv.org/abs/2401.01234v2") == "2401.01234v2"
        assert normalize_arxiv_id("https://arxiv.org/pdf/2401.01234.pdf") == "2401.01234"
        assert normalize_arxiv_id("arxiv:2401.01234") == "2401.01234"
        assert normalize_arxiv_id("2401.01234") == "2401.01234"
        assert normalize_arxiv_id("math.GT/0309136") == "math.GT/0309136"
        assert normalize_arxiv_id(None, doi="10.48550/arXiv.2401.01234") == "2401.01234"
        assert normalize_arxiv_id(None) is None
        assert normalize_arxiv_id("") is None

    def test_is_valid_pdf_bytes_magic_bytes_and_html_rejection(self):
        valid_pdf = b"%PDF-1.7\n" + b"A" * 1024
        assert is_valid_pdf_bytes(valid_pdf) is True

        # Short payload
        assert is_valid_pdf_bytes(b"%PDF-1.4 short") is False
        assert is_valid_pdf_bytes(None) is False
        assert is_valid_pdf_bytes(b"") is False

        # HTML paywall masquerading as 200 OK
        html_payload = b"<!DOCTYPE html><html><head><title>403 Forbidden</title></head><body>Login required</body></html>" + b" " * 1024
        assert is_valid_pdf_bytes(html_payload) is False

        # PDF header containing HTML tag inside header
        fake_pdf = b"%PDF-1.4\n<html><body>Access Denied</body></html>" + b" " * 1024
        assert is_valid_pdf_bytes(fake_pdf) is False

    def test_reconstruct_openalex_abstract(self):
        inverted = {
            "We": [0],
            "propose": [1],
            "an": [2],
            "autonomous": [3],
            "multi-agent": [4],
            "system.": [5],
        }
        reconstructed = reconstruct_openalex_abstract(inverted)
        assert reconstructed == "We propose an autonomous multi-agent system."
        assert reconstruct_openalex_abstract(None) == ""
        assert reconstruct_openalex_abstract({}) == ""

    def test_extract_openalex_mesh_and_concepts(self):
        mesh_raw = [
            {"descriptor_name": "Artificial Intelligence", "qualifier_name": "methods"},
            {"descriptor_name": "Machine Learning", "qualifier_name": None},
        ]
        mesh_terms = extract_openalex_mesh_terms(mesh_raw)
        assert mesh_terms == ["Artificial Intelligence - methods", "Machine Learning"]
        assert extract_openalex_mesh_terms(None) == []

        concepts_raw = [
            {"display_name": "Computer Science", "score": 0.95},
            {"display_name": "Deep Learning", "score": 0.88},
        ]
        concepts = extract_openalex_concepts(concepts_raw)
        assert concepts == ["Computer Science", "Deep Learning"]
        assert extract_openalex_concepts(None) == []


class TestOAResolutionResultModel:
    """Test suite for OAResolutionResult Pydantic and subscripting compatibility."""

    def test_dual_interface_attribute_and_item_access(self):
        res = OAResolutionResult(
            pdf_bytes=b"%PDF-1.5 test",
            source="unpaywall",
            is_full_text=True,
            doi="10.1000/182",
            resolution_tier=1,
        )
        # Attribute access
        assert res.is_full_text is True
        assert res.source == "unpaywall"
        assert res.doi == "10.1000/182"
        assert res.resolution_tier == 1

        # Dict subscripting access
        assert res["is_full_text"] is True
        assert res["source"] == "unpaywall"
        assert res["doi"] == "10.1000/182"
        assert "is_full_text" in res
        assert res.get("source") == "unpaywall"
        assert res.get("non_existent", "default_val") == "default_val"


class TestOAResolverCascadeExecution:
    """Test suite for Tier 1, Tier 2, and Tier 3 cascade resolution paths."""

    @pytest.fixture
    def sample_pdf_bytes(self):
        return b"%PDF-1.7\n" + b"Binary PDF Content Stream..." + b"0" * 1024

    @patch("requests.Session.get")
    def test_tier1_unpaywall_success(self, mock_get, sample_pdf_bytes):
        # 1. Unpaywall API response
        mock_api_resp = MagicMock()
        mock_api_resp.status_code = 200
        mock_api_resp.json.return_value = {
            "is_oa": True,
            "best_oa_location": {
                "url_for_pdf": "https://publisher.org/paper.pdf",
                "url_for_landing_page": "https://publisher.org/paper",
            },
        }

        # 2. PDF Download response
        mock_pdf_resp = MagicMock()
        mock_pdf_resp.status_code = 200
        mock_pdf_resp.iter_content.return_value = [sample_pdf_bytes]

        mock_get.side_effect = [mock_api_resp, mock_pdf_resp]

        resolver = OAResolver()
        result = resolver.resolve_paper(doi="10.1038/s41586-020-2649-2")

        assert result.is_full_text is True
        assert result.source == "unpaywall"
        assert result.resolution_tier == 1
        assert result.pdf_bytes == sample_pdf_bytes
        assert result.pdf_url == "https://publisher.org/paper.pdf"

    @patch("requests.Session.get")
    def test_tier1_openalex_success_when_unpaywall_fails(self, mock_get, sample_pdf_bytes):
        # 1. Unpaywall API 404
        mock_unpaywall_resp = MagicMock()
        mock_unpaywall_resp.status_code = 404

        # 2. OpenAlex API 200
        mock_openalex_resp = MagicMock()
        mock_openalex_resp.status_code = 200
        mock_openalex_resp.json.return_value = {
            "title": "OpenAlex Resolved Paper",
            "doi": "https://doi.org/10.1000/oa.001",
            "best_oa_location": {
                "pdf_url": "https://openalex.org/pdf/001.pdf",
                "landing_page_url": "https://openalex.org/w/001",
            },
            "abstract_inverted_index": {"Study": [0], "results": [1]},
        }

        # 3. PDF Download response
        mock_pdf_resp = MagicMock()
        mock_pdf_resp.status_code = 200
        mock_pdf_resp.iter_content.return_value = [sample_pdf_bytes]

        mock_get.side_effect = [mock_unpaywall_resp, mock_openalex_resp, mock_pdf_resp]

        resolver = OAResolver()
        result = resolver.resolve_paper(doi="10.1000/oa.001")

        assert result.is_full_text is True
        assert result.source == "openalex"
        assert result.resolution_tier == 1
        assert result.pdf_bytes == sample_pdf_bytes

    @patch("requests.Session.get")
    def test_tier2_arxiv_direct_download(self, mock_get, sample_pdf_bytes):
        # Direct arXiv download
        mock_pdf_resp = MagicMock()
        mock_pdf_resp.status_code = 200
        mock_pdf_resp.iter_content.return_value = [sample_pdf_bytes]

        mock_get.return_value = mock_pdf_resp

        resolver = OAResolver()
        result = resolver.resolve_paper(arxiv_id="2401.01234")

        assert result.is_full_text is True
        assert result.source == "arxiv"
        assert result.resolution_tier == 2
        assert result.pdf_bytes == sample_pdf_bytes
        assert result.landing_page_url == "https://arxiv.org/abs/2401.01234"

    @patch("requests.Session.get")
    def test_tier2_semantic_scholar_cdn(self, mock_get, sample_pdf_bytes):
        # 1. Unpaywall 404
        mock_unpaywall_resp = MagicMock()
        mock_unpaywall_resp.status_code = 404

        # 2. OpenAlex 404
        mock_openalex_resp = MagicMock()
        mock_openalex_resp.status_code = 404

        # 3. Semantic Scholar 200 with openAccessPdf
        mock_s2_resp = MagicMock()
        mock_s2_resp.status_code = 200
        mock_s2_resp.json.return_value = {
            "title": "Semantic Scholar OA Preprint",
            "abstract": "Preprint abstract text",
            "openAccessPdf": {
                "url": "https://cdn.semanticscholar.org/preprint.pdf",
                "status": "HYBRID",
            },
        }

        # 4. PDF Download
        mock_pdf_resp = MagicMock()
        mock_pdf_resp.status_code = 200
        mock_pdf_resp.iter_content.return_value = [sample_pdf_bytes]

        mock_get.side_effect = [mock_unpaywall_resp, mock_openalex_resp, mock_s2_resp, mock_pdf_resp]

        resolver = OAResolver()
        result = resolver.resolve_paper(doi="10.1000/s2.oa")

        assert result.is_full_text is True
        assert result.source == "semantic_scholar"
        assert result.resolution_tier == 2
        assert result.pdf_bytes == sample_pdf_bytes

    @patch("requests.Session.get")
    def test_tier3_graceful_paywall_fallback(self, mock_get):
        # 1. Unpaywall returns not OA
        mock_unpaywall = MagicMock()
        mock_unpaywall.status_code = 200
        mock_unpaywall.json.return_value = {"is_oa": False}

        # 2. OpenAlex returns structured abstract but no OA PDF
        mock_openalex = MagicMock()
        mock_openalex.status_code = 200
        mock_openalex.json.return_value = {
            "title": "Paywalled High-Impact Journal Article",
            "publication_year": 2024,
            "abstract_inverted_index": {
                "This": [0],
                "paper": [1],
                "analyzes": [2],
                "paywalled": [3],
                "frontiers.": [4],
            },
            "mesh": [{"descriptor_name": "Neuroscience", "qualifier_name": "trends"}],
            "concepts": [{"display_name": "Brain-Computer Interface"}],
            "best_oa_location": None,
            "primary_location": {"landing_page_url": "https://nature.com/articles/123", "pdf_url": None},
        }

        # 3. S2 returns no OA PDF
        mock_s2 = MagicMock()
        mock_s2.status_code = 200
        mock_s2.json.return_value = {"openAccessPdf": None}

        mock_get.side_effect = [mock_unpaywall, mock_openalex, mock_s2]

        resolver = OAResolver()
        result = resolver.resolve_paper(doi="10.1038/nature12345")

        assert result.is_full_text is False
        assert result.pdf_bytes is None
        assert result.source == "abstract_fallback"
        assert result.resolution_tier == 3
        assert result.abstract_fallback is not None
        assert result.abstract_fallback["title"] == "Paywalled High-Impact Journal Article"
        assert "This paper analyzes paywalled frontiers." in result.abstract_fallback["abstract"]
        assert "Neuroscience - trends" in result.abstract_fallback["mesh_terms"]
        assert "Brain-Computer Interface" in result.abstract_fallback["concepts"]

    @patch("requests.Session.get")
    def test_never_raises_on_network_timeout(self, mock_get):
        # Simulate network timeout
        mock_get.side_effect = requests.exceptions.ConnectTimeout("Connection timed out to Unpaywall")

        resolver = OAResolver()
        result = resolver.resolve_paper(doi="10.1000/timeout.001", title="Resilient Title")

        assert result.is_full_text is False
        assert result.pdf_bytes is None
        assert result.source == "abstract_fallback"
        assert result.resolution_tier == 3
        assert result.title == "Resilient Title"

    @pytest.mark.asyncio
    async def test_async_resolution_and_batch(self):
        resolver = OAResolver()
        with patch.object(resolver, "resolve_paper") as mock_resolve:
            mock_resolve.return_value = OAResolutionResult(
                is_full_text=False,
                source="abstract_fallback",
                title="Batch Paper",
            )
            res_async = await resolver.resolve_paper_async(doi="10.1000/async.001")
            assert res_async.title == "Batch Paper"

            batch_res = resolver.resolve_batch([{"doi": "10.1000/1"}, {"doi": "10.1000/2"}])
            assert len(batch_res) == 2
