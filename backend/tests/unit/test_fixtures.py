# Unit Tests for Comprehensive Test Fixtures
# Verifies db_session, synthetic_scientific_pdf, mock_oa_resolver, and mock_academic_search

from pathlib import Path
import pytest
from sqlalchemy import select, text
import fitz

from models.database import User, ResearchProject, PaperReference


@pytest.mark.unit
class TestConftestFixtures:
    """Test suite verifying all shared fixtures in backend/tests/conftest.py."""

    def test_db_session_in_memory_crud(self, db_session):
        """Verify in-memory SQLite db_session creates tables and supports clean CRUD."""
        # Create a user
        user = User(
            email="researcher@university.edu",
            name="Dr. Alan Turing",
            hashed_password="securehash123",
            institution="Cambridge",
            tier="pro",
            monthly_budget_usd=50.0,
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        assert user.id is not None
        assert user.name == "Dr. Alan Turing"

        # Create a research project
        project = ResearchProject(
            user_id=user.id,
            title="Next-Gen Autonomous Scientific Reasoning",
            research_question="How do autonomous multi-agent DAGs improve literature review quality?",
            keywords=["multi-agent", "literature-review", "reasoning"],
            subtopics=["Architecture", "Empirical Evaluation"],
            status="active",
        )
        db_session.add(project)
        db_session.commit()
        db_session.refresh(project)

        assert project.id is not None
        assert project.user_id == user.id

        # Query back
        stmt = select(ResearchProject).where(ResearchProject.user_id == user.id)
        queried = db_session.execute(stmt).scalars().all()
        assert len(queried) == 1
        assert queried[0].title == "Next-Gen Autonomous Scientific Reasoning"

    def test_synthetic_scientific_pdf_fixture(self, synthetic_scientific_pdf_bytes: bytes, synthetic_scientific_pdf_path: str):
        """Verify synthetic scientific PDF fixture generates multi-page PDF with headings, math, and table."""
        assert len(synthetic_scientific_pdf_bytes) > 500
        assert Path(synthetic_scientific_pdf_path).exists()

        # Parse with PyMuPDF
        doc = fitz.open(stream=synthetic_scientific_pdf_bytes, filetype="pdf")
        assert len(doc) == 2  # 2 pages

        # Verify Page 1 content (Headings, Math, Methodology)
        page1_text = doc[0].get_text()
        assert "Deep Transformer Reasoning in Multi-Agent Scientific Discovery" in page1_text
        assert "# Abstract" in page1_text
        assert "# 1. Introduction" in page1_text
        assert "$E = mc^2$" in page1_text
        assert r"\mathcal{L}_{total}" in page1_text
        assert "# 2. Methodology & Architecture" in page1_text

        # Verify Page 2 content (Results, Table, Limitations, References)
        page2_text = doc[1].get_text()
        assert "# 3. Empirical Results & Evaluation" in page2_text
        assert "| Benchmark | Metric | Baseline | ScholarAgent |" in page2_text
        assert "PubMed-QA" in page2_text
        assert "94.6%" in page2_text
        assert "# 4. Limitations & Threats to Validity" in page2_text
        assert "# 5. References" in page2_text

        doc.close()

    def test_mock_oa_resolver_3tier_cascade(self, mock_oa_resolver):
        """Verify mock OA resolver implements 3-tier cascade and never throws."""
        # Tier 1: OpenAccess DOI
        res_oa = mock_oa_resolver.resolve_paper(doi="10.1000/openaccess.001")
        assert res_oa["is_full_text"] is True
        assert res_oa["source"] == "unpaywall"
        assert res_oa["pdf_bytes"] is not None
        assert res_oa["abstract_fallback"] is None

        # Tier 2: arXiv ID
        res_arxiv = mock_oa_resolver.resolve_paper(arxiv_id="2401.01234")
        assert res_arxiv["is_full_text"] is True
        assert res_arxiv["source"] == "arxiv"
        assert res_arxiv["pdf_bytes"] is not None

        # Tier 3: Paywalled DOI fallback
        res_paywalled = mock_oa_resolver.resolve_paper(doi="10.1000/paywalled.001", title="Paywalled Nature Article")
        assert res_paywalled["is_full_text"] is False
        assert res_paywalled["source"] == "abstract_fallback"
        assert res_paywalled["pdf_bytes"] is None
        assert res_paywalled["abstract_fallback"] is not None
        assert "abstract" in res_paywalled["abstract_fallback"]

    def test_mock_academic_search_and_snowballing(self, mock_academic_search):
        """Verify mock academic search returns deduplicated papers and 1-hop citations."""
        results = mock_academic_search.search("autonomous scientific reasoning", limit=10)
        assert len(results) == 2
        assert results[0]["doi"] == "10.1000/scholar.001"
        assert "A. Turing" in results[0]["authors"]

        # 1-hop snowballing
        snowball = mock_academic_search.traverse_1hop(["paper_001"])
        assert len(snowball) == 1
        assert snowball[0]["doi"] == "10.1000/scholar.003"
