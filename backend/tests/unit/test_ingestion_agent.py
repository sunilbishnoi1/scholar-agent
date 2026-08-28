"""
Unit tests for FullTextIngestionSpecialist (Ingestion Agent).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.agents.core.ingestion import FullTextIngestionSpecialist
from backend.agents.state import create_initial_agent_state
from backend.agents.tools.oa_resolver import OAResolutionResult
from backend.agents.tools.pdf_parser import ParsedDocument, ParsedSection
from backend.models.database import Base, PaperCache
from backend.rag.chunker import ChunkType


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.mark.asyncio
async def test_ingestion_agent_with_pdf_resolution(in_memory_db):
    mock_resolver = MagicMock()
    mock_resolver.resolve_paper.return_value = OAResolutionResult(
        doi="10.1234/test.pdf",
        is_oa=True,
        pdf_url="https://example.com/paper.pdf",
        pdf_bytes=b"%PDF-1.4 Fake PDF Content with sufficient bytes length to exceed 100 characters padding 1234567890 1234567890",
        source="unpaywall",
    )

    mock_parser = MagicMock()
    mock_parser.parse_pdf.return_value = ParsedDocument(
        paper_id="ref_1",
        doi="10.1234/test.pdf",
        title="Test Full Text Paper",
        authors=["Author One"],
        year=2023,
        abstract="Test abstract",
        markdown_text="# Test Full Text Paper\n\n## Methodology\nMethod content.\n\n## Results\nResult content.",
        sections=[
            ParsedSection(heading_level=2, title="Methodology", body="Method content.", chunk_type=ChunkType.METHODOLOGY),
            ParsedSection(heading_level=2, title="Results", body="Result content.", chunk_type=ChunkType.RESULTS),
        ],
        tables=[],
        equations=[],
        is_full_text=True,
    )

    agent = FullTextIngestionSpecialist(
        llm_client=None,
        oa_resolver=mock_resolver,
        pdf_parser=mock_parser,
        db_session=in_memory_db,
    )

    state = create_initial_agent_state(
        project_id="proj_ingest",
        research_question="Question",
    )
    state["papers"] = [
        {"id": "ref_1", "doi": "10.1234/test.pdf", "title": "Test Full Text Paper", "authors": ["Author One"]}
    ]

    new_state = await agent.run(state)

    assert new_state["papers_analyzed_full_text"] == 1
    assert "ref_1" in new_state["parsed_papers"]
    assert new_state["parsed_papers"]["ref_1"]["is_full_text"] is True
    assert len(new_state["paper_chunks"]) > 0

    # Verify cached in DB
    cached = in_memory_db.query(PaperCache).filter(PaperCache.doi == "10.1234/test.pdf").first()
    assert cached is not None
    assert cached.title == "Test Full Text Paper"


@pytest.mark.asyncio
async def test_ingestion_agent_abstract_fallback(in_memory_db):
    mock_resolver = MagicMock()
    mock_resolver.resolve_paper.return_value = OAResolutionResult(
        doi="10.1234/paywalled",
        is_oa=False,
        pdf_url=None,
        pdf_bytes=None,
        abstract_fallback={"abstract": "Fallback abstract content"},
        source="openalex_abstract",
    )

    agent = FullTextIngestionSpecialist(
        llm_client=None,
        oa_resolver=mock_resolver,
        db_session=in_memory_db,
    )

    state = create_initial_agent_state(
        project_id="proj_ingest_abs",
        research_question="Question",
    )
    state["papers"] = [
        {"id": "ref_2", "doi": "10.1234/paywalled", "title": "Paywalled Paper", "abstract": "Fallback abstract content"}
    ]

    new_state = await agent.run(state)

    assert new_state["papers_analyzed_abstract_only"] == 1
    assert new_state["parsed_papers"]["ref_2"]["is_full_text"] is False
    assert len(new_state["paper_chunks"]) >= 1

