"""
Unit tests for EvidenceMatrixBuilder (Matrix Builder Agent).
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.agents.core.matrix_builder import EvidenceMatrixBuilder
from backend.agents.schemas import EvidenceMatrixRow
from backend.agents.state import create_initial_agent_state
from backend.models.database import Base, EvidenceMatrixEntry


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_matrix_builder_isolate_high_signal_context():
    paper = {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": "We introduce BERT...",
        "is_full_text": True,
        "sections": [
            {"heading": "1. Introduction", "content": "Intro noise", "section_type": "introduction"},
            {"heading": "3. BERT Methodology", "content": "Masked language model and next sentence prediction.", "section_type": "methodology"},
            {"heading": "4. Experiments & Results", "content": "GLUE score 80.5%.", "section_type": "results"},
            {"heading": "5. Limitations", "content": "Heavy pre-training compute requirements.", "section_type": "limitations"},
        ],
        "tables": [{"table_id": "tab_1", "caption": "GLUE Benchmark results"}],
    }

    context = EvidenceMatrixBuilder.isolate_high_signal_context(paper)
    assert "BERT Methodology" in context
    assert "GLUE score" in context
    assert "Limitations" in context
    assert "Intro noise" not in context


@pytest.mark.asyncio
async def test_matrix_builder_run(in_memory_db):
    agent = EvidenceMatrixBuilder(llm_client=None, db_session=in_memory_db)

    state = create_initial_agent_state(
        project_id="proj_mat",
        research_question="NLP Models",
    )
    state["parsed_papers"] = {
        "ref_1": {
            "paper_id": "ref_1",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Devlin et al."],
            "year": 2019,
            "abstract": "We introduce BERT transformer architecture.",
            "is_full_text": True,
        }
    }

    new_state = await agent.run(state)

    assert len(new_state["evidence_matrix"]) == 1
    assert new_state["evidence_matrix"][0]["paper_id"] == "ref_1"
    assert "| **[ref_1]** |" in new_state["evidence_matrix_markdown"]

    # Verify persisted in database
    entries = in_memory_db.query(EvidenceMatrixEntry).filter(EvidenceMatrixEntry.project_id == "proj_mat").all()
    assert len(entries) == 1
    assert entries[0].paper_id == "ref_1"
    assert entries[0].title == "BERT: Pre-training of Deep Bidirectional Transformers"

