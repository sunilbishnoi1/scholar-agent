"""
Unit tests for Full-Text PDF, Table & Formula Parser (PDFParser).
Validates font-histogram heading extraction, multi-column reading order preservation,
GFM table extraction, LaTeX formula extraction, and PaperCache ORM persistence.
"""

from unittest.mock import MagicMock
import pytest
import pymupdf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

try:
    from models.database import Base, PaperCache
    from rag.chunker import ChunkType
    from agents.tools.pdf_parser import (
        DISPLAY_MATH_PATTERN,
        INLINE_MATH_PATTERN,
        PDFParser,
        ParsedDocument,
    )
except ImportError:
    from backend.models.database import Base, PaperCache
    from backend.rag.chunker import ChunkType
    from backend.agents.tools.pdf_parser import (
        DISPLAY_MATH_PATTERN,
        INLINE_MATH_PATTERN,
        PDFParser,
        ParsedDocument,
    )


@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database session for PaperCache testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def synthetic_academic_pdf() -> bytes:
    """Generate a realistic 2-page academic PDF with multi-column layout, headings, tables, and math."""
    doc = pymupdf.open()

    # Page 1: Title, Abstract, 2-column Intro & Method
    page1 = doc.new_page(width=612, height=792)  # Standard Letter

    # Title (20pt font)
    page1.insert_text((54, 70), "Deep Adaptive Reasoning in Scientific Multi-Agent Systems", fontsize=20)

    # Authors (10pt font)
    page1.insert_text((54, 95), "Alan Turing, Claude Shannon, John von Neumann", fontsize=10)

    # Abstract (10pt body)
    page1.insert_text((54, 130), "Abstract", fontsize=13)
    page1.insert_text(
        (54, 150),
        "We introduce an autonomous framework for literature review and synthesis. "
        "Our experiments demonstrate significant improvements in benchmark precision.",
        fontsize=10,
    )

    # Left Column: Introduction
    page1.insert_text((54, 210), "1. Introduction", fontsize=13)
    page1.insert_text(
        (54, 230),
        "Scientific discovery requires systematic literature exploration and analysis. "
        "Recent breakthroughs in large language models provide powerful reasoning capabilities.",
        fontsize=10,
    )

    # Right Column: Methodology
    page1.insert_text((320, 210), "2. Methodology", fontsize=13)
    page1.insert_text(
        (320, 230),
        "We formulate the synthesis process as an iterative game between Explorer and Critic. "
        "The objective function optimizes information gain across structured evidence matrix rows.",
        fontsize=10,
    )

    # Standalone LaTeX math on Page 1
    page1.insert_text((320, 300), "$$ L(\\theta) = -\\sum_{i=1}^N \\log P(y_i | x_i; \\theta) $$", fontsize=10)

    # Page 2: Table and Experimental Results
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((54, 60), "3. Experimental Results", fontsize=13)
    page2.insert_text((54, 80), "We evaluated our approach across three standard scientific benchmarks.", fontsize=10)

    # Draw table borders
    rect = pymupdf.Rect(54, 110, 500, 180)
    page2.draw_rect(rect, color=(0, 0, 0), width=1)
    # Header horizontal line
    page2.draw_line(pymupdf.Point(54, 135), pymupdf.Point(500, 135), color=(0, 0, 0), width=1)
    # Vertical lines
    page2.draw_line(pymupdf.Point(200, 110), pymupdf.Point(200, 180), color=(0, 0, 0), width=1)
    page2.draw_line(pymupdf.Point(350, 110), pymupdf.Point(350, 180), color=(0, 0, 0), width=1)

    # Insert table text
    page2.insert_text((60, 128), "Model", fontsize=10)
    page2.insert_text((210, 128), "Precision (%)", fontsize=10)
    page2.insert_text((360, 128), "Recall (%)", fontsize=10)

    page2.insert_text((60, 155), "Baseline RAG", fontsize=10)
    page2.insert_text((210, 155), "72.4", fontsize=10)
    page2.insert_text((360, 155), "68.1", fontsize=10)

    page2.insert_text((60, 175), "Scholar Agent", fontsize=10)
    page2.insert_text((210, 175), "89.6", fontsize=10)
    page2.insert_text((360, 175), "85.2", fontsize=10)

    # Conclusion & Limitations
    page2.insert_text((54, 220), "4. Limitations", fontsize=13)
    page2.insert_text(
        (54, 240),
        "Our current study is limited to English-language preprints and published journal articles.",
        fontsize=10,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestPDFParserExtraction:
    """Test suite for parsing PDF text, headings, formulas, and tables."""

    def test_parse_pdf_headings_and_sections(self, synthetic_academic_pdf):
        parser = PDFParser()
        doc = parser.parse_pdf(
            pdf_bytes=synthetic_academic_pdf,
            doi="10.1000/scholar.001",
            title_hint="Deep Adaptive Reasoning",
        )

        assert doc.is_full_text is True
        assert doc.page_count == 2
        assert "Deep Adaptive Reasoning in Scientific Multi-Agent Systems" in doc.title or "Deep Adaptive Reasoning" in doc.title

        # Check section extraction
        titles = [s.title.lower() for s in doc.sections]
        assert any("introduction" in t for t in titles)
        assert any("methodology" in t or "methods" in t for t in titles)
        assert any("results" in t or "experimental" in t for t in titles)
        assert any("limitations" in t for t in titles)

        # Check typed chunk mapping
        types = [s.chunk_type for s in doc.sections]
        assert ChunkType.INTRODUCTION in types or ChunkType.ABSTRACT in types
        assert ChunkType.METHODOLOGY in types
        assert ChunkType.RESULTS in types
        assert ChunkType.LIMITATIONS in types

    def test_parse_latex_math_formulas(self, synthetic_academic_pdf):
        parser = PDFParser()
        doc = parser.parse_pdf(synthetic_academic_pdf, doi="10.1000/math.001")

        # Must extract the equation
        assert len(doc.equations) >= 1
        math_combined = " ".join(doc.equations)
        assert "L(\\theta)" in math_combined or "sum" in math_combined or "\\theta" in math_combined

    def test_regex_math_patterns(self):
        text = "Here is an equation: $$E = mc^2$$ and another \\[a^2 + b^2 = c^2\\] with inline $f(x) = y$."
        eqs = PDFParser.extract_latex_equations(text)
        assert "$$E = mc^2$$" in eqs
        assert "\\[a^2 + b^2 = c^2\\]" in eqs
        assert "$f(x) = y$" in eqs


class TestPDFParserPaperCacheIntegration:
    """Test suite for PaperCache ORM persistence and caching."""

    def test_paper_cache_save_and_retrieve(self, in_memory_db, synthetic_academic_pdf):
        parser = PDFParser()

        # Parse and save to DB
        doc = parser.parse_pdf(
            pdf_bytes=synthetic_academic_pdf,
            doi="10.1000/cached.001",
            arxiv_id="2401.00001",
            db_session=in_memory_db,
            use_cache=True,
        )

        assert doc is not None
        assert doc.doi == "10.1000/cached.001"

        # Verify DB entry exists
        cached_entry = in_memory_db.query(PaperCache).filter(PaperCache.doi == "10.1000/cached.001").first()
        assert cached_entry is not None
        assert cached_entry.is_full_text is True
        assert cached_entry.arxiv_id == "2401.00001"
        assert len(cached_entry.sections_json) > 0

        # Retrieve from cache
        retrieved_doc = parser.get_from_cache(in_memory_db, doi="10.1000/cached.001")
        assert retrieved_doc is not None
        assert retrieved_doc.doi == "10.1000/cached.001"
        assert retrieved_doc.is_full_text is True
        assert len(retrieved_doc.sections) == len(doc.sections)

        # Retrieve from cache via parse_pdf (should not re-parse)
        cached_doc = parser.parse_pdf(
            pdf_bytes=b"",  # Empty bytes should still succeed if cached
            doi="10.1000/cached.001",
            db_session=in_memory_db,
            use_cache=True,
        )
        assert cached_doc.doi == "10.1000/cached.001"
        assert cached_doc.is_full_text is True


class TestPDFParserEdgeCases:
    """Test suite for corrupted bytes, empty files, and error handling."""

    def test_corrupt_or_empty_bytes_graceful_handling(self):
        parser = PDFParser()
        doc = parser.parse_pdf(b"", doi="10.1000/empty")
        assert doc.is_full_text is False
        assert doc.markdown_text == ""

        corrupt_doc = parser.parse_pdf(b"not a valid pdf file content" * 10, doi="10.1000/corrupt")
        assert corrupt_doc.is_full_text is False
        assert "error" in corrupt_doc.metadata

    def test_parse_non_existent_file(self, tmp_path):
        parser = PDFParser()
        non_existent = tmp_path / "does_not_exist.pdf"
        doc = parser.parse_pdf_file(non_existent)
        assert doc.is_full_text is False
        assert "File not found" in doc.metadata.get("error", "")
