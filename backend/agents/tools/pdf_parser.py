"""
Full-Text PDF, Table & Formula Parser for Academic Literature.

Converts raw PDF bytes into hierarchical structured Markdown with:
- Academic section headings (# Title, ## Section, ### Subsection)
- GitHub-Flavored Markdown tables extracted via PyMuPDF / pdfplumber
- LaTeX mathematical formulas ($...$, $$...$$, \\begin{equation}...\\end{equation})
- Multi-column layout awareness (column-first reading order)
- Integration with PaperCache for DOI/arXiv deduplication and persistent caching
- Direct compatibility with SectionAwareChunker for downstream RAG indexing
"""

from __future__ import annotations

import io
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pymupdf  # PyMuPDF 1.23+
from sqlalchemy.orm import Session

try:
    from models.database import PaperCache
    from rag.chunker import (
        ChunkType,
        ParsedSection,
        SECTION_PATTERNS,
        SectionAwareChunker,
    )
except ImportError:
    from backend.models.database import PaperCache
    from backend.rag.chunker import (
        ChunkType,
        ParsedSection,
        SECTION_PATTERNS,
        SectionAwareChunker,
    )

logger = logging.getLogger(__name__)

# Standard academic section title regexes
NUMBERED_HEADING_REGEX = re.compile(
    r"^(?:(?:Section|Sec\.)\s+)?(\d+(?:\.\d+)*\.?|[IVXLCDM]+\.?)\s+([A-Z][A-Za-z0-9\s\-,:()&]+)$",
    re.MULTILINE,
)

STANDARD_SECTION_NAMES = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "prior work",
    "methodology",
    "methods",
    "materials and methods",
    "system architecture",
    "model architecture",
    "proposed method",
    "approach",
    "experimental setup",
    "experiments",
    "experimental results",
    "results",
    "evaluation",
    "empirical evaluation",
    "discussion",
    "limitations",
    "limitations and future work",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "appendices",
}

MATH_SYMBOLS = set("∑∫∏√∈∉⊆⊂∪∩∀∃→⇒↔⇔≤≥≠≈±×÷·∂∇αβγδεθλμστφωΔΣΩ")

DISPLAY_MATH_PATTERN = re.compile(
    r"(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\begin\{(?:equation|align|gather|matrix|bmatrix|pmatrix)\*?\}[\s\S]*?\\end\{(?:equation|align|gather|matrix|bmatrix|pmatrix)\*?\})",
    re.MULTILINE,
)

INLINE_MATH_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")


@dataclass
class ParsedDocument:
    """
    Structured academic document parsed from PDF bytes.
    Contains hierarchical markdown, typed sections, tables, equations, and metadata.
    """

    paper_id: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    s2_id: Optional[str] = None
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    markdown_text: str = ""
    sections: list[ParsedSection] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    is_full_text: bool = True
    page_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize parsed document to JSON-compatible dictionary."""
        return {
            "paper_id": self.paper_id,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "s2_id": self.s2_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "markdown_text": self.markdown_text,
            "sections": [
                {
                    "heading_level": s.heading_level,
                    "title": s.title,
                    "body": s.body,
                    "chunk_type": s.chunk_type.value if hasattr(s.chunk_type, "value") else str(s.chunk_type),
                    "heading_hierarchy": s.heading_hierarchy,
                    "parent_section": s.parent_section,
                    "start_char": s.start_char,
                    "end_char": s.end_char,
                }
                for s in self.sections
            ],
            "tables": self.tables,
            "equations": self.equations,
            "is_full_text": self.is_full_text,
            "page_count": self.page_count,
            "metadata": self.metadata,
        }

    def to_paper_cache(self) -> PaperCache:
        """Convert parsed document into a SQLAlchemy PaperCache ORM model."""
        cache_key = self.doi or self.arxiv_id or self.paper_id
        return PaperCache(
            doi=cache_key,
            arxiv_id=self.arxiv_id,
            s2_id=self.s2_id,
            title=self.title,
            authors=self.authors,
            abstract=self.abstract,
            parsed_markdown=self.markdown_text,
            sections_json=[
                {
                    "heading_level": s.heading_level,
                    "title": s.title,
                    "body": s.body,
                    "chunk_type": s.chunk_type.value if hasattr(s.chunk_type, "value") else str(s.chunk_type),
                    "heading_hierarchy": s.heading_hierarchy,
                    "parent_section": s.parent_section,
                }
                for s in self.sections
            ],
            tables_json=self.tables,
            source_url=self.metadata.get("source_url"),
            is_full_text=self.is_full_text,
            fetched_at=datetime.now(timezone.utc),
        )

    @classmethod
    def from_paper_cache(cls, cache: PaperCache, paper_id: str = "") -> ParsedDocument:
        """Reconstruct ParsedDocument from an existing PaperCache ORM record."""
        p_id = paper_id or cache.doi or cache.arxiv_id or "cached_paper"

        parsed_sections: list[ParsedSection] = []
        if cache.sections_json:
            for s_dict in cache.sections_json:
                chunk_type_str = s_dict.get("chunk_type", "general")
                parsed_sections.append(
                    ParsedSection(
                        heading_level=s_dict.get("heading_level", 2),
                        title=s_dict.get("title", ""),
                        body=s_dict.get("body", ""),
                        chunk_type=ChunkType.from_str(chunk_type_str),
                        heading_hierarchy=s_dict.get("heading_hierarchy", [s_dict.get("title", "")]),
                        parent_section=s_dict.get("parent_section"),
                    )
                )

        equations: list[str] = []
        if cache.parsed_markdown:
            equations = PDFParser.extract_latex_equations(cache.parsed_markdown)

        return cls(
            paper_id=p_id,
            doi=cache.doi,
            arxiv_id=cache.arxiv_id,
            s2_id=cache.s2_id,
            title=cache.title or "",
            authors=cache.authors or [],
            abstract=cache.abstract,
            markdown_text=cache.parsed_markdown or "",
            sections=parsed_sections,
            tables=cache.tables_json or [],
            equations=equations,
            is_full_text=cache.is_full_text,
            page_count=0,
            metadata={"source": "paper_cache", "source_url": cache.source_url},
        )


class PDFParser:
    """
    Robust Academic PDF to Hierarchical Markdown, Table & LaTeX Parser.
    """

    def __init__(
        self,
        min_heading_ratio: float = 1.12,
        extract_tables: bool = True,
        extract_equations: bool = True,
        max_pages: Optional[int] = None,
    ):
        self.min_heading_ratio = min_heading_ratio
        self.extract_tables = extract_tables
        self.extract_equations = extract_equations
        self.max_pages = max_pages
        self._chunker = SectionAwareChunker()

    @staticmethod
    def extract_latex_equations(markdown_text: str) -> list[str]:
        """Extract all LaTeX display equations, math environments, and inline formulas."""
        if not markdown_text:
            return []

        equations: list[str] = []
        for match in DISPLAY_MATH_PATTERN.finditer(markdown_text):
            eq = match.group(0).strip()
            if eq and eq not in equations:
                equations.append(eq)

        for match in INLINE_MATH_PATTERN.finditer(markdown_text):
            eq = f"${match.group(1).strip()}$"
            if eq and len(eq) > 2 and eq not in equations:
                equations.append(eq)

        return equations

    def _compute_font_profile(self, doc: pymupdf.Document) -> tuple[float, float, float]:
        """Analyze font sizes across the document to determine body, H2, and title thresholds."""
        font_sizes: list[float] = []
        for page_idx, page in enumerate(doc):
            if self.max_pages and page_idx >= self.max_pages:
                break
            blocks_dict = page.get_text("dict")
            for block in blocks_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                size = round(span.get("size", 10.0), 1)
                                font_sizes.extend([size] * len(text))

        if not font_sizes:
            return 10.0, 12.0, 16.0

        counter = Counter(font_sizes)
        body_size = counter.most_common(1)[0][0]
        h2_size = round(body_size * self.min_heading_ratio, 1)
        title_size = round(body_size * 1.45, 1)

        return body_size, h2_size, title_size

    def _clean_table_to_markdown(self, table_matrix: list[list[Any]]) -> str:
        """Convert a 2D list of table cells into a clean GitHub-Flavored Markdown table."""
        if not table_matrix or len(table_matrix) < 1:
            return ""

        num_cols = max(len(r) for r in table_matrix)
        if num_cols == 0:
            return ""

        cleaned_rows: list[list[str]] = []
        for row in table_matrix:
            cleaned_row = []
            for cell in row:
                cell_str = str(cell or "").replace("\n", " ").replace("|", "\\|").strip()
                cleaned_row.append(cell_str)
            while len(cleaned_row) < num_cols:
                cleaned_row.append("")
            cleaned_rows.append(cleaned_row)

        if all(c == "" for c in cleaned_rows[0]):
            cleaned_rows[0] = [f"Col {i+1}" for i in range(num_cols)]

        header = "| " + " | ".join(cleaned_rows[0]) + " |"
        delimiter = "| " + " | ".join(["---"] * num_cols) + " |"
        body_rows = ["| " + " | ".join(row) + " |" for row in cleaned_rows[1:]]

        return "\n".join([header, delimiter] + body_rows)

    def _extract_page_tables(self, page: pymupdf.Page) -> list[tuple[pymupdf.Rect, str]]:
        """Extract tables from a page using PyMuPDF table finder and convert to Markdown."""
        extracted: list[tuple[pymupdf.Rect, str]] = []
        try:
            tab_finder = page.find_tables()
            if tab_finder and tab_finder.tables:
                for tab in tab_finder.tables:
                    matrix = tab.extract()
                    if matrix and len(matrix) >= 2 and len(matrix[0]) >= 2:
                        md_table = self._clean_table_to_markdown(matrix)
                        if md_table:
                            extracted.append((tab.bbox, md_table))
        except Exception as e:
            logger.debug(f"PyMuPDF table extraction notice: {e}")

        return extracted

    def _is_rect_intersecting(self, r1: Sequence[float], r2: pymupdf.Rect | Sequence[float], threshold: float = 0.5) -> bool:
        """Check if block rectangle r1 significantly overlaps with table rectangle r2."""
        try:
            rect1 = pymupdf.Rect(r1[0], r1[1], r1[2], r1[3])
            rect2 = pymupdf.Rect(r2[0], r2[1], r2[2], r2[3])
            intersect = rect1 & rect2
            if intersect.is_empty:
                return False
            return (intersect.get_area() / max(1.0, rect1.get_area())) >= threshold
        except Exception:
            return False

    def _sort_blocks_reading_order(
        self, raw_blocks: list[dict[str, Any]], page_width: float
    ) -> list[dict[str, Any]]:
        """Sort blocks in multi-column academic reading order."""
        mid_x = page_width / 2.0
        left_blocks: list[dict[str, Any]] = []
        right_blocks: list[dict[str, Any]] = []
        spanning_blocks: list[dict[str, Any]] = []

        for b in raw_blocks:
            bbox = b.get("bbox", (0, 0, 0, 0))
            x0, x1 = bbox[0], bbox[2]
            center_x = (x0 + x1) / 2.0

            if x1 < mid_x + 35 and center_x < mid_x:
                left_blocks.append(b)
            elif x0 > mid_x - 35 and center_x > mid_x:
                right_blocks.append(b)
            else:
                spanning_blocks.append(b)

        if left_blocks and right_blocks and (len(left_blocks) + len(right_blocks) >= 3):
            left_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
            right_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

            min_col_y = min(left_blocks[0]["bbox"][1], right_blocks[0]["bbox"][1])

            top_spanning = [b for b in spanning_blocks if b["bbox"][1] < min_col_y]
            top_spanning.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

            bottom_spanning = [b for b in spanning_blocks if b["bbox"][1] >= min_col_y]
            bottom_spanning.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

            return top_spanning + left_blocks + right_blocks + bottom_spanning
        else:
            all_blocks = list(raw_blocks)
            all_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
            return all_blocks

    def _classify_line_as_heading(
        self,
        text: str,
        avg_size: float,
        is_bold: bool,
        body_size: float,
        h2_size: float,
        title_size: float,
        is_first_page: bool,
    ) -> tuple[bool, int, str]:
        """Determine if a line is a heading and assign heading level."""
        clean = text.strip()
        if not clean or len(clean) > 130:
            return False, 0, clean

        clean_lower = clean.lower()

        if is_first_page and avg_size >= title_size and len(clean) > 5:
            return True, 1, clean

        numbered_match = NUMBERED_HEADING_REGEX.match(clean)
        if numbered_match:
            prefix = numbered_match.group(1).rstrip(".")
            dot_count = prefix.count(".")
            level = 2 if dot_count == 0 else min(4, 2 + dot_count)
            return True, level, clean

        if clean_lower in STANDARD_SECTION_NAMES or any(clean_lower.startswith(f"{sec}:") for sec in STANDARD_SECTION_NAMES):
            return True, 2, clean

        if avg_size >= h2_size:
            if not clean.endswith((".", ",", ";")) or clean.isupper() or is_bold:
                level = 2 if avg_size >= (h2_size * 1.1) else 3
                return True, level, clean

        if is_bold and avg_size >= body_size and (clean.isupper() or clean.istitle()):
            if len(clean) < 60 and not clean.endswith((".", ",")):
                return True, 3, clean

        return False, 0, clean

    def _is_standalone_math(self, text: str) -> bool:
        """Detect if text line represents a standalone math equation."""
        clean = text.strip()
        if not clean:
            return False

        has_symbols = any(ch in clean for ch in MATH_SYMBOLS)
        has_eq_num = bool(re.search(r"\(\d+(?:\.\d+)*\)$", clean))
        has_eq_sign = "=" in clean or "\\approx" in clean or "\\le" in clean or "\\in" in clean

        return (has_symbols and has_eq_sign) or (has_eq_num and has_eq_sign)

    def parse_pdf(
        self,
        pdf_bytes: bytes,
        paper_id: str = "",
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        db_session: Optional[Session] = None,
        use_cache: bool = True,
        title_hint: Optional[str] = None,
    ) -> ParsedDocument:
        """Parse raw PDF bytes into a structured ParsedDocument."""
        doc_paper_id = paper_id or doi or arxiv_id or "unnamed_paper"

        # Check PaperCache
        if use_cache and db_session is not None:
            cached_doc = self.get_from_cache(db_session, doi=doi, arxiv_id=arxiv_id, paper_id=doc_paper_id)
            if cached_doc is not None:
                logger.info(f"Retrieved parsed document from PaperCache for DOI={doi}")
                return cached_doc

        if not pdf_bytes or len(pdf_bytes) < 100:
            return ParsedDocument(
                paper_id=doc_paper_id,
                doi=doi,
                arxiv_id=arxiv_id,
                title=title_hint or "Untitled",
                markdown_text="",
                is_full_text=False,
                metadata={"error": "Empty or truncated PDF bytes"},
            )

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            return ParsedDocument(
                paper_id=doc_paper_id,
                doi=doi,
                arxiv_id=arxiv_id,
                title=title_hint or "Untitled",
                markdown_text="",
                is_full_text=False,
                metadata={"error": f"PDF open error: {str(e)}"},
            )

        page_count = len(doc)
        if page_count == 0:
            return ParsedDocument(
                paper_id=doc_paper_id,
                doi=doi,
                arxiv_id=arxiv_id,
                title=title_hint or "Untitled",
                markdown_text="",
                is_full_text=False,
                metadata={"error": "PDF has 0 pages"},
            )

        body_size, h2_size, title_size = self._compute_font_profile(doc)

        extracted_title: Optional[str] = title_hint
        extracted_authors: list[str] = []
        extracted_abstract: Optional[str] = None
        all_tables: list[str] = []
        markdown_blocks: list[str] = []

        for page_num in range(page_count):
            if self.max_pages and page_num >= self.max_pages:
                break

            page = doc[page_num]
            page_width = page.rect.width
            is_first_page = (page_num == 0)

            # Extract Tables
            page_tables = self._extract_page_tables(page) if self.extract_tables else []
            table_rects = [t[0] for t in page_tables]
            for _, md_table in page_tables:
                all_tables.append(md_table)

            # Extract Text Blocks
            page_dict = page.get_text("dict")
            raw_blocks = page_dict.get("blocks", [])

            text_blocks = []
            for b in raw_blocks:
                if b.get("type") != 0 or "lines" not in b:
                    continue
                bbox = b.get("bbox", (0, 0, 0, 0))
                if any(self._is_rect_intersecting(bbox, tr) for tr in table_rects):
                    continue
                text_blocks.append(b)

            sorted_blocks = self._sort_blocks_reading_order(text_blocks, page_width)

            for b in sorted_blocks:
                block_lines = b.get("lines", [])
                if not block_lines:
                    continue

                for line in block_lines:
                    spans = line.get("spans", [])
                    if not spans:
                        continue

                    line_text = "".join(s.get("text", "") for s in spans).strip()
                    if not line_text:
                        continue

                    avg_size = sum(s.get("size", 10.0) * len(s.get("text", "")) for s in spans) / max(1, len(line_text))
                    is_bold = any(bool(s.get("flags", 0) & 2) or "bold" in s.get("font", "").lower() for s in spans)

                    is_heading, level, heading_text = self._classify_line_as_heading(
                        line_text,
                        avg_size,
                        is_bold,
                        body_size,
                        h2_size,
                        title_size,
                        is_first_page,
                    )

                    if is_heading:
                        if level == 1 and not extracted_title:
                            extracted_title = heading_text
                        heading_hashes = "#" * max(1, min(4, level))
                        markdown_blocks.append(f"\n{heading_hashes} {heading_text}\n")
                    else:
                        if self.extract_equations and self._is_standalone_math(line_text):
                            markdown_blocks.append(f"\n$$\n{line_text}\n$$\n")
                        else:
                            markdown_blocks.append(line_text)

            for _, md_table in page_tables:
                markdown_blocks.append(f"\n\n{md_table}\n\n")

        assembled_markdown = "\n".join(markdown_blocks).strip()
        assembled_markdown = re.sub(r"\n{3,}", "\n\n", assembled_markdown)

        equations = self.extract_latex_equations(assembled_markdown) if self.extract_equations else []
        parsed_sections = self._chunker._split_into_hierarchical_sections(assembled_markdown)

        for sec in parsed_sections:
            if sec.chunk_type == ChunkType.ABSTRACT and not extracted_abstract:
                extracted_abstract = sec.body

        parsed_doc = ParsedDocument(
            paper_id=doc_paper_id,
            doi=doi,
            arxiv_id=arxiv_id,
            title=extracted_title or title_hint or "Untitled",
            authors=extracted_authors,
            abstract=extracted_abstract,
            markdown_text=assembled_markdown,
            sections=parsed_sections,
            tables=all_tables,
            equations=equations,
            is_full_text=True,
            page_count=page_count,
            metadata={
                "body_font_size": body_size,
                "h2_font_size": h2_size,
                "title_font_size": title_size,
                "parsed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        if use_cache and db_session is not None:
            try:
                self.save_to_cache(db_session, parsed_doc)
            except Exception as e:
                logger.warning(f"Failed to persist parsed document to PaperCache: {e}")

        return parsed_doc

    def parse_pdf_file(
        self,
        file_path: str | Path,
        paper_id: str = "",
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        db_session: Optional[Session] = None,
        use_cache: bool = True,
    ) -> ParsedDocument:
        """Convenience method to parse a PDF file from local disk."""
        path = Path(file_path)
        if not path.is_file():
            return ParsedDocument(
                paper_id=paper_id or path.stem,
                doi=doi,
                arxiv_id=arxiv_id,
                title=path.stem,
                markdown_text="",
                is_full_text=False,
                metadata={"error": f"File not found: {path}"},
            )
        pdf_bytes = path.read_bytes()
        return self.parse_pdf(
            pdf_bytes=pdf_bytes,
            paper_id=paper_id or path.stem,
            doi=doi,
            arxiv_id=arxiv_id,
            db_session=db_session,
            use_cache=use_cache,
            title_hint=path.stem,
        )

    @staticmethod
    def get_from_cache(
        db: Session,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        paper_id: Optional[str] = None,
    ) -> Optional[ParsedDocument]:
        """Query PaperCache table for an existing parsed document."""
        if not db:
            return None

        cache_entry: Optional[PaperCache] = None
        if doi:
            cache_entry = db.query(PaperCache).filter(PaperCache.doi == doi).first()
        if not cache_entry and arxiv_id:
            cache_entry = db.query(PaperCache).filter(PaperCache.arxiv_id == arxiv_id).first()
        if not cache_entry and paper_id:
            cache_entry = db.query(PaperCache).filter(PaperCache.doi == paper_id).first()

        if cache_entry and cache_entry.is_full_text and cache_entry.parsed_markdown:
            return ParsedDocument.from_paper_cache(cache_entry, paper_id=paper_id or "")

        return None

    @staticmethod
    def save_to_cache(
        db: Session,
        doc: ParsedDocument,
        update_existing: bool = True,
    ) -> PaperCache:
        """Persist a ParsedDocument into the PaperCache table."""
        cache_key = doc.doi or doc.arxiv_id or doc.paper_id
        existing = db.query(PaperCache).filter(PaperCache.doi == cache_key).first()

        if existing:
            if update_existing:
                existing.arxiv_id = doc.arxiv_id or existing.arxiv_id
                existing.s2_id = doc.s2_id or existing.s2_id
                existing.title = doc.title or existing.title
                existing.authors = doc.authors or existing.authors
                existing.abstract = doc.abstract or existing.abstract
                existing.parsed_markdown = doc.markdown_text
                existing.sections_json = [
                    {
                        "heading_level": s.heading_level,
                        "title": s.title,
                        "body": s.body,
                        "chunk_type": s.chunk_type.value if hasattr(s.chunk_type, "value") else str(s.chunk_type),
                        "heading_hierarchy": s.heading_hierarchy,
                        "parent_section": s.parent_section,
                    }
                    for s in doc.sections
                ]
                existing.tables_json = doc.tables
                existing.is_full_text = doc.is_full_text
                existing.fetched_at = datetime.now(timezone.utc)
                db.commit()
                db.refresh(existing)
            return existing
        else:
            new_entry = doc.to_paper_cache()
            db.add(new_entry)
            db.commit()
            db.refresh(new_entry)
            return new_entry


__all__ = [
    "ParsedDocument",
    "PDFParser",
    "DISPLAY_MATH_PATTERN",
    "INLINE_MATH_PATTERN",
]
