"""
Section-Aware Hierarchical Markdown Chunker for Academic Papers.

Implements:
- Hierarchical section parsing with heading stack & breadcrumbs
- Section category detection via comprehensive regex (Methodology, Results, Limitations, Tables, etc.)
- Deterministic citation anchor generation: [ref_{paper_id}#sec_{i}] and [ref_{paper_id}#tab_{j}]
- Atomic block preservation for Markdown tables and LaTeX equations
- Sentence-aware sliding window fallback for headerless/unstructured documents
- Full backward compatibility with SemanticChunker interface
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ChunkType(str, Enum):
    """Types of chunks and sections in academic papers."""

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    LIMITATIONS = "limitations"
    TABLES = "tables"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    GENERAL = "general"
    TITLE = "title"

    @classmethod
    def from_str(cls, val: str) -> "ChunkType":
        """Coerce string or uppercase name to ChunkType."""
        if not val:
            return cls.GENERAL
        clean = val.lower().strip()
        for member in cls:
            if member.value == clean or member.name.lower() == clean:
                return member
        return cls.GENERAL


# Section Importance Weights for RRF ranking & RAG scoring
SECTION_WEIGHTS: dict[ChunkType, float] = {
    ChunkType.TITLE: 2.0,
    ChunkType.ABSTRACT: 1.8,
    ChunkType.LIMITATIONS: 1.6,
    ChunkType.CONCLUSION: 1.5,
    ChunkType.RESULTS: 1.4,
    ChunkType.TABLES: 1.3,
    ChunkType.METHODOLOGY: 1.3,
    ChunkType.INTRODUCTION: 1.1,
    ChunkType.DISCUSSION: 1.1,
    ChunkType.GENERAL: 1.0,
    ChunkType.REFERENCES: 0.5,
}

# Regex patterns for section category classification
SECTION_PATTERNS: dict[ChunkType, re.Pattern] = {
    ChunkType.ABSTRACT: re.compile(
        r"(?i)^(?:#+\s*)?(?:\d+\.?\s*)?(?:abstract|executive\s+summary|synopsis|summary)\b"
    ),
    ChunkType.INTRODUCTION: re.compile(
        r"(?i)^(?:#+\s*)?(?:\d+\.?\s*)?(?:introduction|background|motivation|overview|preliminaries|problem\s+formulation|context)\b"
    ),
    ChunkType.METHODOLOGY: re.compile(
        r"(?i)^(?:#+\s*)?(?:\d+(?:\.\d+)*\.?\s*)?(?:(?:proposed|novel|our|the)\s+)?(?:method(?:ology|s)?|approach(?:es)?|architecture|experimental\s+(?:setup|design|framework|methodology)|materials?\s+and\s+methods?|model(?:\s+architecture)?|algorithm(?:s|ic\s+framework)?|implementation\s+details?|system\s+design|formulation|pipeline)\b"
    ),
    ChunkType.RESULTS: re.compile(
        r"(?i)^(?:#+\s*)?(?:\d+(?:\.\d+)*\.?\s*)?(?:(?:experimental|empirical|main|evaluation)\s+)?(?:results?|findings?|evaluation|experiments?|validation|performance(?:\s+comparison)?|ablation(?:\s+(?:study|experiments?))?|benchmarks?)\b"
    ),
    ChunkType.LIMITATIONS: re.compile(
        r"(?i)^(?:#+\s*)?(?:\d+(?:\.\d+)*\.?\s*)?(?:limitations?(?:\s+and\s+(?:future\s+work|risks|discussion))?|failure\s+modes?|threats\s+to\s+validity|drawbacks|bottlenecks?|weaknesses?|assumptions\s+and\s+limitations?|ethical\s+considerations?|potential\s+risks?|caveats?|open\s+challenges?)\b"
    ),
    ChunkType.TABLES: re.compile(
        r"(?i)^(?:#+\s*)?(?:table\s+\d+|tab\.\s*\d+|tabular\s+data|tables?\b)"
    ),
    ChunkType.DISCUSSION: re.compile(
        r"(?i)^(?:#+\s*)?(?:\d+(?:\.\d+)*\.?\s*)?(?:discussion|implications?|broader\s+impact|qualitative\s+analysis|comparative\s+analysis|perspectives)\b"
    ),
    ChunkType.CONCLUSION: re.compile(
        r"(?i)^(?:#+\s*)?(?:\d+(?:\.\d+)*\.?\s*)?(?:conclusions?(?:\s+and\s+future\s+work)?|concluding\s+remarks?|future\s+work|summary\s+and\s+outlook)\b"
    ),
    ChunkType.REFERENCES: re.compile(
        r"(?i)^(?:#+\s*)?(?:\d+\.?\s*)?(?:references?|bibliography|works\s+cited|citations?|literature\s+cited)\b"
    ),
}


@dataclass
class PaperChunk:
    """Represents a structured semantic chunk of an academic paper."""

    content: str
    chunk_type: ChunkType
    paper_id: str
    paper_title: str
    chunk_index: int
    anchor_tag: str = ""
    section_title: str = ""
    section_index: int = 0
    table_index: Optional[int] = None
    heading_hierarchy: list[str] = field(default_factory=list)
    parent_section: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for storage and API responses."""
        return {
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "chunk_index": self.chunk_index,
            "anchor_tag": self.anchor_tag,
            "section_title": self.section_title,
            "section_index": self.section_index,
            "table_index": self.table_index,
            "heading_hierarchy": self.heading_hierarchy,
            "parent_section": self.parent_section,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "weight": self.weight,
            "metadata": self.metadata,
        }


class BlockType(str, Enum):
    """Semantic block classification within a section."""

    PARAGRAPH = "paragraph"
    TABLE = "table"
    LATEX_MATH = "latex_math"
    CODE = "code"


@dataclass
class SemanticBlock:
    """A contiguous atomic or paragraph block in markdown."""

    text: str
    block_type: BlockType
    is_atomic: bool = True
    token_count: int = 0


@dataclass
class ParsedSection:
    """A hierarchical section extracted from markdown."""

    heading_level: int
    title: str
    body: str
    chunk_type: ChunkType
    heading_hierarchy: list[str] = field(default_factory=list)
    parent_section: Optional[str] = None
    start_char: int = 0
    end_char: int = 0


class SectionAwareChunker:
    """
    Section-Aware Hierarchical Markdown Chunker for Academic Papers.

    Features:
    - Heading hierarchy parsing (# to ###### and numbered sections)
    - Heading context breadcrumbs & category inheritance
    - Robust section classification (Methodology, Results, Limitations, Tables, etc.)
    - Exact anchor generation: [ref_{paper_id}#sec_{i}] and [ref_{paper_id}#tab_{j}]
    - Atomic block tokenizer preserving Markdown tables and LaTeX formulas
    - Header-preserving sub-table splitting when tables exceed max_chunk_size
    - Fallback to sliding window for headerless markdown
    """

    HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    NUMBERED_HEADING_REGEX = re.compile(
        r"^(\d+(?:\.\d+)*\.?)\s+([A-Z][A-Za-z0-9\s\-,:()&]+)$", re.MULTILINE
    )

    LATEX_BLOCK_PATTERN = re.compile(
        r"(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\begin\{(?:equation|align|gather|matrix|bmatrix|pmatrix|table|tabular)\*?\}[\s\S]*?\\end\{(?:equation|align|gather|matrix|bmatrix|pmatrix|table|tabular)\*?\})",
        re.MULTILINE,
    )

    MARKDOWN_TABLE_PATTERN = re.compile(
        r"(?:(?:^\|.+\|\s*\n)+^\|[\s\-:|]+\|\s*\n(?:^\|.+\|\s*(?:\n|$))+)|(?:<table>[\s\S]*?</table>)",
        re.MULTILINE,
    )

    CODE_BLOCK_PATTERN = re.compile(
        r"(```[\w]*\n[\s\S]*?```)",
        re.MULTILINE,
    )

    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 80,
        overlap_size: int = 50,
        include_title_in_chunks: bool = True,
        include_anchors_in_content: bool = True,
    ):
        """
        Initialize SectionAwareChunker.

        Args:
            max_chunk_size: Maximum estimated tokens per chunk
            min_chunk_size: Minimum estimated tokens per chunk
            overlap_size: Overlap tokens between consecutive sub-chunks
            include_title_in_chunks: Prepend paper title and breadcrumbs to chunk text
            include_anchors_in_content: Embed anchor tag directly into chunk content string
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.include_title_in_chunks = include_title_in_chunks
        self.include_anchors_in_content = include_anchors_in_content

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approx 4 chars per token)."""
        return max(1, len(text) // 4)

    def _detect_section_type(self, heading: str) -> ChunkType:
        """Classify section category from heading text using regex rules."""
        clean = heading.strip()
        for stype, pattern in SECTION_PATTERNS.items():
            if pattern.search(clean):
                return stype
        return ChunkType.GENERAL

    def _split_into_hierarchical_sections(self, markdown: str) -> list[ParsedSection]:
        """Split Markdown document into hierarchical sections with parent tracking."""
        heading_matches = []
        for match in self.HEADING_REGEX.finditer(markdown):
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_matches.append((match.start(), match.end(), level, title))

        if not heading_matches:
            # Fallback to numbered headers without markdown hash syntax
            for match in self.NUMBERED_HEADING_REGEX.finditer(markdown):
                num_dots = match.group(1).count(".")
                level = max(1, num_dots)
                title = f"{match.group(1)} {match.group(2)}".strip()
                heading_matches.append((match.start(), match.end(), level, title))
            heading_matches.sort(key=lambda x: x[0])

        if not heading_matches:
            # Headerless document fallback
            return [
                ParsedSection(
                    heading_level=0,
                    title="Body",
                    body=markdown.strip(),
                    chunk_type=ChunkType.GENERAL,
                    heading_hierarchy=["Body"],
                    parent_section=None,
                    start_char=0,
                    end_char=len(markdown),
                )
            ]

        sections = []
        stack: list[tuple[int, str, ChunkType]] = []

        # Handle preamble text before first heading
        if heading_matches[0][0] > 0:
            preamble = markdown[: heading_matches[0][0]].strip()
            if preamble:
                first_line = preamble.splitlines()[0].strip()
                preamble_cat = self._detect_section_type(first_line)
                title = "Abstract" if preamble_cat == ChunkType.ABSTRACT else "Preamble"
                sections.append(
                    ParsedSection(
                        heading_level=0,
                        title=title,
                        body=preamble,
                        chunk_type=preamble_cat,
                        heading_hierarchy=[title],
                        parent_section=None,
                        start_char=0,
                        end_char=heading_matches[0][0],
                    )
                )

        for i, (start_pos, header_end_pos, level, title) in enumerate(heading_matches):
            next_start = (
                heading_matches[i + 1][0] if i + 1 < len(heading_matches) else len(markdown)
            )
            body = markdown[header_end_pos:next_start].strip()

            # Manage heading ancestor stack
            while stack and stack[-1][0] >= level:
                stack.pop()

            detected_category = self._detect_section_type(title)

            # Inherit category from parent if current sub-section is GENERAL
            if detected_category == ChunkType.GENERAL and stack:
                parent_cat = stack[-1][2]
                if parent_cat in (
                    ChunkType.METHODOLOGY,
                    ChunkType.RESULTS,
                    ChunkType.LIMITATIONS,
                    ChunkType.DISCUSSION,
                    ChunkType.CONCLUSION,
                ):
                    detected_category = parent_cat

            hierarchy = [item[1] for item in stack] + [title]
            parent_name = stack[-1][1] if stack else None
            stack.append((level, title, detected_category))

            sections.append(
                ParsedSection(
                    heading_level=level,
                    title=title,
                    body=body,
                    chunk_type=detected_category,
                    heading_hierarchy=hierarchy,
                    parent_section=parent_name,
                    start_char=start_pos,
                    end_char=next_start,
                )
            )

        return sections

    def _tokenize_semantic_blocks(self, text: str) -> list[SemanticBlock]:
        """Tokenize text into atomic blocks (tables, math, code) and paragraphs."""
        if not text.strip():
            return []

        special_regions: list[tuple[int, int, BlockType, str]] = []

        for m in self.LATEX_BLOCK_PATTERN.finditer(text):
            special_regions.append((m.start(), m.end(), BlockType.LATEX_MATH, m.group(0)))

        for m in self.MARKDOWN_TABLE_PATTERN.finditer(text):
            if not any(s <= m.start() < e or s < m.end() <= e for s, e, _, _ in special_regions):
                special_regions.append((m.start(), m.end(), BlockType.TABLE, m.group(0)))

        for m in self.CODE_BLOCK_PATTERN.finditer(text):
            if not any(s <= m.start() < e or s < m.end() <= e for s, e, _, _ in special_regions):
                special_regions.append((m.start(), m.end(), BlockType.CODE, m.group(0)))

        special_regions.sort(key=lambda x: x[0])

        blocks: list[SemanticBlock] = []
        cursor = 0

        for start, end, b_type, block_text in special_regions:
            if start > cursor:
                intervening_text = text[cursor:start].strip()
                if intervening_text:
                    paras = re.split(r"\n\s*\n", intervening_text)
                    for p in paras:
                        p_str = p.strip()
                        if p_str:
                            blocks.append(
                                SemanticBlock(
                                    text=p_str,
                                    block_type=BlockType.PARAGRAPH,
                                    is_atomic=False,
                                    token_count=self._estimate_tokens(p_str),
                                )
                            )

            blocks.append(
                SemanticBlock(
                    text=block_text.strip(),
                    block_type=b_type,
                    is_atomic=True,
                    token_count=self._estimate_tokens(block_text),
                )
            )
            cursor = end

        if cursor < len(text):
            remaining_text = text[cursor:].strip()
            if remaining_text:
                paras = re.split(r"\n\s*\n", remaining_text)
                for p in paras:
                    p_str = p.strip()
                    if p_str:
                        blocks.append(
                            SemanticBlock(
                                text=p_str,
                                block_type=BlockType.PARAGRAPH,
                                is_atomic=False,
                                token_count=self._estimate_tokens(p_str),
                            )
                        )

        return blocks

    def _split_long_paragraph(self, text: str, max_tokens: int) -> list[str]:
        """Split a long paragraph along sentence boundaries."""
        sentences = re.split(r"(?<=[.?!])\s+", text)
        chunks = []
        current = []
        current_tokens = 0

        for s in sentences:
            s_tokens = self._estimate_tokens(s)
            if current_tokens + s_tokens > max_tokens and current:
                chunks.append(" ".join(current))
                current = [s]
                current_tokens = s_tokens
            else:
                current.append(s)
                current_tokens += s_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _split_long_table(self, table_text: str, max_tokens: int) -> list[str]:
        """Split a large table row-by-row while preserving header & delimiter in each sub-table."""
        lines = [line.strip() for line in table_text.splitlines() if line.strip()]
        if len(lines) <= 2:
            return [table_text]

        header = lines[0]
        delimiter = lines[1]
        data_rows = lines[2:]

        sub_tables = []
        current_rows = []
        current_tokens = self._estimate_tokens(f"{header}\n{delimiter}")

        for row in data_rows:
            r_tokens = self._estimate_tokens(row)
            if current_tokens + r_tokens > max_tokens and current_rows:
                sub_table = "\n".join([header, delimiter] + current_rows)
                sub_tables.append(sub_table)
                current_rows = [row]
                current_tokens = self._estimate_tokens(f"{header}\n{delimiter}\n{row}")
            else:
                current_rows.append(row)
                current_tokens += r_tokens

        if current_rows:
            sub_table = "\n".join([header, delimiter] + current_rows)
            sub_tables.append(sub_table)

        return sub_tables

    def chunk_document(
        self,
        markdown: str,
        paper_id: str,
        paper_title: str = "Untitled",
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[PaperChunk]:
        """
        Chunk a Markdown document into section-aware, anchored semantic chunks.

        Args:
            markdown: Full text Markdown of paper
            paper_id: Unique paper identifier
            paper_title: Paper title
            metadata: Additional metadata dictionary

        Returns:
            List of PaperChunk instances
        """
        if not markdown or not markdown.strip():
            return []

        metadata = metadata or {}
        sections = self._split_into_hierarchical_sections(markdown)

        all_chunks: list[PaperChunk] = []
        chunk_counter = 0
        section_counter = 0
        table_counter = 0

        for sec in sections:
            # Skip references section to preserve synthesis budget
            if sec.chunk_type == ChunkType.REFERENCES:
                continue

            section_counter += 1
            sec_anchor = f"[ref_{paper_id}#sec_{section_counter}]"
            breadcrumb_str = " > ".join(sec.heading_hierarchy)

            blocks = self._tokenize_semantic_blocks(sec.body)

            if not blocks:
                content = f"{sec_anchor} ## {breadcrumb_str}"
                all_chunks.append(
                    PaperChunk(
                        content=content,
                        chunk_type=sec.chunk_type,
                        paper_id=paper_id,
                        paper_title=paper_title,
                        chunk_index=chunk_counter,
                        anchor_tag=sec_anchor,
                        section_title=sec.title,
                        section_index=section_counter,
                        table_index=None,
                        heading_hierarchy=sec.heading_hierarchy,
                        parent_section=sec.parent_section,
                        start_char=sec.start_char,
                        end_char=sec.end_char,
                        token_count=self._estimate_tokens(content),
                        weight=SECTION_WEIGHTS.get(sec.chunk_type, 1.0),
                        metadata={**metadata, "section_path": breadcrumb_str},
                    )
                )
                chunk_counter += 1
                continue

            current_chunk_blocks: list[str] = []
            current_tokens = 0

            def flush_chunk(force_table: bool = False):
                nonlocal current_chunk_blocks, current_tokens, chunk_counter, table_counter
                if not current_chunk_blocks:
                    return

                body_text = "\n\n".join(current_chunk_blocks)
                tab_idx = None
                anchor = sec_anchor
                chunk_type = sec.chunk_type

                is_table = (
                    force_table
                    or sec.chunk_type == ChunkType.TABLES
                    or (len(current_chunk_blocks) == 1 and current_chunk_blocks[0].startswith("|"))
                )

                if is_table:
                    table_counter += 1
                    tab_idx = table_counter
                    anchor = f"[ref_{paper_id}#tab_{table_counter}]"
                    chunk_type = ChunkType.TABLES

                if self.include_anchors_in_content:
                    if self.include_title_in_chunks and paper_title:
                        chunk_text = f"{anchor} [{paper_title}] {breadcrumb_str}\n\n{body_text}"
                    else:
                        chunk_text = f"{anchor} {breadcrumb_str}\n\n{body_text}"
                else:
                    chunk_text = (
                        f"[{paper_title}] {body_text}"
                        if self.include_title_in_chunks and paper_title
                        else body_text
                    )

                all_chunks.append(
                    PaperChunk(
                        content=chunk_text,
                        chunk_type=chunk_type,
                        paper_id=paper_id,
                        paper_title=paper_title,
                        chunk_index=chunk_counter,
                        anchor_tag=anchor,
                        section_title=sec.title,
                        section_index=section_counter,
                        table_index=tab_idx,
                        heading_hierarchy=sec.heading_hierarchy,
                        parent_section=sec.parent_section,
                        start_char=sec.start_char,
                        end_char=sec.end_char,
                        token_count=self._estimate_tokens(chunk_text),
                        weight=SECTION_WEIGHTS.get(chunk_type, 1.0),
                        metadata={**metadata, "section_path": breadcrumb_str},
                    )
                )
                chunk_counter += 1
                current_chunk_blocks = []
                current_tokens = 0

            for block in blocks:
                if block.block_type == BlockType.TABLE:
                    # Flush preceding text blocks before table
                    flush_chunk()
                    if block.token_count > self.max_chunk_size:
                        sub_tabs = self._split_long_table(block.text, self.max_chunk_size)
                        for st in sub_tabs:
                            current_chunk_blocks.append(st)
                            flush_chunk(force_table=True)
                    else:
                        current_chunk_blocks.append(block.text)
                        flush_chunk(force_table=True)
                    continue

                if block.token_count > self.max_chunk_size:
                    flush_chunk()
                    if block.block_type == BlockType.PARAGRAPH:
                        sub_paras = self._split_long_paragraph(block.text, self.max_chunk_size)
                        for sp in sub_paras:
                            current_chunk_blocks.append(sp)
                            flush_chunk()
                    else:
                        # LaTeX math or code block: keep intact
                        current_chunk_blocks.append(block.text)
                        flush_chunk()
                    continue

                if current_tokens + block.token_count > self.max_chunk_size and current_chunk_blocks:
                    flush_chunk()

                current_chunk_blocks.append(block.text)
                current_tokens += block.token_count

            flush_chunk()

        return all_chunks

    def chunk_paper(self, paper: dict[str, Any]) -> list[PaperChunk]:
        """
        Chunk paper dictionary containing title, abstract, authors, and full_text.
        """
        paper_id = str(paper.get("id", paper.get("doi", paper.get("url", "unknown"))))
        paper_title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "")
        full_text = paper.get("full_text", "")
        authors = paper.get("authors", [])

        chunks: list[PaperChunk] = []
        chunk_index = 0

        # 1. High-weight Title chunk
        if paper_title:
            title_content = f"Title: {paper_title}"
            if authors:
                author_str = ", ".join(authors[:5])
                if len(authors) > 5:
                    author_str += " et al."
                title_content += f" by {author_str}"

            anchor = f"[ref_{paper_id}#sec_0]"
            display_content = (
                f"{anchor} {title_content}"
                if self.include_anchors_in_content
                else title_content
            )

            chunks.append(
                PaperChunk(
                    content=display_content,
                    chunk_type=ChunkType.TITLE,
                    paper_id=paper_id,
                    paper_title=paper_title,
                    chunk_index=chunk_index,
                    anchor_tag=anchor,
                    section_title="Title",
                    section_index=0,
                    heading_hierarchy=["Title"],
                    token_count=self._estimate_tokens(display_content),
                    weight=SECTION_WEIGHTS[ChunkType.TITLE],
                    metadata={"authors": authors},
                )
            )
            chunk_index += 1

        # 2. Abstract chunk (if provided)
        if abstract:
            abstract_markdown = f"## Abstract\n{abstract}"
            abstract_chunks = self.chunk_document(
                abstract_markdown, paper_id, paper_title, metadata={"is_abstract": True}
            )
            for ac in abstract_chunks:
                ac.chunk_index = chunk_index
                ac.chunk_type = ChunkType.ABSTRACT
                ac.weight = SECTION_WEIGHTS[ChunkType.ABSTRACT]
                chunks.append(ac)
                chunk_index += 1

        # 3. Full-text hierarchical chunks
        if full_text:
            ft_chunks = self.chunk_document(full_text, paper_id, paper_title)
            for fc in ft_chunks:
                fc.chunk_index = chunk_index
                chunks.append(fc)
                chunk_index += 1

        logger.debug(f"Chunked paper '{paper_title[:50]}' into {len(chunks)} chunks")
        return chunks

    def chunk_papers(self, papers: list[dict[str, Any]]) -> list[PaperChunk]:
        """Chunk multiple paper dictionaries into a flat list of PaperChunks."""
        all_chunks = []
        for paper in papers:
            try:
                all_chunks.extend(self.chunk_paper(paper))
            except Exception as e:
                logger.error(f"Failed to chunk paper {paper.get('title', 'unknown')}: {e}")
                continue
        return all_chunks

    def extract_tables(self, markdown: str, paper_id: str) -> list[PaperChunk]:
        """Extract all markdown and HTML tables as standalone table chunks."""
        chunks = self.chunk_document(markdown, paper_id)
        return [c for c in chunks if c.chunk_type == ChunkType.TABLES or c.table_index is not None]


class SemanticChunker(SectionAwareChunker):
    """Backward-compatible alias for SectionAwareChunker."""

    pass


def create_chunker(
    max_chunk_size: int = 512,
    overlap_size: int = 50,
    **kwargs: Any,
) -> SectionAwareChunker:
    """Factory function creating a configured SectionAwareChunker."""
    return SectionAwareChunker(
        max_chunk_size=max_chunk_size,
        overlap_size=overlap_size,
        **kwargs,
    )
