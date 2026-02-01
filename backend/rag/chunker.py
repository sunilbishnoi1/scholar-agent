"""
Semantic Chunker for Academic Papers.

Intelligently chunks academic papers into meaningful sections
for better retrieval and context preservation.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of chunks in academic papers."""

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    GENERAL = "general"
    TITLE = "title"


@dataclass
class PaperChunk:
    """Represents a chunk of a paper."""

    content: str
    chunk_type: ChunkType
    paper_id: str
    paper_title: str
    chunk_index: int
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    weight: float = 1.0  # Importance weight for ranking
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "weight": self.weight,
            "metadata": self.metadata,
        }


class SemanticChunker:
    """
    Semantic chunker for academic papers.

    Features:
    - Section-aware chunking (respects paper structure)
    - Sliding window with overlap for long sections
    - Importance weighting by section type
    - Token counting for context window management
    """

    # Section detection patterns
    SECTION_PATTERNS = {
        ChunkType.ABSTRACT: r"(?i)^\s*(abstract|summary)\s*[:\n]",
        ChunkType.INTRODUCTION: r"(?i)^\s*(1\.?\s*)?(introduction|background)\s*[:\n]?",
        ChunkType.METHODOLOGY: r"(?i)^\s*(\d\.?\s*)?(method(?:ology|s)?|approach|experimental setup|materials? and methods?)\s*[:\n]?",
        ChunkType.RESULTS: r"(?i)^\s*(\d\.?\s*)?(results?|findings?|experiments?)\s*[:\n]?",
        ChunkType.DISCUSSION: r"(?i)^\s*(\d\.?\s*)?(discussion|analysis)\s*[:\n]?",
        ChunkType.CONCLUSION: r"(?i)^\s*(\d\.?\s*)?(conclusion|concluding remarks?|summary)\s*[:\n]?",
        ChunkType.REFERENCES: r"(?i)^\s*(references?|bibliography|citations?)\s*[:\n]?",
    }

    # Importance weights by section type
    SECTION_WEIGHTS = {
        ChunkType.TITLE: 2.0,
        ChunkType.ABSTRACT: 1.8,
        ChunkType.CONCLUSION: 1.5,
        ChunkType.RESULTS: 1.3,
        ChunkType.METHODOLOGY: 1.2,
        ChunkType.INTRODUCTION: 1.1,
        ChunkType.DISCUSSION: 1.1,
        ChunkType.GENERAL: 1.0,
        ChunkType.REFERENCES: 0.5,  # References are less useful for synthesis
    }

    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        overlap_size: int = 50,
        include_title_in_chunks: bool = True,
    ):
        """
        Initialize the semantic chunker.

        Args:
            max_chunk_size: Maximum tokens per chunk
            min_chunk_size: Minimum tokens per chunk (avoid tiny chunks)
            overlap_size: Tokens to overlap between consecutive chunks
            include_title_in_chunks: Whether to prepend paper title to each chunk
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.include_title_in_chunks = include_title_in_chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approx 4 chars per token)."""
        return len(text) // 4

    def _detect_section_type(self, text: str) -> ChunkType:
        """Detect the section type from text content."""
        # Check first 100 chars for section headers
        header = text[:100].strip()

        for section_type, pattern in self.SECTION_PATTERNS.items():
            if re.match(pattern, header):
                return section_type

        return ChunkType.GENERAL

    def _split_into_sections(self, text: str) -> list[tuple[ChunkType, str]]:
        """Split text into sections based on headers."""
        sections = []

        # Find all section boundaries
        section_starts = []
        for section_type, pattern in self.SECTION_PATTERNS.items():
            for match in re.finditer(pattern, text, re.MULTILINE):
                section_starts.append((match.start(), section_type, match.group()))

        # Sort by position
        section_starts.sort(key=lambda x: x[0])

        if not section_starts:
            # No sections detected, treat as general
            return [(ChunkType.GENERAL, text)]

        # Extract sections
        for i, (start, section_type, _) in enumerate(section_starts):
            if i + 1 < len(section_starts):
                end = section_starts[i + 1][0]
            else:
                end = len(text)

            section_text = text[start:end].strip()
            if section_text:
                sections.append((section_type, section_text))

        # Add any text before first section
        if section_starts[0][0] > 0:
            prefix = text[: section_starts[0][0]].strip()
            if prefix:
                sections.insert(0, (ChunkType.GENERAL, prefix))

        return sections

    def _sliding_window_chunk(
        self, text: str, paper_id: str, paper_title: str, chunk_type: ChunkType, start_index: int
    ) -> list[PaperChunk]:
        """Create chunks using sliding window approach."""
        chunks = []

        # Convert to approximate character positions
        char_window = self.max_chunk_size * 4
        char_overlap = self.overlap_size * 4
        char_min = self.min_chunk_size * 4

        pos = 0
        chunk_idx = start_index

        while pos < len(text):
            # Find end position
            end = min(pos + char_window, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 20% of chunk
                look_back = int(char_window * 0.2)
                sentence_end = self._find_sentence_boundary(text[end - look_back : end + look_back])
                if sentence_end >= 0:
                    end = end - look_back + sentence_end

            chunk_text = text[pos:end].strip()

            # Skip if too small
            if len(chunk_text) < char_min and chunks:
                # Append to previous chunk if possible
                if chunks[-1].token_count < self.max_chunk_size * 1.5:
                    chunks[-1].content += " " + chunk_text
                    chunks[-1].token_count = self._estimate_tokens(chunks[-1].content)
                    chunks[-1].end_char = end
                break

            # Add title prefix if configured
            if self.include_title_in_chunks and paper_title:
                display_text = f"[{paper_title}] {chunk_text}"
            else:
                display_text = chunk_text

            chunk = PaperChunk(
                content=display_text,
                chunk_type=chunk_type,
                paper_id=paper_id,
                paper_title=paper_title,
                chunk_index=chunk_idx,
                start_char=pos,
                end_char=end,
                token_count=self._estimate_tokens(display_text),
                weight=self.SECTION_WEIGHTS.get(chunk_type, 1.0),
            )
            chunks.append(chunk)
            chunk_idx += 1

            # Move position with overlap
            pos = end - char_overlap
            if pos <= 0:
                break

        return chunks

    def _find_sentence_boundary(self, text: str) -> int:
        """Find the best sentence boundary in text."""
        # Look for sentence-ending punctuation followed by space
        patterns = [
            r"\. ",
            r"\? ",
            r"! ",
            r"\.\n",
            r"\n\n",
        ]

        best_pos = -1
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                best_pos = max(best_pos, match.end())

        return best_pos

    def chunk_paper(self, paper: dict) -> list[PaperChunk]:
        """
        Chunk a paper into semantic chunks.

        Args:
            paper: Dictionary with paper data including:
                - id: Paper ID
                - title: Paper title
                - abstract: Paper abstract
                - full_text (optional): Full paper text
                - authors (optional): List of authors

        Returns:
            List of PaperChunk objects
        """
        paper_id = paper.get("id", paper.get("url", "unknown"))
        paper_title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "")
        full_text = paper.get("full_text", "")
        authors = paper.get("authors", [])

        chunks = []
        chunk_index = 0

        # 1. Title chunk (high weight)
        if paper_title:
            title_content = f"Title: {paper_title}"
            if authors:
                author_str = ", ".join(authors[:5])  # Limit to 5 authors
                if len(authors) > 5:
                    author_str += " et al."
                title_content += f" by {author_str}"

            chunks.append(
                PaperChunk(
                    content=title_content,
                    chunk_type=ChunkType.TITLE,
                    paper_id=paper_id,
                    paper_title=paper_title,
                    chunk_index=chunk_index,
                    token_count=self._estimate_tokens(title_content),
                    weight=self.SECTION_WEIGHTS[ChunkType.TITLE],
                )
            )
            chunk_index += 1

        # 2. Abstract chunk (high weight, usually fits in one chunk)
        if abstract:
            abstract_content = f"Abstract: {abstract}"

            if self._estimate_tokens(abstract_content) <= self.max_chunk_size:
                chunks.append(
                    PaperChunk(
                        content=abstract_content,
                        chunk_type=ChunkType.ABSTRACT,
                        paper_id=paper_id,
                        paper_title=paper_title,
                        chunk_index=chunk_index,
                        token_count=self._estimate_tokens(abstract_content),
                        weight=self.SECTION_WEIGHTS[ChunkType.ABSTRACT],
                    )
                )
                chunk_index += 1
            else:
                # Abstract too long, use sliding window
                abstract_chunks = self._sliding_window_chunk(
                    abstract_content, paper_id, paper_title, ChunkType.ABSTRACT, chunk_index
                )
                chunks.extend(abstract_chunks)
                chunk_index += len(abstract_chunks)

        # 3. Full text chunks (if available)
        if full_text:
            sections = self._split_into_sections(full_text)

            for section_type, section_text in sections:
                # Skip references section (low value for synthesis)
                if section_type == ChunkType.REFERENCES:
                    continue

                section_chunks = self._sliding_window_chunk(
                    section_text, paper_id, paper_title, section_type, chunk_index
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)

        logger.debug(f"Chunked paper '{paper_title[:50]}...' into {len(chunks)} chunks")

        return chunks

    def chunk_papers(self, papers: list[dict]) -> list[PaperChunk]:
        """
        Chunk multiple papers.

        Args:
            papers: List of paper dictionaries

        Returns:
            List of all PaperChunk objects
        """
        all_chunks = []

        for paper in papers:
            try:
                paper_chunks = self.chunk_paper(paper)
                all_chunks.extend(paper_chunks)
            except Exception as e:
                logger.error(f"Failed to chunk paper {paper.get('title', 'unknown')}: {e}")
                continue

        logger.info(f"Chunked {len(papers)} papers into {len(all_chunks)} total chunks")

        return all_chunks


def create_chunker(max_chunk_size: int = 512, overlap_size: int = 50) -> SemanticChunker:
    """Factory function to create a configured chunker."""
    return SemanticChunker(max_chunk_size=max_chunk_size, overlap_size=overlap_size)
