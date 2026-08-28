"""
Unit tests for SectionAwareChunker and academic paper chunking.

Verifies:
- Heading hierarchy resolution, stack breadcrumbs, and parent category inheritance
- Regex classification for 9 academic section categories
- Exact citation anchor generation: [ref_{paper_id}#sec_{i}] and [ref_{paper_id}#tab_{j}]
- Atomic block preservation for Markdown tables and LaTeX math formulas
- Large table row-by-row splitting with preserved headers
- Sentence-aware sliding window fallback for headerless/unstructured markdown
- References filtering and importance weight calibration
- Full backward compatibility with SemanticChunker
"""

import pytest

from backend.rag.chunker import (
    SECTION_WEIGHTS,
    ChunkType,
    PaperChunk,
    SectionAwareChunker,
    SemanticChunker,
    create_chunker,
)


class TestSectionAwareChunkerClassification:
    """Test regex classification across diverse academic heading formats."""

    @pytest.fixture
    def chunker(self):
        return SectionAwareChunker(max_chunk_size=256, min_chunk_size=40)

    @pytest.mark.parametrize(
        ("heading", "expected_type"),
        [
            ("Abstract", ChunkType.ABSTRACT),
            ("## Executive Summary", ChunkType.ABSTRACT),
            ("# 1. Introduction and Background", ChunkType.INTRODUCTION),
            ("## Problem Formulation", ChunkType.INTRODUCTION),
            ("### 3. Methodology & System Architecture", ChunkType.METHODOLOGY),
            ("## Proposed Approach", ChunkType.METHODOLOGY),
            ("## Experimental Setup", ChunkType.METHODOLOGY),
            ("### 4. Experimental Results and Evaluation", ChunkType.RESULTS),
            ("## Empirical Findings", ChunkType.RESULTS),
            ("### Performance Comparison", ChunkType.RESULTS),
            ("## Limitations and Threats to Validity", ChunkType.LIMITATIONS),
            ("### Open Challenges and Failure Modes", ChunkType.LIMITATIONS),
            ("## Discussion and Broader Impact", ChunkType.DISCUSSION),
            ("## Conclusion and Future Work", ChunkType.CONCLUSION),
            ("## References", ChunkType.REFERENCES),
            ("## Works Cited", ChunkType.REFERENCES),
            ("Table 1: Benchmark Performance", ChunkType.TABLES),
        ],
    )
    def test_section_category_detection(self, chunker, heading, expected_type):
        detected = chunker._detect_section_type(heading)
        assert detected == expected_type


class TestHierarchyAndInheritance:
    """Test heading hierarchy stack, breadcrumb generation, and category inheritance."""

    def test_subheading_inherits_parent_category(self):
        chunker = SectionAwareChunker(max_chunk_size=512)
        markdown = """
# 3. Experimental Results

The overall performance across all benchmarks is strong.

## 3.1 Setup and Baselines
We evaluate against 5 state-of-the-art baselines.

## 3.2 Ablation Study
Removing the section-aware parser reduces retrieval precision by 18.4%.
"""
        chunks = chunker.chunk_document(markdown, paper_id="paper_abc", paper_title="Test Paper")
        assert len(chunks) >= 3

        # Check subheadings inherited RESULTS category
        ablation_chunks = [c for c in chunks if "Ablation Study" in c.content]
        assert len(ablation_chunks) > 0
        for ac in ablation_chunks:
            assert ac.chunk_type == ChunkType.RESULTS
            assert "3. Experimental Results" in ac.heading_hierarchy
            assert "3.2 Ablation Study" in ac.heading_hierarchy
            assert ac.parent_section == "3. Experimental Results"

    def test_anchor_tag_generation(self):
        chunker = SectionAwareChunker(max_chunk_size=512)
        markdown = """
# 1. Introduction
Background information on LLM reasoning.

# 2. Methodology
Our proposed multi-agent coordination protocol.

# 3. Limitations
Hardware constraints and scaling limitations.
"""
        chunks = chunker.chunk_document(markdown, paper_id="p101", paper_title="Orchestration")
        assert len(chunks) == 3
        assert chunks[0].anchor_tag == "[ref_p101#sec_1]"
        assert chunks[1].anchor_tag == "[ref_p101#sec_2]"
        assert chunks[2].anchor_tag == "[ref_p101#sec_3]"
        assert "[ref_p101#sec_1]" in chunks[0].content


class TestBlockPreservation:
    """Test atomic block preservation for LaTeX math, Markdown tables, and code."""

    def test_latex_math_block_preservation(self):
        chunker = SectionAwareChunker(max_chunk_size=100)  # Small max chunk to test preservation
        markdown = """
# 2. Mathematical Formulation

The optimization objective is defined as follows:

$$
\\mathcal{L}_{\\text{total}} = \\sum_{i=1}^{N} \\left( \\alpha \\cdot \\mathcal{L}_{\\text{rec}}(x_i, \\hat{x}_i) + \\beta \\cdot \\mathcal{D}_{\\text{KL}}(q(z|x_i) \\parallel p(z)) \\right) + \\gamma \\cdot \\Omega(\\theta)
$$

Where $\\Omega(\\theta)$ denotes the regularization penalty.
"""
        chunks = chunker.chunk_document(markdown, paper_id="p_math", paper_title="Math Paper")
        # Ensure LaTeX formula is not sliced in half
        math_chunks = [c for c in chunks if "\\mathcal{L}_{\\text{total}}" in c.content]
        assert len(math_chunks) == 1
        assert "\\Omega(\\theta)" in math_chunks[0].content
        assert "$$" in math_chunks[0].content

    def test_markdown_table_preservation_and_table_anchor(self):
        chunker = SectionAwareChunker(max_chunk_size=200)
        markdown = """
# 4. Evaluation Results

Below is the comparative performance table:

| Model | BLEU | ROUGE-1 | Exact Match |
|---|---|---|---|
| Baseline | 24.2 | 41.5 | 52.1 |
| ScholarAgent | 31.8 | 49.2 | 68.4 |

The results show clear statistical superiority.
"""
        chunks = chunker.chunk_document(markdown, paper_id="p_tab", paper_title="Table Paper")
        table_chunks = [c for c in chunks if "| Model |" in c.content]
        assert len(table_chunks) >= 1
        tc = table_chunks[0]
        assert tc.chunk_type == ChunkType.TABLES
        assert tc.table_index == 1
        assert tc.anchor_tag == "[ref_p_tab#tab_1]"
        assert "| ScholarAgent | 31.8 |" in tc.content

    def test_large_table_splitting_preserves_header(self):
        chunker = SectionAwareChunker(max_chunk_size=40)  # Very small chunk size to trigger table splitting
        rows = "\n".join([f"| Model_{i} | Dataset_{i} | Score_{i} | Accuracy_{i} |" for i in range(15)])
        markdown = f"""
# 4. Results

| Model | Dataset | Score | Accuracy |
|---|---|---|---|
{rows}
"""
        chunks = chunker.chunk_document(markdown, paper_id="p_big_tab", paper_title="Big Table Paper")
        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLES]
        assert len(table_chunks) > 1
        for tc in table_chunks:
            assert "| Model | Dataset | Score | Accuracy |" in tc.content
            assert "|---|---|---|---|" in tc.content


class TestEdgeCasesAndFallbacks:
    """Test edge cases: headerless markdown, references exclusion, and empty inputs."""

    def test_headerless_markdown_fallback(self):
        chunker = SectionAwareChunker(max_chunk_size=100, overlap_size=20)
        plain_text = (
            "Scientific discovery requires extensive synthesis across literature. "
            "Autonomous agents can accelerate this process significantly. "
            "However, verifying factual consistency is critical for reliable evidence extraction. "
            "Our benchmark evaluates 500 scientific papers across diverse domains."
        )
        chunks = chunker.chunk_document(plain_text, paper_id="p_plain", paper_title="Plain Text")
        assert len(chunks) >= 1
        for i, c in enumerate(chunks):
            assert c.chunk_type == ChunkType.GENERAL
            assert c.anchor_tag == f"[ref_p_plain#sec_{i + 1}]"
            assert "Scientific discovery" in c.content or "benchmark evaluates" in c.content

    def test_references_section_excluded(self):
        chunker = SectionAwareChunker(max_chunk_size=512)
        markdown = """
# 1. Introduction
Key insights on literature review automation.

# References
[1] Vaswani et al. Attention is All You Need. NeurIPS 2017.
[2] Brown et al. Language Models are Few-Shot Learners. NeurIPS 2020.
"""
        chunks = chunker.chunk_document(markdown, paper_id="p_ref", paper_title="Ref Paper")
        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.INTRODUCTION
        assert "Vaswani" not in [c.content for c in chunks]

    def test_empty_or_whitespace_markdown(self):
        chunker = SectionAwareChunker()
        assert chunker.chunk_document("", paper_id="p0") == []
        assert chunker.chunk_document("   \n\n\t  ", paper_id="p0") == []

    def test_extract_tables_method(self):
        chunker = SectionAwareChunker()
        markdown = """
# 1. Results

| A | B |
|---|---|
| 1 | 2 |

Text in between.

| X | Y |
|---|---|
| 8 | 9 |
"""
        tables = chunker.extract_tables(markdown, paper_id="p_t")
        assert len(tables) == 2
        assert all(t.chunk_type == ChunkType.TABLES for t in tables)


class TestFullPaperChunkingAndBackwardCompatibility:
    """Test chunk_paper, chunk_papers, and SemanticChunker backward-compatible alias."""

    def test_chunk_paper_with_metadata(self):
        chunker = create_chunker(max_chunk_size=256)
        paper = {
            "id": "10.1000/182",
            "title": "Autonomous Scientific Research Agents",
            "authors": ["Alice", "Bob", "Charlie", "Dave", "Eve", "Frank"],
            "abstract": "We present an autonomous agent architecture for multi-step scientific reasoning.",
            "full_text": """
# 1. Introduction
Autonomous discovery is an emerging frontier.

# 2. Methodology
We implement state-bounded DAG supervisors.

# 3. Results
Our approach outperforms traditional RAG by 34%.

# 4. Limitations
Requires access to open-access repositories.
""",
        }
        chunks = chunker.chunk_paper(paper)
        assert len(chunks) >= 4

        # Title chunk check
        title_chunk = chunks[0]
        assert title_chunk.chunk_type == ChunkType.TITLE
        assert title_chunk.weight == SECTION_WEIGHTS[ChunkType.TITLE]
        assert "et al." in title_chunk.content

        # Full-text chunks check
        types = {c.chunk_type for c in chunks}
        assert ChunkType.METHODOLOGY in types
        assert ChunkType.RESULTS in types
        assert ChunkType.LIMITATIONS in types

    def test_semantic_chunker_subclass_compatibility(self):
        chunker = SemanticChunker(max_chunk_size=512)
        assert isinstance(chunker, SectionAwareChunker)

        paper = {
            "id": "p_compat",
            "title": "Compatibility Test",
            "abstract": "Simple abstract.",
            "full_text": "## Methods\nStep 1, Step 2.",
        }
        chunks = chunker.chunk_paper(paper)
        assert len(chunks) >= 2
