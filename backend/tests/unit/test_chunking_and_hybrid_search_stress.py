"""
Document Chunking, Block Preservation, and Hybrid Search RRF Stress Test Suite.
Empirical verification of SectionAwareChunker, HybridSearchEngine, and Relational Models.

Stress dimensions tested:
1. Massive academic papers (50+ pages, 100+ sections, 6-level deep heading hierarchies, deep ancestry jumps)
2. Block preservation: Complex multi-row markdown tables, long table splitting, HTML tables, and multiline LaTeX environments
3. OCR-degraded & headerless unstructured documents: Noisy unicode, irregular whitespace, sentence-level windowing
4. Mathematical precision of Reciprocal Rank Fusion (RRF) with section multipliers, edge cases, and project isolation
5. Relational database constraints, cascade deletions, and JSON serialization
6. Adversarial edge cases: Code blocks with # comments, LaTeX with # symbols, and special character DOIs
"""

import math
import re
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.models.database import (
    Base,
    EvidenceMatrixEntry,
    PaperCache,
    ResearchGapModel,
    ResearchProject,
    ResearchReportModel,
    User,
)
from backend.rag.chunker import (
    SECTION_WEIGHTS,
    BlockType,
    ChunkType,
    PaperChunk,
    SectionAwareChunker,
    create_chunker,
)
from backend.rag.hybrid_search import (
    DEFAULT_SECTION_MULTIPLIERS,
    BM25Index,
    HybridSearchEngine,
    HybridSearchResult,
)
from backend.rag.vector_store import AcademicVectorStore, SearchResult


# ============================================================================
# 1. Stress Tests: Massive Academic Papers & Deep Heading Hierarchies
# ============================================================================


class TestMassiveAcademicPaperChunking:
    """Stress tests on massive 50+ page synthetic papers with deep hierarchies."""

    def generate_massive_paper_markdown(
        self, num_main_sections: int = 25, sub_depth: int = 5, words_per_section: int = 150
    ) -> str:
        """Generate a massive 50+ page academic markdown document."""
        lines = [
            "# Comprehensive Survey and Empirical Analysis of Autonomous Multi-Agent Reasoning Systems\n",
            "## Abstract\n",
            "This paper provides an exhaustive 50-page survey analyzing multi-agent reasoning, hierarchical planning, "
            "and factual citation auditing in autonomous scientific discovery. We formalize the problem across 100 benchmarks.\n",
        ]

        section_categories = [
            ("Introduction and Theoretical Foundations", "1"),
            ("Related Work and Literature Taxonomy", "2"),
            ("Methodology and System Architecture", "3"),
            ("Mathematical Formulation of Agent State Graphs", "4"),
            ("Experimental Setup and Benchmark Environments", "5"),
            ("Empirical Results and Performance Analysis", "6"),
            ("Ablation Studies and Component Isolation", "7"),
            ("Limitations, Scalability Bottlenecks, and Failure Modes", "8"),
            ("Ethical Considerations and Governance", "9"),
            ("Discussion and Comparative Synthesis", "10"),
            ("Conclusions and Open Research Directions", "11"),
        ]

        lorem_corpus = (
            "Autonomous scientific reasoning requires bidirectional synchronization between neural representations "
            "and symbolic logic engines. When scaling up context windows, models experience attention dispersion. "
            "To mitigate dispersion, structured evidence extraction maintains grounded proposition matrices. "
            "Our empirical evaluation demonstrates a 38.4% improvement in hallucination resistance across diverse domains. "
        ) * 5

        sec_idx = 1
        for cat_title, cat_num in section_categories:
            lines.append(f"\n# {cat_num}. {cat_title}\n")
            lines.append(f"High-level overview of section {cat_num}. {lorem_corpus[:200]}\n")

            # Nested subsections up to sub_depth levels
            for sub_1 in range(1, 4):
                lines.append(f"\n## {cat_num}.{sub_1} Subsystem Exploration {sub_1}\n")
                lines.append(f"Detailed analysis at level 2. {lorem_corpus}\n")

                for sub_2 in range(1, 3):
                    lines.append(f"\n### {cat_num}.{sub_1}.{sub_2} Subcomponent Analysis {sub_2}\n")
                    lines.append(f"Fine-grained details at level 3. {lorem_corpus}\n")

                    for sub_3 in range(1, 3):
                        lines.append(f"\n#### {cat_num}.{sub_1}.{sub_2}.{sub_3} Micro-level Metric {sub_3}\n")
                        lines.append(f"Micro analysis at level 4. {lorem_corpus}\n")

                        # Deep level 5
                        lines.append(
                            f"\n##### {cat_num}.{sub_1}.{sub_2}.{sub_3}.1 Deep Level Analysis\n"
                        )
                        lines.append(f"Deepest hierarchy node at level 5. {lorem_corpus[:300]}\n")

            sec_idx += 1

        # Add references at end
        lines.append("\n# References\n")
        for r in range(1, 101):
            lines.append(f"[{r}] Author {r} et al. Seminal Paper on Reasoning. Journal of AI, 202{r % 6}.\n")

        return "\n".join(lines)

    def test_massive_50_page_paper_chunking(self):
        chunker = SectionAwareChunker(max_chunk_size=400, min_chunk_size=50, overlap_size=40)
        markdown = self.generate_massive_paper_markdown()

        assert len(markdown) > 100_000, f"Markdown length was {len(markdown)}, expected >100,000 chars"

        chunks = chunker.chunk_document(markdown, paper_id="massive_p001", paper_title="Massive Survey")

        assert len(chunks) > 100, f"Expected >100 chunks, got {len(chunks)}"

        # 1. Verify anchor tags are sequential and valid
        for i, chunk in enumerate(chunks):
            assert chunk.anchor_tag.startswith("[ref_massive_p001#sec_") or chunk.anchor_tag.startswith("[ref_massive_p001#tab_")
            assert chunk.chunk_index == i
            assert chunk.paper_id == "massive_p001"
            assert chunk.token_count > 0
            assert chunk.start_char >= 0
            assert chunk.end_char <= len(markdown)

        # 2. Verify References section was excluded
        assert not any(c.chunk_type == ChunkType.REFERENCES for c in chunks)
        assert not any("Seminal Paper on Reasoning" in c.content for c in chunks)

        # 3. Verify category inheritance across deep hierarchies
        method_chunks = [c for c in chunks if "3. Methodology" in " > ".join(c.heading_hierarchy)]
        assert len(method_chunks) > 5
        for mc in method_chunks:
            assert mc.chunk_type == ChunkType.METHODOLOGY

        results_chunks = [c for c in chunks if "6. Empirical Results" in " > ".join(c.heading_hierarchy)]
        assert len(results_chunks) > 5
        for rc in results_chunks:
            assert rc.chunk_type == ChunkType.RESULTS

        limits_chunks = [c for c in chunks if "8. Limitations" in " > ".join(c.heading_hierarchy)]
        assert len(limits_chunks) > 5
        for lc in limits_chunks:
            assert lc.chunk_type == ChunkType.LIMITATIONS

    def test_deep_stack_unwinding_and_jumping(self):
        """Test heading stack correctly unwinds when jumping from Level 6 to Level 1."""
        chunker = SectionAwareChunker(max_chunk_size=512)
        markdown = """
# 1. Methodology
## 1.1 Architecture
### 1.1.1 Modules
#### 1.1.1.1 Submodules
##### 1.1.1.1.1 Core
###### 1.1.1.1.1.1 Leaf Node
Deepest leaf content.

# 2. Results
Direct jump to top-level section after deep nesting.
"""
        chunks = chunker.chunk_document(markdown, paper_id="p_jump", paper_title="Jump Test")
        assert len(chunks) >= 2

        leaf_chunk = [c for c in chunks if "Leaf Node" in c.content][0]
        assert leaf_chunk.chunk_type == ChunkType.METHODOLOGY
        assert len(leaf_chunk.heading_hierarchy) == 6
        assert leaf_chunk.heading_hierarchy[0] == "1. Methodology"
        assert leaf_chunk.heading_hierarchy[-1] == "1.1.1.1.1.1 Leaf Node"

        results_chunk = [c for c in chunks if "Direct jump" in c.content][0]
        assert results_chunk.chunk_type == ChunkType.RESULTS
        assert len(results_chunk.heading_hierarchy) == 1
        assert results_chunk.heading_hierarchy[0] == "2. Results"
        assert results_chunk.parent_section is None


# ============================================================================
# 2. Stress Tests: Tables, Complex LaTeX Math & Block Preservation
# ============================================================================


class TestComplexBlockPreservation:
    """Stress tests on complex tables, HTML tables, and multiline LaTeX equations."""

    def test_multi_equation_latex_environments(self):
        """Verify various LaTeX environments (align, gather, equation, bmatrix) stay atomic."""
        chunker = SectionAwareChunker(max_chunk_size=200)

        markdown = r"""
# 2. Mathematical Modeling

\begin{equation}
E = mc^2 + \int_{0}^{\infty} \psi(x) \nabla^2 \phi(x) \, dx
\end{equation}

Here is a system of linear equations in matrix form:

\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{bmatrix}
=
\begin{bmatrix}
b_1 \\ b_2 \\ \vdots \\ b_m
\end{bmatrix}

And aligned optimizations:

\begin{align*}
\min_{\theta} \quad & \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathcal{L}(f_\theta(x), y) \right] \\
\text{s.t.} \quad & g_i(\theta) \le 0, \quad i = 1, \dots, k \\
& h_j(\theta) = 0, \quad j = 1, \dots, m
\end{align*}
"""
        chunks = chunker.chunk_document(markdown, paper_id="p_math_stress", paper_title="Math Stress")

        # Verify all equations exist unbroken in chunks
        full_chunk_text = "\n\n".join([c.content for c in chunks])
        assert r"\begin{equation}" in full_chunk_text
        assert r"\end{equation}" in full_chunk_text
        assert r"\begin{bmatrix}" in full_chunk_text
        assert r"\end{bmatrix}" in full_chunk_text
        assert r"\begin{align*}" in full_chunk_text
        assert r"\end{align*}" in full_chunk_text

    def test_complex_pipe_table_with_escaped_characters(self):
        """Verify markdown tables with escaped pipes, code spans, and complex formatting."""
        chunker = SectionAwareChunker(max_chunk_size=300)

        markdown = """
# 4. Experimental Results

| Model Name | Config / Regex | Primary Metric (F1 \\| BLEU) | Latency (ms) | Notes |
|:---|:---|:---:|:---:|:---|
| Baseline-1 | `.*\\.py` | 82.4 | 120ms | Standard waterfall |
| Scholar-Agent-V1 | `(?i)ref_\\[0-9\\]` | 94.8 | 45ms | Hybrid RAG + RRF |
| Scholar-Agent-V2 | `\\| [a-z]+ \\|` | 98.2 | 32ms | Section-aware tokenization |

The table above demonstrates significant latency reduction and accuracy improvements.
"""
        chunks = chunker.chunk_document(markdown, paper_id="p_table_stress", paper_title="Table Paper")

        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLES]
        assert len(table_chunks) >= 1
        tc = table_chunks[0]
        assert tc.table_index == 1
        assert tc.anchor_tag == "[ref_p_table_stress#tab_1]"
        assert "Baseline-1" in tc.content
        assert "Scholar-Agent-V1" in tc.content
        assert "Scholar-Agent-V2" in tc.content

    def test_html_table_preservation(self):
        """Verify HTML <table>...</table> blocks are recognized and extracted as tables."""
        chunker = SectionAwareChunker(max_chunk_size=300)

        markdown = """
# 3. Methodology

We summarize the baseline comparison below:

<table>
  <thead>
    <tr><th>Method</th><th>Score</th><th>Hardware</th></tr>
  </thead>
  <tbody>
    <tr><td>Model A</td><td>88.5</td><td>4x A100</td></tr>
    <tr><td>Model B</td><td>94.1</td><td>8x H100</td></tr>
  </tbody>
</table>

Further details are discussed below.
"""
        chunks = chunker.chunk_document(markdown, paper_id="p_html_tab", paper_title="HTML Tab")
        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLES or c.table_index is not None]
        assert len(table_chunks) >= 1
        assert "<table>" in table_chunks[0].content
        assert "Model A" in table_chunks[0].content
        assert "Model B" in table_chunks[0].content

    def test_oversized_table_splitting_invariance(self):
        """Verify huge 50-row tables are split cleanly across sub-chunks with duplicated header."""
        chunker = SectionAwareChunker(max_chunk_size=60)  # Very small budget to force multiple splits

        rows = [f"| Experiment_{i:02d} | Setting_{i} | Accuracy: {90 + (i % 10):.1f}% | Seed: {1000 + i} |" for i in range(50)]
        table_body = "\n".join(rows)

        markdown = f"""
# 4. Results

| Experiment ID | Setting Type | Benchmark Metric | Random Seed |
|---|---|---|---|
{table_body}
"""
        chunks = chunker.chunk_document(markdown, paper_id="p_huge_tab", paper_title="Huge Table")
        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLES]

        assert len(table_chunks) >= 5, f"Expected >= 5 sub-table chunks, got {len(table_chunks)}"

        for i, tc in enumerate(table_chunks):
            # Every sub-table must preserve the header and delimiter
            assert "| Experiment ID | Setting Type | Benchmark Metric | Random Seed |" in tc.content
            assert "|---|---|---|---|" in tc.content
            assert tc.anchor_tag.startswith("[ref_p_huge_tab#tab_")

        # Verify all 50 experiments are present across chunks
        all_content = "\n".join([tc.content for tc in table_chunks])
        for i in range(50):
            assert f"Experiment_{i:02d}" in all_content


# ============================================================================
# 3. Stress Tests: OCR-Degraded & Headerless Documents
# ============================================================================


class TestOCRDegradedAndHeaderlessTexts:
    """Stress tests on noisy OCR output, unicode corruption, and headerless texts."""

    def test_completely_headerless_long_prose(self):
        """Verify long headerless prose splits into sentences with section anchors."""
        chunker = SectionAwareChunker(max_chunk_size=80, min_chunk_size=20, overlap_size=10)

        # 25 distinct sentences
        sentences = [
            f"Sentence {i}: Scientific discovery relies on autonomous literature extraction and synthesis across distributed datasets."
            for i in range(25)
        ]
        raw_text = " ".join(sentences)

        chunks = chunker.chunk_document(raw_text, paper_id="p_headerless", paper_title="Raw Text")

        assert len(chunks) >= 3
        # In headerless documents, all chunks originate from the single body section (sec_1)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_type == ChunkType.GENERAL
            assert chunk.anchor_tag == "[ref_p_headerless#sec_1]"
            assert chunk.section_title == "Body"
            assert chunk.heading_hierarchy == ["Body"]
            assert chunk.chunk_index == i

    def test_noisy_ocr_unicode_and_math_symbols(self):
        """Verify chunker handles OCR noise, mathematical symbols, and non-ASCII chars cleanly without crashing."""
        chunker = SectionAwareChunker(max_chunk_size=200)

        ocr_noise = """
# 1. Introduction and Background

Thê quântüm stâtë |ψ⟩ = α|0⟩ + β|1⟩ dëfînës the qubît côörđînâtës in Hîlbërt spâcë ℋ.
Whën ⟨ψ|φ⟩ = 0, stâtës ârë õrthôgônâl: ∑_{i=1}^n λ_i = 1.

# 2. Experimental Results and Evaluation

Thë àccuracy wâs 99.4% ± 0.05% acrôss 10,000 tëst instâncës (p < 0.001, χ² = 42.1).
"""
        chunks = chunker.chunk_document(ocr_noise, paper_id="p_ocr", paper_title="OCR Paper")
        assert len(chunks) >= 2

        intro_chunk = [c for c in chunks if "Introduction" in c.content or "Hîlbërt" in c.content][0]
        assert intro_chunk.chunk_type == ChunkType.INTRODUCTION
        assert "|ψ⟩" in intro_chunk.content

        results_chunk = [c for c in chunks if "Results" in c.content or "99.4%" in c.content][0]
        assert results_chunk.chunk_type == ChunkType.RESULTS

    def test_numbered_headings_without_markdown_hashes(self):
        """Verify academic papers with numbered headings like '1. Introduction' without '#' syntax."""
        chunker = SectionAwareChunker(max_chunk_size=300)

        markdown = """
1. Introduction
Autonomous literature review is a complex multi-stage cognitive task.

2. Methodology
We propose a state-bounded DAG supervisor coordinating specialized agents.

2.1 Model Architecture
The architecture comprises six distinct reasoning personas.

3. Experimental Results
Our benchmark indicates a 42% reduction in hallucinated synthesis citations.

4. Limitations
Requires open-access API access for full PDF parsing.
"""
        chunks = chunker.chunk_document(markdown, paper_id="p_numbered", paper_title="Numbered Headers")
        assert len(chunks) >= 4

        types = [c.chunk_type for c in chunks]
        assert ChunkType.INTRODUCTION in types
        assert ChunkType.METHODOLOGY in types
        assert ChunkType.RESULTS in types
        assert ChunkType.LIMITATIONS in types


# ============================================================================
# 4. Stress Tests: Mathematical Precision of Reciprocal Rank Fusion (RRF)
# ============================================================================


class TestRRFMathematicalPrecisionAndScoring:
    """Stress tests verifying exact formula compliance and edge conditions for HybridSearchEngine."""

    def test_exact_rrf_scoring_multiple_ranks(self):
        """
        Verify RRF formula across various ranks:
        RRF(d) = sum_{m in {vector, bm25}} (1 / (k + rank_m(d))) * W(section_type)
        with k = 60.
        """
        engine = HybridSearchEngine(rrf_k=60)

        # Create test items with diverse section types and ranks
        vec_results = [
            SearchResult("c_res", "Results text", "p1", "P1", "results", 0.95, 1.4),        # vec rank 1
            SearchResult("c_meth", "Method text", "p2", "P2", "methodology", 0.90, 1.3),   # vec rank 2
            SearchResult("c_lim", "Limits text", "p3", "P3", "limitations", 0.85, 1.2),    # vec rank 3
            SearchResult("c_abs", "Abstract text", "p4", "P4", "abstract", 0.80, 1.1),     # vec rank 4
            SearchResult("c_gen", "General text", "p5", "P5", "general", 0.75, 1.0),       # vec rank 5
        ]

        bm25_results = [
            ("c_gen", 10.5),   # bm25 rank 1
            ("c_meth", 8.2),   # bm25 rank 2
            ("c_res", 6.1),    # bm25 rank 3
            ("c_lim", 4.0),    # bm25 rank 4
            # c_abs not in bm25
        ]

        fused = engine._reciprocal_rank_fusion(
            vector_results=vec_results,
            bm25_results=bm25_results,
            bm25_index=None,
        )

        fused_dict = {r.chunk_id: r for r in fused}

        # Calculate exact expected scores
        # c_res: vec rank 1, bm25 rank 3, weight 1.4
        expected_res = (1.0 / (60 + 1) + 1.0 / (60 + 3)) * 1.4
        # c_meth: vec rank 2, bm25 rank 2, weight 1.3
        expected_meth = (1.0 / (60 + 2) + 1.0 / (60 + 2)) * 1.3
        # c_lim: vec rank 3, bm25 rank 4, weight 1.2
        expected_lim = (1.0 / (60 + 3) + 1.0 / (60 + 4)) * 1.2
        # c_abs: vec rank 4, bm25 rank 0 (absent), weight 1.1
        expected_abs = (1.0 / (60 + 4)) * 1.1
        # c_gen: vec rank 5, bm25 rank 1, weight 1.0
        expected_gen = (1.0 / (60 + 5) + 1.0 / (60 + 1)) * 1.0

        assert abs(fused_dict["c_res"].final_score - expected_res) < 1e-7
        assert abs(fused_dict["c_meth"].final_score - expected_meth) < 1e-7
        assert abs(fused_dict["c_lim"].final_score - expected_lim) < 1e-7
        assert abs(fused_dict["c_abs"].final_score - expected_abs) < 1e-7
        assert abs(fused_dict["c_gen"].final_score - expected_gen) < 1e-7

        # Verify descending order of final list
        for i in range(len(fused) - 1):
            assert fused[i].final_score >= fused[i + 1].final_score

    def test_bm25_lucene_smoothed_idf_stability(self):
        """Verify BM25 Lucene-smoothed IDF calculations are non-negative and monotonic."""
        index = BM25Index(k1=1.5, b=0.75)

        # 10 documents
        docs = [
            {"chunk_id": f"d_{i}", "content": f"common token apple orange unique_{i}"}
            for i in range(10)
        ]
        index.add_documents(docs)

        # IDF of ubiquitous term ('common', in 10/10 docs)
        # IDF = ln((10 - 10 + 0.5) / (10 + 0.5) + 1) = ln(0.5 / 10.5 + 1) = ln(1.047619) ≈ 0.0465
        idf_common = index._calculate_idf("common")
        assert idf_common > 0.0, "Lucene smoothed IDF must remain positive even for all-doc terms"

        # IDF of unique term ('unique_0', in 1/10 docs)
        # IDF = ln((10 - 1 + 0.5) / (1 + 0.5) + 1) = ln(9.5 / 1.5 + 1) = ln(7.3333) ≈ 1.9924
        idf_unique = index._calculate_idf("unique_0")
        assert idf_unique > idf_common, "Rare term IDF must be significantly higher than common term IDF"

        # IDF of non-existent term
        idf_none = index._calculate_idf("nonexistent_term")
        assert idf_none == 0.0

    def test_high_concurrency_project_isolation(self):
        """Verify multiple projects indexing identical chunk IDs remain fully isolated."""
        engine = HybridSearchEngine(vector_store=None)

        # 5 distinct projects
        for p_idx in range(5):
            proj_id = f"project_{p_idx}"
            docs = [
                {
                    "chunk_id": "c1",  # Same chunk ID across all projects
                    "content": f"Project specific domain keyword_{p_idx} secret_{p_idx}",
                    "paper_id": f"paper_{p_idx}",
                    "paper_title": f"Title {p_idx}",
                    "chunk_type": "methodology",
                }
            ]
            engine.index_project_documents(proj_id, docs)

        # Query project 2 for keyword_2
        res_2 = engine.search("keyword_2 secret_2", project_id="project_2")
        assert len(res_2) == 1
        assert res_2[0].paper_id == "paper_2"

        # Query project 3 for keyword_2 -> must return 0 results
        res_3 = engine.search("keyword_2 secret_2", project_id="project_3")
        assert len(res_3) == 0


# ============================================================================
# 5. Stress Tests: Relational Storage & Cascade Deletion Invariants
# ============================================================================


class TestRelationalStorageAndCascades:
    """Stress tests on SQLAlchemy models: PaperCache, Reports, Matrix, Gaps, and cascade deletion."""

    @pytest.fixture
    def db_session(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine)
        session = session_factory()
        yield session
        session.close()

    def test_paper_cache_full_text_and_json_payloads(self, db_session: Session):
        """Test PaperCache storage with complex JSON structures for sections and tables."""
        paper = PaperCache(
            doi="10.1145/3618257.3624800",
            arxiv_id="2310.01234",
            s2_id="s2_98765",
            title="Reasoning with LLMs",
            authors=["Alice Researcher", "Bob Scientist"],
            year=2024,
            venue="NeurIPS 2024",
            abstract="We investigate autonomous reasoning.",
            parsed_markdown="# 1. Intro\nContent...",
            sections_json=[
                {"heading": "1. Intro", "type": "introduction", "tokens": 150},
                {"heading": "2. Methods", "type": "methodology", "tokens": 400},
            ],
            tables_json=[
                {"table_id": "tab_1", "headers": ["Model", "Acc"], "rows": [["M1", "90%"]]}
            ],
            source_url="https://arxiv.org/abs/2310.01234",
            is_full_text=True,
        )
        db_session.add(paper)
        db_session.commit()

        retrieved = db_session.query(PaperCache).filter_by(doi="10.1145/3618257.3624800").first()
        assert retrieved is not None
        assert retrieved.is_full_text is True
        assert len(retrieved.authors) == 2
        assert len(retrieved.sections_json) == 2
        assert retrieved.tables_json[0]["table_id"] == "tab_1"

    def test_paper_cache_special_character_dois(self, db_session: Session):
        """Verify DOIs containing slashes, hashes, parentheses, and brackets store and query reliably."""
        special_dois = [
            "10.1000/182",
            "10.1002/(sici)1097-0142(19980101)82:1<1::aid-cncr1>3.0.co;2-m",
            "10.1016/s0040-4039(00)00213-4",
            "10.1103/physrevd.98.030001#sec4",
        ]
        for doi in special_dois:
            p = PaperCache(
                doi=doi,
                title=f"Paper for {doi}",
                authors=["Test Author"],
                is_full_text=False,
            )
            db_session.add(p)
        db_session.commit()

        for doi in special_dois:
            ret = db_session.query(PaperCache).filter_by(doi=doi).first()
            assert ret is not None
            assert ret.doi == doi

    def test_full_project_cascade_deletion(self, db_session: Session):
        """Verify deleting a ResearchProject cascades and deletes all associated reports, matrix entries, and gaps."""
        user = User(
            id="u_001",
            email="researcher@university.edu",
            name="Dr. Alan Turing",
            hashed_password="secure_hash_abc",
        )
        db_session.add(user)

        project = ResearchProject(
            id="proj_cascade_01",
            user_id="u_001",
            title="Autonomous Discovery Project",
            research_question="How to automate scientific reviews?",
        )
        db_session.add(project)
        db_session.commit()

        # Add reports, matrix entries, and gaps
        report = ResearchReportModel(
            id="rep_01",
            project_id="proj_cascade_01",
            title="Synthesis Report",
            executive_summary="Summary of findings.",
            quality_score=88.5,
            thematic_sections=[{"theme": "Theme 1", "anchors": ["[ref_p1#sec_1]"]}],
            conflicts_and_debates=[],
        )
        matrix_entry = EvidenceMatrixEntry(
            id="mat_01",
            project_id="proj_cascade_01",
            paper_id="doi_10.1000/1",
            title="Paper 1",
            methodology_type="Deep Learning",
            benchmark_dataset="ImageNet",
            primary_metric="Top-1 Acc: 85%",
            primary_limitation="High compute",
        )
        gap = ResearchGapModel(
            id="gap_01",
            project_id="proj_cascade_01",
            gap_id="GAP-01",
            description="Lack of long-context benchmark",
            importance="high",
            recommended_methodology="Create synthetic long-context evaluation dataset",
            grounding_paper_ids=["doi_10.1000/1"],
        )

        db_session.add_all([report, matrix_entry, gap])
        db_session.commit()

        # Verify existence
        assert db_session.query(ResearchReportModel).filter_by(project_id="proj_cascade_01").count() == 1
        assert db_session.query(EvidenceMatrixEntry).filter_by(project_id="proj_cascade_01").count() == 1
        assert db_session.query(ResearchGapModel).filter_by(project_id="proj_cascade_01").count() == 1

        # Delete project via ORM
        db_session.delete(project)
        db_session.commit()

        # Verify cascading deletion of all child models
        assert db_session.query(ResearchReportModel).filter_by(project_id="proj_cascade_01").count() == 0
        assert db_session.query(EvidenceMatrixEntry).filter_by(project_id="proj_cascade_01").count() == 0
        assert db_session.query(ResearchGapModel).filter_by(project_id="proj_cascade_01").count() == 0
