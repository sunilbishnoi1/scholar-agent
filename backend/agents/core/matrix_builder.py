"""
Cross-Paper Evidence Matrix Builder Agent for Scholar Agent.

Isolates high-signal sections (Methodology, Results, Limitations, Tables) from acquired papers,
extracts structured comparative schemas (EvidenceMatrixRow), persists records into PostgreSQL,
and renders clean GitHub-Flavored Markdown comparison tables.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from sqlalchemy.orm import Session

try:
    from agents.base import BaseAgent
    from agents.llm.base import BaseLLMClient, ModelTier
    from agents.schemas import EvidenceMatrixExtraction, EvidenceMatrixRow, SectionType
    from agents.state import AgentMessage, AgentResult, AgentState, AgentType
    from models.database import EvidenceMatrixEntry
    from services.cancellation_manager import TaskCancelledException, cancellation_manager
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.agents.llm.base import BaseLLMClient, ModelTier
    from backend.agents.schemas import EvidenceMatrixExtraction, EvidenceMatrixRow, SectionType
    from backend.agents.state import AgentMessage, AgentResult, AgentState, AgentType
    from backend.models.database import EvidenceMatrixEntry
    try:
        from backend.services.cancellation_manager import TaskCancelledException, cancellation_manager
    except ImportError:
        cancellation_manager = None
        TaskCancelledException = Exception

logger = logging.getLogger(__name__)

MATRIX_EXTRACTION_SYSTEM_PROMPT = """You are a rigorous Scientific Meta-Analyst and Methodologist.
Your task is to extract precise, standardized comparative technical parameters from scientific papers.

For each paper, extract:
1. paper_id: Exact reference identifier provided (e.g. "ref_1").
2. title: Full academic title of the paper.
3. authors: List of author names.
4. year: Publication year as integer (or null if not determinable).
5. methodology: Specific technical architecture, mathematical algorithm, or system method (e.g. "Direct Preference Optimization with KL penalty", "Dense Retrieval with Reciprocal Rank Fusion").
6. benchmark_dataset: Key datasets or experimental environments evaluated (e.g. "MMLU, GSM8K, HumanEval", "MS MARCO, BEIR").
7. primary_metric: Quantitative metric and score reported (e.g. "Accuracy 89.2% (+3.4% over baseline)", "NDCG@10 0.542").
8. primary_limitation: Critical constraint, failure mode, or scalability bottleneck reported by authors.
9. is_full_text: True if extracted from full-text sections, False if extracted from abstract only.

Return strictly valid JSON adhering to the EvidenceMatrixRow schema.
"""


class EvidenceMatrixBuilder(BaseAgent):
    """
    Cross-Paper Evidence Matrix Builder Agent.

    Capabilities:
    1. Isolates high-signal sections (Methods, Results, Limitations, Tables) discarding noisy introduction text.
    2. Performs structured comparative extraction across all acquired papers into EvidenceMatrixRow objects.
    3. Persists comparative entries into PostgreSQL / SQLite relational storage.
    4. Renders a formatted GitHub-Flavored Markdown (GFM) comparison table.
    5. Populates state['evidence_matrix'] and state['evidence_matrix_markdown'].
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        db_session: Optional[Session] = None,
        name: str = "matrix_builder",
    ) -> None:
        super().__init__(llm_client=llm_client, name=name)
        self.db_session = db_session

    @staticmethod
    def isolate_high_signal_context(paper_dict: dict[str, Any]) -> str:
        """
        Extract only Methodology, Results, Limitations, and Tables sections from a paper.
        Discards introduction and related work noise to maximize LLM extraction signal.
        """
        title = paper_dict.get("title", "Untitled")
        abstract = paper_dict.get("abstract", "")
        sections = paper_dict.get("sections", paper_dict.get("sections_json", []))
        tables = paper_dict.get("tables", paper_dict.get("tables_json", []))
        is_full_text = paper_dict.get("is_full_text", False)

        context_parts = [f"Title: {title}\nAbstract: {abstract}"]

        if is_full_text and sections:
            high_signal_types = {"methodology", "results", "limitations", "tables", "experiments", "evaluation"}
            filtered_sections = []
            for s in sections:
                stype = str(s.get("section_type", s.get("type", "general"))).lower()
                sheading = s.get("heading", s.get("section_title", s.get("title", "")))
                scontent = s.get("content", s.get("text", ""))

                if any(hst in stype or hst in sheading.lower() for hst in high_signal_types):
                    filtered_sections.append(f"### {sheading} ({stype.upper()})\n{scontent[:2500]}")

            if filtered_sections:
                context_parts.append("\n".join(filtered_sections))
            else:
                # Fallback to first few sections if no explicit type match
                for s in sections[:4]:
                    sheading = s.get("heading", s.get("title", "Section"))
                    scontent = s.get("content", s.get("text", ""))
                    context_parts.append(f"### {sheading}\n{scontent[:1500]}")

        if tables:
            context_parts.append("### Key Tables & Quantitative Results:\n" + "\n".join(str(t) for t in tables[:3]))

        return "\n\n".join(context_parts)

    def extract_single_row(
        self,
        paper_dict: dict[str, Any],
        paper_id: str,
    ) -> EvidenceMatrixRow:
        """Extract a structured EvidenceMatrixRow from a paper's isolated context."""
        title = paper_dict.get("title", "Untitled")
        authors = paper_dict.get("authors", [])
        year = paper_dict.get("year")
        is_full_text = paper_dict.get("is_full_text", False)
        isolated_context = self.isolate_high_signal_context(paper_dict)

        if not self.llm_client:
            return self._fallback_extraction(paper_dict, paper_id)

        prompt = f"""Extract structured comparative parameters for the following scientific paper:

Paper ID: {paper_id}
Full-Text Available: {is_full_text}

=== PAPER HIGH-SIGNAL CONTEXT ===
{isolated_context[:8000]}
"""
        if hasattr(self.llm_client, "generate_structured"):
            try:
                row = self.llm_client.generate_structured(
                    prompt=prompt,
                    schema=EvidenceMatrixRow,
                    system_prompt=MATRIX_EXTRACTION_SYSTEM_PROMPT,
                    model_tier=ModelTier.REASONING,
                )
                if isinstance(row, EvidenceMatrixRow):
                    row.paper_id = paper_id
                    row.title = title or row.title
                    row.authors = authors or row.authors
                    row.year = year or row.year
                    row.doi = paper_dict.get("doi") or getattr(row, "doi", None)
                    row.arxiv_id = paper_dict.get("arxiv_id") or getattr(row, "arxiv_id", None)
                    row.url = paper_dict.get("source_url") or paper_dict.get("url") or getattr(row, "url", None)
                    row.is_full_text = is_full_text
                    return row
            except Exception as e:
                self.logger.warning(f"Structured matrix extraction failed for {paper_id} ({title[:30]}): {e}. Using rule fallback.")

        return self._fallback_extraction(paper_dict, paper_id)


    def _fallback_extraction(
        self,
        paper_dict: dict[str, Any],
        paper_id: str,
    ) -> EvidenceMatrixRow:
        """Deterministic rule-based fallback extraction if LLM is unavailable or fails."""
        title = paper_dict.get("title", "Untitled")
        authors = paper_dict.get("authors", [])
        year = paper_dict.get("year")
        abstract = paper_dict.get("abstract") or ""
        is_full_text = paper_dict.get("is_full_text", False)
        doi = paper_dict.get("doi")
        arxiv_id = paper_dict.get("arxiv_id")
        url = paper_dict.get("source_url") or paper_dict.get("url")

        method = "Empirical machine learning analysis and evaluation"
        dataset = "Standard benchmark datasets"
        metric = "Performance metrics evaluated"
        limitation = "Generalizability across broader domain distributions"

        # Heuristic extraction from abstract keywords
        abs_lower = abstract.lower()
        if "transformer" in abs_lower or "attention" in abs_lower:
            method = "Transformer-based neural architecture"
        elif "reinforcement" in abs_lower or "rlhf" in abs_lower:
            method = "Reinforcement learning from feedback"
        elif "retrieval" in abs_lower or "rag" in abs_lower:
            method = "Retrieval-augmented generation (RAG)"

        return EvidenceMatrixRow(
            paper_id=paper_id,
            title=title,
            authors=authors if isinstance(authors, list) else [],
            year=year if isinstance(year, int) else None,
            methodology=method,
            benchmark_dataset=dataset,
            primary_metric=metric,
            primary_limitation=limitation,
            is_full_text=is_full_text,
            doi=doi,
            arxiv_id=arxiv_id,
            url=url,
        )

    def persist_matrix_entries(
        self,
        rows: Sequence[EvidenceMatrixRow],
        project_id: str,
    ) -> int:
        """Persist extracted matrix rows into database table evidence_matrix_entries."""
        if not self.db_session:
            return len(rows)

        try:
            # Clear old entries for project
            self.db_session.query(EvidenceMatrixEntry).filter(
                EvidenceMatrixEntry.project_id == project_id
            ).delete()

            for row in rows:
                entry = EvidenceMatrixEntry(
                    id=str(uuid.uuid4()),
                    project_id=project_id,
                    paper_id=row.paper_id,
                    title=row.title,
                    methodology_type=row.methodology,
                    benchmark_dataset=row.benchmark_dataset,
                    primary_metric=row.primary_metric,
                    primary_limitation=row.primary_limitation,
                    authors=row.authors if isinstance(row.authors, list) else [],
                    year=row.year if isinstance(row.year, int) else None,
                    doi=getattr(row, "doi", None),
                    url=getattr(row, "url", None),
                    is_full_text=bool(row.is_full_text),
                    created_at=datetime.now(timezone.utc),
                )
                self.db_session.add(entry)

            self.db_session.commit()
            self.logger.info(f"Persisted {len(rows)} evidence matrix entries for project '{project_id}'")
            return len(rows)
        except Exception as e:
            self.logger.warning(f"Failed to persist evidence matrix entries: {e}")
            if self.db_session:
                self.db_session.rollback()
            return 0

    @staticmethod
    def render_markdown_table(rows: Sequence[EvidenceMatrixRow]) -> str:
        """Render extracted matrix rows into a clean GitHub-Flavored Markdown table."""
        if not rows:
            return "*No evidence matrix entries available.*"

        headers = [
            "| Ref ID | Title | Year | Methodology | Benchmark Dataset | Primary Metric | Primary Limitation | Full Text |",
            "| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :---: |",
        ]
        table_lines = list(headers)

        for r in rows:
            ft_str = "✅ Yes" if r.is_full_text else "📄 Abstract"
            year_str = str(r.year) if r.year else "—"
            clean_title = r.title.replace("|", "-")
            clean_meth = r.methodology.replace("|", "-")
            clean_data = r.benchmark_dataset.replace("|", "-")
            clean_metric = r.primary_metric.replace("|", "-")
            clean_limit = r.primary_limitation.replace("|", "-")

            line = f"| **[{r.paper_id}]** | {clean_title} | {year_str} | {clean_meth} | {clean_data} | {clean_metric} | {clean_limit} | {ft_str} |"
            table_lines.append(line)

        return "\n".join(table_lines)

    async def run(self, state: AgentState) -> AgentState:
        """Execute cross-paper evidence matrix extraction workflow."""
        self._log_start(state)
        state["current_agent"] = AgentType.MATRIX_BUILDER

        parsed_papers_obj = state.get("parsed_papers", {})
        project_id = state.get("project_id", "default_project")

        # Convert dict or list to normalized items list
        papers_list: list[dict[str, Any]] = []
        if isinstance(parsed_papers_obj, dict):
            for pid, pdata in parsed_papers_obj.items():
                pcopy = dict(pdata)
                pcopy.setdefault("paper_id", pid)
                papers_list.append(pcopy)
        elif isinstance(parsed_papers_obj, list):
            papers_list = list(parsed_papers_obj)
        else:
            raw_papers = state.get("papers", [])
            papers_list = list(raw_papers)

        if not papers_list:
            self.logger.warning("No papers available in state for matrix building.")
            return state

        self.logger.info(f"Extracting comparative evidence matrix across {len(papers_list)} papers...")

        matrix_rows: list[EvidenceMatrixRow] = []
        for idx, paper in enumerate(papers_list, start=1):
            if cancellation_manager and cancellation_manager.is_cancelled(project_id):
                self.logger.info(f"Matrix extraction cancelled for project '{project_id}' at paper {idx}/{len(papers_list)}")
                raise TaskCancelledException(project_id)

            paper_id = paper.get("paper_id") or paper.get("id") or f"ref_{idx}"
            self.logger.info(f"Extracting matrix parameters {idx}/{len(papers_list)}: [{paper_id}] {paper.get('title', '')[:40]}")
            row = self.extract_single_row(paper_dict=paper, paper_id=paper_id)
            matrix_rows.append(row)

        # Persist to database
        self.persist_matrix_entries(rows=matrix_rows, project_id=project_id)

        # Render Markdown table
        table_md = self.render_markdown_table(matrix_rows)

        # Update state
        state["evidence_matrix"] = [r.model_dump() for r in matrix_rows]
        state["evidence_matrix_markdown"] = table_md

        msg = self._create_message(
            action="evidence_matrix_extraction",
            content={
                "papers_extracted": len(matrix_rows),
                "full_text_papers": sum(1 for r in matrix_rows if r.is_full_text),
            },
        )
        if "messages" not in state or state["messages"] is None:
            state["messages"] = []
        state["messages"].append(msg)

        self._log_complete(state, AgentResult(success=True, data={"matrix_rows_count": len(matrix_rows)}))
        return state


MatrixBuilderAgent = EvidenceMatrixBuilder

