"""
Deterministic Citation Auditor Agent for Scholar Agent.

Validates all factual propositions in the synthesis draft against full-text source chunks,
executes Natural Language Inference (NLI), strips hallucinated anchors, and compiles
the authoritative bibliography.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

try:
    from agents.base import BaseAgent
    from agents.llm.base import BaseLLMClient
    from agents.schemas import (
        BibliographyItem,
        CitationAuditReport,
        NLIVerdict,
        ThematicSection,
        ThematicSynthesisDraft,
    )
    from agents.state import AgentMessage, AgentResult, AgentState, AgentType
    from agents.tools.fact_checker import FactCheckerEngine
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.agents.llm.base import BaseLLMClient
    from backend.agents.schemas import (
        BibliographyItem,
        CitationAuditReport,
        NLIVerdict,
        ThematicSection,
        ThematicSynthesisDraft,
    )
    from backend.agents.state import AgentMessage, AgentResult, AgentState, AgentType
    from backend.agents.tools.fact_checker import FactCheckerEngine

logger = logging.getLogger(__name__)


class DeterministicCitationAuditorAgent(BaseAgent):
    """
    Deterministic Citation Auditor Agent.

    Capabilities:
    1. Extracts atomic propositions linked to citation anchors from synthesis drafts.
    2. Retrieves source section chunks from in-flight working memory.
    3. Runs NLI classification (Entailment / Neutral / Contradiction).
    4. Computes citation precision score and flags hallucinated anchors.
    5. Cleans and canonicalizes synthesis prose, stripping invalid citations.
    6. Compiles a verified, deduplicated academic bibliography.
    7. Updates state['audit_report'], state['audit_precision_score'], state['audit_passed'], state['bibliography'].
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        fact_checker: Optional[FactCheckerEngine] = None,
        name: str = "auditor",
    ) -> None:
        super().__init__(llm_client=llm_client, name=name)
        self.fact_checker = fact_checker or FactCheckerEngine(llm_client=llm_client)

    @staticmethod
    def _build_paper_chunks_map(
        parsed_papers_dict: dict[str, Any],
        paper_chunks_list: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Build mapping of paper_id -> list of section chunks for fast grounded retrieval.
        """
        chunks_map: dict[str, list[dict[str, Any]]] = {}

        # 1. Add chunks from paper_chunks list if present
        for c in paper_chunks_list:
            pid = c.get("paper_id", "")
            if pid:
                chunks_map.setdefault(pid, []).append(c)

        # 2. Add sections from parsed_papers_dict
        for pid, paper in parsed_papers_dict.items():
            if pid not in chunks_map or not chunks_map[pid]:
                sections = paper.get("sections", paper.get("sections_json", []))
                if sections:
                    for idx, s in enumerate(sections):
                        chunk_entry = {
                            "chunk_id": f"{pid}_sec_{idx+1}",
                            "paper_id": pid,
                            "chunk_type": s.get("section_type", "general"),
                            "section_title": s.get("heading", f"Section {idx+1}"),
                            "anchor_tag": s.get("anchor_tag", f"[{pid}#sec_{idx+1}]"),
                            "section_anchor": s.get("anchor_tag", f"{pid}#sec_{idx+1}").strip("[]"),
                            "content": s.get("content", ""),
                        }
                        chunks_map.setdefault(pid, []).append(chunk_entry)
                else:
                    # Fallback to abstract
                    abstract = paper.get("abstract", "")
                    if abstract:
                        chunks_map.setdefault(pid, []).append(
                            {
                                "chunk_id": f"{pid}_abstract",
                                "paper_id": pid,
                                "chunk_type": "abstract",
                                "section_title": "Abstract",
                                "anchor_tag": f"[{pid}#sec_abstract]",
                                "section_anchor": f"{pid}#sec_abstract",
                                "content": abstract,
                            }
                        )

        return chunks_map

    @staticmethod
    def _compile_bibliography(
        parsed_papers_dict: dict[str, Any],
        cited_paper_ids: set[str],
    ) -> list[BibliographyItem]:
        """Compile structured BibliographyItem list for all cited papers."""
        bib_items: list[BibliographyItem] = []
        seen_ids: set[str] = set()

        for pid, paper in parsed_papers_dict.items():
            clean_pid = pid.replace("ref_", "")
            is_cited = pid in cited_paper_ids or clean_pid in cited_paper_ids or f"ref_{clean_pid}" in cited_paper_ids
            if not is_cited and cited_paper_ids:
                continue

            if pid in seen_ids:
                continue
            seen_ids.add(pid)

            bib_items.append(
                BibliographyItem(
                    paper_id=pid,
                    title=paper.get("title", "Untitled"),
                    authors=paper.get("authors", []),
                    year=paper.get("year"),
                    venue=paper.get("venue"),
                    doi=paper.get("doi"),
                    arxiv_id=paper.get("arxiv_id"),
                    url=paper.get("source_url", paper.get("url")),
                    is_open_access=paper.get("is_full_text", False),
                )
            )

        return bib_items

    async def run(self, state: AgentState) -> AgentState:
        """Execute citation auditing and fact-checking workflow."""
        self._log_start(state)
        state["current_agent"] = AgentType.AUDITOR

        try:
            parsed_papers_dict = state.get("parsed_papers", {})
            paper_chunks_list = state.get("paper_chunks", [])
            synthesis_draft_dict = state.get("synthesis_draft")
            draft: ThematicSynthesisDraft | None = None
            if isinstance(synthesis_draft_dict, dict):
                try:
                    draft = ThematicSynthesisDraft.model_validate(synthesis_draft_dict)
                except Exception:
                    draft = None

            thematic_sections_raw = state.get("thematic_sections") or state.get("draft_thematic_sections") or []
            debates_raw = state.get("conflicting_debates") or state.get("debates") or []
            gaps_raw = state.get("research_gaps") or []

            if not draft:
                # Fallback draft reconstruction from available state fields
                draft = ThematicSynthesisDraft(
                    executive_summary=state.get("executive_summary") or state.get("synthesis", "Review completed."),
                    thematic_sections=[
                        ThematicSection(
                            theme_id=s.get("theme_id", f"theme_{i}") if isinstance(s, dict) else getattr(s, "theme_id", f"theme_{i}"),
                            title=s.get("title", f"Theme {i}") if isinstance(s, dict) else getattr(s, "title", f"Theme {i}"),
                            synthesis_prose=s.get("synthesis_prose", s.get("content", "")) if isinstance(s, dict) else getattr(s, "synthesis_prose", ""),
                            key_takeaways=s.get("key_takeaways", []) if isinstance(s, dict) else getattr(s, "key_takeaways", []),
                            cited_paper_ids=s.get("cited_paper_ids", []) if isinstance(s, dict) else getattr(s, "cited_paper_ids", []),
                        )
                        for i, s in enumerate(thematic_sections_raw, 1)
                    ],
                    conflicting_findings_and_debates=debates_raw,
                    actionable_research_gaps=gaps_raw,
                    methodology_overview=state.get("methodology_overview")
                    or {
                        "distribution": {},
                        "dominant_approach": "Empirical",
                        "trend_description": "",
                    },
                )

            # Build paper chunks map for grounding retrieval
            chunks_map = self._build_paper_chunks_map(parsed_papers_dict, paper_chunks_list)
            known_ids = set(parsed_papers_dict.keys()) | set(chunks_map.keys())

            self.logger.info(f"Auditing draft synthesis against {len(known_ids)} known paper references...")

            # Run Fact-Checking Audit
            audit_report: CitationAuditReport = await self.fact_checker.audit_thematic_draft(
                draft=draft,
                paper_chunks_map=chunks_map,
                known_paper_ids=known_ids,
            )

            # Clean and canonicalize prose in thematic sections
            cleaned_sections: list[ThematicSection] = []
            all_cited_ids: set[str] = set()

            for sec in draft.thematic_sections:
                cleaned_prose = self.fact_checker.canonicalize_and_clean_prose(
                    sec.synthesis_prose, audit_report
                )
                # Extract remaining valid cited paper IDs
                cited_in_sec = [
                    prop.paper_id
                    for prop in self.fact_checker.extract_atomic_propositions(cleaned_prose)
                ]

                valid_cited = [pid for pid in cited_in_sec if pid in known_ids or pid.replace("ref_", "") in known_ids]
                all_cited_ids.update(valid_cited)

                cleaned_sec = sec.model_copy(
                    update={
                        "synthesis_prose": cleaned_prose,
                        "cited_paper_ids": list(dict.fromkeys(valid_cited or sec.cited_paper_ids)),
                    }
                )
                cleaned_sections.append(cleaned_sec)

            # Clean executive summary
            cleaned_exec_summary = self.fact_checker.canonicalize_and_clean_prose(
                draft.executive_summary, audit_report
            )

            # Compile bibliography
            bibliography = self._compile_bibliography(
                parsed_papers_dict=parsed_papers_dict,
                cited_paper_ids=all_cited_ids or set(parsed_papers_dict.keys()),
            )

            # Update AgentState
            state["audit_report"] = audit_report
            state["citation_audit_report"] = audit_report.model_dump()
            state["audit_precision_score"] = audit_report.precision_score
            state["audit_passed"] = audit_report.audit_passed

            state["draft_thematic_sections"] = [s.model_dump() for s in cleaned_sections]
            state["thematic_sections"] = [s.model_dump() for s in cleaned_sections]
            state["executive_summary"] = cleaned_exec_summary
            state["bibliography"] = [b.model_dump() for b in bibliography]

            msg = self._create_message(
                action="citation_audit",
                content={
                    "total_propositions": audit_report.total_propositions,
                    "precision_score": audit_report.precision_score,
                    "entailed": audit_report.entailed_count,
                    "neutral": audit_report.neutral_count,
                    "contradictions": audit_report.contradiction_count,
                    "hallucinated_anchors": audit_report.hallucinated_anchors,
                    "audit_passed": audit_report.audit_passed,
                    "bibliography_items": len(bibliography),
                },
            )
            if "messages" not in state or state["messages"] is None:
                state["messages"] = []
            state["messages"].append(msg)

            self._log_complete(
                state,
                AgentResult(
                    success=True,
                    data={
                        "precision_score": audit_report.precision_score,
                        "audit_passed": audit_report.audit_passed,
                        "contradictions": audit_report.contradiction_count,
                        "bibliography_count": len(bibliography),
                    },
                ),
            )
            return state

        except Exception as e:
            return self._handle_error(state, e)


AuditorAgent = DeterministicCitationAuditorAgent

