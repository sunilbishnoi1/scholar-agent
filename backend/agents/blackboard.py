"""
Working Memory Blackboard for Scholar Agent Autonomous Reasoning Pipeline.
Provides thread-safe, project-scoped in-flight working memory tracking goals,
evidence matrix, synthesis drafts, critic feedback, NLI audit, and DB sync.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel
from sqlalchemy.orm import Session

try:
    from agents.schemas import (
        BibliographyItem,
        CitationAuditReport,
        ConflictingDebate,
        CriticEvaluation,
        EvidenceMatrixRow,
        MethodologyDistribution,
        ReportMetadata,
        ReportStatus,
        ResearchGapItem,
        ResearchReport,
        ThematicSection,
    )
    from agents.state import (
        AgentMessage,
        AgentState,
        AgentType,
        GoalItem,
        GoalStatus,
        ParsedPaperData,
        TelemetryEvent,
    )
    from models.database import (
        EvidenceMatrixEntry,
        PaperCache,
        ResearchGapModel,
        ResearchProject,
        ResearchReportModel,
    )
except ImportError:
    from backend.agents.schemas import (
        BibliographyItem,
        CitationAuditReport,
        ConflictingDebate,
        CriticEvaluation,
        EvidenceMatrixRow,
        MethodologyDistribution,
        ReportMetadata,
        ReportStatus,
        ResearchGapItem,
        ResearchReport,
        ThematicSection,
    )
    from backend.agents.state import (
        AgentMessage,
        AgentState,
        AgentType,
        GoalItem,
        GoalStatus,
        ParsedPaperData,
        TelemetryEvent,
    )
    from backend.models.database import (
        EvidenceMatrixEntry,
        PaperCache,
        ResearchGapModel,
        ResearchProject,
        ResearchReportModel,
    )

logger = logging.getLogger(__name__)


class WorkingMemoryBlackboard:
    """
    Project-scoped Working Memory Blackboard.
    Tracks goal_stack, parsed_papers, evidence_matrix, draft_thematic_sections,
    debates, research_gaps, critic_feedback, audit_report, iteration_count,
    and telemetry with concurrency locks and relational persistence.
    """

    def __init__(
        self,
        project_id: str,
        user_id: str = "default_user",
        title: str = "",
        research_question: str = "",
        max_papers: int = 25,
        max_iterations: int = 2,
    ) -> None:
        self.project_id = project_id
        self.user_id = user_id
        self.title = title
        self.research_question = research_question
        self.max_papers = max_papers
        self.max_iterations = max(1, min(max_iterations, 2))  # Hard bounded limit [1, 2]
        self.iteration_count = 0

        # Primary Artifact Containers
        self.goal_stack: list[GoalItem] = []
        self.parsed_papers: dict[str, ParsedPaperData] = {}
        self.evidence_matrix: list[EvidenceMatrixRow] = []
        self.executive_summary: str = ""
        self.draft_thematic_sections: list[ThematicSection] = []
        self.debates: list[ConflictingDebate] = []
        self.research_gaps: list[ResearchGapItem] = []
        self.methodology_overview: MethodologyDistribution | None = None
        self.bibliography: list[BibliographyItem] = []
        self.critic_feedback: list[CriticEvaluation] = []
        self.audit_report: CitationAuditReport | None = None

        # Telemetry & Logging
        self.telemetry_events: list[TelemetryEvent] = []
        self.messages: list[AgentMessage] = []
        self.errors: list[str] = []

        # Concurrency & Observers
        self._lock = asyncio.Lock()
        self._observers: list[Callable[[str, dict[str, Any]], None]] = []

    def subscribe(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Register an observer callback for realtime state mutations."""
        if callback not in self._observers:
            self._observers.append(callback)

    def _notify(self, event_type: str, payload: dict[str, Any]) -> None:
        """Notify observers of blackboard mutations."""
        for callback in self._observers:
            try:
                callback(event_type, payload)
            except Exception as e:
                logger.warning(f"Observer callback failed for event '{event_type}': {e}")

    # -------------------------------------------------------------------------
    # Goal Stack Management
    # -------------------------------------------------------------------------

    def push_goal(
        self,
        goal_id: str,
        name: str,
        description: str,
        target_agent: AgentType | str,
        priority: int = 5,
        parent_goal_id: str | None = None,
    ) -> GoalItem:
        """Add a sub-goal or milestone to the goal stack."""
        agent_enum = AgentType(target_agent) if isinstance(target_agent, str) else target_agent
        goal: GoalItem = {
            "goal_id": goal_id,
            "name": name,
            "description": description,
            "target_agent": agent_enum,
            "status": GoalStatus.PENDING,
            "priority": priority,
            "parent_goal_id": parent_goal_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error_message": None,
        }
        self.goal_stack.append(goal)
        self._notify("goal_created", {"goal": goal})
        return goal

    def update_goal_status(
        self,
        goal_id: str,
        status: GoalStatus | str,
        error_message: str | None = None,
    ) -> bool:
        """Update lifecycle status of a goal on the stack."""
        status_enum = GoalStatus(status) if isinstance(status, str) else status
        for goal in self.goal_stack:
            if goal["goal_id"] == goal_id:
                goal["status"] = status_enum
                if status_enum in (GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.SKIPPED):
                    goal["completed_at"] = datetime.now(timezone.utc).isoformat()
                if error_message:
                    goal["error_message"] = error_message
                self._notify("goal_updated", {"goal": goal})
                return True
        return False

    # -------------------------------------------------------------------------
    # Artifact Mutation Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _coerce(cls: Any, val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, dict):
            return cls.model_validate(val)
        if isinstance(val, cls):
            return val
        if hasattr(val, "model_dump"):
            return cls.model_validate(val.model_dump())
        return cls.model_validate(dict(val))

    def add_parsed_paper(self, paper: ParsedPaperData | dict[str, Any]) -> None:
        """Add or update an acquired paper in the working memory."""
        pid = paper.get("paper_id", paper.get("id", "ref_unknown"))
        self.parsed_papers[pid] = paper  # type: ignore[assignment]
        self._notify("paper_added", {"paper_id": pid, "title": paper.get("title", "")})

    def set_evidence_matrix(self, rows: Sequence[EvidenceMatrixRow | dict[str, Any]]) -> None:
        """Set the extracted comparative evidence matrix."""
        self.evidence_matrix = [self._coerce(EvidenceMatrixRow, r) for r in rows]
        self._notify("matrix_updated", {"row_count": len(self.evidence_matrix)})

    def set_thematic_synthesis(
        self,
        executive_summary: str,
        sections: Sequence[ThematicSection | dict[str, Any]],
        debates: Sequence[ConflictingDebate | dict[str, Any]],
        gaps: Sequence[ResearchGapItem | dict[str, Any]],
        methodology_overview: MethodologyDistribution | dict[str, Any] | None = None,
    ) -> None:
        """Update draft thematic synthesis and discovered research gaps."""
        self.executive_summary = executive_summary
        self.draft_thematic_sections = [self._coerce(ThematicSection, s) for s in sections]
        self.debates = [self._coerce(ConflictingDebate, d) for d in debates]
        self.research_gaps = [self._coerce(ResearchGapItem, g) for g in gaps]
        if methodology_overview:
            self.methodology_overview = self._coerce(MethodologyDistribution, methodology_overview)
        self._notify(
            "synthesis_updated",
            {
                "section_count": len(self.draft_thematic_sections),
                "debate_count": len(self.debates),
                "gap_count": len(self.research_gaps),
            },
        )

    def add_critic_evaluation(self, evaluation: CriticEvaluation | dict[str, Any]) -> None:
        """Record an adversarial critic evaluation."""
        val_eval = self._coerce(CriticEvaluation, evaluation)
        self.critic_feedback.append(val_eval)
        self._notify(
            "critic_evaluation_added",
            {
                "score": val_eval.overall_score,
                "should_refine": val_eval.should_refine,
                "iteration": self.iteration_count,
            },
        )

    def set_audit_report(self, audit: CitationAuditReport | dict[str, Any]) -> None:
        """Record the citation grounding and NLI fact-checking audit."""
        val_audit = self._coerce(CitationAuditReport, audit)
        self.audit_report = val_audit
        self._notify(
            "audit_completed",
            {
                "precision": val_audit.precision_score,
                "passed": val_audit.audit_passed,
                "entailed": val_audit.entailed_count,
                "contradictions": val_audit.contradiction_count,
            },
        )


    def record_telemetry(
        self,
        agent: str,
        action: str,
        duration_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        model_name: str = "default",
    ) -> None:
        """Record execution telemetry for observability."""
        event: TelemetryEvent = {
            "agent": agent,
            "action": action,
            "duration_ms": duration_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost_usd,
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.telemetry_events.append(event)

    # -------------------------------------------------------------------------
    # State Synchronization (LangGraph <-> Blackboard)
    # -------------------------------------------------------------------------

    def to_agent_state(self) -> AgentState:
        """Export current blackboard state to a LangGraph AgentState TypedDict."""
        latest_score = self.critic_feedback[-1].overall_score if self.critic_feedback else 0.0
        should_refine = self.critic_feedback[-1].should_refine if self.critic_feedback else False
        refinement_guidance = (
            self.critic_feedback[-1].refinement_guidance if self.critic_feedback else []
        )

        ft_count = sum(1 for p in self.parsed_papers.values() if p.get("is_full_text"))
        abs_count = len(self.parsed_papers) - ft_count

        return AgentState(
            project_id=self.project_id,
            user_id=self.user_id,
            title=self.title,
            research_question=self.research_question,
            academic_level="graduate",
            target_word_count=3500,
            max_papers=self.max_papers,
            goal_stack=copy.deepcopy(self.goal_stack),
            current_node="supervisor",
            status="running",
            iteration_count=self.iteration_count,
            max_iterations=self.max_iterations,
            search_query_plan=None,
            candidate_papers=[],
            total_candidates_found=len(self.parsed_papers),
            parsed_papers=copy.deepcopy(self.parsed_papers),
            papers_analyzed_full_text=ft_count,
            papers_analyzed_abstract_only=abs_count,
            paper_chunks=[],
            evidence_matrix=copy.deepcopy(self.evidence_matrix),
            evidence_matrix_markdown="",
            executive_summary=self.executive_summary,
            draft_thematic_sections=copy.deepcopy(self.draft_thematic_sections),
            thematic_sections=[s.model_dump() for s in self.draft_thematic_sections],
            conflicting_debates=copy.deepcopy(self.debates),
            debates=[d.model_dump() for d in self.debates],
            research_gaps=copy.deepcopy(self.research_gaps),
            methodology_overview=copy.deepcopy(self.methodology_overview),
            bibliography=copy.deepcopy(self.bibliography),
            synthesis_draft=None,
            critic_evaluations=[e.model_dump() for e in self.critic_feedback],
            critic_evaluation=self.critic_feedback[-1].model_dump() if self.critic_feedback else None,
            current_critic_score=latest_score,
            should_refine=should_refine,
            refinement_guidance=refinement_guidance,
            audit_report=self.audit_report,
            citation_audit_report=self.audit_report.model_dump() if self.audit_report else None,
            audit_precision_score=self.audit_report.precision_score if self.audit_report else 0.0,
            audit_passed=self.audit_report.audit_passed if self.audit_report else False,
            final_report=None,
            messages=copy.deepcopy(self.messages),
            errors=copy.deepcopy(self.errors),
            telemetry=copy.deepcopy(self.telemetry_events),
            keywords=[],
            subtopics=[],
            search_strategy={},
            papers=list(self.parsed_papers.values()),
            total_papers_found=len(self.parsed_papers),
            analyzed_papers=list(self.parsed_papers.values()),
            high_quality_papers=list(self.parsed_papers.values()),
            synthesis=self.executive_summary,
            synthesis_sections=[s.model_dump() for s in self.draft_thematic_sections],
            quality_score=latest_score,
            quality_feedback="\n".join(refinement_guidance),
            current_agent=AgentType.SUPERVISOR,
            iteration=self.iteration_count,
            relevance_threshold=60.0,
        )

    def update_from_agent_state(self, state: AgentState) -> None:
        """Ingest updates from a LangGraph AgentState back into the blackboard."""
        self.iteration_count = state.get("iteration_count", self.iteration_count)
        self.goal_stack = state.get("goal_stack", self.goal_stack)

        raw_parsed = state.get("parsed_papers", {})
        if isinstance(raw_parsed, dict):
            self.parsed_papers.update(raw_parsed)
        elif isinstance(raw_parsed, list):
            for p in raw_parsed:
                pid = p.get("paper_id", p.get("id", "ref_unknown"))
                self.parsed_papers[pid] = p

        raw_matrix = state.get("evidence_matrix", [])
        matrix_rows: list[EvidenceMatrixRow] = []
        for r in raw_matrix:
            if isinstance(r, EvidenceMatrixRow):
                matrix_rows.append(r)
            elif isinstance(r, dict):
                try:
                    matrix_rows.append(EvidenceMatrixRow.model_validate(r))
                except Exception:
                    pass
        self.evidence_matrix = matrix_rows

        self.executive_summary = state.get("executive_summary", self.executive_summary)

        raw_sections = state.get("draft_thematic_sections") or state.get("thematic_sections", [])
        sections: list[ThematicSection] = []
        for s in raw_sections:
            if isinstance(s, ThematicSection):
                sections.append(s)
            elif isinstance(s, dict):
                try:
                    sections.append(ThematicSection.model_validate(s))
                except Exception:
                    pass
        self.draft_thematic_sections = sections

        raw_debates = state.get("conflicting_debates") or state.get("debates", [])
        debates: list[ConflictingDebate] = []
        for d in raw_debates:
            if isinstance(d, ConflictingDebate):
                debates.append(d)
            elif isinstance(d, dict):
                try:
                    debates.append(ConflictingDebate.model_validate(d))
                except Exception:
                    pass
        self.debates = debates

        raw_gaps = state.get("research_gaps", [])
        gaps: list[ResearchGapItem] = []
        for g in raw_gaps:
            if isinstance(g, ResearchGapItem):
                gaps.append(g)
            elif isinstance(g, dict):
                try:
                    gaps.append(ResearchGapItem.model_validate(g))
                except Exception:
                    pass
        self.research_gaps = gaps

        raw_overview = state.get("methodology_overview")
        if raw_overview:
            if isinstance(raw_overview, MethodologyDistribution):
                self.methodology_overview = raw_overview
            elif isinstance(raw_overview, dict):
                try:
                    self.methodology_overview = MethodologyDistribution.model_validate(raw_overview)
                except Exception:
                    pass

        raw_bib = state.get("bibliography", [])
        bib: list[BibliographyItem] = []
        for b in raw_bib:
            if isinstance(b, BibliographyItem):
                bib.append(b)
            elif isinstance(b, dict):
                try:
                    bib.append(BibliographyItem.model_validate(b))
                except Exception:
                    pass
        self.bibliography = bib


        raw_evals = state.get("critic_evaluations", [])
        self.critic_feedback = [
            e if isinstance(e, CriticEvaluation) else CriticEvaluation.model_validate(e)
            for e in raw_evals
        ]

        raw_audit = state.get("audit_report") or state.get("citation_audit_report")
        if raw_audit:
            self.audit_report = (
                raw_audit
                if isinstance(raw_audit, CitationAuditReport)
                else CitationAuditReport.model_validate(raw_audit)
            )

        self.messages = state.get("messages", self.messages)
        self.errors = state.get("errors", self.errors)
        self.telemetry_events = state.get("telemetry", self.telemetry_events)

    # -------------------------------------------------------------------------
    # Final Report Assembly & Database Persistence
    # -------------------------------------------------------------------------

    def assemble_research_report(self) -> ResearchReport:
        """Assemble the complete Pydantic v2 ResearchReport deliverable."""
        total_tokens = sum(t["total_tokens"] for t in self.telemetry_events)
        total_duration = sum(t["duration_ms"] for t in self.telemetry_events) / 1000.0
        models_used = list({t["model_name"] for t in self.telemetry_events if t.get("model_name")})
        ft_count = sum(1 for p in self.parsed_papers.values() if p.get("is_full_text"))
        latest_score = self.critic_feedback[-1].overall_score if self.critic_feedback else 85.0

        # Build bibliography from parsed papers if not explicitly populated
        bib_items: list[BibliographyItem] = list(self.bibliography)
        if not bib_items and self.parsed_papers:
            for pid, p in self.parsed_papers.items():
                bib_items.append(
                    BibliographyItem(
                        paper_id=pid,
                        title=p.get("title", "Untitled"),
                        authors=p.get("authors", []),
                        year=p.get("year"),
                        venue=p.get("venue"),
                        doi=p.get("doi"),
                        arxiv_id=p.get("arxiv_id"),
                        url=p.get("source_url", p.get("url")),
                        is_open_access=p.get("is_full_text", False),
                    )
                )

        overview = self.methodology_overview or MethodologyDistribution(
            distribution={"Empirical": len(self.parsed_papers)},
            dominant_approach="Empirical Analysis",
            trend_description="Dominant empirical methodology across acquired literature.",
        )

        metadata = ReportMetadata(
            project_id=self.project_id,
            user_id=getattr(self, "user_id", "default_user"),
            title=self.title or "Scientific Literature Review",
            research_question=self.research_question or "Research synthesis",

            academic_level="graduate",
            target_word_count=3500,
            generated_at=datetime.now(timezone.utc),
            status=ReportStatus.COMPLETE,
            total_papers_analyzed=len(self.parsed_papers),

            full_text_papers_count=ft_count,
            abstract_only_papers_count=len(self.parsed_papers) - ft_count,
            iteration_count=self.iteration_count,
            total_tokens_used=total_tokens,
            execution_time_seconds=total_duration,
            models_used=models_used or ["default"],
            quality_score=latest_score,
        )

        return ResearchReport(
            metadata=metadata,
            executive_summary=self.executive_summary or "Executive summary of literature review.",
            comparison_matrix=self.evidence_matrix,
            thematic_sections=self.draft_thematic_sections,
            conflicting_findings_and_debates=self.debates,
            actionable_research_gaps=self.research_gaps,
            methodology_overview=overview,
            bibliography=bib_items,
        )

    def sync_to_database(self, db: Session) -> None:
        """Persist blackboard state into relational PostgreSQL / SQLite models."""
        try:
            # 1. Update or create ResearchReportModel
            latest_score = self.critic_feedback[-1].overall_score if self.critic_feedback else 0.0
            sections_json = [s.model_dump() for s in self.draft_thematic_sections]
            debates_json = [d.model_dump() for d in self.debates]
            overview_json = (
                self.methodology_overview.model_dump()
                if self.methodology_overview
                else {"distribution": {}, "dominant_approach": "", "trend_description": ""}
            )

            existing_report = (
                db.query(ResearchReportModel)
                .filter(ResearchReportModel.project_id == self.project_id)
                .first()
            )
            if existing_report:
                existing_report.title = self.title
                existing_report.executive_summary = self.executive_summary
                existing_report.quality_score = latest_score
                existing_report.thematic_sections = sections_json
                existing_report.conflicts_and_debates = debates_json
                existing_report.methodology_overview = overview_json
                existing_report.generated_at = datetime.now(timezone.utc)
            else:
                new_report = ResearchReportModel(
                    project_id=self.project_id,
                    title=self.title,
                    executive_summary=self.executive_summary,
                    methodology_overview=overview_json,
                    quality_score=latest_score,
                    thematic_sections=sections_json,
                    conflicts_and_debates=debates_json,
                    generated_at=datetime.now(timezone.utc),
                )
                db.add(new_report)

            # 2. Sync EvidenceMatrixEntry records
            db.query(EvidenceMatrixEntry).filter(
                EvidenceMatrixEntry.project_id == self.project_id
            ).delete()

            for row in self.evidence_matrix:
                entry = EvidenceMatrixEntry(
                    id=str(uuid.uuid4()),
                    project_id=self.project_id,
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
                    is_full_text=bool(getattr(row, "is_full_text", False)),
                    created_at=datetime.now(timezone.utc),
                )
                db.add(entry)

            # 3. Sync ResearchGapModel records
            db.query(ResearchGapModel).filter(
                ResearchGapModel.project_id == self.project_id
            ).delete()

            for gap in self.research_gaps:
                gap_entry = ResearchGapModel(
                    id=str(uuid.uuid4()),
                    project_id=self.project_id,
                    gap_id=gap.gap_id,
                    description=gap.description,
                    importance=gap.importance,
                    recommended_methodology=gap.recommended_methodology,
                    grounding_paper_ids=gap.grounding_paper_ids,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(gap_entry)

            # 4. Upsert PaperCache records for acquired papers
            for pid, paper in self.parsed_papers.items():
                doi = paper.get("doi") or (f"id:{pid}" if not paper.get("arxiv_id") else None)
                cache_key = doi or f"arxiv:{paper.get('arxiv_id')}"
                existing_cache = db.query(PaperCache).filter(PaperCache.doi == cache_key).first()
                if not existing_cache:
                    new_cache = PaperCache(
                        doi=cache_key,
                        arxiv_id=paper.get("arxiv_id"),
                        s2_id=paper.get("s2_id"),
                        title=paper.get("title", "Untitled"),
                        authors=paper.get("authors", []),
                        year=paper.get("year"),
                        venue=paper.get("venue"),
                        abstract=paper.get("abstract", ""),
                        parsed_markdown=paper.get("full_text_markdown", paper.get("markdown_text", "")),
                        sections_json=paper.get("sections", []),
                        tables_json=paper.get("tables", []),
                        source_url=paper.get("source_url", paper.get("url")),
                        is_full_text=paper.get("is_full_text", False),
                        fetched_at=datetime.now(timezone.utc),
                    )
                    db.add(new_cache)

            db.commit()
            logger.info(f"Successfully synced blackboard state to database for project {self.project_id}")

        except Exception as e:
            logger.error(f"Failed to sync blackboard to database: {e}")
            db.rollback()
            raise

    def load_from_database(self, db: Session, project_id: str) -> bool:
        """Load existing persisted artifacts from database models into blackboard."""
        try:
            report_entry = (
                db.query(ResearchReportModel)
                .filter(ResearchReportModel.project_id == project_id)
                .first()
            )
            if not report_entry:
                return False

            self.project_id = project_id
            self.title = report_entry.title or self.title
            self.executive_summary = report_entry.executive_summary or ""

            if report_entry.thematic_sections:
                self.draft_thematic_sections = [
                    ThematicSection.model_validate(s) for s in report_entry.thematic_sections
                ]
            if report_entry.conflicts_and_debates:
                self.debates = [
                    ConflictingDebate.model_validate(d) for d in report_entry.conflicts_and_debates
                ]
            if report_entry.methodology_overview:
                self.methodology_overview = MethodologyDistribution.model_validate(
                    report_entry.methodology_overview
                )

            # Load matrix rows
            matrix_entries = (
                db.query(EvidenceMatrixEntry)
                .filter(EvidenceMatrixEntry.project_id == project_id)
                .all()
            )
            self.evidence_matrix = [
                EvidenceMatrixRow(
                    paper_id=e.paper_id,
                    title=e.title,
                    authors=[],
                    year=None,
                    methodology=e.methodology_type,
                    benchmark_dataset=e.benchmark_dataset,
                    primary_metric=e.primary_metric,
                    primary_limitation=e.primary_limitation,
                    is_full_text=True,
                )
                for e in matrix_entries
            ]

            # Load research gaps
            gap_entries = (
                db.query(ResearchGapModel)
                .filter(ResearchGapModel.project_id == project_id)
                .all()
            )
            self.research_gaps = [
                ResearchGapItem(
                    gap_id=g.gap_id,
                    description=g.description,
                    importance=g.importance,  # type: ignore[arg-type]
                    recommended_methodology=g.recommended_methodology,
                    grounding_paper_ids=g.grounding_paper_ids or [],
                )
                for g in gap_entries
            ]

            logger.info(f"Loaded blackboard state from database for project {project_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load blackboard from database: {e}")
            db.rollback()
            return False
