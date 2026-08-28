"""
Full-Text Ingestion Specialist Agent for Scholar Agent.

Executes the 3-tier Open-Access resolution cascade, parses PDFs into structured
Markdown, tables, and LaTeX equations via PDFParser, populates the PostgreSQL PaperCache,
chunks documents with SectionAwareChunker, and indexes vectors and BM25 records.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from sqlalchemy.orm import Session

try:
    from agents.base import BaseAgent
    from agents.llm.base import BaseLLMClient
    from agents.schemas import AcademicPaperCandidate, SectionType
    from agents.state import AgentMessage, AgentResult, AgentState, AgentType, ParsedPaperData
    from agents.tools.oa_resolver import OAResolutionResult, OAResolver
    from agents.tools.pdf_parser import ParsedDocument, ParsedSection, PDFParser
    from models.database import PaperCache
    from rag.chunker import ChunkType, PaperChunk, SectionAwareChunker
    from rag.service import RAGService
    from rag.vector_store import AcademicVectorStore
    from services.cancellation_manager import TaskCancelledException, cancellation_manager
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.agents.llm.base import BaseLLMClient
    from backend.agents.schemas import AcademicPaperCandidate, SectionType
    from backend.agents.state import AgentMessage, AgentResult, AgentState, AgentType, ParsedPaperData
    from backend.agents.tools.oa_resolver import OAResolutionResult, OAResolver
    from backend.agents.tools.pdf_parser import ParsedDocument, ParsedSection, PDFParser
    from backend.models.database import PaperCache
    from backend.rag.chunker import ChunkType, PaperChunk, SectionAwareChunker
    from backend.rag.service import RAGService
    from backend.rag.vector_store import AcademicVectorStore
    try:
        from backend.services.cancellation_manager import TaskCancelledException, cancellation_manager
    except ImportError:
        cancellation_manager = None
        TaskCancelledException = Exception

logger = logging.getLogger(__name__)


class FullTextIngestionSpecialist(BaseAgent):
    """
    Full-Text Ingestion Specialist Agent.

    Capabilities:
    1. Checks PostgreSQL PaperCache for pre-ingested papers by DOI or arXiv ID.
    2. Resolves open-access full-text PDFs using the 3-Tier OA cascade (Unpaywall -> arXiv/S2 -> Abstract Fallback).
    3. Parses binary PDFs into structured Markdown with headings, tables, and math equations via PDFParser.
    4. Persists parsed documents into global PostgreSQL PaperCache.
    5. Chunks full-text and abstract documents using SectionAwareChunker with anchor tags [ref_X#secY].
    6. Ingests embeddings into Qdrant vector store and builds BM25 index.
    7. Populates state['parsed_papers'] and state['paper_chunks'] for downstream synthesis and evidence matrix extraction.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        oa_resolver: Optional[OAResolver] = None,
        pdf_parser: Optional[PDFParser] = None,
        chunker: Optional[SectionAwareChunker] = None,
        vector_store: Optional[AcademicVectorStore] = None,
        rag_service: Optional[RAGService] = None,
        db_session: Optional[Session] = None,
        name: str = "ingestion",
    ) -> None:
        super().__init__(llm_client=llm_client, name=name)
        self.oa_resolver = oa_resolver or OAResolver()
        self.pdf_parser = pdf_parser or PDFParser()
        self.chunker = chunker or SectionAwareChunker()
        self.vector_store = vector_store
        self.rag_service = rag_service
        self.db_session = db_session

    def get_cached_paper(
        self,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        paper_id: str = "",
    ) -> Optional[ParsedDocument]:
        """Check PostgreSQL PaperCache for an existing parsed document."""
        if not self.db_session:
            return None

        try:
            query = self.db_session.query(PaperCache)
            entry = None
            if doi:
                entry = query.filter(PaperCache.doi == doi).first()
            if not entry and arxiv_id:
                entry = query.filter(PaperCache.arxiv_id == arxiv_id).first()

            if entry and entry.parsed_markdown:
                sections: list[ParsedSection] = []
                if entry.sections_json:
                    for s in entry.sections_json:
                        sec_type = ChunkType.from_str(s.get("section_type", s.get("chunk_type", "general")))
                        sections.append(
                            ParsedSection(
                                heading_level=s.get("level", s.get("heading_level", 1)),
                                title=s.get("heading", s.get("title", "")),
                                body=s.get("content", s.get("body", "")),
                                chunk_type=sec_type,
                            )
                        )

                return ParsedDocument(
                    paper_id=paper_id or (entry.doi or "cached_paper"),
                    doi=entry.doi,
                    arxiv_id=entry.arxiv_id,
                    s2_id=entry.s2_id,
                    title=entry.title,
                    authors=entry.authors or [],
                    year=entry.year,
                    venue=entry.venue,
                    abstract=entry.abstract,
                    markdown_text=entry.parsed_markdown,
                    sections=sections,
                    tables=entry.tables_json or [],
                    is_full_text=entry.is_full_text,
                    metadata={"source": "paper_cache", "source_url": entry.source_url},
                )
        except Exception as e:
            self.logger.warning(f"Error querying PaperCache: {e}")
            if self.db_session:
                self.db_session.rollback()

        return None

    def cache_paper(self, doc: ParsedDocument, source_url: Optional[str] = None) -> None:
        """Persist or update a parsed document in PostgreSQL PaperCache."""
        if not self.db_session:
            return

        try:
            cache_doi = doc.doi or (f"arxiv:{doc.arxiv_id}" if doc.arxiv_id else f"id:{doc.paper_id}")
            existing = self.db_session.query(PaperCache).filter(PaperCache.doi == cache_doi).first()

            sections_data = [
                {
                    "heading": getattr(s, "title", getattr(s, "heading", "")),
                    "title": getattr(s, "title", getattr(s, "heading", "")),
                    "content": getattr(s, "body", getattr(s, "content", "")),
                    "body": getattr(s, "body", getattr(s, "content", "")),
                    "section_type": (
                        s.chunk_type.value
                        if hasattr(s, "chunk_type") and hasattr(s.chunk_type, "value")
                        else (
                            s.section_type.value
                            if hasattr(s, "section_type") and hasattr(s.section_type, "value")
                            else str(getattr(s, "chunk_type", getattr(s, "section_type", "general")))
                        )
                    ),
                    "level": getattr(s, "heading_level", getattr(s, "level", 1)),
                    "heading_level": getattr(s, "heading_level", getattr(s, "level", 1)),
                    "tables": getattr(s, "tables", []),
                    "equations": getattr(s, "equations", []),
                }
                for s in doc.sections
            ]


            if existing:
                existing.title = doc.title or existing.title
                existing.authors = doc.authors or existing.authors
                existing.abstract = doc.abstract or existing.abstract
                existing.parsed_markdown = doc.markdown_text
                existing.sections_json = sections_data
                existing.tables_json = doc.tables
                existing.is_full_text = doc.is_full_text
                existing.source_url = source_url or existing.source_url
            else:
                new_entry = PaperCache(
                    doi=cache_doi,
                    arxiv_id=doc.arxiv_id,
                    s2_id=doc.s2_id,
                    title=doc.title,
                    authors=doc.authors,
                    year=getattr(doc, "year", None),
                    venue=getattr(doc, "venue", None),
                    abstract=doc.abstract,
                    parsed_markdown=doc.markdown_text,
                    sections_json=sections_data,
                    tables_json=doc.tables,
                    source_url=source_url,
                    is_full_text=doc.is_full_text,
                )
                self.db_session.add(new_entry)

            self.db_session.commit()
        except Exception as e:
            self.logger.warning(f"Failed to commit PaperCache entry for {doc.paper_id}: {e}")
            if self.db_session:
                self.db_session.rollback()

    def ingest_single_paper(
        self,
        candidate: dict[str, Any] | AcademicPaperCandidate,
        paper_id: str,
    ) -> ParsedDocument:
        """Resolve and parse an individual paper candidate into a ParsedDocument."""
        cand_dict = candidate.model_dump() if isinstance(candidate, AcademicPaperCandidate) else candidate
        doi = cand_dict.get("doi")
        arxiv_id = cand_dict.get("arxiv_id")
        title = cand_dict.get("title", "")
        s2_id = cand_dict.get("s2_id")
        abstract = cand_dict.get("abstract", "")
        authors = cand_dict.get("authors", [])
        year = cand_dict.get("year")
        venue = cand_dict.get("venue")

        # 1. Check PaperCache first
        cached = self.get_cached_paper(doi=doi, arxiv_id=arxiv_id, paper_id=paper_id)
        if cached:
            cached.paper_id = paper_id
            if not cached.authors and authors:
                cached.authors = authors
            if not getattr(cached, "year", None) and year:
                cached.year = year
            if not getattr(cached, "venue", None) and venue:
                cached.venue = venue
            if not getattr(cached, "doi", None) and doi:
                cached.doi = doi
            if not getattr(cached, "arxiv_id", None) and arxiv_id:
                cached.arxiv_id = arxiv_id
            return cached

        # 2. Execute 3-Tier OA Resolution Cascade
        resolution = self.oa_resolver.resolve_paper(
            doi=doi,
            arxiv_id=arxiv_id,
            title=title,
            s2_id=s2_id,
        )

        # 3. Parse PDF if bytes acquired (Tier 1/2)
        if resolution.pdf_bytes and len(resolution.pdf_bytes) >= 100:
            parsed_doc = self.pdf_parser.parse_pdf(
                pdf_bytes=resolution.pdf_bytes,
                paper_id=paper_id,
                doi=doi,
                arxiv_id=arxiv_id,
                title_hint=title,
            )
            parsed_doc.authors = authors or parsed_doc.authors
            parsed_doc.year = year
            parsed_doc.venue = venue
            if not parsed_doc.abstract and abstract:
                parsed_doc.abstract = abstract

            # Cache the parsed document
            self.cache_paper(parsed_doc, source_url=resolution.pdf_url)
            return parsed_doc

        # 4. Construct Tier 3 Structured Abstract Fallback Document
        fallback_abstract = (
            resolution.abstract_fallback.get("abstract", "")
            if resolution.abstract_fallback
            else abstract
        )
        fallback_md = f"# {title}\n\n## Abstract\n{fallback_abstract}\n"

        abstract_section = ParsedSection(
            heading_level=2,
            title="Abstract",
            body=fallback_abstract,
            chunk_type=ChunkType.ABSTRACT,
        )

        doc = ParsedDocument(
            paper_id=paper_id,
            doi=doi,
            arxiv_id=arxiv_id,
            s2_id=s2_id,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            abstract=fallback_abstract,
            markdown_text=fallback_md,
            sections=[abstract_section],
            tables=[],
            equations=[],
            is_full_text=False,
            metadata={"source": resolution.source, "fallback": True},
        )

        self.cache_paper(doc, source_url=resolution.landing_page_url)
        return doc

    async def run(self, state: AgentState) -> AgentState:
        """Execute full-text ingestion workflow on discovered candidate papers."""
        self._log_start(state)
        state["current_agent"] = AgentType.INGESTION

        raw_papers = state.get("papers", [])
        project_id = state.get("project_id", "default_project")

        if not raw_papers:
            self.logger.warning("No papers in state to ingest.")
            return state

        parsed_docs: list[ParsedDocument] = []
        all_chunks: list[PaperChunk] = []

        full_text_count = 0
        abstract_only_count = 0
        parsed_papers_dict: dict[str, Any] = {}

        for idx, paper in enumerate(raw_papers, start=1):
            if cancellation_manager and cancellation_manager.is_cancelled(project_id):
                self.logger.info(f"Ingestion cancelled for project '{project_id}' at paper {idx}/{len(raw_papers)}")
                raise TaskCancelledException(project_id)

            paper_id = paper.get("id") or f"ref_{idx}"
            self.logger.info(f"Ingesting paper {idx}/{len(raw_papers)}: [{paper_id}] {paper.get('title', '')[:50]}")

            doc = self.ingest_single_paper(candidate=paper, paper_id=paper_id)
            parsed_docs.append(doc)

            if doc.is_full_text:
                full_text_count += 1
            else:
                abstract_only_count += 1

            # Chunk document into typed section chunks with anchor tags
            chunks = self.chunker.chunk_document(
                markdown=doc.markdown_text,
                paper_id=paper_id,
                metadata={
                    "title": doc.title,
                    "authors": doc.authors,
                    "year": getattr(doc, "year", None),
                    "doi": doc.doi,
                    "arxiv_id": doc.arxiv_id,
                    "is_full_text": doc.is_full_text,
                },
            )
            all_chunks.extend(chunks)

            parsed_papers_dict[paper_id] = {
                "paper_id": doc.paper_id,
                "doi": doc.doi,
                "arxiv_id": doc.arxiv_id,
                "s2_id": doc.s2_id,
                "title": doc.title,
                "authors": doc.authors,
                "year": getattr(doc, "year", None),
                "venue": getattr(doc, "venue", None),
                "abstract": doc.abstract,
                "full_text_markdown": doc.markdown_text,
                "markdown_text": doc.markdown_text,
                "sections": [
                    {
                        "heading": getattr(s, "title", getattr(s, "heading", "")),
                        "title": getattr(s, "title", getattr(s, "heading", "")),
                        "content": getattr(s, "body", getattr(s, "content", "")),
                        "body": getattr(s, "body", getattr(s, "content", "")),
                        "section_type": (
                            s.chunk_type.value
                            if hasattr(s, "chunk_type") and hasattr(s.chunk_type, "value")
                            else (
                                s.section_type.value
                                if hasattr(s, "section_type") and hasattr(s.section_type, "value")
                                else str(getattr(s, "chunk_type", getattr(s, "section_type", "general")))
                            )
                        ),
                        "level": getattr(s, "heading_level", getattr(s, "level", 1)),
                        "anchor_tag": f"[{paper_id}#sec_{i+1}]",
                    }
                    for i, s in enumerate(doc.sections)
                ],
                "tables": doc.tables,
                "equations": doc.equations,
                "source_url": getattr(doc, "source_url", None),
                "is_full_text": doc.is_full_text,
                "citation_count": paper.get("citation_count"),
                "relevance_score": paper.get("relevance_score"),
            }


        # Index chunks into Vector Store and BM25 index if available
        chunks_indexed = 0
        if self.vector_store and all_chunks:
            try:
                chunks_indexed = self.vector_store.ingest_chunks(all_chunks, project_id=project_id)
                self.logger.info(f"Ingested {chunks_indexed} chunks into vector store for project '{project_id}'")
            except Exception as e:
                self.logger.warning(f"Vector store chunk indexing failed: {e}")

        # Update state
        state["parsed_papers"] = parsed_papers_dict
        state["papers_analyzed_full_text"] = full_text_count
        state["papers_analyzed_abstract_only"] = abstract_only_count
        state["paper_chunks"] = [
            {
                "chunk_id": getattr(c, "chunk_id", f"{c.paper_id}_chunk_{i+1}"),
                "paper_id": c.paper_id,
                "chunk_type": c.chunk_type.value if hasattr(c.chunk_type, "value") else str(c.chunk_type),
                "section_title": c.section_title,
                "anchor_tag": c.anchor_tag,
                "content": c.content,
                "token_count": c.token_count,
            }
            for i, c in enumerate(all_chunks)
        ]


        # Update backward compatibility fields
        state["analyzed_papers"] = list(parsed_papers_dict.values())
        state["high_quality_papers"] = list(parsed_papers_dict.values())

        msg = self._create_message(
            action="full_text_ingestion",
            content={
                "total_papers": len(parsed_docs),
                "full_text_count": full_text_count,
                "abstract_only_count": abstract_only_count,
                "total_chunks_created": len(all_chunks),
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
                    "full_text_count": full_text_count,
                    "abstract_only_count": abstract_only_count,
                    "chunks_count": len(all_chunks),
                },
            ),
        )
        return state


IngestionAgent = FullTextIngestionSpecialist

