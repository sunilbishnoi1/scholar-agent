"""
1-Hop Citation Graph Traversal Tool for Scholar Agent.
Discovers seminal baseline papers (backward references) and follow-up breakthroughs (forward citations)
via Semantic Scholar and OpenAlex with strict deduplication and bounded traversal limits.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence, Union

import requests

try:
    from agents.schemas import AcademicPaperCandidate
except ImportError:
    from backend.agents.schemas import AcademicPaperCandidate

try:
    from agents.tools.academic_search import (
        deduplicate_and_merge_candidates,
        normalize_arxiv_id,
        normalize_doi,
        normalize_title,
        reconstruct_openalex_abstract,
    )
except ImportError:
    from backend.agents.tools.academic_search import (
        deduplicate_and_merge_candidates,
        normalize_arxiv_id,
        normalize_doi,
        normalize_title,
        reconstruct_openalex_abstract,
    )

logger = logging.getLogger(__name__)
USER_AGENT = "ScholarAgent/1.0 (mailto:contact@scholar-agent.com)"
MAX_GRAPH_LIMIT = 40


class CitationGraphTraverser:
    """
    1-Hop Citation Graph Traversal engine for seminal paper snowballing.
    Traverses forward citations (papers citing seed) and backward references (bibliography).
    """

    def __init__(
        self,
        s2_api_key: Optional[str] = None,
        timeout: float = 12.0,
    ):
        self.s2_api_key = s2_api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.timeout = timeout

    def _get_s2_forward_citations(self, s2_or_doi_id: str, limit: int = 15) -> list[AcademicPaperCandidate]:
        """Fetch papers citing the seed paper from Semantic Scholar."""
        url = f"https://api.semanticscholar.org/graph/v1/paper/{s2_or_doi_id}/citations"
        params = {
            "limit": min(limit, 30),
            "fields": "paperId,title,abstract,authors,year,venue,citationCount,externalIds,url",
        }
        headers = {"User-Agent": USER_AGENT}
        if self.s2_api_key:
            headers["x-api-key"] = self.s2_api_key

        candidates: list[AcademicPaperCandidate] = []
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            data = resp.json() or {}
            items = data.get("data") or []
            for item in items:
                if not isinstance(item, dict):
                    continue
                citing = item.get("citingPaper") or {}
                if not citing:
                    continue
                title = citing.get("title")
                if not title:
                    continue
                s2_id = citing.get("paperId")
                ext_ids = citing.get("externalIds") or {}
                doi = normalize_doi(ext_ids.get("DOI"))
                arxiv_id = normalize_arxiv_id(ext_ids.get("ArXiv"))
                paper_id = (
                    f"doi:{doi}"
                    if doi
                    else (f"arxiv:{arxiv_id}" if arxiv_id else f"s2:{s2_id}")
                )
                raw_authors = citing.get("authors") or []
                authors: list[str] = []
                for a in raw_authors:
                    if isinstance(a, dict) and a.get("name"):
                        authors.append(str(a["name"]).strip())
                    elif isinstance(a, str) and a.strip():
                        authors.append(a.strip())

                origin_url = (
                    f"https://doi.org/{doi}"
                    if doi
                    else (
                        f"https://arxiv.org/abs/{arxiv_id}"
                        if arxiv_id
                        else (citing.get("url") or (f"https://www.semanticscholar.org/paper/{s2_id}" if s2_id else ""))
                    )
                )

                candidates.append(
                    AcademicPaperCandidate(
                        paper_id=paper_id,
                        title=title,
                        authors=authors,
                        abstract=citing.get("abstract") or "",
                        year=citing.get("year"),
                        venue=citing.get("venue"),
                        doi=doi,
                        arxiv_id=arxiv_id,
                        s2_id=s2_id,
                        url=origin_url,
                        citation_count=citing.get("citationCount"),
                        source="semanticscholar_forward",
                        relevance_score=None,
                    )
                )
        except Exception as e:
            logger.warning(f"Error getting S2 forward citations for {s2_or_doi_id}: {e}")
        return candidates

    def _get_s2_backward_references(self, s2_or_doi_id: str, limit: int = 15) -> list[AcademicPaperCandidate]:
        """Fetch papers referenced in the bibliography of the seed paper from Semantic Scholar."""
        url = f"https://api.semanticscholar.org/graph/v1/paper/{s2_or_doi_id}/references"
        params = {
            "limit": min(limit, 30),
            "fields": "paperId,title,abstract,authors,year,venue,citationCount,externalIds,url",
        }
        headers = {"User-Agent": USER_AGENT}
        if self.s2_api_key:
            headers["x-api-key"] = self.s2_api_key

        candidates: list[AcademicPaperCandidate] = []
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            data = resp.json() or {}
            items = data.get("data") or []
            for item in items:
                if not isinstance(item, dict):
                    continue
                cited = item.get("citedPaper") or {}
                if not cited:
                    continue
                title = cited.get("title")
                if not title:
                    continue
                s2_id = cited.get("paperId")
                ext_ids = cited.get("externalIds") or {}
                doi = normalize_doi(ext_ids.get("DOI"))
                arxiv_id = normalize_arxiv_id(ext_ids.get("ArXiv"))
                paper_id = (
                    f"doi:{doi}"
                    if doi
                    else (f"arxiv:{arxiv_id}" if arxiv_id else f"s2:{s2_id}")
                )
                raw_authors = cited.get("authors") or []
                authors: list[str] = []
                for a in raw_authors:
                    if isinstance(a, dict) and a.get("name"):
                        authors.append(str(a["name"]).strip())
                    elif isinstance(a, str) and a.strip():
                        authors.append(a.strip())

                origin_url = (
                    f"https://doi.org/{doi}"
                    if doi
                    else (
                        f"https://arxiv.org/abs/{arxiv_id}"
                        if arxiv_id
                        else (cited.get("url") or (f"https://www.semanticscholar.org/paper/{s2_id}" if s2_id else ""))
                    )
                )

                candidates.append(
                    AcademicPaperCandidate(
                        paper_id=paper_id,
                        title=title,
                        authors=authors,
                        abstract=cited.get("abstract") or "",
                        year=cited.get("year"),
                        venue=cited.get("venue"),
                        doi=doi,
                        arxiv_id=arxiv_id,
                        s2_id=s2_id,
                        url=origin_url,
                        citation_count=cited.get("citationCount"),
                        source="semanticscholar_backward",
                        relevance_score=None,
                    )
                )
        except Exception as e:
            logger.warning(f"Error getting S2 backward references for {s2_or_doi_id}: {e}")
        return candidates

    def _get_openalex_forward_citations(self, doi_or_openalex_id: str, limit: int = 15) -> list[AcademicPaperCandidate]:
        """Fetch papers citing the seed paper from OpenAlex."""
        clean_id = doi_or_openalex_id.strip()
        if clean_id.lower().startswith("doi:"):
            clean_id = clean_id[4:].strip()
        if clean_id.startswith("https://doi.org/"):
            clean_id = clean_id.replace("https://doi.org/", "").strip()
        elif clean_id.startswith("http://doi.org/"):
            clean_id = clean_id.replace("http://doi.org/", "").strip()

        if not (clean_id.startswith("10.") or clean_id.upper().startswith("W")):
            return []

        url = "https://api.openalex.org/works"
        params = {
            "filter": f"cites:{clean_id}",
            "per-page": min(limit, 30),
            "sort": "cited_by_count:desc",
            "mailto": "contact@scholar-agent.com",
        }
        headers = {"User-Agent": USER_AGENT}
        candidates: list[AcademicPaperCandidate] = []
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            data = resp.json() or {}
            results = data.get("results") or []
            for item in results:
                if not isinstance(item, dict):
                    continue
                title = item.get("display_name") or item.get("title")
                if not title:
                    continue
                doi = normalize_doi(item.get("doi"))
                raw_id = item.get("id", "")
                openalex_id = raw_id.split("/")[-1] if raw_id else ""
                paper_id = f"doi:{doi}" if doi else f"openalex:{openalex_id}"
                authors = [
                    auth.get("author", {}).get("display_name")
                    for auth in (item.get("authorships") or [])
                    if isinstance(auth, dict) and isinstance(auth.get("author"), dict) and auth["author"].get("display_name")
                ]
                abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
                arxiv_id = normalize_arxiv_id(item.get("ids", {}).get("arxiv")) if item.get("ids") else None

                origin_url = (
                    f"https://doi.org/{doi}"
                    if doi
                    else (
                        f"https://arxiv.org/abs/{arxiv_id}"
                        if arxiv_id
                        else (item.get("doi") or (f"https://openalex.org/{openalex_id}" if openalex_id else raw_id))
                    )
                )

                candidates.append(
                    AcademicPaperCandidate(
                        paper_id=paper_id,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        year=item.get("publication_year"),
                        venue=(
                            item.get("primary_location", {}).get("source", {}).get("display_name")
                            if item.get("primary_location") and item.get("primary_location").get("source")
                            else None
                        ),
                        doi=doi,
                        arxiv_id=arxiv_id,
                        s2_id=None,
                        url=origin_url,
                        citation_count=item.get("cited_by_count"),
                        source="openalex_forward",
                        relevance_score=None,
                    )
                )
        except Exception as e:
            logger.warning(f"Error getting OpenAlex forward citations for {doi_or_openalex_id}: {e}")
        return candidates

    def traverse_1hop(
        self,
        seed_paper_ids: Sequence[Union[str, AcademicPaperCandidate]],
        include_forward: bool = True,
        include_backward: bool = True,
        limit_per_seed: int = 10,
        total_limit: int = 25,
    ) -> list[AcademicPaperCandidate]:
        """
        Execute 1-hop bidirectional citation graph traversal.
        Extracts forward and backward citations, deduplicates, excludes seeds, and caps total results.
        """
        if not seed_paper_ids:
            return []

        bounded_total = max(1, min(total_limit, MAX_GRAPH_LIMIT))
        seed_excluded_dois: set[str] = set()
        seed_excluded_arxiv: set[str] = set()
        seed_excluded_s2_ids: set[str] = set()
        seed_excluded_paper_ids: set[str] = set()
        seed_excluded_titles: set[str] = set()
        clean_seed_identifiers: list[str] = []

        for seed in seed_paper_ids:
            if isinstance(seed, str):
                cleaned = seed.strip()
                if not cleaned:
                    continue
                norm_doi = normalize_doi(cleaned)
                if norm_doi:
                    seed_excluded_dois.add(norm_doi)
                    clean_seed_identifiers.append(norm_doi)
                else:
                    norm_arxiv = normalize_arxiv_id(cleaned)
                    if norm_arxiv and ("." in norm_arxiv or "/" in norm_arxiv):
                        seed_excluded_arxiv.add(norm_arxiv)
                        clean_seed_identifiers.append(f"ARXIV:{norm_arxiv}")
                    else:
                        clean_s2 = cleaned
                        if clean_s2.lower().startswith("s2:"):
                            clean_s2 = clean_s2[3:].strip()
                        seed_excluded_s2_ids.add(clean_s2)
                        seed_excluded_paper_ids.add(cleaned)
                        clean_seed_identifiers.append(cleaned)

            elif hasattr(seed, "doi") or hasattr(seed, "paper_id") or hasattr(seed, "abstract") or type(seed).__name__ == "AcademicPaperCandidate":
                doi_val = getattr(seed, "doi", None)
                arxiv_val = getattr(seed, "arxiv_id", None)
                s2_val = getattr(seed, "s2_id", None)
                title_val = getattr(seed, "title", None)
                pid_val = getattr(seed, "paper_id", None)

                norm_doi = normalize_doi(doi_val) if doi_val else None
                norm_arxiv = normalize_arxiv_id(arxiv_val) if arxiv_val else None

                if norm_doi:
                    seed_excluded_dois.add(norm_doi)
                if norm_arxiv:
                    seed_excluded_arxiv.add(norm_arxiv)
                if s2_val:
                    clean_s2 = s2_val.strip()
                    if clean_s2.lower().startswith("s2:"):
                        clean_s2 = clean_s2[3:].strip()
                    seed_excluded_s2_ids.add(clean_s2)
                if pid_val:
                    clean_pid = str(pid_val).strip()
                    seed_excluded_paper_ids.add(clean_pid)
                    if clean_pid.lower().startswith("s2:"):
                        seed_excluded_s2_ids.add(clean_pid[3:].strip())
                if title_val and isinstance(title_val, str):
                    seed_excluded_titles.add(normalize_title(title_val))

                # Select single best query identifier per seed
                if norm_doi:
                    clean_seed_identifiers.append(norm_doi)
                elif s2_val:
                    clean_seed_identifiers.append(s2_val.strip())
                elif norm_arxiv:
                    clean_seed_identifiers.append(f"ARXIV:{norm_arxiv}")
                elif pid_val and not str(pid_val).startswith("ref_"):
                    clean_seed_identifiers.append(str(pid_val).strip())

        raw_candidates: list[AcademicPaperCandidate] = []

        for seed_id in clean_seed_identifiers:
            if include_forward:
                raw_candidates.extend(self._get_s2_forward_citations(seed_id, limit=limit_per_seed))
                raw_candidates.extend(self._get_openalex_forward_citations(seed_id, limit=limit_per_seed))
            if include_backward:
                raw_candidates.extend(self._get_s2_backward_references(seed_id, limit=limit_per_seed))

        if not raw_candidates:
            return []

        # Deduplicate
        unique = deduplicate_and_merge_candidates(raw_candidates)

        # Exclude seed papers
        filtered: list[AcademicPaperCandidate] = []
        for cand in unique:
            cand_doi = normalize_doi(cand.doi)
            if cand_doi and cand_doi in seed_excluded_dois:
                continue
            cand_arxiv = normalize_arxiv_id(cand.arxiv_id)
            if cand_arxiv and cand_arxiv in seed_excluded_arxiv:
                continue
            if cand.s2_id:
                clean_s2 = cand.s2_id.strip()
                if clean_s2.lower().startswith("s2:"):
                    clean_s2 = clean_s2[3:].strip()
                if clean_s2 in seed_excluded_s2_ids:
                    continue
            if cand.paper_id:
                clean_pid = str(cand.paper_id).strip()
                if clean_pid in seed_excluded_paper_ids or clean_pid in seed_excluded_s2_ids:
                    continue
                if clean_pid.lower().startswith("s2:") and clean_pid[3:].strip() in seed_excluded_s2_ids:
                    continue
            cand_title = normalize_title(cand.title)
            if cand_title and cand_title in seed_excluded_titles:
                continue
            filtered.append(cand)

        # Sort by citation count descending
        filtered.sort(key=lambda x: (x.citation_count or 0), reverse=True)

        return filtered[:bounded_total]


__all__ = [
    "CitationGraphTraverser",
]
