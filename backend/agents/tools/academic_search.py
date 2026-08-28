"""
Multi-Source Academic Search Tool for Scholar Agent.
Executes federated search across OpenAlex, Semantic Scholar, arXiv, and PubMed
with strict DOI deduplication, normalized title matching, and bounded candidate retrieval.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Optional
import urllib.parse
import xml.etree.ElementTree as ET

import requests

try:
    from agents.schemas import AcademicPaperCandidate
except ImportError:
    from backend.agents.schemas import AcademicPaperCandidate

logger = logging.getLogger(__name__)

OPENALEX_EMAIL = (
    os.environ.get("OPENALEX_EMAIL")
    or os.environ.get("OPENALEX_MAILTO")
    or "scholar_agent@scholar-agent.org"
)
USER_AGENT = f"ScholarAgent/1.0 (mailto:{OPENALEX_EMAIL})"
MAX_CANDIDATE_CAP = 40
MIN_CANDIDATE_CAP = 1
DEFAULT_SEARCH_LIMIT = 25


def sanitize_arxiv_query(query: str) -> str:
    """
    Sanitizes complex boolean queries into arXiv-compatible syntax.
    Extracts core quoted phrases or keywords and formats with 'AND all:' clauses.
    """
    if not query or not query.strip():
        return ""
    q = query.strip()
    if not re.search(r"[()\"']", q) and " " not in q:
        return f"all:{q}"

    # Extract quoted phrases
    quoted_phrases = re.findall(r'"([^"]+)"', q)
    if quoted_phrases:
        unique_phrases = list(dict.fromkeys(p.strip() for p in quoted_phrases if p.strip()))[:3]
        return " AND ".join(f'all:"{p}"' for p in unique_phrases)

    # Clean boolean operators and parentheses
    cleaned = re.sub(r"\b(AND|OR|NOT)\b", " ", q, flags=re.IGNORECASE)
    cleaned = re.sub(r"[()\"']", " ", cleaned)
    words = [w.strip() for w in cleaned.split() if len(w.strip()) > 2]
    if not words:
        return f"all:{q}"

    terms = words[:4]
    return " AND ".join(f"all:{t}" for t in terms)


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalize and validate DOI strings."""
    if not doi or not isinstance(doi, str):
        return None
    cleaned = doi.strip().lower()
    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
        "doi.org/",
    ]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    cleaned = cleaned.strip("/ ")
    if re.match(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$", cleaned):
        return cleaned
    return cleaned if cleaned.startswith("10.") and "/" in cleaned else None


def normalize_arxiv_id(arxiv_id: Optional[str]) -> Optional[str]:
    """Normalize arXiv identifier by stripping URLs, prefixes, and version suffixes."""
    if not arxiv_id or not isinstance(arxiv_id, str):
        return None
    cleaned = arxiv_id.strip()
    cleaned = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^arxiv:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\.pdf$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"v\d+$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() or None


def normalize_title(title: Optional[str]) -> str:
    """Normalize paper title for fuzzy/exact string comparison."""
    if not title or not isinstance(title, str):
        return ""
    norm = title.lower().strip()
    norm = re.sub(r"[^a-z0-9\s]", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def titles_match(t1: str, t2: str, threshold: float = 0.88) -> bool:
    """Check if two titles match using exact normalized equality or token Jaccard similarity."""
    n1 = normalize_title(t1)
    n2 = normalize_title(t2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    w1 = set(n1.split())
    w2 = set(n2.split())
    if not w1 or not w2:
        return False
    jaccard = len(w1 & w2) / len(w1 | w2)
    if jaccard >= threshold and len(n1) > 15 and len(n2) > 15:
        return True
    if len(n1) >= 15 and len(n2) >= 15 and (n1 in n2 or n2 in n1):
        return True
    return False


def reconstruct_openalex_abstract(inverted_index: Optional[dict[str, list[int]]]) -> str:
    """Reconstruct plain text abstract from OpenAlex inverted index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if isinstance(positions, (list, tuple)):
            for pos in positions:
                if isinstance(pos, int) and pos >= 0:
                    word_positions.append((pos, str(word)))
    if not word_positions:
        return ""
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions).strip()


def merge_candidate_into(target: AcademicPaperCandidate, incoming: AcademicPaperCandidate) -> None:
    """Merge incoming candidate metadata into existing target record."""
    if not target.doi and incoming.doi:
        target.doi = incoming.doi
    if not target.arxiv_id and incoming.arxiv_id:
        target.arxiv_id = incoming.arxiv_id
    if not target.s2_id and incoming.s2_id:
        target.s2_id = incoming.s2_id
    if not target.abstract and incoming.abstract:
        target.abstract = incoming.abstract
    elif incoming.abstract and len(incoming.abstract) > len(target.abstract):
        target.abstract = incoming.abstract
    if not target.year and incoming.year:
        target.year = incoming.year
    if not target.venue and incoming.venue:
        target.venue = incoming.venue
    if incoming.citation_count is not None:
        if target.citation_count is None or incoming.citation_count > target.citation_count:
            target.citation_count = incoming.citation_count
    if not target.url and incoming.url:
        target.url = incoming.url
    if not target.authors and incoming.authors:
        target.authors = incoming.authors
    # Append distinct source
    incoming_sources = [s.strip() for s in incoming.source.split(",") if s.strip()]
    target_sources = [s.strip() for s in target.source.split(",") if s.strip()]
    for s in incoming_sources:
        if s not in target_sources:
            target_sources.append(s)
    target.source = ",".join(target_sources)


def deduplicate_and_merge_candidates(
    candidates: list[AcademicPaperCandidate],
) -> list[AcademicPaperCandidate]:
    """Strictly deduplicate candidates by DOI, arXiv ID, and normalized title."""
    merged_by_doi: dict[str, AcademicPaperCandidate] = {}
    merged_by_arxiv: dict[str, AcademicPaperCandidate] = {}
    unique_candidates: list[AcademicPaperCandidate] = []

    for cand in candidates:
        norm_doi = normalize_doi(cand.doi)
        norm_arxiv = normalize_arxiv_id(cand.arxiv_id)

        if norm_doi and norm_doi in merged_by_doi:
            merge_candidate_into(merged_by_doi[norm_doi], cand)
            continue

        if norm_arxiv and norm_arxiv in merged_by_arxiv:
            merge_candidate_into(merged_by_arxiv[norm_arxiv], cand)
            continue

        matched = False
        for existing in unique_candidates:
            if titles_match(existing.title, cand.title):
                merge_candidate_into(existing, cand)
                matched = True
                if norm_doi and norm_doi not in merged_by_doi:
                    merged_by_doi[norm_doi] = existing
                if norm_arxiv and norm_arxiv not in merged_by_arxiv:
                    merged_by_arxiv[norm_arxiv] = existing
                break

        if not matched:
            unique_candidates.append(cand)
            if norm_doi:
                merged_by_doi[norm_doi] = cand
            if norm_arxiv:
                merged_by_arxiv[norm_arxiv] = cand

    return unique_candidates


class MultiSourceAcademicSearch:
    """
    Federated multi-source search engine querying OpenAlex, Semantic Scholar, Crossref, arXiv, and PubMed.
    Guarantees strict DOI deduplication, normalized title matching, and bounded candidate limits.
    """

    def __init__(
        self,
        s2_api_key: Optional[str] = None,
        openalex_api_key: Optional[str] = None,
        timeout: float = 12.0,
        enable_openalex: bool = True,
        enable_semanticscholar: bool = True,
        enable_crossref: bool = True,
        enable_arxiv: bool = True,
        enable_pubmed: bool = True,
    ):
        import time
        self.s2_api_key = s2_api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.openalex_api_key = openalex_api_key or os.environ.get("OPENALEX_API_KEY")
        self.timeout = timeout
        self.enable_openalex = enable_openalex
        self.enable_semanticscholar = enable_semanticscholar
        self.enable_crossref = enable_crossref
        self.enable_arxiv = enable_arxiv
        self.enable_pubmed = enable_pubmed
        self._last_s2_time = 0.0
        self._s2_cooldown_until = 0.0

    def search_openalex(self, query: str, limit: int = 15) -> list[AcademicPaperCandidate]:
        """Query OpenAlex Works API with Polite Pool, API key support, and retry backoff."""
        if not self.enable_openalex or not query.strip():
            return []
        url = "https://api.openalex.org/works"
        params: dict[str, Any] = {
            "search": query,
            "per-page": min(limit, 50),
            "sort": "relevance_score:desc",
            "mailto": OPENALEX_EMAIL,
        }
        headers = {"User-Agent": USER_AGENT}
        if self.openalex_api_key:
            params["api_key"] = self.openalex_api_key
            headers["api-key"] = self.openalex_api_key

        candidates: list[AcademicPaperCandidate] = []

        import time
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
                if resp.status_code == 429 and attempt < max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_time = float(retry_after) + 0.5 if retry_after else (1.5 * (attempt + 1))
                    time.sleep(min(15.0, sleep_time))
                    continue
                if resp.status_code != 200:
                    logger.warning(f"OpenAlex returned status {resp.status_code}")
                    return []
                data = resp.json() or {}
                for item in (data.get("results") or []):
                    if not isinstance(item, dict):
                        continue
                    title = item.get("display_name") or item.get("title") or ""
                    if not title:
                        continue
                    doi = normalize_doi(item.get("doi"))
                    raw_id = item.get("id", "")
                    openalex_id = raw_id.split("/")[-1] if raw_id else ""
                    paper_id = f"doi:{doi}" if doi else f"openalex:{openalex_id}"
                    authors: list[str] = []
                    for auth in (item.get("authorships") or []):
                        if isinstance(auth, dict) and isinstance(auth.get("author"), dict) and auth["author"].get("display_name"):
                            authors.append(str(auth["author"]["display_name"]).strip())
                    abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
                    year = item.get("publication_year")
                    venue = (
                        item.get("primary_location", {})
                        .get("source", {})
                        .get("display_name")
                        if item.get("primary_location") and item.get("primary_location").get("source")
                        else None
                    )
                    citation_count = item.get("cited_by_count")
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
                            year=year,
                            venue=venue,
                            doi=doi,
                            arxiv_id=arxiv_id,
                            s2_id=None,
                            url=origin_url,
                            citation_count=citation_count,
                            source="openalex",
                            relevance_score=None,
                        )
                    )
                break
            except Exception as e:
                logger.warning(f"OpenAlex search error for query '{query}': {e}")
                break
        return candidates

    def search_semantic_scholar(self, query: str, limit: int = 15) -> list[AcademicPaperCandidate]:
        """Query Semantic Scholar Graph API with rate limit throttling and retry backoff."""
        if not self.enable_semanticscholar or not query.strip():
            return []
        import time

        # Check provider cooldown
        if time.time() < self._s2_cooldown_until:
            logger.debug("Semantic Scholar in temporary cooldown. Skipping.")
            return []

        # Throttling interval for unauthenticated requests
        if not self.s2_api_key:
            elapsed = time.time() - self._last_s2_time
            if elapsed < 1.2:
                time.sleep(1.2 - elapsed)
            self._last_s2_time = time.time()

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 50),
            "fields": "paperId,title,abstract,authors,year,venue,citationCount,externalIds,url",
        }
        headers = {"User-Agent": USER_AGENT}
        if self.s2_api_key:
            headers["x-api-key"] = self.s2_api_key

        candidates: list[AcademicPaperCandidate] = []
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
                if resp.status_code == 429 and attempt < max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_time = float(retry_after) + 0.5 if retry_after else (2.0 * (attempt + 1))
                    time.sleep(min(15.0, sleep_time))
                    continue
                if resp.status_code == 429:
                    logger.warning("Semantic Scholar rate limited (429). Activating temporary cooldown.")
                    self._s2_cooldown_until = time.time() + 20.0
                    return []
                if resp.status_code != 200:
                    logger.warning(f"Semantic Scholar returned status {resp.status_code}")
                    return []
                data = resp.json() or {}
                for item in (data.get("data") or []):
                    if not isinstance(item, dict):
                        continue
                    title = item.get("title") or ""
                    if not title:
                        continue
                    s2_id = item.get("paperId")
                    ext_ids = item.get("externalIds") or {}
                    doi = normalize_doi(ext_ids.get("DOI"))
                    arxiv_id = normalize_arxiv_id(ext_ids.get("ArXiv"))
                    paper_id = (
                        f"doi:{doi}"
                        if doi
                        else (f"arxiv:{arxiv_id}" if arxiv_id else f"s2:{s2_id}")
                    )
                    authors: list[str] = []
                    for a in (item.get("authors") or []):
                        if isinstance(a, dict) and a.get("name"):
                            authors.append(str(a["name"]).strip())
                        elif isinstance(a, str) and a.strip():
                            authors.append(a.strip())
                    abstract = item.get("abstract") or ""
                    year = item.get("year")
                    venue = item.get("venue")
                    citation_count = item.get("citationCount")

                    origin_url = (
                        f"https://doi.org/{doi}"
                        if doi
                        else (
                            f"https://arxiv.org/abs/{arxiv_id}"
                            if arxiv_id
                            else (item.get("url") or (f"https://www.semanticscholar.org/paper/{s2_id}" if s2_id else ""))
                        )
                    )

                    candidates.append(
                        AcademicPaperCandidate(
                            paper_id=paper_id,
                            title=title,
                            authors=authors,
                            abstract=abstract,
                            year=year,
                            venue=venue,
                            doi=doi,
                            arxiv_id=arxiv_id,
                            s2_id=s2_id,
                            url=origin_url,
                            citation_count=citation_count,
                            source="semanticscholar",
                            relevance_score=None,
                        )
                    )
                break
            except Exception as e:
                logger.warning(f"Semantic Scholar search error for query '{query}': {e}")
                break
        return candidates

    def search_crossref(self, query: str, limit: int = 15) -> list[AcademicPaperCandidate]:
        """Query Crossref Works API as a high-availability fallback."""
        if not self.enable_crossref or not query.strip():
            return []
        url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": min(limit, 30),
            "sort": "relevance",
            "mailto": OPENALEX_EMAIL,
        }
        headers = {"User-Agent": USER_AGENT}
        candidates: list[AcademicPaperCandidate] = []
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            data = resp.json() or {}
            items = data.get("message", {}).get("items", [])
            for item in items:
                title_list = item.get("title", [])
                title = title_list[0] if title_list else ""
                if not title:
                    continue
                doi = normalize_doi(item.get("DOI"))
                authors: list[str] = []
                for auth in item.get("author", []):
                    given = auth.get("given", "")
                    family = auth.get("family", "")
                    full = f"{given} {family}".strip()
                    if full:
                        authors.append(full)

                raw_abstract = item.get("abstract", "")
                abstract = re.sub(r"<[^>]+>", "", raw_abstract).strip() if raw_abstract else ""

                year = None
                published = item.get("published") or item.get("issued") or {}
                date_parts = published.get("date-parts", [[]])
                if date_parts and date_parts[0] and isinstance(date_parts[0][0], int):
                    year = date_parts[0][0]

                venue = None
                container = item.get("container-title", [])
                if container and isinstance(container, list):
                    venue = container[0]

                citation_count = item.get("is-referenced-by-count")
                url_link = f"https://doi.org/{doi}" if doi else item.get("URL", "")

                candidates.append(
                    AcademicPaperCandidate(
                        paper_id=f"doi:{doi}" if doi else f"crossref:{doi or title[:30]}",
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        year=year,
                        venue=venue,
                        doi=doi,
                        arxiv_id=None,
                        s2_id=None,
                        url=url_link,
                        citation_count=citation_count,
                        source="crossref",
                        relevance_score=None,
                    )
                )
        except Exception as e:
            logger.warning(f"Crossref search error for query '{query}': {e}")
        return candidates

    def search_arxiv(self, query: str, limit: int = 15) -> list[AcademicPaperCandidate]:
        """Query arXiv API with robust timeout and query sanitization."""
        if not self.enable_arxiv or not query.strip():
            return []
        sanitized = sanitize_arxiv_query(query)
        encoded_query = urllib.parse.quote_plus(sanitized) if not sanitized.startswith("all:") else sanitized
        if encoded_query.startswith("all:"):
            # URL encode parameter value after all:
            pass
        url = (
            f"https://export.arxiv.org/api/query?search_query={urllib.parse.quote_plus(sanitized)}"
            f"&start=0&max_results={min(limit, 50)}&sortBy=relevance&sortOrder=descending"
        )
        headers = {"User-Agent": USER_AGENT}
        candidates: list[AcademicPaperCandidate] = []
        timeout_val = max(self.timeout, 25.0)

        try:
            resp = requests.get(url, headers=headers, timeout=timeout_val)
            if resp.status_code != 200:
                logger.warning(f"arXiv returned status {resp.status_code}")
                return []
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

            for entry in root.findall("atom:entry", ns):
                id_elem = entry.find("atom:id", ns)
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                if id_elem is None or title_elem is None:
                    continue

                raw_url = id_elem.text.strip() if id_elem.text else ""
                raw_arxiv_id = raw_url.split("/abs/")[-1] if "/abs/" in raw_url else raw_url
                arxiv_id = normalize_arxiv_id(raw_arxiv_id)
                title = re.sub(r"\s+", " ", title_elem.text or "").strip()
                abstract = re.sub(r"\s+", " ", summary_elem.text or "").strip() if summary_elem is not None else ""

                authors: list[str] = []
                for auth in entry.findall("atom:author", ns):
                    name_elem = auth.find("atom:name", ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())

                year = None
                pub_elem = entry.find("atom:published", ns)
                if pub_elem is not None and pub_elem.text:
                    match = re.search(r"\b(19\d\d|20\d\d)\b", pub_elem.text)
                    if match:
                        year = int(match.group(1))

                doi = None
                doi_elem = entry.find("arxiv:doi", ns)
                if doi_elem is not None and doi_elem.text:
                    doi = normalize_doi(doi_elem.text)

                venue = None
                jr_elem = entry.find("arxiv:journal_ref", ns)
                if jr_elem is not None and jr_elem.text:
                    venue = jr_elem.text.strip()

                paper_id = f"doi:{doi}" if doi else (f"arxiv:{arxiv_id}" if arxiv_id else raw_url)
                origin_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (f"https://doi.org/{doi}" if doi else raw_url)

                candidates.append(
                    AcademicPaperCandidate(
                        paper_id=paper_id,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        year=year,
                        venue=venue,
                        doi=doi,
                        arxiv_id=arxiv_id,
                        s2_id=None,
                        url=origin_url,
                        citation_count=None,
                        source="arxiv",
                        relevance_score=None,
                    )
                )
        except Exception as e:
            logger.warning(f"arXiv search error for query '{query}': {e}")
        return candidates

    def search_pubmed(self, query: str, limit: int = 15) -> list[AcademicPaperCandidate]:
        """Query PubMed via NCBI E-utilities with efetch for complete abstracts."""
        if not self.enable_pubmed or not query.strip():
            return []
        candidates: list[AcademicPaperCandidate] = []
        try:
            esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": min(limit, 20),
                "sort": "relevance",
            }
            headers = {"User-Agent": USER_AGENT}
            resp = requests.get(esearch_url, params=params, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            esearch_data = resp.json()
            id_list = esearch_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
            }
            fetch_resp = requests.get(efetch_url, params=fetch_params, headers=headers, timeout=max(self.timeout, 15.0))
            if fetch_resp.status_code == 200:
                # 1. Try XML parsing
                if fetch_resp.text and "<PubmedArticle" in fetch_resp.text:
                    try:
                        root = ET.fromstring(fetch_resp.text)
                        for article in root.findall(".//PubmedArticle"):
                            medline = article.find(".//MedlineCitation")
                            if medline is None:
                                continue
                            pmid_elem = medline.find("PMID")
                            pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else ""

                            article_elem = medline.find(".//Article")
                            if article_elem is None:
                                continue

                            title_elem = article_elem.find("ArticleTitle")
                            title = re.sub(r"\s+", " ", title_elem.text or "").strip().rstrip(".") if title_elem is not None else ""
                            if not title:
                                continue

                            abstract_parts: list[str] = []
                            for abs_elem in article_elem.findall(".//AbstractText"):
                                label = abs_elem.get("Label")
                                txt = "".join(abs_elem.itertext()).strip()
                                if txt:
                                    if label:
                                        abstract_parts.append(f"{label}: {txt}")
                                    else:
                                        abstract_parts.append(txt)
                            abstract = " ".join(abstract_parts).strip()

                            authors: list[str] = []
                            for auth in article_elem.findall(".//AuthorList/Author"):
                                last = auth.find("LastName")
                                fore = auth.find("ForeName")
                                if last is not None and last.text:
                                    if fore is not None and fore.text:
                                        authors.append(f"{fore.text} {last.text}")
                                    else:
                                        authors.append(last.text)

                            year = None
                            year_elem = article_elem.find(".//Journal/JournalIssue/PubDate/Year")
                            if year_elem is not None and year_elem.text:
                                try:
                                    year = int(year_elem.text)
                                except ValueError:
                                    pass
                            if not year:
                                medline_date = article_elem.find(".//Journal/JournalIssue/PubDate/MedlineDate")
                                if medline_date is not None and medline_date.text:
                                    match = re.search(r"\b(19\d\d|20\d\d)\b", medline_date.text)
                                    if match:
                                        year = int(match.group(1))

                            venue = None
                            journal_elem = article_elem.find(".//Journal/Title")
                            if journal_elem is not None and journal_elem.text:
                                venue = journal_elem.text.strip()

                            doi = None
                            for aid in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
                                if aid.get("IdType") == "doi" and aid.text:
                                    doi = normalize_doi(aid.text)
                                    break

                            paper_id = f"doi:{doi}" if doi else f"pubmed:{pmid}"
                            url_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                            candidates.append(
                                AcademicPaperCandidate(
                                    paper_id=paper_id,
                                    title=title,
                                    authors=authors,
                                    abstract=abstract,
                                    year=year,
                                    venue=venue,
                                    doi=doi,
                                    arxiv_id=None,
                                    s2_id=None,
                                    url=url_link,
                                    citation_count=None,
                                    source="pubmed",
                                    relevance_score=None,
                                )
                            )
                    except Exception as xml_err:
                        logger.warning(f"Error parsing PubMed XML: {xml_err}")

                # 2. Try JSON summary parsing on fetch_resp if it returned JSON (e.g. from ESummary mock)
                if not candidates:
                    try:
                        json_body = fetch_resp.json()
                        if isinstance(json_body, dict) and "result" in json_body:
                            sum_data = json_body.get("result", {})
                            for pmid in id_list:
                                item = sum_data.get(pmid)
                                if not item or not isinstance(item, dict):
                                    continue
                                title = item.get("title", "").rstrip(".")
                                if not title:
                                    continue
                                authors = [a.get("name") for a in item.get("authors", []) if a and a.get("name")]
                                year = None
                                pubdate = item.get("pubdate", "")
                                match = re.search(r"\b(19\d\d|20\d\d)\b", pubdate)
                                if match:
                                    year = int(match.group(1))
                                venue = item.get("source")
                                doi = None
                                for article_id in item.get("articleids", []):
                                    if article_id.get("idtype") == "doi":
                                        doi = normalize_doi(article_id.get("value"))
                                        break
                                paper_id = f"doi:{doi}" if doi else f"pubmed:{pmid}"
                                url_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                                candidates.append(
                                    AcademicPaperCandidate(
                                        paper_id=paper_id,
                                        title=title,
                                        authors=authors,
                                        abstract="",
                                        year=year,
                                        venue=venue,
                                        doi=doi,
                                        arxiv_id=None,
                                        s2_id=None,
                                        url=url_link,
                                        citation_count=None,
                                        source="pubmed",
                                        relevance_score=None,
                                    )
                                )
                    except Exception:
                        pass

            if not candidates:
                # Fallback to ESummary
                esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                sum_params = {
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "json",
                }
                sum_resp = requests.get(esummary_url, params=sum_params, headers=headers, timeout=self.timeout)
                if sum_resp.status_code == 200:
                    try:
                        sum_data = sum_resp.json().get("result", {})
                        for pmid in id_list:
                            item = sum_data.get(pmid)
                            if not item or not isinstance(item, dict):
                                continue
                            title = item.get("title", "").rstrip(".")
                            if not title:
                                continue
                            authors = [a.get("name") for a in item.get("authors", []) if a and a.get("name")]
                            year = None
                            pubdate = item.get("pubdate", "")
                            match = re.search(r"\b(19\d\d|20\d\d)\b", pubdate)
                            if match:
                                year = int(match.group(1))
                            venue = item.get("source")
                            doi = None
                            for article_id in item.get("articleids", []):
                                if article_id.get("idtype") == "doi":
                                    doi = normalize_doi(article_id.get("value"))
                                    break
                            paper_id = f"doi:{doi}" if doi else f"pubmed:{pmid}"
                            url_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            candidates.append(
                                AcademicPaperCandidate(
                                    paper_id=paper_id,
                                    title=title,
                                    authors=authors,
                                    abstract="",
                                    year=year,
                                    venue=venue,
                                    doi=doi,
                                    arxiv_id=None,
                                    s2_id=None,
                                    url=url_link,
                                    citation_count=None,
                                    source="pubmed",
                                    relevance_score=None,
                                )
                            )
                    except Exception as sum_err:
                        logger.warning(f"Error parsing PubMed ESummary JSON: {sum_err}")
        except Exception as e:
            logger.warning(f"PubMed search error for query '{query}': {e}")
        return candidates

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[AcademicPaperCandidate]:
        """
        Execute federated multi-source search across all active providers,
        deduplicate candidates strictly by DOI and normalized title,
        and return a bounded, ranked candidate list.
        """
        if not query or not query.strip():
            return []

        bounded_limit = max(MIN_CANDIDATE_CAP, min(limit, MAX_CANDIDATE_CAP))
        per_provider_limit = min(30, max(10, int(bounded_limit * 0.75)))

        raw_candidates: list[AcademicPaperCandidate] = []
        raw_candidates.extend(self.search_openalex(query, limit=per_provider_limit))
        raw_candidates.extend(self.search_semantic_scholar(query, limit=per_provider_limit))
        raw_candidates.extend(self.search_crossref(query, limit=per_provider_limit))
        raw_candidates.extend(self.search_arxiv(query, limit=per_provider_limit))
        raw_candidates.extend(self.search_pubmed(query, limit=per_provider_limit))

        if not raw_candidates:
            return []

        # Deduplicate
        unique = deduplicate_and_merge_candidates(raw_candidates)

        # Composite relevance scoring and ranking
        total_unique = len(unique)
        for rank, cand in enumerate(unique):
            # Base rank score from retrieval order
            base_score = max(0.1, (total_unique - rank) / max(1, total_unique) * 0.5)
            # Multi-provider consensus bonus
            source_count = len(cand.source.split(","))
            multi_source_bonus = 0.25 if source_count > 1 else 0.0
            # Citation count bonus
            citation_bonus = 0.0
            if cand.citation_count and cand.citation_count > 0:
                citation_bonus = min(0.20, math.log10(cand.citation_count + 1) / 20.0)
            # Recency bonus
            recency_bonus = 0.05 if (cand.year and cand.year >= 2022) else 0.0

            cand.relevance_score = round(min(1.0, base_score + multi_source_bonus + citation_bonus + recency_bonus), 4)

        # Sort descending by relevance score
        unique.sort(key=lambda x: (x.relevance_score or 0.0), reverse=True)

        return unique[:bounded_limit]


__all__ = [
    "AcademicPaperCandidate",
    "MultiSourceAcademicSearch",
    "sanitize_arxiv_query",
    "normalize_doi",
    "normalize_arxiv_id",
    "normalize_title",
    "titles_match",
    "reconstruct_openalex_abstract",
    "merge_candidate_into",
    "deduplicate_and_merge_candidates",
]
