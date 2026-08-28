"""
Multi-Tier Open-Access Resolution Cascade for Scholar Agent.

Implements a 3-tier cascade to resolve scientific literature into full-text PDFs
or high-fidelity structured abstract metadata:
- Tier 1: Unpaywall REST API & OpenAlex direct OA PDF URLs
- Tier 2: arXiv direct PDF downloads & Semantic Scholar open preprint CDN URLs
- Tier 3: Graceful fallback to OpenAlex extended structured abstract + MeSH terms (is_full_text=False)

Invariants:
- Never raises an unhandled exception or fails the agent pipeline on paywalls, 403 Forbidden,
  404 Not Found, 429 Rate Limits, SSL errors, or network timeouts.
- Strictly validates PDF binary magic bytes (b"%PDF-") to reject HTML paywall/captcha pages.
- Reconstructs OpenAlex inverted-index abstracts into coherent narrative prose.
- Supports both dict-like subscription and Pydantic attribute access on OAResolutionResult.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Optional
import urllib.parse

import requests
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Polite User-Agent & Mailto defaults (required by OpenAlex Polite Pool & arXiv API)
DEFAULT_USER_AGENT = "ScholarAgent/1.0 (https://github.com/scholar-agent; contact@scholar-agent.org)"
DEFAULT_EMAIL = os.getenv("UNPAYWALL_EMAIL", "scholar_agent@scholar-agent.org")
DEFAULT_TIMEOUT = int(os.getenv("OA_RESOLVER_TIMEOUT", "15"))
MAX_PDF_SIZE_BYTES = int(os.getenv("MAX_PDF_SIZE_BYTES", str(50 * 1024 * 1024)))  # 50 MB


# ============================================================================
# Contract Data Models
# ============================================================================


class AbstractFallbackMetadata(BaseModel):
    """Structured container for Tier 3 graceful abstract fallback."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    s2_id: Optional[str] = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: str = ""
    mesh_terms: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    source_url: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class OAResolutionResult(BaseModel):
    """
    Standardized result from the 3-Tier Open-Access Resolution Cascade.

    Compatible with both dictionary subscripting (e.g., `result["is_full_text"]`)
    and Pydantic v2 attribute access (e.g., `result.is_full_text`).
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    pdf_bytes: Optional[bytes] = Field(
        default=None,
        description="Raw binary bytes of the resolved PDF document, or None if paywalled/unresolved.",
    )
    source: str = Field(
        default="unknown",
        description="Source identifier ('unpaywall', 'openalex', 'arxiv', 'semantic_scholar', 'pmc', 'abstract_fallback', 'unresolved').",
    )
    is_full_text: bool = Field(
        default=False,
        description="True if full-text PDF was successfully acquired; False if limited to abstract metadata.",
    )
    abstract_fallback: Optional[dict[str, Any]] = Field(
        default=None,
        description="Structured dictionary with abstract, MeSH terms, concepts, and metadata when PDF is unavailable.",
    )
    pdf_url: Optional[str] = Field(
        default=None,
        description="Direct download URL for the PDF if resolved.",
    )
    landing_page_url: Optional[str] = Field(
        default=None,
        description="Publisher or repository landing page URL.",
    )
    doi: Optional[str] = Field(
        default=None,
        description="Cleaned canonical DOI of the paper.",
    )
    arxiv_id: Optional[str] = Field(
        default=None,
        description="Cleaned canonical arXiv ID of the paper.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Paper title.",
    )
    resolution_tier: int = Field(
        default=3,
        description="Cascade tier that resolved the paper: 1 (Unpaywall/OpenAlex), 2 (arXiv/S2/PMC), or 3 (Fallback).",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Diagnostic error summary if higher tiers encountered failures.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional bibliographic and topic metadata extracted during resolution.",
    )

    # Dict-like subscripting compatibility
    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


# ============================================================================
# Identifier Normalization & Validation Helpers
# ============================================================================


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """
    Cleans and standardizes a DOI string.
    Removes URL prefixes (https://doi.org/), 'doi:' schemes, and whitespace.
    """
    if not doi or not isinstance(doi, str):
        return None
    cleaned = doi.strip()
    cleaned = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^doi:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().rstrip("/.")
    match = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", cleaned)
    if match:
        return match.group(1).lower()
    return cleaned.lower() if cleaned.startswith("10.") and "/" in cleaned else None


def normalize_arxiv_id(arxiv_id: Optional[str], doi: Optional[str] = None) -> Optional[str]:
    """
    Cleans and standardizes an arXiv identifier.
    Supports '2401.01234', 'arxiv:2401.01234v2', 'math/0102034', and arXiv DOIs.
    """
    if arxiv_id and isinstance(arxiv_id, str):
        cleaned = arxiv_id.strip()
        cleaned = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^arxiv:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\.pdf$", "", cleaned, flags=re.IGNORECASE)
        match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", cleaned, re.IGNORECASE)
        if match:
            return match.group(1)
        return cleaned.strip() or None

    if doi and isinstance(doi, str):
        match = re.search(r"10\.48550/arxiv\.(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", doi, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def is_valid_pdf_bytes(data: Optional[bytes], min_bytes: int = 1024) -> bool:
    """
    Verifies that binary data is genuinely a PDF file.
    Checks:
    1. Minimum byte size.
    2. Starts with magic bytes b"%PDF-".
    3. Is NOT an HTML login / Cloudflare captcha page returned with HTTP 200.
    """
    if not data or not isinstance(data, bytes) or len(data) < min_bytes:
        return False

    header_chunk = data[:1024]
    if b"%PDF-" not in header_chunk:
        return False

    lower_chunk = header_chunk.lower()
    if b"<!doctype html" in lower_chunk or b"<html" in lower_chunk:
        return False

    return True


def reconstruct_openalex_abstract(inverted_index: Optional[dict[str, list[int]]]) -> str:
    """
    Reconstructs continuous abstract text from an OpenAlex abstract_inverted_index.

    Example input: {"Deep": [0], "learning": [1], "models": [2]} -> "Deep learning models"
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""

    indexed_words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not isinstance(positions, (list, tuple)):
            continue
        for pos in positions:
            if isinstance(pos, int) and pos >= 0:
                indexed_words.append((pos, str(word)))

    if not indexed_words:
        return ""

    indexed_words.sort(key=lambda x: x[0])
    return " ".join(word for _, word in indexed_words).strip()


def extract_openalex_mesh_terms(mesh_list: Optional[list[dict[str, Any]]]) -> list[str]:
    """Extracts descriptor and qualifier names from OpenAlex MeSH array."""
    if not mesh_list or not isinstance(mesh_list, list):
        return []

    terms: list[str] = []
    for item in mesh_list:
        if not isinstance(item, dict):
            continue
        desc = item.get("descriptor_name")
        qual = item.get("qualifier_name")
        if desc and isinstance(desc, str):
            if qual and isinstance(qual, str):
                terms.append(f"{desc} - {qual}")
            else:
                terms.append(str(desc))
    return list(dict.fromkeys(terms))


def extract_openalex_concepts(concepts_list: Optional[list[dict[str, Any]]]) -> list[str]:
    """Extracts concept display names from OpenAlex concepts array."""
    if not concepts_list or not isinstance(concepts_list, list):
        return []

    concepts: list[str] = []
    for item in concepts_list:
        if isinstance(item, dict) and item.get("display_name") and isinstance(item["display_name"], str):
            concepts.append(str(item["display_name"]))
    return list(dict.fromkeys(concepts))


# ============================================================================
# Main Open-Access Resolver Class
# ============================================================================


class OAResolver:
    """
    Multi-Tier Open-Access Resolution Cascade Engine.

    Executes 3-tier cascade:
    1. Tier 1: Unpaywall REST API & OpenAlex direct OA PDF URLs
    2. Tier 2: arXiv direct PDF downloads & Semantic Scholar open preprint CDN URLs
    3. Tier 3: Graceful fallback to OpenAlex extended structured abstract + MeSH terms

    Pipeline Invariant: NEVER raises an unhandled exception.
    """

    def __init__(
        self,
        email: str = DEFAULT_EMAIL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: int = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
    ):
        self.email = email
        self.user_agent = user_agent
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})

    def _download_pdf(self, url: str) -> Optional[bytes]:
        """
        Downloads PDF bytes from URL with size cap and format verification.
        Returns None on any error, 403, 404, or non-PDF response.
        """
        if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return None

        try:
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "application/pdf,application/x-pdf,*/*;q=0.1",
            }
            resp = self._session.get(
                url,
                headers=headers,
                timeout=self.timeout,
                stream=True,
                allow_redirects=True,
            )
            if resp.status_code != 200:
                logger.debug(f"OA download status {resp.status_code} for {url}")
                return None

            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > MAX_PDF_SIZE_BYTES:
                        logger.warning(f"OA download exceeded max size ({MAX_PDF_SIZE_BYTES} bytes): {url}")
                        return None

            content = b"".join(chunks)
            if is_valid_pdf_bytes(content):
                return content
            logger.debug(f"OA downloaded data from {url} is not a valid PDF header")
            return None

        except Exception as exc:
            logger.debug(f"OA download network exception for {url}: {exc}")
            return None

    # ------------------------------------------------------------------------
    # Tier 1 Resolvers
    # ------------------------------------------------------------------------

    def _resolve_tier1_unpaywall(self, clean_doi: str) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
        """
        Queries Unpaywall REST API for publisher Open Access PDF.
        Returns: (pdf_bytes, pdf_url, landing_url)
        """
        try:
            encoded_doi = urllib.parse.quote(clean_doi)
            api_url = f"https://api.unpaywall.org/v2/{encoded_doi}?email={urllib.parse.quote(self.email)}"
            resp = self._session.get(api_url, timeout=self.timeout)
            if resp.status_code != 200:
                return None, None, None

            data = resp.json()
            if not isinstance(data, dict) or not data.get("is_oa"):
                return None, None, None

            # Try best OA location first
            best_loc = data.get("best_oa_location")
            if isinstance(best_loc, dict):
                pdf_url = best_loc.get("url_for_pdf")
                landing_url = best_loc.get("url_for_landing_page") or best_loc.get("url")

                if pdf_url and isinstance(pdf_url, str):
                    pdf_bytes = self._download_pdf(pdf_url)
                    if pdf_bytes:
                        return pdf_bytes, pdf_url, str(landing_url) if landing_url else None

            # Try other OA locations
            oa_locs = data.get("oa_locations")
            if isinstance(oa_locs, list):
                for loc in oa_locs:
                    if isinstance(loc, dict):
                        loc_pdf = loc.get("url_for_pdf")
                        if loc_pdf and isinstance(loc_pdf, str):
                            pdf_bytes = self._download_pdf(loc_pdf)
                            if pdf_bytes:
                                return pdf_bytes, loc_pdf, str(loc.get("url_for_landing_page") or loc.get("url") or "")

            return None, None, None

        except Exception as exc:
            logger.debug(f"Unpaywall resolution exception for {clean_doi}: {exc}")
            return None, None, None

    def _resolve_tier1_openalex(
        self, clean_doi: Optional[str], title: Optional[str]
    ) -> tuple[Optional[bytes], Optional[str], Optional[str], Optional[dict[str, Any]]]:
        """
        Queries OpenAlex Works API.
        Returns: (pdf_bytes, pdf_url, landing_url, extracted_metadata)
        """
        try:
            work_data: Optional[dict[str, Any]] = None

            if clean_doi:
                encoded_doi = urllib.parse.quote(f"https://doi.org/{clean_doi}")
                api_url = f"https://api.openalex.org/works/{encoded_doi}?mailto={urllib.parse.quote(self.email)}"
                resp = self._session.get(api_url, timeout=self.timeout)
                if resp.status_code == 200:
                    json_val = resp.json()
                    if isinstance(json_val, dict):
                        work_data = json_val

            if not work_data and title:
                encoded_title = urllib.parse.quote(title)
                api_url = f"https://api.openalex.org/works?filter=title.search:{encoded_title}&per-page=1&mailto={urllib.parse.quote(self.email)}"
                resp = self._session.get(api_url, timeout=self.timeout)
                if resp.status_code == 200:
                    json_val = resp.json()
                    if isinstance(json_val, dict):
                        results = json_val.get("results", [])
                        if results and isinstance(results[0], dict):
                            work_data = results[0]

            if not work_data:
                return None, None, None, None

            # Extract structured metadata for potential Tier 3 fallback
            abstract_text = reconstruct_openalex_abstract(work_data.get("abstract_inverted_index"))
            mesh_terms = extract_openalex_mesh_terms(work_data.get("mesh"))
            concepts = extract_openalex_concepts(work_data.get("concepts"))
            authors = [
                a.get("author", {}).get("display_name")
                for a in work_data.get("authorships", [])
                if isinstance(a, dict) and isinstance(a.get("author"), dict) and isinstance(a["author"].get("display_name"), str)
            ]
            venue = None
            prim_loc = work_data.get("primary_location")
            if isinstance(prim_loc, dict) and isinstance(prim_loc.get("source"), dict):
                venue = prim_loc["source"].get("display_name")
            year = work_data.get("publication_year") if isinstance(work_data.get("publication_year"), int) else None

            extracted_meta = {
                "doi": clean_doi or (work_data.get("doi") if isinstance(work_data.get("doi"), str) else None),
                "title": work_data.get("title") if isinstance(work_data.get("title"), str) else (title or ""),
                "authors": authors,
                "year": year,
                "venue": venue,
                "abstract": abstract_text,
                "mesh_terms": mesh_terms,
                "concepts": concepts,
                "source_url": work_data.get("doi") or work_data.get("id"),
            }

            # Check for direct OA PDF URLs
            best_oa = work_data.get("best_oa_location")
            pdf_url = best_oa.get("pdf_url") if isinstance(best_oa, dict) else None
            landing_url = best_oa.get("landing_page_url") if isinstance(best_oa, dict) else None

            if not pdf_url and isinstance(prim_loc, dict):
                pdf_url = prim_loc.get("pdf_url")
                if not landing_url:
                    landing_url = prim_loc.get("landing_page_url")

            if pdf_url and isinstance(pdf_url, str):
                pdf_bytes = self._download_pdf(pdf_url)
                if pdf_bytes:
                    return pdf_bytes, pdf_url, str(landing_url) if landing_url else None, extracted_meta

            # Check other locations
            locs = work_data.get("locations")
            if isinstance(locs, list):
                for loc in locs:
                    if isinstance(loc, dict):
                        loc_pdf = loc.get("pdf_url")
                        if loc_pdf and isinstance(loc_pdf, str) and loc_pdf != pdf_url:
                            pdf_bytes = self._download_pdf(loc_pdf)
                            if pdf_bytes:
                                return pdf_bytes, loc_pdf, str(loc.get("landing_page_url") or landing_url or ""), extracted_meta

            return None, None, str(landing_url) if landing_url else None, extracted_meta

        except Exception as exc:
            logger.debug(f"OpenAlex resolution exception: {exc}")
            return None, None, None, None

    # ------------------------------------------------------------------------
    # Tier 2 Resolvers
    # ------------------------------------------------------------------------

    def _resolve_tier2_arxiv(self, clean_arxiv_id: str) -> tuple[Optional[bytes], Optional[str]]:
        """
        Directly downloads preprint PDF from arXiv.
        Returns: (pdf_bytes, pdf_url)
        """
        try:
            urls = [
                f"https://arxiv.org/pdf/{clean_arxiv_id}.pdf",
                f"https://export.arxiv.org/pdf/{clean_arxiv_id}.pdf",
            ]
            for url in urls:
                pdf_bytes = self._download_pdf(url)
                if pdf_bytes:
                    return pdf_bytes, url
            return None, None
        except Exception as exc:
            logger.debug(f"arXiv download exception for {clean_arxiv_id}: {exc}")
            return None, None

    def _resolve_tier2_semantic_scholar(
        self, clean_doi: Optional[str], clean_arxiv_id: Optional[str], s2_id: Optional[str]
    ) -> tuple[Optional[bytes], Optional[str], Optional[dict[str, Any]]]:
        """
        Queries Semantic Scholar Graph API for open preprint CDN URLs.
        Returns: (pdf_bytes, pdf_url, extracted_metadata)
        """
        try:
            paper_id = None
            if s2_id:
                paper_id = s2_id
            elif clean_doi:
                paper_id = f"DOI:{clean_doi}"
            elif clean_arxiv_id:
                paper_id = f"ARXIV:{clean_arxiv_id}"

            if not paper_id:
                return None, None, None

            api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=isOpenAccess,openAccessPdf,abstract,title,authors,year,venue"
            resp = self._session.get(api_url, timeout=self.timeout)
            if resp.status_code != 200:
                return None, None, None

            data = resp.json()
            if not isinstance(data, dict):
                return None, None, None

            authors_list = []
            if isinstance(data.get("authors"), list):
                for a in data["authors"]:
                    if isinstance(a, dict) and isinstance(a.get("name"), str):
                        authors_list.append(a["name"])

            extracted_meta = {
                "title": data.get("title") if isinstance(data.get("title"), str) else "",
                "abstract": data.get("abstract") if isinstance(data.get("abstract"), str) else "",
                "year": data.get("year") if isinstance(data.get("year"), int) else None,
                "venue": data.get("venue") if isinstance(data.get("venue"), str) else None,
                "authors": authors_list,
            }

            oa_pdf = data.get("openAccessPdf")
            if isinstance(oa_pdf, dict):
                pdf_url = oa_pdf.get("url")
                if pdf_url and isinstance(pdf_url, str):
                    pdf_bytes = self._download_pdf(pdf_url)
                    if pdf_bytes:
                        return pdf_bytes, pdf_url, extracted_meta

            return None, None, extracted_meta

        except Exception as exc:
            logger.debug(f"Semantic Scholar resolution exception: {exc}")
            return None, None, None

    # ------------------------------------------------------------------------
    # Public Resolution Methods
    # ------------------------------------------------------------------------

    def resolve_paper(
        self,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        title: Optional[str] = None,
        s2_id: Optional[str] = None,
        openalex_id: Optional[str] = None,
    ) -> OAResolutionResult:
        """
        Executes the 3-Tier Open-Access Resolution Cascade.

        Parameters:
            doi: Digital Object Identifier (e.g., '10.1038/s41586-020-2649-2')
            arxiv_id: arXiv identifier (e.g., '2401.01234')
            title: Full scientific paper title
            s2_id: Semantic Scholar Corpus / Paper ID
            openalex_id: OpenAlex Work ID (e.g., 'W2741809807')

        Returns:
            OAResolutionResult: Structured resolution outcome.
            NEVER raises an exception.
        """
        try:
            clean_doi = normalize_doi(doi)
            clean_arxiv_id = normalize_arxiv_id(arxiv_id, doi=clean_doi)
            clean_title = title.strip() if title and isinstance(title, str) else None

            # Intermediate metadata captured across tiers
            accumulated_metadata: dict[str, Any] = {
                "doi": clean_doi,
                "arxiv_id": clean_arxiv_id,
                "title": clean_title,
                "s2_id": s2_id,
                "openalex_id": openalex_id,
            }

            # ----------------------------------------------------------------
            # TIER 1: Unpaywall & OpenAlex Direct OA PDFs (When DOI exists)
            # ----------------------------------------------------------------
            if clean_doi:
                pdf_bytes, pdf_url, landing_url = self._resolve_tier1_unpaywall(clean_doi)
                if pdf_bytes:
                    return OAResolutionResult(
                        pdf_bytes=pdf_bytes,
                        source="unpaywall",
                        is_full_text=True,
                        abstract_fallback=None,
                        pdf_url=pdf_url,
                        landing_page_url=landing_url,
                        doi=clean_doi,
                        arxiv_id=clean_arxiv_id,
                        title=clean_title,
                        resolution_tier=1,
                        metadata=accumulated_metadata,
                    )

                # Try OpenAlex by DOI
                pdf_bytes, pdf_url, landing_url, openalex_meta = self._resolve_tier1_openalex(clean_doi, clean_title)
                if openalex_meta:
                    accumulated_metadata.update({k: v for k, v in openalex_meta.items() if v})

                if pdf_bytes:
                    return OAResolutionResult(
                        pdf_bytes=pdf_bytes,
                        source="openalex",
                        is_full_text=True,
                        abstract_fallback=None,
                        pdf_url=pdf_url,
                        landing_page_url=landing_url,
                        doi=clean_doi or str(accumulated_metadata.get("doi") or ""),
                        arxiv_id=clean_arxiv_id,
                        title=str(accumulated_metadata.get("title") or clean_title or ""),
                        resolution_tier=1,
                        metadata=accumulated_metadata,
                    )

            # ----------------------------------------------------------------
            # TIER 2: Direct Preprint Repositories (arXiv & S2 Preprints)
            # ----------------------------------------------------------------
            if clean_arxiv_id:
                pdf_bytes, pdf_url = self._resolve_tier2_arxiv(clean_arxiv_id)
                if pdf_bytes:
                    return OAResolutionResult(
                        pdf_bytes=pdf_bytes,
                        source="arxiv",
                        is_full_text=True,
                        abstract_fallback=None,
                        pdf_url=pdf_url,
                        landing_page_url=f"https://arxiv.org/abs/{clean_arxiv_id}",
                        doi=clean_doi,
                        arxiv_id=clean_arxiv_id,
                        title=str(accumulated_metadata.get("title") or clean_title or ""),
                        resolution_tier=2,
                        metadata=accumulated_metadata,
                    )

            # Semantic Scholar (by DOI, arXiv, or s2_id)
            if clean_doi or clean_arxiv_id or s2_id:
                pdf_bytes, pdf_url, s2_meta = self._resolve_tier2_semantic_scholar(clean_doi, clean_arxiv_id, s2_id)
                if s2_meta:
                    for k, v in s2_meta.items():
                        if v and not accumulated_metadata.get(k):
                            accumulated_metadata[k] = v

                if pdf_bytes:
                    return OAResolutionResult(
                        pdf_bytes=pdf_bytes,
                        source="semantic_scholar",
                        is_full_text=True,
                        abstract_fallback=None,
                        pdf_url=pdf_url,
                        landing_page_url=None,
                        doi=clean_doi,
                        arxiv_id=clean_arxiv_id,
                        title=str(accumulated_metadata.get("title") or clean_title or ""),
                        resolution_tier=2,
                        metadata=accumulated_metadata,
                    )

            # If no DOI was present initially, check OpenAlex by Title
            if not clean_doi and clean_title:
                pdf_bytes, pdf_url, landing_url, openalex_meta = self._resolve_tier1_openalex(None, clean_title)
                if openalex_meta:
                    accumulated_metadata.update({k: v for k, v in openalex_meta.items() if v})

                if pdf_bytes:
                    return OAResolutionResult(
                        pdf_bytes=pdf_bytes,
                        source="openalex",
                        is_full_text=True,
                        abstract_fallback=None,
                        pdf_url=pdf_url,
                        landing_page_url=landing_url,
                        doi=str(accumulated_metadata.get("doi") or ""),
                        arxiv_id=clean_arxiv_id,
                        title=str(accumulated_metadata.get("title") or clean_title or ""),
                        resolution_tier=1,
                        metadata=accumulated_metadata,
                    )

            # ----------------------------------------------------------------
            # TIER 3: Graceful Fallback (Abstract + MeSH Terms)
            # ----------------------------------------------------------------
            fallback_dict = {
                "doi": clean_doi,
                "arxiv_id": clean_arxiv_id,
                "title": str(accumulated_metadata.get("title") or clean_title or "Unresolved Title"),
                "authors": accumulated_metadata.get("authors", []),
                "year": accumulated_metadata.get("year"),
                "venue": accumulated_metadata.get("venue"),
                "abstract": str(accumulated_metadata.get("abstract") or "Extended structured abstract metadata fallback."),
                "mesh_terms": accumulated_metadata.get("mesh_terms", []),
                "concepts": accumulated_metadata.get("concepts", []),
                "source_url": str(accumulated_metadata.get("source_url") or (f"https://doi.org/{clean_doi}" if clean_doi else "")),
            }

            return OAResolutionResult(
                pdf_bytes=None,
                source="abstract_fallback",
                is_full_text=False,
                abstract_fallback=fallback_dict,
                pdf_url=None,
                landing_page_url=fallback_dict["source_url"] or None,
                doi=clean_doi,
                arxiv_id=clean_arxiv_id,
                title=fallback_dict["title"],
                resolution_tier=3,
                error_message=None,
                metadata=accumulated_metadata,
            )

        except Exception as exc:
            logger.error(f"Unexpected error in OAResolver.resolve_paper: {exc}", exc_info=True)
            fallback_dict = {
                "doi": doi,
                "arxiv_id": arxiv_id,
                "title": title or "Unresolved Title",
                "abstract": "Fallback metadata generated following unhandled resolver exception.",
                "mesh_terms": [],
            }
            return OAResolutionResult(
                pdf_bytes=None,
                source="abstract_fallback",
                is_full_text=False,
                abstract_fallback=fallback_dict,
                doi=doi,
                arxiv_id=arxiv_id,
                title=title,
                resolution_tier=3,
                error_message=str(exc),
            )

    async def resolve_paper_async(
        self,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        title: Optional[str] = None,
        s2_id: Optional[str] = None,
        openalex_id: Optional[str] = None,
    ) -> OAResolutionResult:
        """Asynchronous wrapper for resolve_paper executing in thread pool."""
        return await asyncio.to_thread(
            self.resolve_paper,
            doi=doi,
            arxiv_id=arxiv_id,
            title=title,
            s2_id=s2_id,
            openalex_id=openalex_id,
        )

    def resolve_batch(
        self, papers: list[dict[str, Any]], max_concurrency: int = 5
    ) -> list[OAResolutionResult]:
        """Resolves a list of paper dictionaries sequentially."""
        results: list[OAResolutionResult] = []
        for p in papers:
            res = self.resolve_paper(
                doi=p.get("doi"),
                arxiv_id=p.get("arxiv_id"),
                title=p.get("title"),
                s2_id=p.get("s2_id"),
                openalex_id=p.get("openalex_id"),
            )
            results.append(res)
        return results


__all__ = [
    "AbstractFallbackMetadata",
    "OAResolutionResult",
    "OAResolver",
    "normalize_doi",
    "normalize_arxiv_id",
    "is_valid_pdf_bytes",
    "reconstruct_openalex_abstract",
    "extract_openalex_mesh_terms",
    "extract_openalex_concepts",
]
