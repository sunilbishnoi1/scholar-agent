import logging
import os
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import requests

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# User-Agent header to identify our application (required by arXiv and helps avoid blocks)
USER_AGENT = "ScholarAgent/1.0 (https://github.com/scholar-agent; contact@scholar-agent.com)"


class PaperRetriever:
    """
    A class to retrieve academic papers from arXiv and Semantic Scholar APIs.

    Implements proper rate limiting and retry logic:
    - arXiv: 3+ seconds between requests (per their API guidelines)
    - Semantic Scholar: Exponential backoff on 429 errors
    """

    # Rate limiting constants
    ARXIV_DELAY_SECONDS = 3.5  # arXiv recommends 3 seconds minimum
    SEMANTIC_SCHOLAR_DELAY_SECONDS = 1.0  # Base delay for Semantic Scholar
    MAX_RETRIES = 3
    ARXIV_TIMEOUT = 30  # Increased timeout for arXiv (can be slow)
    SEMANTIC_SCHOLAR_TIMEOUT = 15

    # Adaptive rate limiting thresholds
    MAX_CONSECUTIVE_FAILURES = 5  # Disable API after this many consecutive failures
    MAX_BACKOFF_SECONDS = 30  # Don't wait longer than this between calls

    def __init__(self):
        self.arxiv_base_url = "http://export.arxiv.org/api/query?"
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        # Optional: Semantic Scholar API key for higher rate limits
        self.semantic_scholar_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    def _clean_text(self, text):
        """Remove newlines and extra whitespace from text."""
        if text:
            return re.sub(r"\s+", " ", text).strip()
        return ""

    def _search_arxiv(self, query: str, max_results: int = 10):
        """
        Searches the arXiv API and returns a list of paper dictionaries.

        Implements proper User-Agent header and retry logic for reliability.
        arXiv API guidelines: https://info.arxiv.org/help/api/user-manual.html
        """
        papers = []
        search_query = (
            f"search_query=all:{urllib.parse.quote_plus(query)}&start=0&max_results={max_results}"
        )
        url = self.arxiv_base_url + search_query

        for attempt in range(self.MAX_RETRIES):
            try:
                # Create request with proper User-Agent header (critical for arXiv)
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

                with urllib.request.urlopen(request, timeout=self.ARXIV_TIMEOUT) as response:
                    xml_data = response.read().decode("utf-8")
                    root = ET.fromstring(xml_data)

                    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                        title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                        abstract_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                        url_elem = entry.find("{http://www.w3.org/2005/Atom}id")

                        # Skip if essential elements are missing
                        if title_elem is None or abstract_elem is None or url_elem is None:
                            continue

                        title = self._clean_text(title_elem.text)
                        abstract = self._clean_text(abstract_elem.text)
                        paper_url = url_elem.text
                        authors = [
                            author.find("{http://www.w3.org/2005/Atom}name").text
                            for author in entry.findall("{http://www.w3.org/2005/Atom}author")
                            if author.find("{http://www.w3.org/2005/Atom}name") is not None
                        ]

                        papers.append(
                            {
                                "title": title,
                                "abstract": abstract,
                                "authors": authors,
                                "url": paper_url,
                                "source": "arXiv",
                            }
                        )

                logging.info(f"Found {len(papers)} papers on arXiv for query: '{query}'")
                return papers

            except urllib.error.HTTPError as e:
                if e.code == 503:
                    # Service unavailable - arXiv is overloaded, back off
                    wait_time = (attempt + 1) * 5
                    logging.warning(
                        f"arXiv service unavailable (503), waiting {wait_time}s before retry {attempt + 1}/{self.MAX_RETRIES}"
                    )
                    time.sleep(wait_time)
                else:
                    logging.error(f"HTTP error fetching from arXiv: {e.code} - {e.reason}")
                    break
            except urllib.error.URLError as e:
                # Network/timeout issues - retry with backoff
                wait_time = (attempt + 1) * 3
                logging.warning(
                    f"Network error fetching from arXiv: {e.reason}. Retry {attempt + 1}/{self.MAX_RETRIES} after {wait_time}s"
                )
                time.sleep(wait_time)
            except Exception as e:
                logging.error(f"Error fetching from arXiv: {e}")
                break

        return papers

    def _search_semantic_scholar(
        self, query: str, max_results: int = 10
    ) -> tuple[list[dict], bool]:
        """
        Searches the Semantic Scholar API and returns a list of paper dictionaries.

        Implements exponential backoff for rate limiting (429 errors).
        API docs: https://api.semanticscholar.org/api-docs/

        Returns:
            tuple: (papers_list, was_rate_limited)
                - papers_list: List of paper dictionaries found
                - was_rate_limited: True if request failed due to 429 rate limit
        """
        papers = []
        was_rate_limited = False
        params = {"query": query, "limit": max_results, "fields": "title,authors.name,abstract,url"}

        # Set up headers with User-Agent and optional API key
        headers = {"User-Agent": USER_AGENT}
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(
                    self.semantic_scholar_base_url,
                    params=params,
                    headers=headers,
                    timeout=self.SEMANTIC_SCHOLAR_TIMEOUT,
                )

                # Handle rate limiting with exponential backoff
                if response.status_code == 429:
                    was_rate_limited = True
                    # Exponential backoff: 2^attempt * base + random jitter
                    wait_time = (2**attempt) * 2 + random.uniform(0, 1)
                    logging.warning(
                        f"Semantic Scholar rate limited (429). "
                        f"Waiting {wait_time:.1f}s before retry {attempt + 1}/{self.MAX_RETRIES}"
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                results = response.json()

                for item in results.get("data", []):
                    if item.get("title") and item.get("abstract") and item.get("url"):
                        papers.append(
                            {
                                "title": self._clean_text(item["title"]),
                                "abstract": self._clean_text(item["abstract"]),
                                "authors": [
                                    author["name"]
                                    for author in item.get("authors", [])
                                    if author and "name" in author
                                ],
                                "url": item["url"],
                                "source": "Semantic Scholar",
                            }
                        )

                logging.info(f"Found {len(papers)} papers on Semantic Scholar for query: '{query}'")
                return papers, False  # Success, not rate limited

            except requests.exceptions.Timeout:
                logging.warning(
                    f"Semantic Scholar timeout for query '{query}'. Retry {attempt + 1}/{self.MAX_RETRIES}"
                )
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 2
                    logging.warning(
                        f"Semantic Scholar request error: {e}. Retry {attempt + 1}/{self.MAX_RETRIES} after {wait_time}s"
                    )
                    time.sleep(wait_time)
                else:
                    logging.error(
                        f"Error fetching from Semantic Scholar after {self.MAX_RETRIES} attempts: {e}"
                    )

        return papers, was_rate_limited

    def search_papers(self, search_terms: list[str], max_papers: int):
        """
        Orchestrates searching across multiple academic APIs using a list of search terms.

        Implements adaptive rate limiting with smart API switching:
        - Tracks consecutive failures per API
        - Skips APIs that are consistently failing (don't waste time)
        - Adjusts delays dynamically based on rate limit feedback
        - Stops early if both APIs are unavailable (proceed with papers found so far)
        - arXiv is prioritized as it rarely rate-limits
        """
        if not search_terms:
            logging.error("No search terms provided to search_papers method.")
            return []

        logging.info(f"Received search terms: {search_terms}")

        # Calculate how many papers to fetch per search term from each source
        papers_per_query = max(1, max_papers // (len(search_terms) * 2)) if search_terms else 0

        all_papers = []

        # Track API health across calls (not just within a single call)
        arxiv_consecutive_failures = 0
        s2_consecutive_failures = 0
        s2_current_backoff = self.SEMANTIC_SCHOLAR_DELAY_SECONDS

        arxiv_enabled = True
        s2_enabled = True

        for i, term in enumerate(search_terms):
            term = term.strip().replace('"', "")
            if not term:
                continue

            # --- arXiv (usually reliable, rarely rate-limits) ---
            if arxiv_enabled:
                arxiv_papers = self._search_arxiv(term, papers_per_query)
                if arxiv_papers:
                    all_papers.extend(arxiv_papers)
                    arxiv_consecutive_failures = 0  # Reset on success
                else:
                    arxiv_consecutive_failures += 1
                    if arxiv_consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                        logging.warning(
                            f"arXiv disabled after {self.MAX_CONSECUTIVE_FAILURES} consecutive failures. "
                            f"Continuing with Semantic Scholar only."
                        )
                        arxiv_enabled = False

                # Standard delay after arXiv call (only if we'll make more calls)
                if s2_enabled or (i < len(search_terms) - 1 and arxiv_enabled):
                    time.sleep(self.ARXIV_DELAY_SECONDS)

            # --- Semantic Scholar (often rate-limited without API key) ---
            if s2_enabled:
                s2_papers, was_rate_limited = self._search_semantic_scholar(term, papers_per_query)

                if s2_papers:
                    all_papers.extend(s2_papers)
                    s2_consecutive_failures = 0  # Reset on success
                    s2_current_backoff = self.SEMANTIC_SCHOLAR_DELAY_SECONDS  # Reset backoff
                elif was_rate_limited:
                    s2_consecutive_failures += 1
                    # Increase backoff on rate limit, capped at MAX_BACKOFF_SECONDS
                    s2_current_backoff = min(self.MAX_BACKOFF_SECONDS, s2_current_backoff * 2)
                    logging.info(f"Semantic Scholar backoff increased to {s2_current_backoff:.1f}s")

                    if s2_consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                        logging.warning(
                            "Semantic Scholar disabled due to persistent rate limiting. "
                            "Continuing with arXiv only."
                        )
                        s2_enabled = False
                else:
                    # Other error (not rate limit)
                    s2_consecutive_failures += 1
                    if s2_consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                        s2_enabled = False

                # Wait before next iteration (only if more terms and S2 still enabled)
                if i < len(search_terms) - 1 and s2_enabled:
                    time.sleep(s2_current_backoff)

            # --- Early exit if both APIs are down ---
            if not arxiv_enabled and not s2_enabled:
                logging.warning(
                    f"Both APIs unavailable after processing {i + 1}/{len(search_terms)} terms. "
                    f"Proceeding with {len(all_papers)} papers found so far."
                )
                break

            # --- Log progress ---
            if (i + 1) % 3 == 0:  # Log every 3 terms
                logging.info(
                    f"Progress: {i + 1}/{len(search_terms)} terms processed, "
                    f"{len(all_papers)} papers found. "
                    f"APIs: arXiv={'OK' if arxiv_enabled else 'DISABLED'}, "
                    f"S2={'OK' if s2_enabled else 'DISABLED'}"
                )

        # Deduplicate papers
        unique_papers = []
        seen_titles = set()
        for paper in all_papers:
            normalized_title = paper["title"].lower().strip()
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_papers.append(paper)

        logging.info(
            f"Paper retrieval complete. "
            f"Total: {len(all_papers)}, Unique: {len(unique_papers)}, "
            f"Returning: {min(len(unique_papers), max_papers)}"
        )

        return unique_papers[:max_papers]
