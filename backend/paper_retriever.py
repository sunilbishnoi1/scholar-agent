import requests
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import re
import logging
import time 

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PaperRetriever:
    """
    A class to retrieve academic papers from arXiv and Semantic Scholar APIs.
    """
    def __init__(self):
        self.arxiv_base_url = "http://export.arxiv.org/api/query?"
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def _clean_text(self, text):
        """Remove newlines and extra whitespace from text."""
        if text:
            return re.sub(r'\s+', ' ', text).strip()
        return ""

    def _search_arxiv(self, query: str, max_results: int = 10):
        """Searches the arXiv API and returns a list of paper dictionaries."""
        papers = []
        # Using quote_plus to handle spaces and special characters appropriately.
        search_query = f'search_query=all:{urllib.parse.quote_plus(query)}&start=0&max_results={max_results}'
        try:
            with urllib.request.urlopen(self.arxiv_base_url + search_query, timeout=15) as response:
                xml_data = response.read().decode('utf-8')
                root = ET.fromstring(xml_data)
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    title = self._clean_text(entry.find('{http://www.w3.org/2005/Atom}title').text)
                    abstract = self._clean_text(entry.find('{http://www.w3.org/2005/Atom}summary').text)
                    url = entry.find('{http://www.w3.org/2005/Atom}id').text
                    authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]

                    papers.append({
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "url": url,
                        "source": "arXiv"
                    })
            logging.info(f"Found {len(papers)} papers on arXiv for query: '{query}'")
        except Exception as e:
            logging.error(f"Error fetching from arXiv: {e}")
        return papers

    def _search_semantic_scholar(self, query: str, max_results: int = 10):
        """Searches the Semantic Scholar API and returns a list of paper dictionaries."""
        papers = []
        params = {
            'query': query,
            'limit': max_results,
            'fields': 'title,authors.name,abstract,url'
        }
        try:
            response = requests.get(self.semantic_scholar_base_url, params=params, timeout=15)
            response.raise_for_status()
            results = response.json()
            
            for item in results.get('data', []):
                # Ensure all required fields are present to avoid errors
                if item.get('title') and item.get('abstract') and item.get('url'):
                    papers.append({
                        "title": self._clean_text(item['title']),
                        "abstract": self._clean_text(item['abstract']),
                        "authors": [author['name'] for author in item.get('authors', []) if author and 'name' in author],
                        "url": item['url'],
                        "source": "Semantic Scholar"
                    })
            logging.info(f"Found {len(papers)} papers on Semantic Scholar for query: '{query}'")
        except Exception as e:
            logging.error(f"Error fetching from Semantic Scholar: {e}")
        return papers

    def search_papers(self, planner_response: str, max_papers: int, fallback_keywords: list[str] = None):
        """
        Orchestrates searching across multiple academic APIs based on a planner's strategy.
        It parses search terms from the planner, queries the sources, and de-duplicates the results.
        If parsing fails, it uses the provided fallback keywords.
        """
        search_terms = []
        try:
            strategy_section = re.search(r'1\.\s+Search Strategy:([\s\S]*?)(?=\n\s*\d\.\s|\Z)', planner_response, re.IGNORECASE)
            
            if strategy_section:
                content = strategy_section.group(1)
                search_terms = re.findall(r'^\s*[\*\-\d]+\.?\s*(.*)', content, re.MULTILINE)
                search_terms = [term.strip() for term in search_terms if term.strip()]
        except Exception as e:
            logging.error(f"Failed to parse planner response for search terms: {e}")
        
        if not search_terms:
            logging.warning("Could not parse specific search terms from planner. Using fallback keywords.")
            if fallback_keywords:
                search_terms = fallback_keywords
            else:
                logging.error("Parsing failed and no fallback keywords were provided.")
                return []
            
        logging.info(f"Extracted search terms: {search_terms}")
        
        if len(search_terms) == 1 and len(search_terms[0]) > 100:
            logging.warning("Single long search term detected, might be an unformatted list. Using keywords as a safer alternative.")
            search_terms = fallback_keywords if fallback_keywords else []

        if not search_terms:
             logging.error("No valid search terms found after parsing and fallback.")
             return []

        papers_per_query = max(1, max_papers // (len(search_terms) * 2)) if search_terms else 0
        
        all_papers = []
        for term in search_terms:
            term = term.strip().replace('"', '')
            if not term: continue
            
            all_papers.extend(self._search_arxiv(term, papers_per_query))
            all_papers.extend(self._search_semantic_scholar(term, papers_per_query))
            
            time.sleep(1) # Wait for 1 second before the next set of API calls

        unique_papers = []
        seen_titles = set()
        for paper in all_papers:
            normalized_title = paper['title'].lower().strip()
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_papers.append(paper)
        
        logging.info(f"Total papers found: {len(all_papers)}. Unique papers after de-duplication: {len(unique_papers)}")

        return unique_papers[:max_papers]