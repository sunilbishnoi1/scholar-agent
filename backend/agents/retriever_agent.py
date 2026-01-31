# Paper Retriever Agent (LangGraph Compatible)
# Role: Retrieves papers from academic databases based on search strategy

import logging
import time
from typing import Optional

from agents.base import ToolEnabledAgent
from agents.state import AgentState, AgentResult, AgentType, PaperData
from paper_retriever import PaperRetriever

# RAG integration (optional - gracefully handles if not available)
try:
    from rag import get_rag_service, RAGService
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class PaperRetrieverAgent(ToolEnabledAgent):
    """
    Retriever Agent responsible for:
    - Fetching papers from arXiv and Semantic Scholar
    - Deduplicating results
    - Managing API rate limits
    - Ingesting papers into the RAG vector store
    
    This agent wraps the existing PaperRetriever class and integrates it
    into the LangGraph pipeline.
    """
    
    def __init__(self, llm_client=None, enable_rag: bool = True):
        # LLM is optional for retriever - mainly uses APIs
        super().__init__(llm_client, name="retriever")
        self.paper_retriever = PaperRetriever()
        self.enable_rag = enable_rag and RAG_AVAILABLE
        self._rag_service: Optional[RAGService] = None
        self._register_tools()
    
    @property
    def rag_service(self) -> Optional["RAGService"]:
        """Lazy-load RAG service."""
        if self.enable_rag and self._rag_service is None:
            try:
                self._rag_service = get_rag_service()
            except Exception as e:
                self.logger.warning(f"Failed to initialize RAG service: {e}")
                self.enable_rag = False
        return self._rag_service
    
    def _register_tools(self):
        """Register the tools this agent can use."""
        self.register_tool(
            "search_arxiv",
            self._search_arxiv,
            "Search arXiv for academic papers"
        )
        self.register_tool(
            "search_semantic_scholar",
            self._search_semantic_scholar,
            "Search Semantic Scholar for academic papers"
        )
        self.register_tool(
            "search_all_sources",
            self._search_all_sources,
            "Search all academic sources"
        )
        self.register_tool(
            "ingest_to_rag",
            self._ingest_to_rag,
            "Ingest papers into the RAG vector store"
        )
    
    async def run(self, state: AgentState) -> AgentState:
        """
        Execute the retriever agent.
        
        Workflow:
        1. Get search strategy from state
        2. Search papers from multiple sources
        3. Deduplicate results
        4. Ingest into RAG vector store (if enabled)
        5. Store in state
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated state with retrieved papers
        """
        self._log_start(state)
        
        try:
            state["current_agent"] = AgentType.RETRIEVER
            
            keywords = state.get("keywords", [])
            max_papers = state.get("max_papers", 50)
            project_id = state.get("project_id", "default")
            
            if not keywords:
                self.logger.error("No keywords provided for search")
                state["errors"].append("No keywords available for paper search")
                return state
            
            # Search for papers
            papers = await self.invoke_tool(
                "search_all_sources",
                keywords=keywords,
                max_papers=max_papers
            )
            
            # Convert to PaperData format and deduplicate
            unique_papers = self._deduplicate_papers(papers)
            
            # Ingest into RAG system if enabled
            rag_stats = None
            if self.enable_rag and unique_papers:
                rag_stats = await self.invoke_tool(
                    "ingest_to_rag",
                    papers=unique_papers,
                    project_id=project_id
                )
            
            # Update state
            state["papers"] = unique_papers
            state["total_papers_found"] = len(unique_papers)
            
            message = f"Retrieved {len(unique_papers)} unique papers from {len(papers)} total results"
            if rag_stats:
                message += f" (indexed {rag_stats.get('chunks_ingested', 0)} chunks for semantic search)"
            
            state["messages"] = [self._create_message("search_papers", message)]
            
            # Log result
            result = AgentResult(
                success=True,
                data={
                    "paper_count": len(unique_papers),
                    "rag_stats": rag_stats
                },
                metadata={
                    "sources": ["arXiv", "Semantic Scholar"],
                    "keywords_used": len(keywords),
                    "rag_enabled": self.enable_rag
                }
            )
            self._log_complete(state, result)
            
            return state
            
        except Exception as e:
            return self._handle_error(state, e)
    
    def _search_arxiv(self, query: str, max_results: int = 10) -> list[dict]:
        """Search arXiv API."""
        return self.paper_retriever._search_arxiv(query, max_results)
    
    def _search_semantic_scholar(self, query: str, max_results: int = 10) -> list[dict]:
        """Search Semantic Scholar API."""
        return self.paper_retriever._search_semantic_scholar(query, max_results)
    
    def _search_all_sources(self, keywords: list[str], max_papers: int) -> list[dict]:
        """
        Search all academic sources with the given keywords.
        
        This is a synchronous method that handles rate limiting internally.
        """
        return self.paper_retriever.search_papers(keywords, max_papers)
    
    def _ingest_to_rag(self, papers: list[dict], project_id: str) -> dict:
        """
        Ingest papers into the RAG vector store.
        
        Args:
            papers: List of paper dictionaries
            project_id: Project ID for isolation
            
        Returns:
            Dictionary with ingestion statistics
        """
        if not self.rag_service:
            self.logger.warning("RAG service not available, skipping ingestion")
            return {"chunks_ingested": 0, "error": "RAG service not available"}
        
        try:
            # Convert PaperData format to RAG format
            rag_papers = [
                {
                    "id": paper.get("id", paper.get("url", f"paper_{i}")),
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "authors": paper.get("authors", []),
                    "url": paper.get("url", ""),
                }
                for i, paper in enumerate(papers)
            ]
            
            return self.rag_service.ingest_papers(rag_papers, project_id)
            
        except Exception as e:
            self.logger.error(f"Failed to ingest papers to RAG: {e}")
            return {"chunks_ingested": 0, "error": str(e)}
    
    def _deduplicate_papers(self, papers: list[dict]) -> list[PaperData]:
        """
        Remove duplicate papers based on title similarity.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            List of unique PaperData objects
        """
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            # Normalize title for comparison
            normalized_title = paper.get("title", "").lower().strip()
            
            if normalized_title and normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                
                # Convert to PaperData format
                paper_data: PaperData = {
                    "id": f"paper_{len(unique_papers)}",
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "authors": paper.get("authors", []),
                    "url": paper.get("url", ""),
                    "source": paper.get("source", "unknown"),
                    "relevance_score": None,
                    "analysis": None
                }
                unique_papers.append(paper_data)
        
        self.logger.info(f"Deduplicated {len(papers)} papers to {len(unique_papers)} unique papers")
        return unique_papers
