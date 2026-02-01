#!/usr/bin/env python
"""
End-to-end test for RAG pipeline.
Run with: python test_rag_e2e.py
"""

import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set/override specific test variables
os.environ["QDRANT_URL"] = "http://localhost:6333"

# Verify API key is set
if not os.environ.get("GEMINI_API_KEY"):
    print("❌ Error: GEMINI_API_KEY not found in environment variables")
    print("   Please set it in your .env file")
    sys.exit(1)

from rag import get_rag_service


def test_rag_pipeline():
    """Test the complete RAG pipeline."""
    print("🚀 Testing RAG Pipeline End-to-End\n")

    # Sample papers
    papers = [
        {
            "id": "paper1",
            "title": "Machine Learning for Educational Assessment",
            "abstract": "This paper explores how machine learning algorithms can be applied to automate student assessment and provide personalized feedback.",
            "authors": ["Alice Johnson", "Bob Smith"],
        },
        {
            "id": "paper2",
            "title": "Deep Learning in Healthcare Diagnosis",
            "abstract": "We present a deep learning model for medical image analysis that achieves state-of-the-art results in disease detection.",
            "authors": ["Carol Chen"],
        },
        {
            "id": "paper3",
            "title": "Natural Language Processing for Sentiment Analysis",
            "abstract": "Our research focuses on transformer-based models for sentiment classification in social media posts.",
            "authors": ["David Lee", "Emma Wilson"],
        },
    ]

    project_id = "test_project_e2e"

    try:
        # Initialize RAG service
        print("1️⃣  Initializing RAG service...")
        rag = get_rag_service()
        print("   ✓ RAG service initialized\n")

        # Ingest papers
        print("2️⃣  Ingesting papers...")
        stats = rag.ingest_papers(papers, project_id=project_id, rebuild_bm25=True)
        print(f"   ✓ Ingested {stats['chunks_ingested']} chunks from {stats['papers_processed']} papers")
        print(f"   ✓ Average: {stats['avg_chunks_per_paper']:.1f} chunks/paper\n")

        # Vector-only search
        print("3️⃣  Testing vector-only search...")
        results = rag.search(
            query="machine learning assessment education",
            project_id=project_id,
            top_k=3,
            use_hybrid=False,
            use_reranker=False,
        )
        print(f"   ✓ Found {len(results)} results (vector-only)")
        if results:
            print(f"   Top result: {results[0]['paper_title'][:50]}... (score: {results[0].get('score', 0):.3f})\n")

        # Hybrid search
        print("4️⃣  Testing hybrid search (vector + BM25)...")
        results = rag.search(
            query="how can AI help with medical diagnosis",
            project_id=project_id,
            top_k=5,
            use_hybrid=True,
            use_reranker=True,
        )
        print(f"   ✓ Found {len(results)} results (hybrid)")
        for i, r in enumerate(results[:3], 1):
            print(f"   [{i}] {r['paper_title'][:40]}... | Score: {r.get('final_score', r.get('score', 0)):.3f}")
        print()

        # Get project stats
        print("5️⃣  Checking project statistics...")
        stats = rag.get_project_stats(project_id)
        print(f"   ✓ Total chunks in project: {stats.get('total_chunks', 0)}\n")

        # Embedding stats
        print("6️⃣  Checking embedding cache stats...")
        embed_stats = rag.get_embedding_stats()
        print(f"   ✓ Total requests: {embed_stats.get('total_requests', 0)}")
        print(f"   ✓ Cache hits: {embed_stats.get('cache_hits', 0)}")
        print(f"   ✓ Cache hit rate: {embed_stats.get('cache_hit_rate', 0):.1%}\n")

        # Cleanup
        print("7️⃣  Cleaning up test data...")
        rag.delete_project_data(project_id)
        print("   ✓ Test data deleted\n")

        print("✅ All RAG pipeline tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_rag_pipeline()
    sys.exit(0 if success else 1)
