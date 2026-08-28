# Pytest Configuration and Shared Fixtures
# Comprehensive test fixtures for Scholar Agent test suite

import asyncio
from collections.abc import Generator
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any, TypeVar, get_args, get_origin
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

# Set test database URL BEFORE importing any app modules
# This ensures SQLite is used instead of PostgreSQL during tests
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

# Add the backend directory and project root to the Python path for test imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(backend_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

from models.database import Base

try:
    from agents.llm.base import BaseLLMClient, LLMConfig, LLMResponse
except ImportError:
    # Fallback if base LLM client class is being refactored
    class LLMConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class LLMResponse:
        def __init__(self, text: str = "", model: str = "mock-model", provider: str = "mock", **kwargs):
            self.text = text
            self.model = model
            self.provider = provider
            for k, v in kwargs.items():
                setattr(self, k, v)

    class BaseLLMClient:
        def __init__(self, config=None):
            self.config = config or LLMConfig()


# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]

T = TypeVar("T", bound=BaseModel)


# ============================================
# Session-scoped fixtures
# ============================================


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================
# Database fixtures (Tier 1 & Tier 2)
# ============================================


@pytest.fixture
def db_session():
    """
    In-memory SQLite database session fixture for isolated unit and integration tests.
    Creates all tables declared on Base.metadata and disposes cleanly.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def in_memory_db(db_session):
    """Backwards-compatible alias for db_session."""
    return db_session


@pytest.fixture
def mock_db_session():
    """Create a mock database session for fast unit tests without DB engine."""
    mock = MagicMock()
    mock.query.return_value.filter.return_value.first.return_value = None
    mock.query.return_value.filter.return_value.all.return_value = []
    mock.add = MagicMock()
    mock.commit = MagicMock()
    mock.refresh = MagicMock()
    mock.rollback = MagicMock()
    return mock


# ============================================
# Deterministic MockLLMClient
# ============================================


class DeterministicMockLLMClient(BaseLLMClient):
    """
    Deterministic Mock LLM client supporting both legacy chat methods
    and v3.2 high-capacity structured output generation for any Pydantic schema.
    """

    def __init__(self, config: LLMConfig | None = None):
        super().__init__(config)
        self.calls: list[dict[str, Any]] = []
        self._structured_responses: dict[Any, Any] = {}
        self._text_responses: list[str] = []
        self._error_to_raise: Exception | None = None

    def _setup_client(self) -> None:
        pass

    def set_error(self, error: Exception | None) -> None:
        """Inject a transient or non-retryable error for resilience testing."""
        self._error_to_raise = error

    def set_text_response(self, response: str) -> None:
        """Set next text response."""
        self._text_responses.append(response)

    def set_structured_response(self, schema_cls: type[BaseModel], response: BaseModel | dict[str, Any]) -> None:
        """Register a canned response for a specific Pydantic schema."""
        self._structured_responses[schema_cls] = response

    def _check_error(self) -> None:
        if self._error_to_raise is not None:
            err = self._error_to_raise
            self._error_to_raise = None
            raise err

    def _generate_synthetic_instance(self, schema: type[T]) -> T:
        """
        Dynamically construct a deterministic, valid synthetic instance
        for any Pydantic v2 BaseModel schema.
        """
        if schema in self._structured_responses:
            val = self._structured_responses[schema]
            if isinstance(val, schema):
                return val
            if isinstance(val, dict):
                return schema.model_validate(val)

        fields_data: dict[str, Any] = {}
        # Support Pydantic v2 model_fields
        fields = getattr(schema, "model_fields", getattr(schema, "__fields__", {}))

        for field_name, field_info in fields.items():
            annotation = getattr(field_info, "annotation", getattr(field_info, "type_", None))
            origin = get_origin(annotation)
            args = get_args(annotation)

            # Check if field has a default value or default_factory
            default_val = getattr(field_info, "default", PydanticUndefined)
            if default_val is not PydanticUndefined and default_val is not ... and default_val is not None:
                fields_data[field_name] = default_val
                continue

            fields_data[field_name] = self._generate_field_value(field_name, annotation, origin, args)

        return schema.model_validate(fields_data)

    def _generate_field_value(self, name: str, annotation: Any, origin: Any, args: tuple[Any, ...]) -> Any:
        """Generate deterministic value based on field name and type annotation."""
        # Handle Union / Optional (e.g. int | None, str | None)
        if origin is not None and (origin is type(int | str) or str(origin) == "typing.Union"):
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                target_type = non_none[0]
                return self._generate_field_value(name, target_type, get_origin(target_type), get_args(target_type))

        # Handle Literal types (e.g. Literal["high", "medium", "low"])
        if origin is not None and "Literal" in str(origin):
            return args[0] if args else "high"

        # Handle Lists
        if origin is list or annotation is list:
            item_type = args[0] if args else str
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                return [self._generate_synthetic_instance(item_type)]
            if name.endswith("_ids") or name == "paper_ids":
                return ["paper_001", "paper_002"]
            if name == "authors":
                return ["A. Turing", "C. Shannon"]
            if name == "key_takeaways":
                return ["First key takeaway finding.", "Second key takeaway finding."]
            return [f"sample_{name}_item_1", f"sample_{name}_item_2"]

        # Handle Dicts
        if origin is dict or annotation is dict:
            return {"quantitative": 10, "qualitative": 5, "theoretical": 2}

        # Handle Enums / StrEnums
        if isinstance(annotation, type) and hasattr(annotation, "__members__"):
            members = list(annotation.__members__.values())
            return members[0] if members else "complete"

        # Handle Nested BaseModel
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return self._generate_synthetic_instance(annotation)

        # Handle datetime
        if annotation is datetime:
            return datetime(2026, 8, 25, 12, 0, 0)

        # Handle Primitives by name and type
        if annotation is bool or annotation == "bool":
            return True

        if annotation is int or annotation == "int":
            if "year" in name:
                return 2024
            if "score" in name:
                return 88
            if "count" in name:
                return 15
            return 10

        if annotation is float or annotation == "float":
            if "score" in name:
                return 88.5
            if "duration" in name:
                return 12.4
            if "value" in name or "metric" in name:
                return 94.5
            return 0.95

        if annotation is str or annotation == "str" or annotation is None:
            if "doi" in name:
                return "10.1000/182"
            if "id" in name or name.endswith("_id"):
                return f"mock_{name}_001"
            if "title" in name:
                return "Deep Autonomous Scientific Discovery with Large Language Models"
            if "url" in name:
                return "https://arxiv.org/abs/2401.01234"
            if "methodology" in name:
                return "Section-aware multi-agent synthesis with dense RRF vector retrieval"
            if "metric" in name:
                return "94.6% Accuracy on PubMed-QA"
            if "dataset" in name or "benchmark" in name:
                return "PubMed-QA & SciFact"
            if "limitation" in name:
                return "Evaluated on English scientific literature"
            if "prose" in name or "synthesis" in name:
                return "Autonomous agents formulate hypotheses and verify claims [ref_1#sec2] against full text."
            if "summary" in name or "description" in name or "abstract" in name:
                return f"Detailed synthetic description for {name}."
            if "status" in name:
                return "complete"
            return f"synthetic_{name}_value"

        return f"mock_{name}"

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        model_tier: str = "fast",
        **kwargs: Any,
    ) -> str:
        """Generate deterministic text output."""
        self._check_error()
        self.calls.append({
            "method": "generate_text",
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model_tier": model_tier,
            "kwargs": kwargs,
        })
        if self._text_responses:
            return self._text_responses.pop(0)
        return f"# Synthetic Scientific Response\n\nAnalyzed prompt: {prompt[:100]}...\nConclusion: Hypothesis validated."

    async def generate_text_async(
        self,
        prompt: str,
        system_prompt: str = "",
        model_tier: str = "fast",
        **kwargs: Any,
    ) -> str:
        """Async variant of generate_text."""
        return self.generate_text(prompt, system_prompt=system_prompt, model_tier=model_tier, **kwargs)

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str = "",
        model_tier: str = "fast",
        **kwargs: Any,
    ) -> T:
        """Generate deterministic validated Pydantic model instance."""
        self._check_error()
        self.calls.append({
            "method": "generate_structured",
            "prompt": prompt,
            "schema": schema,
            "system_prompt": system_prompt,
            "model_tier": model_tier,
            "kwargs": kwargs,
        })
        return self._generate_synthetic_instance(schema)

    async def generate_structured_async(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str = "",
        model_tier: str = "fast",
        **kwargs: Any,
    ) -> T:
        """Async variant of generate_structured."""
        return self.generate_structured(prompt, schema=schema, system_prompt=system_prompt, model_tier=model_tier, **kwargs)

    def chat(
        self,
        prompt: str,
        task_type: str = "general",
        complexity_hint: str | None = None,
        max_latency_ms: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Legacy chat method compatibility."""
        self._check_error()
        self.calls.append({
            "method": "chat",
            "prompt": prompt,
            "task_type": task_type,
            "complexity_hint": complexity_hint,
            "kwargs": kwargs,
        })
        if self._text_responses:
            return self._text_responses.pop(0)
        return '{"keywords": ["quantum", "transformers"], "subtopics": ["Architecture", "Empirical Evaluation"]}'

    def chat_with_response(
        self,
        prompt: str,
        task_type: str = "general",
        complexity_hint: str | None = None,
        max_latency_ms: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Legacy chat_with_response compatibility."""
        text = self.chat(prompt, task_type=task_type, complexity_hint=complexity_hint, max_latency_ms=max_latency_ms, **kwargs)
        return LLMResponse(
            text=text,
            model="mock-gemini-2.0-flash",
            provider="mock",
            input_tokens=120,
            output_tokens=80,
            total_tokens=200,
            estimated_cost=0.00005,
            latency_ms=45,
        )

    def get_provider_name(self) -> str:
        return "mock"

    def get_usage_stats(self) -> dict[str, Any]:
        return {"total_calls": len(self.calls), "spent_usd": 0.0, "remaining_budget": 100.0}

    def reset_budget(self, new_budget: float | None = None) -> None:
        pass


@pytest.fixture
def mock_llm_client():
    """Fixture providing a deterministic MockLLMClient instance."""
    return DeterministicMockLLMClient()


@pytest.fixture
def mock_async_llm_client():
    """Async mock LLM client fixture."""
    mock = AsyncMock()
    mock.chat = AsyncMock(return_value='{"keywords": ["test1", "test2"]}')
    mock.generate_text = AsyncMock(return_value="Synthetic LLM response")
    mock.generate_structured = AsyncMock()
    return mock


# ============================================
# Synthetic Scientific PDF Fixture (PyMuPDF / fitz)
# ============================================


@pytest.fixture
def synthetic_scientific_pdf_bytes() -> bytes:
    """
    Generate a deterministic synthetic 2-page scientific PDF containing:
    - Title and Authors metadata
    - Abstract section
    - Markdown headings (# 1. Introduction, # 2. Methodology, # 3. Results, # 4. Limitations)
    - LaTeX mathematical formulas ($E = mc^2$, $$\\mathcal{L}_{total} = \\alpha \\mathcal{L}_{NLI} + \\beta \\mathcal{L}_{RRF}$$)
    - Structured benchmark results table with headers and data rows.
    """
    if fitz is None:
        return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 595 842]>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \n0000000101 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n162\n%%EOF"

    doc = fitz.open()

    # Page 1: Metadata, Abstract, Introduction, Math, Methodology
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((50, 60), "Deep Transformer Reasoning in Multi-Agent Scientific Discovery", fontsize=15)
    page1.insert_text((50, 85), "Authors: A. Turing, J. von Neumann, C. Shannon (doi: 10.1000/scholar.2026.01)", fontsize=9)
    page1.insert_text((50, 115), "# Abstract", fontsize=12)
    page1.insert_text(
        (50, 135),
        "Autonomous multi-agent architectures enhance literature synthesis by executing section-aware retrieval.\n"
        "We demonstrate a 16.4% gain in claim verification accuracy over traditional monolithic LLM prompts.",
        fontsize=10,
    )
    page1.insert_text((50, 180), "# 1. Introduction", fontsize=12)
    page1.insert_text(
        (50, 200),
        "Scientific literature expansion necessitates scalable autonomous reasoning pipelines.\n"
        "Prior monolithic pipelines suffer from context dilution and citation hallucination.",
        fontsize=10,
    )
    page1.insert_text((50, 245), "## 1.1 Mathematical Formulation", fontsize=11)
    page1.insert_text(
        (50, 265),
        "The composite loss function incorporates Natural Language Inference (NLI) and Reciprocal Rank Fusion:\n"
        "Formula: $E = mc^2$\n"
        "$$\\mathcal{L}_{total} = \\alpha \\mathcal{L}_{NLI} + \\beta \\mathcal{L}_{RRF}$$",
        fontsize=10,
    )
    page1.insert_text((50, 320), "# 2. Methodology & Architecture", fontsize=12)
    page1.insert_text(
        (50, 340),
        "We implement a bounded LangGraph DAG with 6 specialist nodes. The section-aware chunker\n"
        "identifies Methodology, Results, and Limitations sections with anchor tags [ref_1#sec2].",
        fontsize=10,
    )

    # Page 2: Results & Benchmarks Table, Limitations, References
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 60), "# 3. Empirical Results & Evaluation", fontsize=12)
    page2.insert_text(
        (50, 80),
        "We evaluated ScholarAgent against state-of-the-art baselines across multiple biomedical and CS benchmarks:",
        fontsize=10,
    )

    # Structured Table
    table_lines = [
        "| Benchmark | Metric | Baseline | ScholarAgent |",
        "| :--- | :--- | :--- | :--- |",
        "| PubMed-QA | Accuracy | 78.2% | 94.6% |",
        "| SciFact | F1-Score | 68.4% | 88.1% |",
        "| BioASQ | MRR | 0.65 | 0.89 |",
    ]
    y = 110
    for line in table_lines:
        page2.insert_text((50, y), line, fontsize=9)
        y += 18

    page2.insert_text((50, y + 20), "# 4. Limitations & Threats to Validity", fontsize=12)
    page2.insert_text(
        (50, y + 40),
        "Primary limitation: The system currently relies on high-quality PDF rendering.\n"
        "Scanned documents with poor OCR resolution require pre-processing.",
        fontsize=10,
    )

    page2.insert_text((50, y + 80), "# 5. References", fontsize=12)
    page2.insert_text((50, y + 100), "[1] Vaswani et al. Attention is All You Need. NeurIPS 2017.", fontsize=9)
    page2.insert_text((50, y + 115), "[2] Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers. NAACL 2019.", fontsize=9)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def synthetic_scientific_pdf_path(tmp_path, synthetic_scientific_pdf_bytes: bytes) -> str:
    """Writes the synthetic scientific PDF to a temporary file path."""
    pdf_file = tmp_path / "sample_scientific_paper.pdf"
    pdf_file.write_bytes(synthetic_scientific_pdf_bytes)
    return str(pdf_file)


# ============================================
# Mock OA Resolver and Search fixtures
# ============================================


@pytest.fixture
def mock_oa_resolver(synthetic_scientific_pdf_bytes: bytes):
    """
    Mock Open-Access Resolver simulating 3-tier cascade:
    - Tier 1 (Unpaywall OA) -> PDF bytes returned
    - Tier 2 (arXiv / Semantic Scholar) -> PDF bytes returned
    - Tier 3 (Paywalled Fallback) -> Structured abstract metadata, is_full_text=False
    - Invariant: Never throws unhandled exception
    """
    class MockOAResolver:
        def __init__(self, sample_bytes: bytes):
            self.sample_bytes = sample_bytes
            self.resolution_log: list[dict[str, Any]] = []

        def resolve_paper(
            self,
            doi: str | None = None,
            arxiv_id: str | None = None,
            title: str | None = None,
        ) -> dict[str, Any]:
            self.resolution_log.append({"doi": doi, "arxiv_id": arxiv_id, "title": title})

            if doi and "paywall" in doi.lower():
                return {
                    "pdf_bytes": None,
                    "source": "abstract_fallback",
                    "is_full_text": False,
                    "abstract_fallback": {
                        "doi": doi,
                        "title": title or "Paywalled Paper Title",
                        "abstract": "This is an extended structured abstract fallback for a paywalled paper.",
                        "mesh_terms": ["Computer Science", "Artificial Intelligence"],
                    },
                }

            if arxiv_id:
                return {
                    "pdf_bytes": self.sample_bytes,
                    "source": "arxiv",
                    "is_full_text": True,
                    "abstract_fallback": None,
                }

            # Default: Tier 1 Unpaywall success
            return {
                "pdf_bytes": self.sample_bytes,
                "source": "unpaywall",
                "is_full_text": True,
                "abstract_fallback": None,
            }

    return MockOAResolver(synthetic_scientific_pdf_bytes)


@pytest.fixture
def mock_academic_search():
    """
    Mock academic search querying OpenAlex, Semantic Scholar, arXiv, and PubMed
    with deduplicated results and citation snowballing.
    """
    class MockAcademicSearch:
        def __init__(self):
            self.query_log: list[str] = []

        def search(self, query: str, limit: int = 25) -> list[dict[str, Any]]:
            self.query_log.append(query)
            return [
                {
                    "paper_id": "paper_001",
                    "doi": "10.1000/scholar.001",
                    "arxiv_id": "2401.00001",
                    "title": f"Advancements in Autonomous Multi-Agent Reasoning for {query}",
                    "authors": ["A. Turing", "C. Shannon"],
                    "year": 2024,
                    "venue": "NeurIPS",
                    "citation_count": 45,
                    "abstract": "We present a comprehensive study on multi-agent scientific reasoning.",
                    "url": "https://arxiv.org/abs/2401.00001",
                    "source": "arXiv",
                },
                {
                    "paper_id": "paper_002",
                    "doi": "10.1000/scholar.002",
                    "arxiv_id": None,
                    "title": f"Benchmarking Section-Aware RAG in Scientific Synthesis for {query}",
                    "authors": ["J. von Neumann"],
                    "year": 2023,
                    "venue": "ICLR",
                    "citation_count": 89,
                    "abstract": "Dense BM25 hybrid search outperforms pure vector search on long documents.",
                    "url": "https://doi.org/10.1000/scholar.002",
                    "source": "OpenAlex",
                },
            ][:limit]

        def traverse_1hop(self, seed_paper_ids: list[str]) -> list[dict[str, Any]]:
            return [
                {
                    "paper_id": f"cited_by_{seed_paper_ids[0]}",
                    "doi": "10.1000/scholar.003",
                    "title": "Snowballed Citation: Foundations of Scientific Verification",
                    "authors": ["K. Gödel"],
                    "year": 2022,
                    "citation_count": 150,
                    "abstract": "Foundations of formal claim verification.",
                    "source": "Semantic Scholar",
                }
            ]

    return MockAcademicSearch()


# ============================================
# Sample Data Fixtures (Backwards-compatible)
# ============================================


@pytest.fixture
def sample_research_question():
    """Provide a sample research question for testing."""
    return "How does artificial intelligence affect student learning outcomes in higher education?"


@pytest.fixture
def sample_title():
    """Provide a sample project title for testing."""
    return "AI in Higher Education: A Literature Review"


@pytest.fixture
def sample_paper_data():
    """Provide sample paper data for testing."""
    return {
        "id": "test_paper_1",
        "title": "Machine Learning for Predicting Student Performance",
        "abstract": "This paper presents a comprehensive study on using machine learning algorithms to predict student academic performance. We analyzed data from 1000 students across multiple institutions and found that ensemble methods outperform traditional statistical approaches.",
        "authors": ["John Smith", "Jane Doe", "Bob Johnson"],
        "url": "https://arxiv.org/abs/2024.12345",
        "source": "arXiv",
        "relevance_score": None,
        "analysis": None,
    }


@pytest.fixture
def sample_papers_list(sample_paper_data):
    """Provide a list of sample papers for testing."""
    return [
        sample_paper_data,
        {
            "id": "test_paper_2",
            "title": "Deep Learning in Educational Assessment",
            "abstract": "We propose a novel deep learning approach for automated essay scoring that achieves state-of-the-art results on multiple benchmarks.",
            "authors": ["Alice Williams"],
            "url": "https://semanticscholar.org/paper/abc123",
            "source": "Semantic Scholar",
            "relevance_score": None,
            "analysis": None,
        },
        {
            "id": "test_paper_3",
            "title": "Adaptive Learning Systems: A Review",
            "abstract": "This systematic review examines the effectiveness of AI-powered adaptive learning systems in improving student engagement and outcomes.",
            "authors": ["Chris Brown", "Diana Lee"],
            "url": "https://arxiv.org/abs/2024.67890",
            "source": "arXiv",
            "relevance_score": None,
            "analysis": None,
        },
    ]


@pytest.fixture
def sample_paper_analysis():
    """Provide a sample paper analysis result."""
    return {
        "relevance_score": 85,
        "justification": "The paper directly addresses machine learning applications in educational settings.",
        "key_findings": [
            "Ensemble methods achieve 15% better accuracy than logistic regression",
            "Feature engineering significantly impacts model performance",
            "Student engagement metrics are strong predictors of success",
        ],
        "methodology": "Quantitative study using gradient boosting and random forests on a dataset of 1000 students",
        "limitations": [
            "Single institution study",
            "Limited to STEM courses",
            "No consideration of socioeconomic factors",
        ],
        "contribution": "Provides a benchmark for ML-based student performance prediction",
        "key_quotes": [
            "Our results suggest that early intervention based on ML predictions could improve retention rates by up to 20%"
        ],
    }


@pytest.fixture
def sample_synthesis():
    """Provide a sample synthesis text."""
    return """
    # Literature Review: AI in Higher Education
    
    ## Introduction
    
    The application of artificial intelligence in higher education has gained significant attention
    in recent years. This literature review synthesizes findings from multiple studies examining
    the impact of AI on student learning outcomes.
    
    ## Key Findings
    
    Research consistently demonstrates that AI-powered adaptive learning systems can improve
    student engagement and outcomes. Machine learning algorithms have shown promise in predicting
    student performance, enabling early intervention strategies.
    
    ## Challenges and Limitations
    
    Despite promising results, several challenges remain, including concerns about data privacy,
    the need for large training datasets, and questions about algorithmic bias.
    
    ## Future Directions
    
    Future research should focus on longitudinal studies, cross-institutional validation,
    and the development of more interpretable AI models for educational applications.
    """


@pytest.fixture
def mock_paper_retriever():
    """Create a mock paper retriever."""
    mock = Mock()
    mock.search_papers = Mock(
        return_value=[
            {
                "title": "Test Paper 1",
                "abstract": "Abstract 1",
                "authors": ["Author 1"],
                "url": "http://example.com/1",
                "source": "arXiv",
            },
            {
                "title": "Test Paper 2",
                "abstract": "Abstract 2",
                "authors": ["Author 2"],
                "url": "http://example.com/2",
                "source": "Semantic Scholar",
            },
        ]
    )
    return mock


# Markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers",
        "real_llm: mark test as requiring real LLM API calls (needs GROQ_API_KEY)",
    )
