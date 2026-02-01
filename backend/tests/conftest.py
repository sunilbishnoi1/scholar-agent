# Pytest Configuration and Shared Fixtures
# Common test fixtures and configuration for the test suite

import os
import sys

# Set test database URL BEFORE importing any app modules
# This ensures SQLite is used instead of PostgreSQL during tests
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

# Add the backend directory to the Python path for test imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from collections.abc import Generator
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

# Configure pytest-asyncio
pytest_plugins = ['pytest_asyncio']


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
# Database fixtures
# ============================================

@pytest.fixture
def mock_db_session():
    """Create a mock database session for unit tests."""
    mock = MagicMock()
    mock.query.return_value.filter.return_value.first.return_value = None
    mock.query.return_value.filter.return_value.all.return_value = []
    mock.add = MagicMock()
    mock.commit = MagicMock()
    mock.refresh = MagicMock()
    mock.rollback = MagicMock()
    return mock


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for integration tests."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from models.database import Base

    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()
    engine.dispose()


# ============================================
# LLM Client fixtures
# ============================================

@pytest.fixture
def mock_llm_client():
    """
    Create a mock LLM client for testing.
    
    The mock is configured to return valid JSON responses by default.
    Tests can override the return value for specific scenarios.
    """
    mock = Mock()
    mock.chat = Mock(return_value='{"keywords": ["test1", "test2"], "subtopics": ["Subtopic 1"]}')
    return mock


@pytest.fixture
def mock_async_llm_client():
    """Create an async mock LLM client."""
    mock = AsyncMock()
    mock.chat = AsyncMock(return_value='{"keywords": ["test1", "test2"]}')
    return mock


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
        "analysis": None
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
            "analysis": None
        },
        {
            "id": "test_paper_3",
            "title": "Adaptive Learning Systems: A Review",
            "abstract": "This systematic review examines the effectiveness of AI-powered adaptive learning systems in improving student engagement and outcomes.",
            "authors": ["Chris Brown", "Diana Lee"],
            "url": "https://arxiv.org/abs/2024.67890",
            "source": "arXiv",
            "relevance_score": None,
            "analysis": None
        }
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
            "Student engagement metrics are strong predictors of success"
        ],
        "methodology": "Quantitative study using gradient boosting and random forests on a dataset of 1000 students",
        "limitations": [
            "Single institution study",
            "Limited to STEM courses",
            "No consideration of socioeconomic factors"
        ],
        "contribution": "Provides a benchmark for ML-based student performance prediction",
        "key_quotes": [
            "Our results suggest that early intervention based on ML predictions could improve retention rates by up to 20%"
        ]
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
    mock.search_papers = Mock(return_value=[
        {
            "title": "Test Paper 1",
            "abstract": "Abstract 1",
            "authors": ["Author 1"],
            "url": "http://example.com/1",
            "source": "arXiv"
        },
        {
            "title": "Test Paper 2",
            "abstract": "Abstract 2",
            "authors": ["Author 2"],
            "url": "http://example.com/2",
            "source": "Semantic Scholar"
        }
    ])
    return mock


# Markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
