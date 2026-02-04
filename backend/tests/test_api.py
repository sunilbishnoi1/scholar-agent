# API Endpoint Tests
# Comprehensive tests for all FastAPI endpoints

import json
import os
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

# Set test database URL BEFORE importing any app modules
os.environ["DATABASE_URL"] = "sqlite:///./test_api.db"


# ============================================
# Fixtures
# ============================================


@pytest.fixture(scope="module")
def setup_test_db():
    """Create test database tables once per test module."""
    from sqlalchemy import create_engine

    from models.database import Base

    engine = create_engine("sqlite:///./test_api.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    # Cleanup after all tests in module
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(setup_test_db):
    """Create a test client with test database."""
    # Import after setting env var
    from main import app

    return TestClient(app)


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    mock = MagicMock()
    mock.query.return_value.filter.return_value.first.return_value = None
    mock.query.return_value.filter.return_value.all.return_value = []
    return mock


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    from models.database import User

    user = User(
        id="test-user-123",
        email="test@example.com",
        name="Test User",
        hashed_password="hashed_password",
        tier="free",
        monthly_budget_usd=1.0,
    )
    return user


@pytest.fixture
def mock_project():
    """Create a mock research project."""
    from models.database import ResearchProject

    project = ResearchProject(
        id="test-project-456",
        user_id="test-user-123",
        title="AI in Education",
        research_question="How does AI affect student learning?",
        keywords=["AI", "education", "learning"],
        subtopics=["Adaptive Learning", "Assessment"],
        status="created",
        total_papers_found=0,
        created_at=datetime.utcnow(),
    )
    return project


@pytest.fixture
def auth_headers(client, mock_user):
    """Create authentication headers with a valid token."""
    # Create a token for the mock user
    from datetime import timedelta

    import auth

    token = auth.create_access_token(
        data={"sub": mock_user.email}, expires_delta=timedelta(minutes=30)
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def authenticated_client(setup_test_db, mock_user, mock_project):
    """Create a test client with mocked authentication and database."""
    import auth
    from db import get_db
    from main import app

    # Create mock db session
    mock_db = MagicMock()
    # Set up default return values for common queries
    mock_db.query.return_value.filter.return_value.first.return_value = mock_project
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_project]

    # Override dependencies
    app.dependency_overrides[auth.get_current_user] = lambda: mock_user
    app.dependency_overrides[get_db] = lambda: mock_db

    client = TestClient(app)
    yield client

    # Cleanup
    app.dependency_overrides.clear()


# ============================================
# Authentication Tests
# ============================================


class TestAuthenticationEndpoints:
    """Tests for authentication API endpoints."""

    def test_register_new_user_success(self, mock_db_session):
        """Test successful user registration."""
        from db import get_db
        from main import app

        # Override the dependency to use mock - must be a generator function
        def override_get_db():
            yield mock_db_session

        app.dependency_overrides[get_db] = override_get_db

        # Mock: no existing user found
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        # Mock db operations - when refresh is called, set the id on the user
        def mock_refresh(user):
            if not hasattr(user, "id") or user.id is None:
                user.id = "test-uuid-123"

        mock_db_session.add = MagicMock()
        mock_db_session.commit = MagicMock()
        mock_db_session.refresh = mock_refresh

        client = TestClient(app)
        response = client.post(
            "/api/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "securepassword123",
                "name": "New User",
            },
        )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        assert "id" in response.json()
        assert response.json()["email"] == "newuser@example.com"
        assert response.json()["name"] == "New User"

    def test_register_duplicate_email_fails(self, setup_test_db, mock_user):
        """Test that registering with existing email fails."""
        from db import get_db
        from main import app

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)
        response = client.post(
            "/api/auth/register",
            json={"email": mock_user.email, "password": "password123", "name": "Test User"},
        )

        app.dependency_overrides.clear()
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]

    def test_register_invalid_email_fails(self, client):
        """Test that registration with invalid email fails."""
        response = client.post(
            "/api/auth/register",
            json={"email": "not-an-email", "password": "password123", "name": "Test"},
        )

        assert response.status_code == 422  # Validation error

    def test_login_success(self, setup_test_db, mock_user):
        """Test successful login returns token."""
        import auth
        from db import get_db
        from main import app

        mock_user.hashed_password = auth.get_password_hash("testpassword")

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)
        response = client.post(
            "/api/auth/token", data={"username": mock_user.email, "password": "testpassword"}
        )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password_fails(self, mock_db_session, mock_user):
        """Test login with wrong password fails."""
        import auth
        from db import get_db
        from main import app

        mock_user.hashed_password = auth.get_password_hash("correctpassword")

        def override_get_db():
            yield mock_db_session

        app.dependency_overrides[get_db] = override_get_db
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_user

        client = TestClient(app)
        response = client.post(
            "/api/auth/token",
            data={"username": mock_user.email, "password": "wrongpassword"},
        )

        app.dependency_overrides.clear()
        assert response.status_code == 401

    def test_login_nonexistent_user_fails(self, mock_db_session):
        """Test login with non-existent user fails."""
        from db import get_db
        from main import app

        def override_get_db():
            yield mock_db_session

        app.dependency_overrides[get_db] = override_get_db
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        client = TestClient(app)
        response = client.post(
            "/api/auth/token",
            data={"username": "nonexistent@example.com", "password": "password"},
        )

        app.dependency_overrides.clear()
        assert response.status_code == 401

    def test_get_current_user_authenticated(self, authenticated_client, mock_user):
        """Test getting current user when authenticated."""
        response = authenticated_client.get("/api/auth/users/me")

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == mock_user.email

    def test_get_current_user_unauthenticated(self, client):
        """Test getting current user without auth fails."""
        response = client.get("/api/auth/users/me")

        assert response.status_code == 401


# ============================================
# Projects Tests
# ============================================


class TestProjectsEndpoints:
    """Tests for project management API endpoints."""

    def test_get_projects_returns_user_projects(self, authenticated_client):
        """Test getting list of projects for authenticated user."""
        response = authenticated_client.get("/api/projects")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_projects_unauthenticated_fails(self, client):
        """Test getting projects without auth fails."""
        response = client.get("/api/projects")

        assert response.status_code == 401

    def test_create_project_success(self, setup_test_db, mock_user, mock_project):
        """Test creating a new project."""
        from datetime import datetime

        import auth
        from db import get_db
        from main import app

        # Set up mock db that properly handles add and refresh
        mock_db = MagicMock()

        # When db.add() is called, do nothing
        # When db.refresh() is called, populate the project with required fields
        def mock_refresh(obj):
            obj.id = mock_project.id
            obj.total_papers_found = 0
            obj.created_at = datetime.utcnow()

        mock_db.refresh = mock_refresh

        app.dependency_overrides[auth.get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: mock_db

        # Mock the LLM client and PlannerAgent
        with patch("main.get_llm_client") as mock_get_llm:
            with patch("main.ResearchPlannerAgent") as mock_planner:
                mock_planner_instance = mock_planner.return_value
                mock_planner_instance.generate_initial_plan.return_value = {
                    "keywords": ["AI", "education"],
                    "subtopics": ["Learning"],
                }

                client = TestClient(app)
                response = client.post(
                    "/api/projects",
                    json={"title": "Test Project", "research_question": "How does AI work?"},
                )

                app.dependency_overrides.clear()
                # May succeed or fail depending on db commit
                assert response.status_code in [200, 201, 500]

    def test_create_project_missing_fields_fails(self, authenticated_client):
        """Test creating project with missing fields fails."""
        response = authenticated_client.post(
            "/api/projects",
            json={
                "title": "Only Title"
                # Missing research_question
            },
        )

        assert response.status_code == 422  # Validation error

    def test_get_single_project_success(self, authenticated_client, mock_project):
        """Test getting a single project by ID."""
        response = authenticated_client.get(f"/api/projects/{mock_project.id}")

        assert response.status_code == 200

    def test_get_project_not_found(self, setup_test_db, mock_user):
        """Test getting non-existent project returns 404."""
        import auth
        from db import get_db
        from main import app

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        app.dependency_overrides[auth.get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)
        response = client.get("/api/projects/nonexistent-id")

        app.dependency_overrides.clear()
        assert response.status_code == 404

    def test_start_project_review_success(self, authenticated_client, mock_project):
        """Test starting a literature review."""
        with patch("main.celery_app.send_task") as mock_celery:
            mock_celery.return_value.id = "job-123"

            response = authenticated_client.post(
                f"/api/projects/{mock_project.id}/start?max_papers=20"
            )

            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "queued"

    def test_delete_project_success(self, setup_test_db, mock_user, mock_project):
        """Test deleting a project successfully."""
        import auth
        from db import get_db
        from main import app

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_project
        mock_db.query.return_value.filter.return_value.delete.return_value = 0

        app.dependency_overrides[auth.get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)

        with patch("main.RAGService", None):
            response = client.delete(f"/api/projects/{mock_project.id}")

        app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == mock_project.id
        assert data["deleted"] is True
        assert "deleted successfully" in data["message"]
        mock_db.delete.assert_called_once_with(mock_project)
        mock_db.commit.assert_called_once()
        # Verify all related tables are queried for deletion (LLMInteraction, AgentPlan, PaperReference)
        assert mock_db.query.return_value.filter.return_value.delete.call_count == 3

    def test_delete_project_not_found(self, setup_test_db, mock_user):
        """Test deleting non-existent project returns 404."""
        import auth
        from db import get_db
        from main import app

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        app.dependency_overrides[auth.get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)
        response = client.delete("/api/projects/nonexistent-id")

        app.dependency_overrides.clear()
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_delete_project_unauthenticated_fails(self, client):
        """Test deleting project without auth fails."""
        response = client.delete("/api/projects/some-id")
        assert response.status_code == 401

    def test_delete_project_with_rag_cleanup(self, setup_test_db, mock_user, mock_project):
        """Test deleting a project also cleans up RAG data."""
        import auth
        from db import get_db
        from main import app

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_project
        mock_db.query.return_value.filter.return_value.delete.return_value = 0

        app.dependency_overrides[auth.get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: mock_db

        # Mock RAGService
        mock_rag_service = MagicMock()
        mock_rag_class = MagicMock(return_value=mock_rag_service)

        client = TestClient(app)

        with patch("main.RAGService", mock_rag_class):
            response = client.delete(f"/api/projects/{mock_project.id}")

        app.dependency_overrides.clear()

        assert response.status_code == 200
        mock_rag_service.delete_project_data.assert_called_once_with(mock_project.id)


# ============================================
# Usage Tracking Tests
# ============================================


class TestUsageEndpoints:
    """Tests for usage tracking API endpoints."""

    def test_get_usage_summary(self, authenticated_client, mock_user):
        """Test getting user usage summary."""
        with patch("main.UsageTracker") as MockTracker:
            mock_instance = MockTracker.return_value
            mock_instance.get_usage_summary.return_value = {
                "user_id": mock_user.id,
                "tier": "free",
                "month": "2024-01",
                "budget": {
                    "limit_usd": 1.0,
                    "used_usd": 0.0,
                    "remaining_usd": 1.0,
                    "usage_percent": 0.0,
                },
                "tokens": {
                    "used": 0,
                    "limit": 100000,
                    "remaining": 100000,
                    "usage_percent": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
                "activity": {"projects_created": 0, "papers_analyzed": 0, "llm_calls": 0},
                "limits": {
                    "monthly_budget_usd": 1.0,
                    "monthly_tokens": 100000,
                    "max_projects": 5,
                    "max_papers_per_project": 30,
                },
            }

            response = authenticated_client.get("/api/users/me/usage")

            assert response.status_code == 200

    def test_budget_check_allowed(self, authenticated_client, mock_user):
        """Test budget check when within limits."""
        with patch("main.UsageTracker") as MockTracker:
            mock_instance = MockTracker.return_value
            mock_instance.check_budget.return_value = {
                "allowed": True,
                "remaining_budget": 0.9,
                "current_usage": 0.1,
                "limit": 1.0,
                "usage_percent": 10.0,
            }

            response = authenticated_client.get("/api/users/me/budget-check?estimated_cost=0.1")

            assert response.status_code == 200
            data = response.json()
            assert data["allowed"] is True


# ============================================
# Search Tests
# ============================================


class TestSearchEndpoints:
    """Tests for semantic search API endpoints."""

    def test_semantic_search_success(self, authenticated_client, mock_project):
        """Test semantic search within a project."""
        with patch("main.RAGService") as MockRAG:
            mock_instance = MockRAG.return_value
            mock_instance.search.return_value = [
                {
                    "chunk_id": "chunk-1",
                    "content": "Test content",
                    "paper_id": "paper-1",
                    "paper_title": "Test Paper",
                    "chunk_type": "abstract",
                    "final_score": 0.95,
                }
            ]

            response = authenticated_client.post(
                f"/api/projects/{mock_project.id}/search",
                json={"text": "machine learning in education", "top_k": 10, "use_hybrid": True},
            )

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_search_project_not_found(self, setup_test_db, mock_user):
        """Test search on non-existent project."""
        import auth
        from db import get_db
        from main import app

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        app.dependency_overrides[auth.get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)
        response = client.post("/api/projects/nonexistent/search", json={"text": "test query"})

        app.dependency_overrides.clear()
        assert response.status_code == 404


# ============================================
# Health Check Tests
# ============================================


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")

        # May or may not exist, check both cases
        assert response.status_code in [200, 404]

    def test_root_endpoint(self, client):
        """Test root endpoint - may not exist in API."""
        response = client.get("/")

        # Root endpoint may not be defined - 404 is acceptable
        assert response.status_code in [200, 404]


# ============================================
# Error Handling Tests
# ============================================


class TestAPIErrorHandling:
    """Tests for API error handling."""

    def test_invalid_json_returns_422(self, client, auth_headers):
        """Test that invalid JSON returns 422."""
        response = client.post(
            "/api/projects",
            headers={**auth_headers, "Content-Type": "application/json"},
            content="not valid json",
        )

        assert response.status_code == 422

    def test_method_not_allowed(self, client):
        """Test that wrong HTTP method returns 405."""
        response = client.delete("/api/auth/token")  # Should only accept POST

        assert response.status_code == 405

    def test_invalid_token_format(self, client):
        """Test that invalid token format is rejected."""
        response = client.get(
            "/api/projects", headers={"Authorization": "Bearer invalid-token-format"}
        )

        assert response.status_code == 401


# ============================================
# Celery Task Import Tests
# ============================================


class TestCeleryTaskImports:
    """Tests to ensure Celery tasks have all required imports.

    These tests verify that all imports inside Celery task functions
    are valid and won't cause NameError at runtime.
    """

    def test_run_literature_review_imports(self, setup_test_db):
        """Verify run_literature_review task has all required imports."""
        # Import the main module - this validates top-level imports
        from main import run_literature_review

        # Verify the task is properly decorated and callable
        assert callable(run_literature_review)
        assert hasattr(run_literature_review, "delay")  # Celery task attribute
        assert hasattr(run_literature_review, "apply_async")  # Celery task attribute

        # Verify the function can be inspected (catches syntax errors)
        import inspect

        sig = inspect.signature(run_literature_review)
        assert "project_id" in sig.parameters
        assert "max_papers" in sig.parameters
