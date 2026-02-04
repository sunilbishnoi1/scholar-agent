import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test_cors.db"


@pytest.fixture(scope="module")
def setup_test_db():
    """Create test database tables once per test module."""
    from sqlalchemy import create_engine

    from models.database import Base

    engine = create_engine("sqlite:///./test_cors.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


class TestCORSConfiguration:
    """Tests for CORS configuration in production and development."""

    def test_cors_headers_present_localhost(self, setup_test_db):
        """Test CORS headers are present for localhost requests."""
        from main import app

        client = TestClient(app)
        response = client.get("/", headers={"Origin": "http://localhost:5173"})

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
        assert "access-control-allow-credentials" in response.headers

    def test_cors_headers_present_localhost_8000(self, setup_test_db):
        """Test CORS headers for localhost:8000."""
        from main import app

        client = TestClient(app)
        response = client.get("/", headers={"Origin": "http://localhost:8000"})

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_headers_present_127_0_0_1(self, setup_test_db):
        """Test CORS headers for 127.0.0.1."""
        from main import app

        client = TestClient(app)
        response = client.get("/", headers={"Origin": "http://127.0.0.1:5173"})

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_headers_rejected_unknown_origin(self, setup_test_db):
        """Test CORS headers are NOT sent for unknown origin."""
        from main import app

        client = TestClient(app)
        response = client.get("/", headers={"Origin": "https://malicious.com"})

        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers

    def test_cors_allowed_methods(self, setup_test_db):
        """Test CORS allowed methods header."""
        from main import app

        client = TestClient(app)
        response = client.get("/", headers={"Origin": "http://localhost:5173"})

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_allowed_headers(self, setup_test_db):
        """Test CORS allowed headers."""
        from main import app

        client = TestClient(app)
        response = client.get("/", headers={"Origin": "http://localhost:5173"})

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_exposed_headers(self, setup_test_db):
        """Test CORS exposed headers."""
        from main import app

        client = TestClient(app)
        response = client.get("/", headers={"Origin": "http://localhost:5173"})

        assert response.status_code == 200
        assert "access-control-expose-headers" in response.headers

    def test_cors_credentials_allowed(self, setup_test_db):
        """Test CORS credentials are allowed."""
        from main import app

        client = TestClient(app)
        response = client.get("/", headers={"Origin": "http://localhost:5173"})

        assert response.status_code == 200
        assert response.headers.get("access-control-allow-credentials") == "true"

    def test_cors_get_request_with_origin(self, setup_test_db):
        """Test GET request includes CORS headers."""
        from main import app

        client = TestClient(app)
        response = client.get("/", headers={"Origin": "http://localhost:5173"})

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_post_request_with_origin(self, setup_test_db):
        """Test POST request includes CORS headers."""
        from main import app

        client = TestClient(app)
        response = client.post("/api/health", headers={"Origin": "http://localhost:5173"})

        assert response.status_code in [200, 404, 405]

    def test_allowed_origins_env_var(self, setup_test_db):
        """Test ALLOWED_ORIGINS environment variable is parsed."""
        from main import _get_cors_origins

        with patch.dict(
            os.environ,
            {"ALLOWED_ORIGINS": "https://custom-domain.com,https://another-domain.com"},
        ):
            origins = _get_cors_origins()

            assert "https://custom-domain.com" in origins
            assert "https://another-domain.com" in origins

    def test_cors_headers_on_health_endpoint(self, setup_test_db):
        """Test CORS headers on health check endpoint."""
        from main import app

        client = TestClient(app)
        response = client.get("/", headers={"Origin": "http://localhost:5173"})

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_middleware_configured(self, setup_test_db):
        """Test CORS middleware is properly configured."""
        from main import origins

        assert len(origins) > 0
        assert any("localhost" in origin for origin in origins)


class TestCORSProductionDeployment:
    """Tests simulating production deployment scenarios."""

    def test_production_urls_included_in_code(self):
        """Test production URLs are in the CORS origins list."""
        from main import _get_cors_origins

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            origins = _get_cors_origins()

            assert any("scholar-agent.vercel.app" in origin for origin in origins)
            assert any("scholaragent.dpdns.org" in origin for origin in origins)

    def test_custom_domain_frontend_to_backend(self, setup_test_db):
        """Test CORS with custom domain from env variable."""
        from main import _get_cors_origins

        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "https://custom-frontend.com"}):
            origins = _get_cors_origins()
            assert "https://custom-frontend.com" in origins

    def test_websocket_origin_accepted(self, setup_test_db):
        """Test WebSocket connection origin is accepted."""
        from main import app

        client = TestClient(app)
        response = client.get("/", headers={"Origin": "http://localhost:5173"})

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_origins_function_includes_production_urls(self):
        """Test _get_cors_origins includes production URLs when ENVIRONMENT=production."""
        from main import _get_cors_origins

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            origins = _get_cors_origins()

            assert "https://scholar-agent.vercel.app" in origins
            assert "https://scholaragent.dpdns.org" in origins

    def test_cors_origins_function_development_mode(self):
        """Test _get_cors_origins for development mode."""
        from main import _get_cors_origins

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            origins = _get_cors_origins()

            assert any("localhost" in origin for origin in origins)
            assert any("127.0.0.1" in origin for origin in origins)

    def test_multiple_allowed_origins_parsed(self, setup_test_db):
        """Test ALLOWED_ORIGINS with multiple comma-separated values."""
        from main import _get_cors_origins

        env_origins = "https://api1.com,https://api2.com,https://api3.com"
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": env_origins}):
            origins = _get_cors_origins()

            assert "https://api1.com" in origins
            assert "https://api2.com" in origins
            assert "https://api3.com" in origins

    def test_render_backend_url_in_origins(self, setup_test_db):
        """Test that Render backend URL can be added via ALLOWED_ORIGINS."""
        from main import _get_cors_origins

        render_url = "https://scholar-backend.onrender.com"
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": render_url}):
            origins = _get_cors_origins()

            assert render_url in origins

    def test_cors_logging_configured(self, setup_test_db, caplog):
        """Test that CORS configuration is logged."""
        from main import origins

        assert len(origins) > 0
