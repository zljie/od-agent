"""Unit tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from src.app import create_app


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_readiness_check(self, client):
        """Test readiness endpoint returns ready status."""
        response = client.get("/readiness")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_liveness_check(self, client):
        """Test liveness endpoint returns alive status."""
        response = client.get("/liveness")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestWelcomePage:
    """Test welcome page endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns welcome information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "endpoints" in data


class TestChatEndpoint:
    """Test chat endpoint."""

    def test_chat_endpoint(self, client):
        """Test chat endpoint accepts messages."""
        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    def test_chat_endpoint_with_session(self, client):
        """Test chat endpoint accepts session ID."""
        response = client.post(
            "/chat",
            json={"message": "Hello", "session_id": "test-session"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("session_id") == "test-session"


class TestProcessEndpoint:
    """Test process endpoint."""

    def test_process_endpoint(self, client):
        """Test process endpoint accepts input."""
        response = client.post(
            "/process",
            json={"input": [{"role": "user", "content": "Hello"}]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert "status" in data

    def test_process_endpoint_requires_user_message(self, client):
        """Test process endpoint requires user message."""
        response = client.post(
            "/process",
            json={"input": [{"role": "assistant", "content": "Hi"}]}
        )
        assert response.status_code == 400
