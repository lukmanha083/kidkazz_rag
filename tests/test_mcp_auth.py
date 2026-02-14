"""Tests for API key authentication middleware."""

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.mcp_server.auth import ApiKeyMiddleware

pytestmark = pytest.mark.mcp

API_KEY = "test-secret-key-123"


def _make_app(api_key: str = API_KEY) -> TestClient:
    """Create a test app wrapped with ApiKeyMiddleware."""

    async def mcp_endpoint(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def health_endpoint(request: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy"})

    app = Starlette(
        routes=[
            Route("/mcp", mcp_endpoint, methods=["POST"]),
            Route("/health", health_endpoint, methods=["GET"]),
        ]
    )
    wrapped = ApiKeyMiddleware(app, api_key)
    return TestClient(wrapped)


class TestApiKeyMiddleware:
    """Tests for ApiKeyMiddleware."""

    def test_rejects_missing_auth(self):
        """401 when no Authorization header is provided."""
        client = _make_app()
        resp = client.post("/mcp")
        assert resp.status_code == 401
        assert "Missing" in resp.json()["error"]

    def test_rejects_invalid_format(self):
        """401 when Authorization header is not Bearer format."""
        client = _make_app()
        resp = client.post("/mcp", headers={"Authorization": "Basic abc123"})
        assert resp.status_code == 401
        assert "Bearer" in resp.json()["error"]

    def test_rejects_wrong_key(self):
        """403 when wrong API key is provided."""
        client = _make_app()
        resp = client.post(
            "/mcp", headers={"Authorization": "Bearer wrong-key"}
        )
        assert resp.status_code == 403
        assert "Invalid" in resp.json()["error"]

    def test_allows_valid_key(self):
        """200 when correct API key is provided."""
        client = _make_app()
        resp = client.post(
            "/mcp", headers={"Authorization": f"Bearer {API_KEY}"}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_is_public(self):
        """Health endpoint accessible without authentication."""
        client = _make_app()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_health_ignores_bad_auth(self):
        """Health endpoint works even with a bad auth header."""
        client = _make_app()
        resp = client.get(
            "/health", headers={"Authorization": "Bearer wrong-key"}
        )
        assert resp.status_code == 200
