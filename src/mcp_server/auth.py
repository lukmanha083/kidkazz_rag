"""API key authentication middleware for HTTP transport."""

import hmac
from typing import Any, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse


class ApiKeyMiddleware:
    """ASGI middleware that enforces Bearer token authentication.

    Checks Authorization: Bearer <token> on all HTTP requests except
    the /health endpoint (which must be public for Fly.io health checks).
    """

    def __init__(self, app: Any, api_key: str) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        # Pass through non-HTTP scopes (lifespan, websocket)
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Allow /health without auth (Fly.io health checks)
        path = scope.get("path", "")
        if path == "/health":
            await self.app(scope, receive, send)
            return

        # Extract Authorization header
        headers = dict(scope.get("headers", []))
        auth_value = headers.get(b"authorization", b"").decode()

        if not auth_value:
            response = JSONResponse(
                {"error": "Missing Authorization header"},
                status_code=401,
            )
            await response(scope, receive, send)
            return

        # Expect "Bearer <token>"
        if not auth_value.startswith("Bearer "):
            response = JSONResponse(
                {"error": "Invalid Authorization format, expected: Bearer <token>"},
                status_code=401,
            )
            await response(scope, receive, send)
            return

        token = auth_value[7:]  # Strip "Bearer "
        if not hmac.compare_digest(token, self.api_key):
            response = JSONResponse(
                {"error": "Invalid API key"},
                status_code=403,
            )
            await response(scope, receive, send)
            return

        # Auth passed — forward to app
        await self.app(scope, receive, send)
