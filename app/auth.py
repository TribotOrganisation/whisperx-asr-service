import os
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

API_KEY = os.getenv("API_KEY", "").strip()
UNAUTHENTICATED_PATHS = frozenset({"/health"})


def _extract_api_key(request: Request) -> str | None:
    authorization = request.headers.get("Authorization")
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    api_key_header = request.headers.get("X-API-Key")
    if api_key_header:
        return api_key_header.strip()
    return None


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not API_KEY:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path in UNAUTHENTICATED_PATHS:
            return await call_next(request)
        provided_key = _extract_api_key(request)
        if not provided_key or not secrets.compare_digest(provided_key, API_KEY):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
        return await call_next(request)


def install_api_key_auth(app) -> None:
    if API_KEY:
        app.add_middleware(ApiKeyMiddleware)
