"""API-key authentication: `X-API-Key: <key>` or `Authorization: Bearer <key>`."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, status

from .config import settings


def require_api_key(request: Request) -> None:
    if settings.auth_disabled:
        return
    key = request.headers.get("x-api-key", "")
    if not key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth[7:].strip()
    if key and any(secrets.compare_digest(key.encode(), k.encode()) for k in settings.api_keys):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key (send it in the X-API-Key header).",
    )
