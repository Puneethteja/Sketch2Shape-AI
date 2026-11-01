"""Vercel FastAPI entrypoint; the application and generation pipeline live in api.main."""

from api.main import app

__all__ = ["app"]
