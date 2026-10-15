"""Compatibility entrypoint for the canonical FastAPI application.

Use ``ai_impact_classifier.api:app`` in production. This shim keeps
``uvicorn api:app`` working without maintaining a second API implementation.
"""

from ai_impact_classifier.api import app

__all__ = ["app"]
