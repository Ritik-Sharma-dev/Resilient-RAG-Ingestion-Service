from functools import lru_cache

from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

from app.config import settings

__all__ = ["get_langfuse_client", "observe", "log_step"]


@lru_cache
def get_langfuse_client() -> Langfuse:
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def log_step(
    *,
    name: str,
    doc_id: str | None = None,
    latency_ms: float | None = None,
    token_usage: dict | None = None,
    metadata: dict | None = None,
) -> None:
    """Attach metadata to the current observation. Must be called from
    within a function decorated with @observe() so langfuse_context has an
    active observation to attach to — otherwise this is a silent no-op."""
    payload = {"doc_id": doc_id, "latency_ms": latency_ms, **(metadata or {})}
    payload = {k: v for k, v in payload.items() if v is not None}

    langfuse_context.update_current_observation(
        name=name,
        metadata=payload,
        usage=token_usage,
    )
