import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.resilience.circuit_breaker import CircuitBreaker
from app.resilience.retry import vector_db_retry

logger = logging.getLogger("app.store")

# Pinecone caps a single upsert request at ~2MB; batch conservatively to stay
# well under that regardless of embedding dimension or metadata text size.
UPSERT_BATCH_SIZE = 50


@dataclass
class VectorRecord:
    id: str
    values: list[float]
    metadata: dict[str, Any]


@dataclass
class VectorMatch:
    id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None: ...

    @abstractmethod
    def query(self, vector: list[float], top_k: int) -> list[VectorMatch]:
        ...


class PineconeVectorStore(VectorStore):
    """Pinecone wrapper. Retry handles transient failures on individual
    calls; the circuit breaker sits above retry so that once a call has
    exhausted its retries repeatedly, further calls fail fast instead of
    each one paying the full retry cost against a dead backend."""

    def __init__(self):
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._ensure_index()
        self._index = self._pc.Index(settings.pinecone_index_name)
        self._breaker = CircuitBreaker(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
        )

    def _ensure_index(self) -> None:
        existing = {idx["name"] for idx in self._pc.list_indexes()}
        if settings.pinecone_index_name not in existing:
            self._pc.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.pinecone_embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )

    @vector_db_retry
    def _upsert_with_retry(self, records: list[VectorRecord]) -> None:
        self._index.upsert(
            vectors=[(r.id, r.values, r.metadata) for r in records]
        )

    def upsert(self, records: list[VectorRecord]) -> None:
        for i in range(0, len(records), UPSERT_BATCH_SIZE):
            batch = records[i : i + UPSERT_BATCH_SIZE]
            logger.info("Upserting batch %d-%d of %d vectors", i, i + len(batch), len(records))
            self._breaker.call(lambda batch=batch: self._upsert_with_retry(batch))

    @vector_db_retry
    def _query_with_retry(self, vector: list[float], top_k: int):
        return self._index.query(vector=vector, top_k=top_k, include_metadata=True)

    def query(self, vector: list[float], top_k: int) -> list[VectorMatch]:
        result = self._breaker.call(lambda: self._query_with_retry(vector, top_k))
        return [
            VectorMatch(id=match["id"], score=match["score"], metadata=match.get("metadata", {}))
            for match in result["matches"]
        ]
