import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger("app.store")


@dataclass
class RegistryEntry:
    doc_hash: str
    doc_id: str
    filename: str
    chunk_count: int
    ingested_at: str


class IngestionRegistry:
    """Tracks ingested document hashes for idempotency.

    SQLite-backed so it survives process restarts during the demo; a single
    file swapped for Postgres/Redis in a multi-instance deployment (see
    DESIGN_BRIEF.md "The Gap").
    """

    def __init__(self, db_path: str = ".ingestion_registry.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingested_documents (
                doc_hash TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                ingested_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, doc_hash: str) -> RegistryEntry | None:
        row = self._conn.execute(
            "SELECT doc_hash, doc_id, filename, chunk_count, ingested_at "
            "FROM ingested_documents WHERE doc_hash = ?",
            (doc_hash,),
        ).fetchone()
        if row is None:
            return None
        return RegistryEntry(*row)

    def record(self, doc_hash: str, doc_id: str, filename: str, chunk_count: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO ingested_documents "
            "(doc_hash, doc_id, filename, chunk_count, ingested_at) VALUES (?, ?, ?, ?, ?)",
            (doc_hash, doc_id, filename, chunk_count, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
        logger.info(
            "Document written to registry DB: doc_id=%s filename=%s chunk_count=%d hash=%s...",
            doc_id, filename, chunk_count, doc_hash[:12],
        )

    def list_all(self) -> list[RegistryEntry]:
        rows = self._conn.execute(
            "SELECT doc_hash, doc_id, filename, chunk_count, ingested_at "
            "FROM ingested_documents ORDER BY ingested_at DESC"
        ).fetchall()
        return [RegistryEntry(*row) for row in rows]
