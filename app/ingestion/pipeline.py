import logging
import time
import uuid
from dataclasses import dataclass

from app.config import settings
from app.ingestion.chunker import Chunk, chunk_pages, normalize_text
from app.ingestion.dedup import compute_doc_hash
from app.ingestion.embedder import Embedder
from app.ingestion.pdf_extractor import ExtractedPage, extract_pdf
from app.store.ingestion_registry import IngestionRegistry
from app.store.vector_store import PineconeVectorStore, VectorRecord
from app.telemetry.langfuse_client import log_step, observe

logger = logging.getLogger("app.ingestion")


@dataclass
class IngestResult:
    status: str
    doc_id: str
    chunk_count: int
    message: str = ""


@observe(name="extract_pdf")
def _extract_step(file_bytes: bytes, filename: str) -> list[ExtractedPage]:
    start = time.perf_counter()
    pages = extract_pdf(file_bytes)
    latency_ms = (time.perf_counter() - start) * 1000

    table_count = sum(len(p.tables) for p in pages)
    logger.info("[extract] filename=%s pages=%d tables=%d took=%.0fms", filename, len(pages), table_count, latency_ms)
    log_step(
        name="extract_pdf",
        latency_ms=latency_ms,
        metadata={"filename": filename, "page_count": len(pages), "table_count": table_count},
    )
    return pages


@observe(name="chunk_pages")
def _chunk_step(pages: list[ExtractedPage]) -> list[Chunk]:
    start = time.perf_counter()
    chunks = chunk_pages(pages, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    latency_ms = (time.perf_counter() - start) * 1000

    prose_count = sum(1 for c in chunks if c.chunk_type == "prose")
    table_count = sum(1 for c in chunks if c.chunk_type == "table")
    logger.info("[chunk] chunks=%d (prose=%d table=%d) took=%.0fms", len(chunks), prose_count, table_count, latency_ms)
    log_step(
        name="chunk_pages",
        latency_ms=latency_ms,
        metadata={"chunk_count": len(chunks), "prose_chunks": prose_count, "table_chunks": table_count},
    )
    return chunks


@observe(name="embed_chunks")
def _embed_step(embedder: Embedder, chunks: list[Chunk]) -> list[list[float]]:
    start = time.perf_counter()
    try:
        vectors = embedder.embed_batch([c.text for c in chunks])
    except Exception:
        logger.error("[embed] AI call failed after retries (chunk_count=%d)", len(chunks), exc_info=True)
        raise
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info("[embed] chunks=%d model=%s took=%.0fms", len(chunks), settings.embedding_model, latency_ms)
    log_step(
        name="embed_chunks",
        latency_ms=latency_ms,
        metadata={"chunk_count": len(chunks), "model": settings.embedding_model},
    )
    return vectors


@observe(name="upsert_vectors")
def _upsert_step(vector_store: PineconeVectorStore, records: list[VectorRecord]) -> None:
    start = time.perf_counter()
    try:
        vector_store.upsert(records)
    except Exception:
        logger.error("[upsert] AI call failed after retries (vector_count=%d)", len(records), exc_info=True)
        raise
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info("[upsert] vectors=%d took=%.0fms", len(records), latency_ms)
    log_step(name="upsert_vectors", latency_ms=latency_ms, metadata={"vector_count": len(records)})


@observe(name="ingest_document")
def run_ingest_pipeline(
    *,
    file_bytes: bytes,
    filename: str,
    embedder: Embedder,
    vector_store: PineconeVectorStore,
    registry: IngestionRegistry,
) -> IngestResult:
    overall_start = time.perf_counter()

    pages = _extract_step(file_bytes, filename)

    full_text = " ".join(normalize_text(p.prose_text) for p in pages)
    doc_hash = compute_doc_hash(full_text)

    existing = registry.get(doc_hash)
    if existing:
        logger.info("[dedup] hash=%s... already ingested as doc_id=%s — skipping", doc_hash[:12], existing.doc_id)
        log_step(
            name="ingest_document",
            doc_id=existing.doc_id,
            metadata={"status": "already_ingested", "filename": filename},
        )
        return IngestResult(
            status="already_ingested",
            doc_id=existing.doc_id,
            chunk_count=existing.chunk_count,
            message=f"Document already ingested as {existing.doc_id}",
        )

    doc_id = str(uuid.uuid4())
    chunks = _chunk_step(pages)

    if not chunks:
        logger.warning("[chunk] no extractable text or tables in %s — rejecting", filename)
        raise ValueError("No extractable text or tables found in PDF")

    vectors = _embed_step(embedder, chunks)

    records = [
        VectorRecord(
            id=f"{doc_id}-{chunk.chunk_index}",
            values=vector,
            metadata={
                "doc_id": doc_id,
                "chunk_type": chunk.chunk_type,
                "page_number": chunk.page_number,
                "text": chunk.text,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    _upsert_step(vector_store, records)

    registry.record(doc_hash, doc_id, filename, len(chunks))

    total_latency_ms = (time.perf_counter() - overall_start) * 1000
    logger.info("[ingest] doc_id=%s chunks=%d total_took=%.0fms", doc_id, len(chunks), total_latency_ms)
    log_step(
        name="ingest_document",
        doc_id=doc_id,
        latency_ms=total_latency_ms,
        metadata={"status": "ingested", "filename": filename, "chunk_count": len(chunks)},
    )

    return IngestResult(status="ingested", doc_id=doc_id, chunk_count=len(chunks))
