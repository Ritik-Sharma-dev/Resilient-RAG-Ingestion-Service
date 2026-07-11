import logging
import time
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.graph.nodes import GraphDependencies
from app.graph.workflow import run_query_pipeline
from app.ingestion.embedder import Embedder
from app.ingestion.pipeline import run_ingest_pipeline
from app.logging_config import setup_logging
from app.models import DocumentRecord, IngestResponse, QueryRequest, StructuredOutput
from app.store.ingestion_registry import IngestionRegistry
from app.store.vector_store import PineconeVectorStore
from openai import OpenAI

setup_logging()
logger = logging.getLogger("app.main")

app = FastAPI(title="Resilient RAG Ingestion Service")


@app.exception_handler(Exception)
async def log_unhandled_exceptions(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache
def get_vector_store() -> PineconeVectorStore:
    return PineconeVectorStore()


@lru_cache
def get_registry() -> IngestionRegistry:
    return IngestionRegistry(settings.ingestion_registry_path)


@lru_cache
def get_llm_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def get_graph_dependencies(
    embedder: Embedder = Depends(get_embedder),
    vector_store: PineconeVectorStore = Depends(get_vector_store),
    llm_client: OpenAI = Depends(get_llm_client),
) -> GraphDependencies:
    return GraphDependencies(embedder=embedder, vector_store=vector_store, llm_client=llm_client)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile,
    embedder: Embedder = Depends(get_embedder),
    vector_store: PineconeVectorStore = Depends(get_vector_store),
    registry: IngestionRegistry = Depends(get_registry),
):
    logger.info("Ingest started: filename=%s content_type=%s", file.filename, file.content_type)
    file_bytes = await file.read()

    try:
        result = run_ingest_pipeline(
            file_bytes=file_bytes,
            filename=file.filename or "unknown",
            embedder=embedder,
            vector_store=vector_store,
            registry=registry,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return IngestResponse(
        status=result.status,
        doc_id=result.doc_id,
        chunk_count=result.chunk_count,
        message=result.message,
    )


@app.get("/documents", response_model=list[DocumentRecord])
async def list_documents(registry: IngestionRegistry = Depends(get_registry)):
    entries = registry.list_all()
    return [
        DocumentRecord(
            doc_id=e.doc_id,
            filename=e.filename,
            chunk_count=e.chunk_count,
            ingested_at=e.ingested_at,
        )
        for e in entries
    ]


@app.post("/query", response_model=StructuredOutput)
async def query(
    request: QueryRequest,
    deps: GraphDependencies = Depends(get_graph_dependencies),
):
    start = time.perf_counter()
    logger.info("Query started: question=%r top_k=%s", request.question, request.top_k)

    result = run_query_pipeline(deps, request.question, request.top_k)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Query complete: sources=%d confidence=%s took=%.0fms",
        len(result["structured_output"]["sources"]),
        result["structured_output"]["confidence"],
        elapsed_ms,
    )
    return StructuredOutput(**result["structured_output"])


if __name__ == "__main__":
    import uvicorn

    # Scope the reload watcher to source code only. Without this, uvicorn
    # watches the whole project directory, including .ingestion_registry.db
    # and log files that change on every request — each change triggers a
    # reload, which kills the worker process before Langfuse's background
    # queue can flush queued events over the network.
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["app"],
    )
