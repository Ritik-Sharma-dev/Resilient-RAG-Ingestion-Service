import json
import logging
import time
from dataclasses import dataclass

from openai import OpenAI

from app.config import settings
from app.graph.state import GraphState
from app.ingestion.embedder import Embedder
from app.resilience.retry import llm_retry
from app.store.vector_store import VectorStore
from app.telemetry.langfuse_client import log_step, observe

logger = logging.getLogger("app.graph")

SYSTEM_PROMPT = (
    "You are a RAG assistant. Answer the question using ONLY the provided "
    "context chunks. Respond with a JSON object: "
    '{"answer": str, "confidence": "low" | "medium" | "high"}. '
    "If the context doesn't contain the answer, say so and use low confidence."
)


@dataclass
class GraphDependencies:
    embedder: Embedder
    vector_store: VectorStore
    llm_client: OpenAI


def make_retrieve_node(deps: GraphDependencies):
    @observe(name="retrieve")
    def retrieve_node(state: GraphState) -> dict:
        start = time.perf_counter()
        top_k = state.get("top_k") or settings.retrieval_top_k

        try:
            query_vector = deps.embedder.embed(state["question"])
            matches = deps.vector_store.query(query_vector, top_k=top_k)
        except Exception:
            logger.error(
                "[retrieve] AI call failed after retries: embedding or vector DB query error",
                exc_info=True,
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        match_doc_ids = [m.metadata.get("doc_id") for m in matches]
        logger.info(
            "[retrieve] top_k=%d matches=%d doc_ids=%s took=%.0fms",
            top_k, len(matches), match_doc_ids, latency_ms,
        )
        log_step(
            name="retrieve",
            latency_ms=latency_ms,
            metadata={"top_k": top_k, "matches_returned": len(matches), "match_doc_ids": match_doc_ids},
        )

        return {"retrieved_chunks": matches}

    return retrieve_node


def make_generate_node(deps: GraphDependencies):
    @llm_retry
    def _call_llm(context: str, question: str):
        return deps.llm_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            response_format={"type": "json_object"},
        )

    @observe(name="generate")
    def generate_node(state: GraphState) -> dict:
        start = time.perf_counter()

        chunks = state.get("retrieved_chunks", [])
        context = "\n\n---\n\n".join(m.metadata.get("text", "") for m in chunks)

        try:
            response = _call_llm(context, state["question"])
        except Exception:
            logger.error(
                "[generate] AI call failed after retries: LLM call error (model=%s)",
                settings.llm_model, exc_info=True,
            )
            raise

        raw_answer = response.choices[0].message.content
        usage = response.usage

        latency_ms = (time.perf_counter() - start) * 1000
        token_usage = {
            "input": usage.prompt_tokens,
            "output": usage.completion_tokens,
            "total": usage.total_tokens,
        }
        logger.info(
            "[generate] model=%s context_chunks=%d tokens=%s took=%.0fms",
            settings.llm_model, len(chunks), token_usage, latency_ms,
        )
        log_step(
            name="generate",
            latency_ms=latency_ms,
            token_usage=token_usage,
            metadata={"model": settings.llm_model, "context_chunks": len(chunks)},
        )

        return {"raw_answer": raw_answer, "token_usage": token_usage}

    return generate_node


def make_format_node(deps: GraphDependencies):
    @observe(name="format")
    def format_node(state: GraphState) -> dict:
        start = time.perf_counter()

        try:
            parsed = json.loads(state["raw_answer"])
            answer = parsed.get("answer", "")
            confidence = parsed.get("confidence", "medium")
        except (json.JSONDecodeError, TypeError):
            logger.warning("[format] LLM did not return valid JSON — falling back to raw text, confidence=low")
            answer = state.get("raw_answer", "")
            confidence = "low"

        sources = [
            {
                "doc_id": m.metadata.get("doc_id"),
                "chunk_id": m.id,
                "chunk_type": m.metadata.get("chunk_type", "prose"),
                "text": m.metadata.get("text", ""),
                "score": m.score,
            }
            for m in state.get("retrieved_chunks", [])
        ]

        structured_output = {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("[format] sources=%d confidence=%s took=%.0fms", len(sources), confidence, latency_ms)
        log_step(name="format", latency_ms=latency_ms, metadata={"source_count": len(sources), "confidence": confidence})

        return {"structured_output": structured_output}

    return format_node
