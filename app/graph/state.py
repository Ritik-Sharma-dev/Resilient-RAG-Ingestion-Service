from typing import TypedDict

from app.store.vector_store import VectorMatch


class GraphState(TypedDict, total=False):
    question: str
    top_k: int
    retrieved_chunks: list[VectorMatch]
    raw_answer: str
    token_usage: dict
    structured_output: dict
