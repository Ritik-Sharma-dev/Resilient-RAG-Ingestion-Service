from typing import Literal

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    status: Literal["ingested", "already_ingested"]
    doc_id: str
    chunk_count: int = 0
    message: str = ""


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None


class SourceChunk(BaseModel):
    doc_id: str
    chunk_id: str
    chunk_type: Literal["prose", "table"]
    text: str
    score: float


class StructuredOutput(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: Literal["low", "medium", "high"]


class DocumentRecord(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    ingested_at: str
