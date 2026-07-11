import logging

from openai import OpenAI

from app.config import settings
from app.resilience.retry import embedding_retry

logger = logging.getLogger("app.ingestion")

_client = OpenAI(api_key=settings.openai_api_key)

# OpenAI's embeddings endpoint caps a single request at 2048 inputs and a
# per-request token budget; batch well under both so a large PDF (hundreds
# of chunks) never produces an oversized request, mirroring the Pinecone
# upsert batching in vector_store.py.
EMBED_BATCH_SIZE = 100


class Embedder:
    """Thin wrapper around the OpenAI embeddings call, isolated from the
    vector store so retry/circuit-breaker policy for embedding failures
    stays independent of vector DB failure policy."""

    def __init__(self, model: str = settings.embedding_model, dimensions: int = settings.pinecone_embedding_dimension):
        self.model = model
        self.dimensions = dimensions

    @embedding_retry
    def embed(self, text: str) -> list[float]:
        response = _client.embeddings.create(model=self.model, input=text, dimensions=self.dimensions)
        return response.data[0].embedding

    @embedding_retry
    def _embed_batch_call(self, texts: list[str]) -> list[list[float]]:
        response = _client.embeddings.create(model=self.model, input=texts, dimensions=self.dimensions)
        return [item.embedding for item in response.data]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            logger.info("Embedding batch %d-%d of %d chunks", i, i + len(batch), len(texts))
            all_embeddings.extend(self._embed_batch_call(batch))
        return all_embeddings
