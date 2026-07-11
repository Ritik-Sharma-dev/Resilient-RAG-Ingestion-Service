from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_embedder, get_llm_client, get_registry, get_vector_store
from app.store.ingestion_registry import IngestionRegistry

SAMPLE_INVOICE_PATH = Path(__file__).parent.parent / "wordpress-pdf-invoice-plugin-sample.pdf"


@pytest.fixture
def sample_invoice_bytes() -> bytes:
    return SAMPLE_INVOICE_PATH.read_bytes()


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 3072
    embedder.embed_batch.side_effect = lambda texts: [[0.1] * 3072 for _ in texts]
    return embedder


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.query.return_value = []
    return store


@pytest.fixture
def mock_llm_client():
    return MagicMock()


@pytest.fixture
def temp_registry(tmp_path) -> IngestionRegistry:
    return IngestionRegistry(str(tmp_path / "test_registry.db"))


@pytest.fixture
def client(mock_embedder, mock_vector_store, mock_llm_client, temp_registry):
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client
    app.dependency_overrides[get_registry] = lambda: temp_registry

    yield TestClient(app)

    app.dependency_overrides.clear()
