from unittest.mock import patch

from app.ingestion.embedder import EMBED_BATCH_SIZE, Embedder


def _fake_openai_response(n: int):
    from types import SimpleNamespace

    return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * 3) for _ in range(n)])


def test_embed_batch_splits_large_input_into_multiple_openai_calls():
    texts = [f"chunk {i}" for i in range(250)]  # > EMBED_BATCH_SIZE (100)
    embedder = Embedder()

    with patch("app.ingestion.embedder._client") as mock_client:
        mock_client.embeddings.create.side_effect = lambda **kwargs: _fake_openai_response(len(kwargs["input"]))

        result = embedder.embed_batch(texts)

    assert len(result) == 250
    assert mock_client.embeddings.create.call_count == 3  # 100 + 100 + 50
    call_sizes = [len(call.kwargs["input"]) for call in mock_client.embeddings.create.call_args_list]
    assert call_sizes == [100, 100, 50]


def test_embed_batch_single_call_when_under_batch_size():
    texts = [f"chunk {i}" for i in range(10)]
    embedder = Embedder()

    with patch("app.ingestion.embedder._client") as mock_client:
        mock_client.embeddings.create.side_effect = lambda **kwargs: _fake_openai_response(len(kwargs["input"]))

        result = embedder.embed_batch(texts)

    assert len(result) == 10
    assert mock_client.embeddings.create.call_count == 1


def test_embed_batch_size_constant_is_conservative():
    # OpenAI's embeddings endpoint caps requests at 2048 inputs; our batch
    # size must stay comfortably under that.
    assert EMBED_BATCH_SIZE < 2048
