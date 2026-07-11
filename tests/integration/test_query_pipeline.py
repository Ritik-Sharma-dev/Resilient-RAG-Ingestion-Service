import json
from types import SimpleNamespace

from app.store.vector_store import VectorMatch


def _fake_llm_response(answer: str, confidence: str = "high"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"answer": answer, "confidence": confidence})))],
        usage=SimpleNamespace(prompt_tokens=42, completion_tokens=8, total_tokens=50),
    )


def test_query_runs_retrieve_generate_format_and_returns_structured_output(
    client, mock_embedder, mock_vector_store, mock_llm_client
):
    mock_vector_store.query.return_value = [
        VectorMatch(
            id="doc-1-0",
            score=0.92,
            metadata={"doc_id": "doc-1", "chunk_type": "prose", "text": "Web Design costs $85.00."},
        )
    ]
    mock_llm_client.chat.completions.create.return_value = _fake_llm_response(
        "The web design service costs $85.00."
    )

    response = client.post("/query", json={"question": "How much does web design cost?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The web design service costs $85.00."
    assert body["confidence"] == "high"
    assert len(body["sources"]) == 1
    assert body["sources"][0]["doc_id"] == "doc-1"

    mock_embedder.embed.assert_called_once_with("How much does web design cost?")
    mock_vector_store.query.assert_called_once()
    mock_llm_client.chat.completions.create.assert_called_once()


def test_query_falls_back_to_low_confidence_on_malformed_llm_json(
    client, mock_embedder, mock_vector_store, mock_llm_client
):
    mock_vector_store.query.return_value = []
    mock_llm_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not valid json"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    response = client.post("/query", json={"question": "Anything in the docs?"})

    assert response.status_code == 200
    body = response.json()
    assert body["confidence"] == "low"
    assert body["sources"] == []
