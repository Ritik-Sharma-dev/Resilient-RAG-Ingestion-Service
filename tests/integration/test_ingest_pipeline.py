from pathlib import Path

ARXIV_SAMPLE_PATH = Path(__file__).parent.parent.parent / "2601.00033v1.pdf"


def test_ingest_new_document_embeds_and_upserts(client, mock_embedder, mock_vector_store, sample_invoice_bytes):
    response = client.post(
        "/ingest",
        files={"file": ("invoice.pdf", sample_invoice_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ingested"
    assert body["chunk_count"] > 0

    mock_embedder.embed_batch.assert_called_once()
    mock_vector_store.upsert.assert_called_once()
    upserted_records = mock_vector_store.upsert.call_args.args[0]
    assert len(upserted_records) == body["chunk_count"]
    # table content (line item) should survive as its own chunk, tagged correctly
    assert any(r.metadata["chunk_type"] == "table" for r in upserted_records)


def test_ingest_duplicate_document_skips_reembedding(client, mock_embedder, mock_vector_store, sample_invoice_bytes):
    first = client.post(
        "/ingest", files={"file": ("invoice.pdf", sample_invoice_bytes, "application/pdf")}
    ).json()

    second = client.post(
        "/ingest", files={"file": ("invoice-renamed.pdf", sample_invoice_bytes, "application/pdf")}
    ).json()

    assert second["status"] == "already_ingested"
    assert second["doc_id"] == first["doc_id"]
    # dedup check must happen before any embedding work
    mock_embedder.embed_batch.assert_called_once()
    mock_vector_store.upsert.assert_called_once()


def test_ingest_different_document_is_treated_as_new(client, mock_embedder, sample_invoice_bytes):
    first = client.post(
        "/ingest", files={"file": ("invoice.pdf", sample_invoice_bytes, "application/pdf")}
    ).json()

    second = client.post(
        "/ingest",
        files={"file": ("paper.pdf", ARXIV_SAMPLE_PATH.read_bytes(), "application/pdf")},
    ).json()

    assert second["status"] == "ingested"
    assert second["doc_id"] != first["doc_id"]
    assert mock_embedder.embed_batch.call_count == 2


def test_list_documents_returns_all_ingested_docs(client, sample_invoice_bytes):
    empty = client.get("/documents").json()
    assert empty == []

    client.post("/ingest", files={"file": ("invoice.pdf", sample_invoice_bytes, "application/pdf")})
    client.post(
        "/ingest",
        files={"file": ("paper.pdf", ARXIV_SAMPLE_PATH.read_bytes(), "application/pdf")},
    )

    documents = client.get("/documents").json()

    assert len(documents) == 2
    filenames = {d["filename"] for d in documents}
    assert filenames == {"invoice.pdf", "paper.pdf"}
    assert all(d["chunk_count"] > 0 for d in documents)
