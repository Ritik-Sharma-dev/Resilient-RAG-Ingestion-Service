from app.ingestion.chunker import chunk_pages, normalize_text
from app.ingestion.pdf_extractor import ExtractedPage, ExtractedTable


def test_normalize_text_collapses_whitespace():
    assert normalize_text("hello   \n\n world  \t!") == "hello world !"


def test_short_prose_yields_single_chunk():
    page = ExtractedPage(page_number=1, prose_text="short paragraph.", tables=[])
    chunks = chunk_pages([page], chunk_size=1000, chunk_overlap=150)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "prose"
    assert chunks[0].text == "short paragraph."


def test_long_prose_splits_with_overlap():
    long_text = "word " * 500  # well over chunk_size
    page = ExtractedPage(page_number=1, prose_text=long_text, tables=[])
    chunks = chunk_pages([page], chunk_size=1000, chunk_overlap=150)

    assert len(chunks) > 1
    assert all(c.chunk_type == "prose" for c in chunks)
    # consecutive chunks overlap
    overlap_region = chunks[0].text[-150:]
    assert overlap_region in chunks[1].text


def test_table_produces_separate_chunk_not_mangled_with_prose():
    table = ExtractedTable(
        page_number=1,
        rows=[["Hrs/Qty", "Service", "Rate"], ["1.00", "Web Design", "$85.00"]],
    )
    page = ExtractedPage(page_number=1, prose_text="Invoice from Acme Corp.", tables=[table])
    chunks = chunk_pages([page], chunk_size=1000, chunk_overlap=150)

    prose_chunks = [c for c in chunks if c.chunk_type == "prose"]
    table_chunks = [c for c in chunks if c.chunk_type == "table"]

    assert len(prose_chunks) == 1
    assert len(table_chunks) == 1
    assert "Web Design" in table_chunks[0].text
    assert "Web Design" not in prose_chunks[0].text


def test_empty_page_yields_no_chunks():
    page = ExtractedPage(page_number=1, prose_text="   ", tables=[])
    chunks = chunk_pages([page])

    assert chunks == []
