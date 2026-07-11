from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_extractor import ExtractedPage, ExtractedTable


def test_table_to_text_serializes_rows_pipe_delimited():
    table = ExtractedTable(
        page_number=1,
        rows=[["Invoice Number", "INV-3337"], ["Total Due", "$93.50"]],
    )

    text = table.to_text()

    assert text == "Invoice Number | INV-3337\nTotal Due | $93.50"


def test_table_to_text_handles_none_cells():
    table = ExtractedTable(page_number=1, rows=[["Name", None, "Value"]])

    assert table.to_text() == "Name |  | Value"


def test_chunk_index_is_sequential_across_pages():
    pages = [
        ExtractedPage(page_number=1, prose_text="First page text.", tables=[]),
        ExtractedPage(page_number=2, prose_text="Second page text.", tables=[]),
    ]
    chunks = chunk_pages(pages)

    assert [c.chunk_index for c in chunks] == [0, 1]
    assert [c.page_number for c in chunks] == [1, 2]


def test_chunk_metadata_tags_table_vs_prose_correctly():
    table = ExtractedTable(page_number=1, rows=[["A", "B"]])
    page = ExtractedPage(page_number=1, prose_text="Some narrative text here.", tables=[table])

    chunks = chunk_pages([page])
    types = {c.chunk_type for c in chunks}

    assert types == {"prose", "table"}
    assert all(c.page_number == 1 for c in chunks)
