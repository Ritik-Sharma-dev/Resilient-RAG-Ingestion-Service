from app.ingestion.dedup import compute_doc_hash, normalize_document_text


def test_same_content_different_whitespace_produces_same_hash():
    text_a = "Hello   World.\n\nThis is a document."
    text_b = "Hello World. This is a document."

    assert compute_doc_hash(text_a) == compute_doc_hash(text_b)


def test_same_content_different_case_produces_same_hash():
    assert compute_doc_hash("Hello World") == compute_doc_hash("hello world")


def test_edited_content_produces_different_hash():
    original = "The quick brown fox jumps over the lazy dog."
    edited = "The quick brown fox jumps over the lazy dog, and then runs away."

    assert compute_doc_hash(original) != compute_doc_hash(edited)


def test_normalize_document_text_strips_and_lowercases():
    assert normalize_document_text("  Foo\tBAR  \n") == "foo bar"


def test_hash_is_deterministic():
    text = "Consistent input text."
    assert compute_doc_hash(text) == compute_doc_hash(text)
