import hashlib
import re


def normalize_document_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def compute_doc_hash(text: str) -> str:
    normalized = normalize_document_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
