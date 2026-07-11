import re
from dataclasses import dataclass
from typing import Literal

from app.ingestion.pdf_extractor import ExtractedPage


@dataclass
class Chunk:
    text: str
    chunk_type: Literal["prose", "table"]
    page_number: int
    chunk_index: int


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_pages(
    pages: list[ExtractedPage],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    for page in pages:
        prose = normalize_text(page.prose_text)
        if prose:
            for piece in _split_fixed_size(prose, chunk_size, chunk_overlap):
                chunks.append(
                    Chunk(
                        text=piece,
                        chunk_type="prose",
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

        for table in page.tables:
            table_text = table.to_text()
            if table_text.strip():
                chunks.append(
                    Chunk(
                        text=table_text,
                        chunk_type="table",
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

    return chunks


def _split_fixed_size(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    pieces = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = start + chunk_size
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return pieces
