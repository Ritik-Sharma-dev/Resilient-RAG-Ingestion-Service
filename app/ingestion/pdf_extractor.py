import io
from dataclasses import dataclass

import pdfplumber


@dataclass
class ExtractedTable:
    page_number: int
    rows: list[list[str | None]]

    def to_text(self) -> str:
        lines = []
        for row in self.rows:
            cells = [cell.strip() if cell else "" for cell in row]
            lines.append(" | ".join(cells))
        return "\n".join(lines)


@dataclass
class ExtractedPage:
    page_number: int
    prose_text: str
    tables: list[ExtractedTable]


def extract_pdf(file_bytes: bytes) -> list[ExtractedPage]:
    pages: list[ExtractedPage] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            raw_tables = page.extract_tables()
            tables = [
                ExtractedTable(page_number=page_number, rows=rows)
                for rows in raw_tables
                if rows
            ]

            table_bboxes = [t.bbox for t in page.find_tables()]
            prose_text = _extract_prose_excluding_tables(page, table_bboxes)

            pages.append(
                ExtractedPage(page_number=page_number, prose_text=prose_text, tables=tables)
            )
    return pages


def _extract_prose_excluding_tables(page, table_bboxes) -> str:
    if not table_bboxes:
        return page.extract_text() or ""

    filtered = page
    for bbox in table_bboxes:
        filtered = filtered.outside_bbox(bbox)
    return filtered.extract_text() or ""
