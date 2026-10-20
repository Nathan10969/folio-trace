from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Protocol

from .config import Settings
from .contracts import NoTextError, Page, RLMQAError
from .ocr import GLMOCRClient


PAGE_MARKER_RE = re.compile(r"^\s*第\s*\d+\s*页")


class PDFOCRClient(Protocol):
    def extract_pdf_pages(self, path: Path, source_id: str, *, api_key: str) -> list[Page]:
        ...


def extract_pages(
    path: Path,
    source_id: str,
    settings: Settings,
    pdf_ocr: PDFOCRClient | None = None,
    *,
    ocr_api_key: str | None = None,
) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if not ocr_api_key:
            raise RLMQAError("PDF OCR must be called with a leased API key")
        return (pdf_ocr or GLMOCRClient(settings)).extract_pdf_pages(path, source_id, api_key=ocr_api_key)
    if suffix in {".txt", ".md", ".markdown"}:
        return _extract_text_pages(path, source_id, settings)
    raise ValueError(f"unsupported file type: {suffix}")


def _extract_text_pages(path: Path, source_id: str, settings: Settings) -> list[Page]:
    text = path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff")
    if not text.strip():
        raise NoTextError("document has no readable text")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and sum(1 for line in lines if PAGE_MARKER_RE.match(line)) >= 2:
        return _pages_from_segments(lines, source_id)

    segments = [
        text[index : index + settings.virtual_page_chars]
        for index in range(0, len(text), settings.virtual_page_chars)
    ]
    return _pages_from_segments(segments, source_id)


def _extract_pdf_pages(path: Path, source_id: str, settings: Settings) -> list[Page]:
    completed = subprocess.run(
        [settings.pdftotext_cmd, "-layout", "-enc", "UTF-8", str(path), "-"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise NoTextError(completed.stderr.strip() or "pdftotext failed")
    raw_pages = [page.strip() for page in completed.stdout.split("\f")]
    pages = [page for page in raw_pages if page]
    if not pages:
        raise NoTextError("document has no readable text")
    return _pages_from_segments(pages, source_id)


def _pages_from_segments(segments: list[str], source_id: str) -> list[Page]:
    pages: list[Page] = []
    cursor = 0
    for index, segment in enumerate(segments, start=1):
        text = segment.strip()
        if not text:
            cursor += len(segment)
            continue
        start = cursor
        end = cursor + len(segment)
        pages.append(Page(source_id=source_id, page_number=index, text=text, char_start=start, char_end=end))
        cursor = end
    if not pages:
        raise NoTextError("document has no readable text")
    return pages
