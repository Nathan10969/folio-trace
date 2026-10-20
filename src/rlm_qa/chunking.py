from __future__ import annotations

from .config import Settings
from .contracts import Chunk, DocProfile, Evidence, Page
from .ids import new_id


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)


def build_chunks(pages: list[Page], settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        text = page.text
        start = 0
        while start < len(text):
            end = min(len(text), start + settings.chunk_chars)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=new_id("chk"),
                        source_id=page.source_id,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        text=chunk_text,
                    )
                )
            if end >= len(text):
                break
            start = max(end - settings.chunk_overlap_chars, start + 1)
    return chunks


def build_evidence(chunks: list[Chunk]) -> tuple[list[Chunk], list[Evidence]]:
    evidences: list[Evidence] = []
    hydrated_chunks: list[Chunk] = []
    for chunk in chunks:
        quote = first_quote(chunk.text)
        evidence = Evidence(
            evidence_id=new_id("ev"),
            source_id=chunk.source_id,
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_start,
            quote=quote,
            char_start=0,
            char_end=len(quote),
        )
        evidences.append(evidence)
        hydrated_chunks.append(
            Chunk(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                evidence_ids=[evidence.evidence_id],
            )
        )
    return hydrated_chunks, evidences


def build_profile(file_name: str, source_id: str, pages: list[Page], chunks: list[Chunk], evidences: list[Evidence], settings: Settings) -> DocProfile:
    page_count = len(pages)
    token_estimate = sum(estimate_tokens(page.text) for page in pages)
    evidence_count = len(evidences)
    direct_ready = page_count <= settings.direct_max_pages and token_estimate <= settings.direct_max_tokens and evidence_count > 0
    rlm_ready = len(chunks) > 0 and evidence_count > 0
    status = "direct_ready" if direct_ready else "rlm_ready" if rlm_ready else "no_text"
    return DocProfile(
        source_id=source_id,
        file_name=file_name,
        page_count=page_count,
        token_estimate=token_estimate,
        chunk_count=len(chunks),
        evidence_count=evidence_count,
        direct_ready=direct_ready,
        rlm_ready=rlm_ready,
        status=status,
    )


def no_text_profile(
    file_name: str,
    source_id: str,
    *,
    status: str = "no_text",
    error_code: str = "DOC_NO_TEXT",
) -> DocProfile:
    return DocProfile(
        source_id=source_id,
        file_name=file_name,
        page_count=0,
        token_estimate=0,
        chunk_count=0,
        evidence_count=0,
        direct_ready=False,
        rlm_ready=False,
        status=status,
        error_code=error_code,
    )


def first_quote(text: str) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:280]
