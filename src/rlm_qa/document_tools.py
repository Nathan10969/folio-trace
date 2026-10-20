from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from .contracts import Chunk, DocProfile, DocumentPack, Evidence, Page
from .tools import search_context as keyword_search_context


DEFAULT_PACK_TARGET_TOKENS = 80_000
DEFAULT_PACK_HARD_MAX_TOKENS = 100_000


def make_document_context(
    profile: DocProfile,
    pages: list[Page],
    chunks: list[Chunk],
    evidences: list[Evidence],
    *,
    question: str = "",
) -> dict[str, Any]:
    return make_multi_document_context([profile], pages, chunks, evidences, question=question)


def make_multi_document_context(
    profiles: list[DocProfile],
    pages: list[Page],
    chunks: list[Chunk],
    evidences: list[Evidence],
    *,
    question: str = "",
) -> dict[str, Any]:
    profile_rows = [asdict(profile) for profile in profiles]
    page_rows = [asdict(page) for page in pages]
    chunk_rows = [asdict(chunk) for chunk in chunks]
    evidence_rows = [asdict(evidence) for evidence in evidences]
    profiles_by_source = {profile.source_id: profile for profile in profiles}
    packs: list[DocumentPack] = []
    for profile in profiles:
        source_pages = [page for page in pages if page.source_id == profile.source_id]
        source_chunks = [chunk for chunk in chunks if chunk.source_id == profile.source_id]
        source_evidences = [evidence for evidence in evidences if evidence.source_id == profile.source_id]
        packs.extend(build_document_packs(profile, source_pages, source_chunks, source_evidences))
    pack_rows = [_pack_row(pack, profiles_by_source) for pack in packs]
    return {
        "question": question,
        "doc_profile": profile_rows[0] if profile_rows else {},
        "doc_profiles": profile_rows,
        "doc_profiles_by_source": {str(profile["source_id"]): profile for profile in profile_rows},
        "pack_policy": {
            "target_tokens": DEFAULT_PACK_TARGET_TOKENS,
            "hard_max_tokens": DEFAULT_PACK_HARD_MAX_TOKENS,
            "semantic_unit": "macro_pack_for_sub_llm",
            "citation_unit": "evidence_span",
        },
        "pages": page_rows,
        "pages_by_number": {str(page["page_number"]): page for page in page_rows},
        "chunks": chunk_rows,
        "chunks_by_id": {str(chunk["chunk_id"]): chunk for chunk in chunk_rows},
        "evidence": evidence_rows,
        "evidence_by_id": {str(evidence["evidence_id"]): evidence for evidence in evidence_rows},
        "packs": pack_rows,
        "packs_by_id": {str(pack["pack_id"]): pack for pack in pack_rows},
    }


def make_document_tools(
    profile: DocProfile,
    pages: list[Page],
    chunks: list[Chunk],
    evidences: list[Evidence],
) -> dict[str, dict[str, Any]]:
    return make_multi_document_tools([profile], pages, chunks, evidences)


def make_multi_document_tools(
    profiles: list[DocProfile],
    pages: list[Page],
    chunks: list[Chunk],
    evidences: list[Evidence],
) -> dict[str, dict[str, Any]]:
    profiles_by_source = {profile.source_id: profile for profile in profiles}
    pages_by_number = {page.page_number: page for page in pages}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    evidences_by_id = {evidence.evidence_id: evidence for evidence in evidences}
    packs: list[DocumentPack] = []
    for profile in profiles:
        source_pages = [page for page in pages if page.source_id == profile.source_id]
        source_chunks = [chunk for chunk in chunks if chunk.source_id == profile.source_id]
        source_evidences = [evidence for evidence in evidences if evidence.source_id == profile.source_id]
        packs.extend(build_document_packs(profile, source_pages, source_chunks, source_evidences))
    packs_by_id = {pack.pack_id: pack for pack in packs}

    def read_pages(page_numbers: list[int | str]) -> dict[str, dict[str, Any]]:
        rows = {}
        for page_number in page_numbers:
            page = pages_by_number.get(int(page_number))
            if page is not None:
                rows[str(page.page_number)] = asdict(page)
        return rows

    def read_chunks(chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        rows = {}
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(str(chunk_id))
            if chunk is not None:
                rows[str(chunk.chunk_id)] = asdict(chunk)
        return rows

    def lookup_evidence(evidence_ids: list[str]) -> dict[str, dict[str, Any]]:
        rows = {}
        for evidence_id in evidence_ids:
            evidence = evidences_by_id.get(str(evidence_id))
            if evidence is not None:
                row = asdict(evidence)
                row["file_name"] = profiles_by_source[evidence.source_id].file_name
                rows[str(evidence.evidence_id)] = row
        return rows

    def search_context(query: str, top_k: int = 8) -> list[dict[str, Any]]:
        limit = max(1, min(int(top_k), 20))
        hits = keyword_search_context(chunks, query, limit=limit)
        if not hits:
            hits = _fallback_token_search(chunks, query, limit=limit)
        rows = []
        for hit in hits:
            row = asdict(hit.chunk)
            row["score"] = hit.score
            rows.append(row)
        return rows

    def list_packs() -> list[dict[str, Any]]:
        return [_pack_row(pack, profiles_by_source, include_text=False) for pack in packs]

    def read_packs(pack_ids: list[str]) -> dict[str, dict[str, Any]]:
        rows = {}
        for pack_id in pack_ids:
            pack = packs_by_id.get(str(pack_id))
            if pack is not None:
                rows[pack.pack_id] = _pack_row(pack, profiles_by_source, include_text=True)
        return rows

    def expand_evidence(evidence_ids: list[str], window_chars: int = 4000) -> dict[str, dict[str, Any]]:
        limit = max(500, min(int(window_chars), 12_000))
        rows = {}
        for evidence_id in evidence_ids:
            evidence = evidences_by_id.get(str(evidence_id))
            if evidence is None:
                continue
            chunk = chunks_by_id.get(evidence.chunk_id)
            text = chunk.text if chunk is not None else evidence.quote
            rows[evidence.evidence_id] = {
                "evidence_id": evidence.evidence_id,
                "source_id": evidence.source_id,
                "file_name": profiles_by_source[evidence.source_id].file_name,
                "chunk_id": evidence.chunk_id,
                "page_number": evidence.page_number,
                "quote": evidence.quote,
                "text": text[:limit],
            }
        return rows

    return {
        "list_packs": {
            "tool": list_packs,
            "description": "Return metadata for macro document packs. Use these packs as the normal unit for sub-LLM reading.",
        },
        "read_packs": {
            "tool": read_packs,
            "description": "Return full macro-pack text for requested pack ids. Feed pack text to llm_query or llm_query_batched.",
        },
        "read_pages": {
            "tool": read_pages,
            "description": "Return full page records for requested PDF page numbers.",
        },
        "read_chunks": {
            "tool": read_chunks,
            "description": "Return chunk records for requested chunk ids.",
        },
        "lookup_evidence": {
            "tool": lookup_evidence,
            "description": "Return verified evidence records for requested evidence ids.",
        },
        "search_context": {
            "tool": search_context,
            "description": "Cheap lexical chunk search for narrowing or citation recovery only; do not use as the main long-document reading strategy.",
        },
        "expand_evidence": {
            "tool": expand_evidence,
            "description": "Return expanded chunk text around verified evidence ids for final grounding and citation checks.",
        },
    }


def _pack_row(
    pack: DocumentPack,
    profiles_by_source: dict[str, DocProfile],
    *,
    include_text: bool = True,
) -> dict[str, Any]:
    profile = profiles_by_source.get(pack.source_id)
    document_page_count = profile.page_count if profile is not None else pack.page_end - pack.page_start + 1
    row = {
        "pack_id": pack.pack_id,
        "source_id": pack.source_id,
        "doc_source": pack.source_id,
        "file_name": pack.file_name,
        "page_start": pack.page_start,
        "page_end": pack.page_end,
        "page_range": f"{pack.page_start}-{pack.page_end}",
        "page_count": document_page_count,
        "doc_page_count": document_page_count,
        "pack_page_count": pack.page_end - pack.page_start + 1,
        "token_estimate": pack.token_estimate,
        "chunk_count": len(pack.chunk_ids),
        "evidence_count": len(pack.evidence_ids),
        "chunk_ids": pack.chunk_ids,
        "evidence_ids": pack.evidence_ids,
    }
    if include_text:
        row["text"] = pack.text
    return row


def build_document_packs(
    profile: DocProfile,
    pages: list[Page],
    chunks: list[Chunk],
    evidences: list[Evidence],
    *,
    target_tokens: int = DEFAULT_PACK_TARGET_TOKENS,
    hard_max_tokens: int = DEFAULT_PACK_HARD_MAX_TOKENS,
) -> list[DocumentPack]:
    del pages
    if not chunks:
        return []
    target = max(1, int(target_tokens))
    hard_max = max(target, int(hard_max_tokens))
    evidence_by_chunk_id = {evidence.chunk_id: evidence.evidence_id for evidence in evidences}
    ordered_chunks = sorted(chunks, key=lambda chunk: (chunk.page_start, chunk.page_end, chunk.chunk_id))
    packs: list[DocumentPack] = []
    current: list[Chunk] = []
    current_tokens = 0

    for chunk in ordered_chunks:
        chunk_tokens = _estimate_tokens(chunk.text)
        should_flush = bool(current) and (
            current_tokens >= target or current_tokens + chunk_tokens > hard_max
        )
        if should_flush:
            packs.append(_make_pack(profile, current, evidence_by_chunk_id, len(packs) + 1))
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += chunk_tokens

    if current:
        packs.append(_make_pack(profile, current, evidence_by_chunk_id, len(packs) + 1))
    return packs


def _make_pack(
    profile: DocProfile,
    chunks: list[Chunk],
    evidence_by_chunk_id: dict[str, str],
    index: int,
) -> DocumentPack:
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    pack_id = f"pack_{profile.source_id}_{index:04d}"
    evidence_ids = [
        evidence_id
        for chunk in chunks
        for evidence_id in chunk.evidence_ids or ([evidence_by_chunk_id[chunk.chunk_id]] if chunk.chunk_id in evidence_by_chunk_id else [])
    ]
    page_start = min(chunk.page_start for chunk in chunks)
    page_end = max(chunk.page_end for chunk in chunks)
    body_parts = [f"PACK_ID={pack_id}\n[{profile.file_name} | pages {page_start}-{page_end}]"]
    for chunk in chunks:
        ids = chunk.evidence_ids or ([evidence_by_chunk_id[chunk.chunk_id]] if chunk.chunk_id in evidence_by_chunk_id else [])
        body_parts.append(
            f"[chunk {chunk.chunk_id} | pages {chunk.page_start}-{chunk.page_end} | evidence {', '.join(ids)}]\n{chunk.text}"
        )
    return DocumentPack(
        pack_id=pack_id,
        source_id=profile.source_id,
        file_name=profile.file_name,
        page_start=page_start,
        page_end=page_end,
        token_estimate=sum(_estimate_tokens(chunk.text) for chunk in chunks),
        chunk_ids=chunk_ids,
        evidence_ids=evidence_ids,
        text="\n\n".join(body_parts),
    )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)


def _fallback_token_search(chunks: list[Chunk], query: str, *, limit: int):
    from .tools import SearchHit

    tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", query)]
    scored = []
    for chunk in chunks:
        haystack = chunk.text.lower()
        score = sum(1 for token in tokens if token and token in haystack)
        if score:
            scored.append(SearchHit(chunk=chunk, score=score))
    scored.sort(key=lambda hit: (-hit.score, hit.chunk.page_start, hit.chunk.chunk_id))
    return scored[:limit]
