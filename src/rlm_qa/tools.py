from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import Chunk, Evidence, EvidenceRef


SUMMARY_TERMS = (
    "总结",
    "概括",
    "讲什么",
    "讲了什么",
    "说了什么",
    "主要内容",
    "内容是什么",
    "主要讲",
    "主要说",
    "主要介绍",
    "这篇文档",
    "这份文档",
    "这篇文章",
    "这篇论文",
    "这篇专利",
    "这篇文献",
    "这个pdf",
    "这个PDF",
    "摘要",
)
OVERVIEW_SECTION_TERMS = (
    "abstract",
    "summary",
    "field",
    "background",
    "detailed description",
    "claims",
    "(54) title",
    "title:",
)
STOP_TERMS = {"什么", "怎么", "应该", "请问", "请", "的是", "一个", "一种", "材料"}


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: int


def search_context(chunks: list[Chunk], question: str, *, limit: int = 8) -> list[SearchHit]:
    if is_summary_question(question):
        return _summary_hits(chunks, limit=limit)

    terms = query_terms(question)
    hits: list[SearchHit] = []
    for chunk in chunks:
        haystack = chunk.text.lower()
        score = sum(1 for term in terms if term and term.lower() in haystack)
        if score > 0:
            hits.append(SearchHit(chunk=chunk, score=score))
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:limit]


def evidence_refs_for_hits(hits: list[SearchHit], evidences: list[Evidence], file_name: str, *, limit: int = 5) -> list[EvidenceRef]:
    evidence_by_id = {evidence.evidence_id: evidence for evidence in evidences}
    refs: list[EvidenceRef] = []
    seen: set[str] = set()
    for hit in hits:
        for evidence_id in hit.chunk.evidence_ids:
            if evidence_id in seen:
                continue
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            refs.append(
                EvidenceRef(
                    evidence_id=evidence.evidence_id,
                    source_id=evidence.source_id,
                    chunk_id=evidence.chunk_id,
                    page_number=evidence.page_number,
                    quote=evidence.quote,
                    file_name=file_name,
                )
            )
            seen.add(evidence_id)
            if len(refs) >= limit:
                return refs
    return refs


def is_summary_question(question: str) -> bool:
    return any(term in question for term in SUMMARY_TERMS)


def _summary_hits(chunks: list[Chunk], *, limit: int) -> list[SearchHit]:
    selected: dict[str, SearchHit] = {}

    def add(chunk: Chunk, score: int) -> None:
        current = selected.get(chunk.chunk_id)
        if current is None or score > current.score:
            selected[chunk.chunk_id] = SearchHit(chunk=chunk, score=score)

    for term in OVERVIEW_SECTION_TERMS:
        found_for_term = 0
        for chunk in chunks:
            if term in chunk.text.lower():
                add(chunk, 4)
                found_for_term += 1
                if found_for_term >= 2:
                    break

    for chunk in chunks[:3]:
        add(chunk, 2)

    if chunks:
        sample_indexes = {0, len(chunks) - 1}
        if len(chunks) >= 4:
            sample_indexes.update({len(chunks) // 4, len(chunks) // 2, (len(chunks) * 3) // 4})
        for index in sorted(sample_indexes):
            add(chunks[index], 1)

    return sorted(selected.values(), key=lambda hit: (-hit.score, hit.chunk.page_start, hit.chunk.chunk_id))[:limit]


def query_terms(question: str) -> list[str]:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", question)
    terms = set()
    for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", normalized):
        if token in STOP_TERMS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            for size in range(2, min(8, len(token)) + 1):
                for index in range(0, len(token) - size + 1):
                    piece = token[index : index + size]
                    if piece not in STOP_TERMS:
                        terms.add(piece)
        else:
            terms.add(token.lower())
    return sorted(terms, key=len, reverse=True)
