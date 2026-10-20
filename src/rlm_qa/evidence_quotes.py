from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .config import Settings
from .contracts import Chunk, DocProfile, EvidenceRef, Page


def normalize_evidence_refs(
    refs: list[EvidenceRef],
    chunks: list[Chunk],
    pages: list[Page],
    *,
    question: str,
    answer: str,
    max_quote_chars: int = 600,
) -> list[EvidenceRef]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    pages_by_key = {(page.source_id, page.page_number): page for page in pages}
    query_terms = _terms(f"{question}\n{answer}")
    normalized: list[EvidenceRef] = []
    for ref in refs:
        chunk = chunks_by_id.get(ref.chunk_id)
        if chunk is not None and chunk.text.strip():
            best = _best_sentence_span(chunk.text, query_terms)
            if best is not None:
                start, end, quote = best
                normalized.append(
                    replace(
                        ref,
                        quote=_clip_quote(quote, max_quote_chars),
                        quote_source="exact_span",
                        char_start=start,
                        char_end=end,
                    )
                )
                continue
            paragraph = _first_paragraph(chunk.text)
            if paragraph:
                start = chunk.text.find(paragraph)
                normalized.append(
                    replace(
                        ref,
                        quote=_clip_quote(paragraph, max_quote_chars),
                        quote_source="paragraph_fallback",
                        char_start=max(0, start),
                        char_end=max(0, start) + len(paragraph),
                    )
                )
                continue

        page = pages_by_key.get((ref.source_id, ref.page_number))
        if page is not None and page.text.strip():
            quote = _clip_quote(_first_paragraph(page.text) or page.text.strip(), max_quote_chars)
            normalized.append(
                replace(
                    ref,
                    quote=quote,
                    quote_source="page_fallback",
                    char_start=page.char_start,
                    char_end=page.char_start + len(quote),
                )
            )
            continue

        normalized.append(
            replace(
                ref,
                quote=_clip_quote(ref.quote, max_quote_chars),
                quote_source=ref.quote_source or "paragraph_fallback",
            )
        )
    return normalized


def select_display_evidence_refs(
    refs: list[EvidenceRef],
    *,
    question: str,
    profiles: list[DocProfile],
    settings: Settings,
) -> tuple[list[EvidenceRef], dict[str, Any]]:
    policy = _display_policy(question, profiles)
    if policy == "multi_document_overview":
        selected = _take_per_source(
            refs,
            per_source=max(1, settings.rlm_evidence_multi_per_doc),
            global_limit=max(1, settings.rlm_evidence_multi_global),
        )
    else:
        limit = _policy_limit(policy, settings)
        selected = refs[:limit]
    return selected, {
        "policy": policy,
        "available_count": len(refs),
        "displayed_count": len(selected),
        "display_limit": len(selected) if policy == "multi_document_overview" else _policy_limit(policy, settings),
    }


def _display_policy(question: str, profiles: list[DocProfile]) -> str:
    normalized = question.strip().lower()
    if len(profiles) > 1 and _contains_any(question, normalized, _MULTI_TERMS) and _contains_any(
        question, normalized, _SUMMARY_TERMS
    ):
        return "multi_document_overview"
    if _contains_any(question, normalized, _SUMMARY_TERMS):
        return "summary"
    if _contains_any(question, normalized, _DETAIL_TERMS):
        return "detail"
    return "simple_fact"


def _policy_limit(policy: str, settings: Settings) -> int:
    if policy == "summary":
        return max(1, settings.rlm_evidence_summary_limit)
    if policy == "detail":
        return max(1, settings.rlm_evidence_detail_limit)
    if policy == "simple_fact":
        return max(1, settings.rlm_evidence_simple_limit)
    return max(1, settings.rlm_evidence_default_limit)


def _take_per_source(refs: list[EvidenceRef], *, per_source: int, global_limit: int) -> list[EvidenceRef]:
    counts: dict[str, int] = {}
    selected: list[EvidenceRef] = []
    for ref in refs:
        if len(selected) >= global_limit:
            break
        count = counts.get(ref.source_id, 0)
        if count >= per_source:
            continue
        selected.append(ref)
        counts[ref.source_id] = count + 1
    return selected


def _best_sentence_span(text: str, terms: set[str]) -> tuple[int, int, str] | None:
    best: tuple[int, int, str, int] | None = None
    for start, end, sentence in _sentence_spans(text):
        score = _score(sentence, terms)
        if score <= 0:
            continue
        if best is None or score > best[3] or (score == best[3] and len(sentence) < len(best[2])):
            best = (start, end, sentence, score)
    if best is None:
        return None
    return best[0], best[1], best[2]


def _sentence_spans(text: str):
    pattern = re.compile(r"[^。！？.!?\n]+[。！？.!?]?")
    for match in pattern.finditer(text):
        sentence = " ".join(match.group(0).split())
        if sentence:
            yield match.start(), match.end(), sentence


def _first_paragraph(text: str) -> str:
    for paragraph in re.split(r"\n\s*\n|\r\n\s*\r\n", text):
        cleaned = " ".join(paragraph.split())
        if cleaned:
            return cleaned
    return ""


def _terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text.lower()):
        if token not in _STOP_TERMS:
            terms.add(token)
    return terms


def _score(text: str, terms: set[str]) -> int:
    haystack = text.lower()
    return sum(1 for term in terms if term in haystack)


def _clip_quote(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(1, limit - 1)].rstrip() + "…"


def _contains_any(question: str, normalized: str, terms: tuple[str, ...]) -> bool:
    return any(term in question or term in normalized for term in terms)


_SUMMARY_TERMS = (
    "说了什么",
    "讲了什么",
    "总结",
    "概括",
    "大意",
    "主要内容",
    "overview",
    "summarize",
    "summary",
    "about",
)

_MULTI_TERMS = ("这些", "这几篇", "这几份", "多篇", "一篇一篇", "分别", "these", "multiple")

_DETAIL_TERMS = ("为什么", "如何", "怎么", "原因", "步骤", "机理", "对比", "比较", "why", "how", "compare")

_STOP_TERMS = {
    "what",
    "which",
    "when",
    "where",
    "this",
    "that",
    "these",
    "those",
    "with",
    "from",
    "after",
    "the",
    "and",
    "for",
    "are",
    "was",
    "were",
}
