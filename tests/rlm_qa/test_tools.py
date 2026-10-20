from __future__ import annotations

from rlm_qa.contracts import Chunk
from rlm_qa.tools import search_context


def test_article_overview_search_prioritizes_abstract_and_summary_over_cover_noise() -> None:
    chunks = [
        Chunk(chunk_id=f"cover_{index}", source_id="src", page_start=1, page_end=1, text=f"PCT cover metadata {index}")
        for index in range(5)
    ]
    chunks.extend(
        [
            Chunk(
                chunk_id="abstract",
                source_id="src",
                page_start=1,
                page_end=1,
                text="ABSTRACT This invention provides oxazolidone functional compounds for coating compositions.",
            ),
            Chunk(
                chunk_id="summary",
                source_id="src",
                page_start=2,
                page_end=2,
                text="SUMMARY The coating compositions may include resins, additives, pigments, or fillers.",
            ),
        ]
    )

    hits = search_context(chunks, "这篇文章说了什么？", limit=5)

    assert [hit.chunk.chunk_id for hit in hits[:2]] == ["abstract", "summary"]
