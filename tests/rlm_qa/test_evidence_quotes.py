from __future__ import annotations

from rlm_qa.config import Settings
from rlm_qa.contracts import Chunk, DocProfile, EvidenceRef, Page
from rlm_qa.evidence_quotes import normalize_evidence_refs, select_display_evidence_refs


def test_quote_normalizer_extracts_answer_related_exact_span() -> None:
    refs = [
        EvidenceRef(
            evidence_id="ev_1",
            source_id="src_a",
            chunk_id="chk_1",
            page_number=2,
            quote="Background sentence.",
            file_name="paper.pdf",
        )
    ]
    chunks = [
        Chunk(
            chunk_id="chk_1",
            source_id="src_a",
            page_start=2,
            page_end=2,
            text=(
                "Background sentence. "
                "The coating adhesion improved after plasma activation and water wetting. "
                "Other details follow."
            ),
            evidence_ids=["ev_1"],
        )
    ]

    normalized = normalize_evidence_refs(
        refs,
        chunks,
        pages=[],
        question="What improved coating adhesion?",
        answer="The coating adhesion improved after plasma activation and water wetting.",
    )

    assert normalized[0].quote == "The coating adhesion improved after plasma activation and water wetting."
    assert normalized[0].quote_source == "exact_span"
    assert normalized[0].char_start > 0
    assert normalized[0].char_end > normalized[0].char_start


def test_quote_normalizer_uses_page_fallback_when_chunk_is_missing() -> None:
    refs = [
        EvidenceRef(
            evidence_id="ev_1",
            source_id="src_a",
            chunk_id="missing",
            page_number=3,
            quote="",
            file_name="paper.pdf",
        )
    ]
    pages = [Page(source_id="src_a", page_number=3, text="Page fallback text for the selected evidence.")]

    normalized = normalize_evidence_refs(refs, chunks=[], pages=pages, question="", answer="")

    assert normalized[0].quote == "Page fallback text for the selected evidence."
    assert normalized[0].quote_source == "page_fallback"


def test_display_evidence_limits_simple_fact_to_three_refs_without_changing_retrieval_pool() -> None:
    refs = [
        EvidenceRef(f"ev_{index}", "src_a", f"chk_{index}", index + 1, f"quote {index}", "a.pdf")
        for index in range(8)
    ]

    selected, policy = select_display_evidence_refs(
        refs,
        question="What is the coating material?",
        profiles=[DocProfile("src_a", "a.pdf", 8, 800, 8, 8, False, True, "rlm_ready")],
        settings=Settings(),
    )

    assert len(selected) == 3
    assert policy["available_count"] == 8
    assert policy["displayed_count"] == 3
    assert policy["policy"] == "simple_fact"


def test_display_evidence_limits_multi_document_overview_per_doc_and_global() -> None:
    refs = [
        EvidenceRef(f"ev_a_{index}", "src_a", f"chk_a_{index}", index + 1, f"a quote {index}", "a.pdf")
        for index in range(8)
    ] + [
        EvidenceRef(f"ev_b_{index}", "src_b", f"chk_b_{index}", index + 1, f"b quote {index}", "b.pdf")
        for index in range(8)
    ]

    selected, policy = select_display_evidence_refs(
        refs,
        question="这些文章说了什么？一篇一篇列出来给我",
        profiles=[
            DocProfile("src_a", "a.pdf", 8, 800, 8, 8, False, True, "rlm_ready"),
            DocProfile("src_b", "b.pdf", 8, 800, 8, 8, False, True, "rlm_ready"),
        ],
        settings=Settings(rlm_evidence_multi_per_doc=3, rlm_evidence_multi_global=20),
    )

    assert len(selected) == 6
    assert sum(1 for ref in selected if ref.source_id == "src_a") == 3
    assert sum(1 for ref in selected if ref.source_id == "src_b") == 3
    assert policy["policy"] == "multi_document_overview"
