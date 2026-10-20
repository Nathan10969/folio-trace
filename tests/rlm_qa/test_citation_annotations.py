from __future__ import annotations

from dataclasses import replace
import json

import rlm_qa.citation_annotations as citation_annotations
from rlm_qa.citation_annotations import (
    CitationVerifier,
    ReferenceMarkerStreamFilter,
    append_reference_markers,
    public_annotations,
    strip_reference_markers,
)
from rlm_qa.contracts import Chunk, EvidenceRef
from rlm_qa.llm import ChatRequest, ChatResponse, LLMClient


class VerifierLLM(LLMClient):
    def __init__(self, response: object = None, *, error: Exception | None = None) -> None:
        self.response = response if response is not None else {"citations": []}
        self.error = error
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if isinstance(self.response, str):
            return ChatResponse(answer=self.response)
        return ChatResponse(answer=json.dumps(self.response))


class MetadataAwareVerifierLLM(LLMClient):
    def __init__(self) -> None:
        self.candidates: list[dict[str, object]] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        payload = json.loads(request.user)
        self.candidates = payload["candidates"]
        selected = next(candidate for candidate in self.candidates if candidate["file_name"] == "beta.pdf")
        return ChatResponse(
            answer=json.dumps(
                {
                    "citations": [
                        {
                            "evidence_id": selected["evidence_id"],
                            "score": 0.93,
                            "quote": "Shared supporting sentence.",
                        }
                    ]
                }
            )
        )


def _ref(
    evidence_id: str,
    *,
    source_id: str = "src_a",
    chunk_id: str = "chk_a",
    page: int = 4,
    quote: str = "Exact source sentence.",
    file_name: str = "paper.pdf",
    doc_id: str = "501",
) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        source_id=source_id,
        chunk_id=chunk_id,
        page_number=page,
        quote=quote,
        file_name=file_name,
        quote_source="exact_span",
        doc_id=doc_id,
        page_end=page,
    )


def _chunk(
    chunk_id: str = "chk_a",
    *,
    source_id: str = "src_a",
    page: int = 4,
    text: str = "Introduction. Exact source sentence. Additional source detail.",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_id=source_id,
        page_start=page,
        page_end=page,
        text=text,
        evidence_ids=["ev_a"],
    )


def test_public_annotations_use_bound_platform_metadata_and_exact_frontend_shape() -> None:
    ref = EvidenceRef(
        evidence_id="ev_internal",
        source_id="src_a",
        chunk_id="chk_a",
        page_number=4,
        quote="Exact source sentence.",
        file_name="paper.pdf",
        quote_source="exact_span",
        doc_id="501",
        page_end=4,
        ref_id="ref_001",
        score=0.94,
    )

    assert public_annotations([ref]) == [
        {
            "id": "ref_001",
            "sourceType": "pdf",
            "docId": "501",
            "fileName": "paper.pdf",
            "page": 4,
            "endPage": 4,
            "displayText": "paper.pdf[4]",
            "quote": "Exact source sentence.",
            "score": 0.94,
        }
    ]


def test_public_annotations_format_inclusive_page_range() -> None:
    ranged = EvidenceRef(
        evidence_id="ev_range",
        source_id="src_a",
        chunk_id="chk_range",
        page_number=7,
        quote="Range quote.",
        file_name="range.pdf",
        doc_id="777",
        page_end=9,
        ref_id="ref_001",
        score=0.5,
    )

    assert public_annotations([ranged])[0]["displayText"] == "range.pdf[7-9]"


def test_public_annotations_keep_missing_support_score_null() -> None:
    unscored = EvidenceRef(
        evidence_id="ev_unscored",
        source_id="src_a",
        chunk_id="chk_unscored",
        page_number=2,
        quote="Grounded but not independently scored.",
        file_name="unscored.pdf",
        doc_id="202",
        page_end=2,
        ref_id="ref_001",
    )

    assert public_annotations([unscored])[0]["score"] is None


def _paragraph_ref(
    index: int,
    *,
    doc_id: str,
    page: int,
    page_end: int | None = None,
) -> EvidenceRef:
    return replace(
        _ref(
            f"ev_{index}",
            source_id=f"src_{doc_id}",
            chunk_id=f"chk_{index}",
            page=page,
            file_name=f"{doc_id}.pdf",
            doc_id=doc_id,
        ),
        page_end=page_end or page,
        ref_id=f"ref_{index:03d}",
    )


def test_paragraph_deduplicator_only_hides_same_document_and_page_range() -> None:
    refs = [
        _paragraph_ref(1, doc_id="A", page=1),
        _paragraph_ref(2, doc_id="A", page=1),
        _paragraph_ref(3, doc_id="A", page=2),
        _paragraph_ref(4, doc_id="B", page=1),
        _paragraph_ref(5, doc_id="A", page=1, page_end=2),
    ]

    answer = citation_annotations.deduplicate_paragraph_references(
        "A [[ref:ref_001]]. Duplicate [[ref:ref_002]]. "
        "Other page [[ref:ref_003]]. Other document [[ref:ref_004]]. "
        "Page range [[ref:ref_005]].",
        refs,
    )

    assert answer == (
        "A [[ref:ref_001]]. Duplicate. "
        "Other page [[ref:ref_003]]. Other document [[ref:ref_004]]. "
        "Page range [[ref:ref_005]]."
    )


def test_paragraph_deduplicator_allows_same_document_page_in_next_paragraph() -> None:
    refs = [
        _paragraph_ref(1, doc_id="A", page=1),
        _paragraph_ref(2, doc_id="A", page=1),
    ]

    answer = citation_annotations.deduplicate_paragraph_references(
        "First [[ref:ref_001]]. Duplicate [[ref:ref_002]].\n\n"
        "Second paragraph [[ref:ref_002]].",
        refs,
    )

    assert answer == (
        "First [[ref:ref_001]]. Duplicate.\n\n"
        "Second paragraph [[ref:ref_002]]."
    )


def test_paragraph_deduplicator_tracks_blank_lines_and_markers_across_stream_deltas() -> None:
    refs = [
        _paragraph_ref(1, doc_id="A", page=1),
        _paragraph_ref(2, doc_id="A", page=1),
    ]
    marker_filter = ReferenceMarkerStreamFilter(refs)
    paragraph_filter = citation_annotations.ParagraphReferenceDeduplicator(refs)
    output = []

    for delta in (
        "First [[ref:ref_001]].\n",
        "Same paragraph continuation [[ref:ref_002]].\n",
        "\nSecond paragraph [[ref:ref_",
        "002]].",
    ):
        output.append(paragraph_filter.feed(marker_filter.feed(delta)))
    output.append(paragraph_filter.feed(marker_filter.finish()))

    assert "".join(output) == (
        "First [[ref:ref_001]].\nSame paragraph continuation.\n\n"
        "Second paragraph [[ref:ref_002]]."
    )


def test_verifier_keeps_allowed_ids_in_order_and_ignores_model_metadata() -> None:
    first = _ref("ev_a")
    second = _ref(
        "ev_b",
        source_id="src_b",
        chunk_id="chk_b",
        page=8,
        quote="Second exact sentence.",
    )
    llm = VerifierLLM(
        {
            "citations": [
                {
                    "evidence_id": "ev_b",
                    "score": 1.4,
                    "quote": "Second exact sentence.",
                    "docId": "forged",
                    "fileName": "forged.pdf",
                    "page": 999,
                },
                {"evidence_id": "ev_unknown", "score": 0.99, "quote": "Unknown."},
                {"evidence_id": "ev_b", "score": 0.2, "quote": "Second exact sentence."},
                {"evidence_id": "ev_a", "score": 0.81, "quote": "Exact source sentence."},
            ]
        }
    )

    result = CitationVerifier(llm, model="deepseek-v4-flash", max_annotations=20).verify(
        question="What supports the answer?",
        answer="The documents support the answer.",
        refs=[first, second],
        chunks=[
            _chunk(),
            _chunk(
                "chk_b",
                source_id="src_b",
                page=8,
                text="Background. Second exact sentence. End.",
            ),
        ],
        run_id="run_test",
        cancelled=lambda: False,
    )

    assert [ref.evidence_id for ref in result.refs] == ["ev_b", "ev_a"]
    assert [ref.ref_id for ref in result.refs] == ["ref_001", "ref_002"]
    assert [ref.score for ref in result.refs] == [1.0, 0.81]
    assert result.refs[0].doc_id == "501"
    assert result.refs[0].file_name == "paper.pdf"
    assert result.refs[0].page_number == 8
    assert result.trace["discarded_unknown_count"] == 1
    assert result.trace["discarded_duplicate_count"] == 1
    assert llm.requests[0].system.startswith("CITATION_VERIFIER")
    assert "untrusted source data" in llm.requests[0].system
    assert llm.requests[0].purpose == "subagent"
    assert llm.requests[0].model == "deepseek-v4-flash"
    assert llm.requests[0].run_id == "run_test"


def test_verifier_accepts_whitespace_equivalent_source_quote() -> None:
    llm = VerifierLLM(
        {"citations": [{"evidence_id": "ev_a", "score": 0.75, "quote": "Alpha beta gamma."}]}
    )
    ref = _ref("ev_a", quote="Fallback source quote.")
    chunk = _chunk(text="Prefix. Alpha\n\t beta   gamma. Suffix.")

    result = CitationVerifier(llm, model="model").verify(
        question="q",
        answer="a",
        refs=[ref],
        chunks=[chunk],
    )

    assert result.refs[0].quote == "Alpha beta gamma."
    assert result.refs[0].quote_source == "exact_span"
    assert result.trace["quote_fallback_count"] == 0


def test_verifier_can_disambiguate_identical_text_using_trusted_document_metadata() -> None:
    llm = MetadataAwareVerifierLLM()
    refs = [
        _ref(
            "ev_alpha",
            source_id="src_alpha",
            chunk_id="chk_alpha",
            page=3,
            quote="Shared supporting sentence.",
            file_name="alpha.pdf",
            doc_id="101",
        ),
        _ref(
            "ev_beta",
            source_id="src_beta",
            chunk_id="chk_beta",
            page=7,
            quote="Shared supporting sentence.",
            file_name="beta.pdf",
            doc_id="202",
        ),
    ]
    chunks = [
        _chunk("chk_alpha", source_id="src_alpha", page=3, text="Shared supporting sentence."),
        _chunk("chk_beta", source_id="src_beta", page=7, text="Shared supporting sentence."),
    ]

    result = CitationVerifier(llm, model="model").verify(
        question="What does beta.pdf conclude?",
        answer="beta.pdf reports the shared result.",
        refs=refs,
        chunks=chunks,
    )

    assert result.refs[0].evidence_id == "ev_beta"
    assert result.refs[0].doc_id == "202"
    assert result.refs[0].file_name == "beta.pdf"
    assert result.refs[0].page_number == 7
    assert llm.candidates == [
        {
            "evidence_id": "ev_alpha",
            "doc_id": "101",
            "file_name": "alpha.pdf",
            "page": 3,
            "end_page": 3,
            "text": "Shared supporting sentence.",
        },
        {
            "evidence_id": "ev_beta",
            "doc_id": "202",
            "file_name": "beta.pdf",
            "page": 7,
            "end_page": 7,
            "text": "Shared supporting sentence.",
        },
    ]


def test_verifier_rejects_invented_quote_and_retains_grounded_fallback() -> None:
    llm = VerifierLLM(
        {"citations": [{"evidence_id": "ev_a", "score": 0.6, "quote": "Invented model quotation."}]}
    )
    ref = _ref("ev_a")

    result = CitationVerifier(llm, model="model").verify(
        question="q",
        answer="a",
        refs=[ref],
        chunks=[_chunk()],
    )

    assert result.refs[0].quote == "Exact source sentence."
    assert result.refs[0].quote_source == "exact_span"
    assert result.trace["quote_fallback_count"] == 1


def test_verifier_failure_and_empty_selection_fail_open_without_fabricated_refs() -> None:
    failed = CitationVerifier(VerifierLLM(error=RuntimeError("provider failed")), model="model").verify(
        question="q",
        answer="valid answer",
        refs=[_ref("ev_a")],
        chunks=[_chunk()],
    )
    empty = CitationVerifier(VerifierLLM({"citations": []}), model="model").verify(
        question="q",
        answer="valid answer",
        refs=[_ref("ev_a")],
        chunks=[_chunk()],
    )

    assert failed.refs == []
    assert failed.trace["status"] == "failed"
    assert failed.trace["error_code"] == "RuntimeError"
    assert empty.refs == []
    assert empty.trace["status"] == "empty"


def test_verifier_has_no_legacy_three_reference_cap() -> None:
    refs = [
        _ref(
            f"ev_{index}",
            source_id=f"src_{index}",
            chunk_id=f"chk_{index}",
            page=index,
            quote=f"Supporting sentence {index}.",
        )
        for index in range(1, 6)
    ]
    chunks = [
        _chunk(
            f"chk_{index}",
            source_id=f"src_{index}",
            page=index,
            text=f"Supporting sentence {index}.",
        )
        for index in range(1, 6)
    ]
    llm = VerifierLLM(
        {
            "citations": [
                {
                    "evidence_id": f"ev_{index}",
                    "score": 0.8,
                    "quote": f"Supporting sentence {index}.",
                }
                for index in range(1, 6)
            ]
        }
    )

    result = CitationVerifier(llm, model="model", max_annotations=20).verify(
        question="Summarize all supported claims.",
        answer="Five claims are supported.",
        refs=refs,
        chunks=chunks,
    )

    assert len(result.refs) == 5
    assert [ref.ref_id for ref in result.refs] == [f"ref_{index:03d}" for index in range(1, 6)]


def test_marker_helpers_remove_model_markers_and_append_only_bound_aliases() -> None:
    refs = [
        EvidenceRef(
            evidence_id="ev_a",
            source_id="src_a",
            chunk_id="chk_a",
            page_number=4,
            quote="quote",
            ref_id="ref_001",
        ),
        EvidenceRef(
            evidence_id="ev_b",
            source_id="src_b",
            chunk_id="chk_b",
            page_number=8,
            quote="quote",
            ref_id="ref_002",
        ),
    ]

    final, suffix = append_reference_markers("Claim [[ref:made_up]]", refs)

    assert final == "Claim [[ref:ref_001]] [[ref:ref_002]]"
    assert suffix == " [[ref:ref_001]] [[ref:ref_002]]"
    assert strip_reference_markers(final) == "Claim"


def test_canonicalizer_normalizes_allowlisted_shorthand_and_removes_unknown_ids() -> None:
    refs = [replace(_ref("ev_a"), ref_id="ref_001")]

    answer = citation_annotations.canonicalize_reference_markers(
        "Supported [[ref_001]]. Canonical [[ref:ref_001]]. Unknown [[ref_999]].",
        refs,
    )

    assert answer == "Supported [[ref:ref_001]]. Canonical [[ref:ref_001]]. Unknown."


def test_stream_filter_normalizes_allowlisted_shorthand_split_across_provider_deltas() -> None:
    refs = [replace(_ref("ev_a"), ref_id="ref_001")]
    stream_filter = ReferenceMarkerStreamFilter(refs)

    outputs = [
        stream_filter.feed("Supported [[re"),
        stream_filter.feed("f_00"),
        stream_filter.feed("1]] claim. Unknown [[ref_999]] removed."),
        stream_filter.finish(),
    ]

    assert "".join(outputs) == "Supported [[ref:ref_001]] claim. Unknown removed."


def test_stream_filter_removes_complete_markers_split_across_provider_deltas() -> None:
    stream_filter = ReferenceMarkerStreamFilter()
    outputs = [
        stream_filter.feed("  Claim "),
        stream_filter.feed("[[re"),
        stream_filter.feed("f:made_up]] supported "),
        stream_filter.feed("answer. "),
        stream_filter.finish(),
    ]

    assert "".join(outputs) == "Claim supported answer."


def test_stream_filter_preserves_incomplete_marker_like_plain_text() -> None:
    stream_filter = ReferenceMarkerStreamFilter()

    output = stream_filter.feed("Claim [[ref:unfinished") + stream_filter.finish()

    assert output == "Claim [[ref:unfinished"


def test_stream_filter_invalidates_oversized_unfinished_marker_and_resumes_output() -> None:
    stream_filter = ReferenceMarkerStreamFilter()

    first = stream_filter.feed("Claim [[ref:" + "x" * 129)
    rest = stream_filter.feed("]] remains visible.") + stream_filter.finish()
    output = first + rest

    assert first
    assert "[[ref:" not in output
    assert "[ [ref:" in output
    assert output.endswith("]] remains visible.")


def test_stream_filter_invalidates_multiline_unfinished_marker() -> None:
    stream_filter = ReferenceMarkerStreamFilter()

    output = stream_filter.feed("Claim [[ref:bad\nvalue") + stream_filter.finish()

    assert "[[ref:" not in output
    assert "[ [ref:bad\nvalue" in output
