from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rlm_qa.app_service import QAService
from rlm_qa.config import Settings
from rlm_qa.contracts import DocProfile, EvidenceRef, NoTextError, Page
from rlm_qa.document_adapter import DocumentRLMAnswer
from rlm_qa.key_pool import KeyStatus
from rlm_qa.llm import ChatRequest, ChatResponse, FakeLLM, LLMClient
from rlm_qa.ocr import OCRProviderError


class ScriptedLLM(LLMClient):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("model called more times than expected")
        return ChatResponse(answer=self.responses.pop(0))


class MapReduceLLM(LLMClient):
    def __init__(
        self,
        *,
        final_answer: str,
        deep_answer: str = "Map-reduce deep answer.",
        fail_deep: bool = False,
    ):
        self.final_answer = final_answer
        self.deep_answer = deep_answer
        self.fail_deep = fail_deep
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.system == "MAP_REDUCE_PLAN":
            return ChatResponse(answer=json.dumps({"coverage": "relevant_subset", "answer_shape": "synthesis"}))
        if request.system == "MAP_REDUCE_TRIAGE":
            evidence_ids = _evidence_ids_from_text(request.user)
            pack_id = _pack_id_from_text(request.user)
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": pack_id,
                        "relevance_score": 0.9,
                        "has_answer": True,
                        "evidence_ids": evidence_ids,
                        "short_reason": "fixture relevant",
                    }
                )
            )
        if request.system == "MAP_REDUCE_DEEP_ANSWER":
            if self.fail_deep:
                return ChatResponse(answer="not json")
            evidence_ids = _evidence_ids_from_text(request.user)
            return ChatResponse(
                answer=json.dumps(
                    {
                        "answer": self.deep_answer,
                        "evidence_ids": evidence_ids,
                        "metadata_refs": [],
                        "confidence": 0.9,
                    }
                )
            )
        if request.system == "MAP_REDUCE_FINALIZER":
            return ChatResponse(answer=self.final_answer)
        raise AssertionError(f"unexpected system: {request.system}")


class TraceableMapReduceLLM(MapReduceLLM):
    def __init__(self, *, final_answer: str):
        super().__init__(final_answer=final_answer)
        self.trace: list[dict[str, str]] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.trace.append(
            {
                "event": "llm_call",
                "run_id": request.run_id,
                "system": request.system,
                "purpose": request.purpose,
                "key_masked": "sk...fake",
            }
        )
        return super().chat(request)

    def key_pool_trace(self) -> list[dict[str, str]]:
        return list(self.trace)


class AllSelectedSummaryLLM(MapReduceLLM):
    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.system == "MAP_REDUCE_PLAN":
            self.requests.append(request)
            return ChatResponse(answer=json.dumps({"coverage": "all_selected", "answer_shape": "per_document"}))
        if request.system == "MAP_REDUCE_FINALIZER":
            self.requests.append(request)
            match = re.search(r"Required source ids: (\[[^\n]+\])", request.user)
            assert match is not None
            source_ids = json.loads(match.group(1))
            return ChatResponse(
                answer=json.dumps(
                    {
                        "items": [
                            {"source_id": source_id, "answer": f"summary for {source_id}", "evidence_ids": []}
                            for source_id in source_ids
                        ]
                    }
                )
            )
        return super().chat(request)


def _evidence_ids_from_text(text: str) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for evidence_id in re.findall(r"\bev_[A-Za-z0-9_:-]+\b", text):
        if evidence_id not in seen:
            seen.add(evidence_id)
            ids.append(evidence_id)
    return ids


def _pack_id_from_text(text: str) -> str:
    match = re.search(r"PACK_ID=([A-Za-z0-9_:-]+)", text)
    return match.group(1) if match else "pack_unknown"


class FakePDFOCR:
    def __init__(self, pages: list[str]):
        self.pages = pages
        self.calls: list[tuple[str, str]] = []

    def extract_pdf_pages(self, path: Path, source_id: str, *, api_key: str) -> list[Page]:
        self.calls.append((path.name, source_id))
        return [
            Page(source_id=source_id, page_number=index, text=text, char_start=0, char_end=len(text))
            for index, text in enumerate(self.pages, start=1)
        ]


class FailingPDFOCR:
    def extract_pdf_pages(self, path: Path, source_id: str, *, api_key: str) -> list[Page]:
        raise NoTextError("ocr failed in fixture")


class UnauthorizedPDFOCR:
    def extract_pdf_pages(self, path: Path, source_id: str, *, api_key: str) -> list[Page]:
        raise OCRProviderError("unauthorized", status_code=401, error_code="http_401")


def rlm_context_answer(text: str = "The runtime inspected document context.") -> str:
    return (
        "```repl\n"
        "ev = context['evidence'][0]\n"
        "answer['content'] = '{\"answer\": \""
        + text
        + "\", \"evidence_ids\": [\"' + ev['evidence_id'] + '\"]}'\n"
        "answer['ready'] = True\n"
        "```"
    )


def rlm_full_pack_answer(text: str = "The runtime inspected the full pack.") -> str:
    return (
        "```repl\n"
        "packs = list_packs()\n"
        "pack_id = packs[0]['pack_id']\n"
        "full = read_packs([pack_id])\n"
        "_ = llm_query(full[pack_id]['text'], model='glm-5.1')\n"
        "ev = packs[0]['evidence_ids'][0]\n"
        "answer['content'] = {'answer': '"
        + text
        + "', 'evidence_ids': [ev]}\n"
        "answer['ready'] = True\n"
        "```"
    )


def rlm_metadata_answer(text: str = "The runtime read verified document metadata.") -> str:
    return (
        "```repl\n"
        "profiles = context.get('doc_profiles') or [context['doc_profile']]\n"
        "refs = [f\"doc_profile:{profile['source_id']}:page_count\" for profile in profiles]\n"
        "answer['content'] = {'answer': '"
        + text
        + "', 'evidence_ids': [], 'metadata_refs': refs}\n"
        "answer['ready'] = True\n"
        "```"
    )


def rlm_batched_multi_document_answer() -> str:
    return (
        "```repl\n"
        "import json\n"
        "packs = list_packs()\n"
        "full = read_packs([pack['pack_id'] for pack in packs])\n"
        "prompts = []\n"
        "for pack in packs:\n"
        "    text = full[pack['pack_id']]['text']\n"
        "    prompts.append('Summarize this selected document pack. Return JSON with answer and evidence_ids.\\n' + text)\n"
        "drafts = llm_query_batched(prompts, model='deepseek-v4-flash')\n"
        "parts = []\n"
        "evidence_ids = []\n"
        "for index, draft in enumerate(drafts):\n"
        "    data = json.loads(draft)\n"
        "    parts.append(data['answer'])\n"
        "    ids = data.get('evidence_ids') or packs[index].get('evidence_ids', [])[:1]\n"
        "    evidence_ids.extend(ids)\n"
        "answer['content'] = {'answer': '\\n'.join(parts), 'evidence_ids': evidence_ids}\n"
        "answer['ready'] = True\n"
        "```"
    )


def make_service(tmp_path: Path, *, virtual_page_chars: int = 5000) -> QAService:
    settings = Settings(
        runtime_dir=tmp_path / "runtime",
        virtual_page_chars=virtual_page_chars,
        direct_max_pages=10,
        direct_max_tokens=30_000,
    )
    return QAService(settings=settings, llm=FakeLLM(answer="A general base-model answer."))


def test_upload_text_document_builds_runtime_artifacts(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    doc = tmp_path / "swcnt.txt"
    doc.write_text(
        "Single-walled carbon nanotubes are one-dimensional carbon materials with useful conductivity.",
        encoding="utf-8",
    )

    result = service.upload_file(doc)

    assert result.status == "direct_ready"
    assert result.source_id.startswith("src_")
    doc_dir = service.store.doc_dir(result.source_id)
    assert (doc_dir / "pages.jsonl").exists()
    assert (doc_dir / "chunks.jsonl").exists()
    assert (doc_dir / "evidence.jsonl").exists()
    assert (doc_dir / "doc_profile.json").exists()
    assert result.doc_profile.page_count == 1
    assert result.doc_profile.evidence_count >= 1


def test_upload_utf8_bom_text_strips_bom_from_evidence(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    doc = tmp_path / "bom.txt"
    doc.write_text("\ufeffSingle-walled carbon nanotubes have useful conductivity.", encoding="utf-8")

    result = service.upload_file(doc)
    evidence_rows = service.store.read_jsonl(service.store.doc_dir(result.source_id) / "evidence.jsonl")

    assert evidence_rows[0]["quote"].startswith("Single-walled carbon nanotubes")


def test_upload_pdf_always_uses_glm_ocr_even_when_pdftotext_is_unavailable(tmp_path: Path) -> None:
    fake_ocr = FakePDFOCR(["OCR page one about SWCNT.", "OCR page two with catalyst details."])
    service = QAService(
        settings=Settings(
            runtime_dir=tmp_path / "runtime",
            pdftotext_cmd="missing-pdftotext-command",
            glm_ocr_api_keys=("fake-ocr-key",),
        ),
        llm=FakeLLM(answer="A general base-model answer."),
        pdf_ocr=fake_ocr,
    )
    doc = tmp_path / "paper.pdf"
    doc.write_bytes(b"%PDF-1.7 embedded text should not be read locally")

    result = service.upload_file(doc)

    assert result.status == "ocr_queued"
    assert fake_ocr.calls == []
    assert service.process_next_ocr_job_for_test() is True
    completed = service.get_document(result.source_id)
    assert completed is not None
    assert completed.status == "direct_ready"
    assert fake_ocr.calls == [("original.pdf", result.source_id)]
    page_rows = service.store.read_jsonl(service.store.doc_dir(result.source_id) / "pages.jsonl")
    assert [row["text"] for row in page_rows] == fake_ocr.pages


def test_pdf_upload_persists_queued_status_before_background_ocr_runs(tmp_path: Path) -> None:
    fake_ocr = FakePDFOCR(["OCR page one about SWCNT."])
    service = QAService(
        settings=Settings(runtime_dir=tmp_path / "runtime", glm_ocr_api_keys=("fake-ocr-key",)),
        llm=FakeLLM(answer="A general base-model answer."),
        pdf_ocr=fake_ocr,
    )

    result = service.upload_bytes("queued.pdf", b"%PDF-1.7")
    job_path = service.store.doc_dir(result.source_id) / "ocr_job.json"

    assert result.status == "ocr_queued"
    assert result.doc_profile.rlm_ready is False
    assert fake_ocr.calls == []
    assert job_path.exists()
    assert service.get_document(result.source_id).status == "ocr_queued"


def test_service_recovers_fresh_running_job_when_a_new_local_process_starts(tmp_path: Path) -> None:
    settings = Settings(runtime_dir=tmp_path / "runtime", glm_ocr_api_keys=("fake-ocr-key",))
    first = QAService(settings=settings, llm=FakeLLM(answer="unused"), pdf_ocr=FakePDFOCR(["OCR page"]))
    uploaded = first.upload_bytes("restart.pdf", b"%PDF-1.7")
    job_path = first.store.doc_dir(uploaded.source_id) / "ocr_job.json"
    job = first.store.read_json(job_path)
    job["state"] = "ocr_running"
    job["started_at"] = datetime.now(timezone.utc).isoformat()
    first.store.write_json(job_path, job)

    second = QAService(settings=settings, llm=FakeLLM(answer="unused"), pdf_ocr=FakePDFOCR(["OCR page"]))
    try:
        assert second.start_ocr_workers() == 1
    finally:
        second.stop_ocr_workers()


def test_service_restores_persisted_dead_ocr_key_after_restart(tmp_path: Path) -> None:
    settings = Settings(runtime_dir=tmp_path / "runtime", glm_ocr_api_keys=("fake-ocr-key",))
    first = QAService(settings=settings, llm=FakeLLM(answer="unused"), pdf_ocr=UnauthorizedPDFOCR())
    uploaded = first.upload_bytes("unauthorized.pdf", b"%PDF-1.7")

    assert first.process_next_ocr_job_for_test() is True
    assert first.get_document(uploaded.source_id).status == "ocr_failed"
    health_path = first.store.ocr_key_health_path()
    assert health_path.exists()
    health_text = health_path.read_text(encoding="utf-8")
    assert "fake-ocr-key" not in health_text
    assert "dead_key_ids" in health_text

    second = QAService(settings=settings, llm=FakeLLM(answer="unused"), pdf_ocr=FakePDFOCR(["OCR page"]))
    assert second.ocr_runner is not None
    assert second.ocr_runner.key_pool.snapshot()[0].status == KeyStatus.DEAD


def test_short_document_chat_returns_direct_answer_with_page_evidence(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.llm = FakeLLM(answer="SWCNTs are one-dimensional carbon materials.")
    doc = tmp_path / "swcnt.txt"
    doc.write_text(
        "Single-walled carbon nanotubes are one-dimensional carbon materials used in conductive composites.",
        encoding="utf-8",
    )
    uploaded = service.upload_file(doc)

    answer = service.chat(uploaded.source_id, "nanotubes")

    assert answer.mode == "direct"
    assert answer.answer_type == "document_grounded"
    assert answer.citation_policy == "pdf_page_citation"
    assert answer.evidences
    assert answer.evidences[0].page_number == 1
    assert "Single-walled carbon nanotubes" in answer.evidences[0].quote


def test_short_document_direct_uses_full_page_context_for_summary_without_keyword_hit(tmp_path: Path) -> None:
    service = make_service(tmp_path, virtual_page_chars=80)
    service.llm = ScriptedLLM(["Direct answer from the whole short document."])
    first_page = "Alpha catalyst platform. " + ("A" * 56)
    second_page = "Critical conclusion: the coating remains stable after heating."
    doc = tmp_path / "short_summary.txt"
    doc.write_text(first_page + second_page, encoding="utf-8")
    uploaded = service.upload_file(doc)

    answer = service.chat(uploaded.source_id, "What is this document about?")

    assert uploaded.doc_profile.direct_ready is True
    assert answer.mode == "direct"
    assert answer.answer_type == "document_grounded"
    assert answer.answer == "Direct answer from the whole short document."
    assert {ref.page_number for ref in answer.evidences} == {1, 2}
    assert "[short_summary.txt | page 1]" in service.llm.requests[0].user
    assert "[short_summary.txt | page 2]" in service.llm.requests[0].user
    assert "Critical conclusion: the coating remains stable after heating." in service.llm.requests[0].user


def test_short_document_still_uses_direct_path_when_rlm_failure_handling_changes(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.llm = ScriptedLLM(["Direct answer only."])
    doc = tmp_path / "short_direct_guard.txt"
    doc.write_text("A short document about carbon nanotubes.", encoding="utf-8")
    uploaded = service.upload_file(doc)

    answer = service.chat(uploaded.source_id, "What is this document about?")

    assert uploaded.doc_profile.direct_ready is True
    assert answer.mode == "direct"
    assert answer.answer == "Direct answer only."
    assert len(service.llm.requests) == 1


def test_long_document_chat_uses_rlm_default_program(tmp_path: Path) -> None:
    service = make_service(tmp_path, virtual_page_chars=80)
    service.llm = MapReduceLLM(final_answer="Final grounded long document answer.")
    long_text = "\n".join(
        f"page {i}. SWCNT sample {i} has conductivity and structural stability." for i in range(1, 15)
    )
    doc = tmp_path / "long_swcnt.txt"
    doc.write_text(long_text, encoding="utf-8")
    uploaded = service.upload_file(doc)

    answer = service.chat(uploaded.source_id, "What is this document about?")

    assert uploaded.status == "rlm_ready"
    assert answer.mode == "rlm"
    assert answer.answer_type == "document_grounded"
    assert answer.evidences
    assert answer.answer == "Final grounded long document answer."
    assert answer.debug is not None
    assert answer.debug.program_steps == ["rlm_map_reduce"]
    assert answer.debug.runtime_status == "rlm_success"
    assert answer.debug.trajectory["strategy"] == "map_reduce"


def test_long_document_chat_uses_deterministic_map_reduce_executor(tmp_path: Path) -> None:
    service = make_service(tmp_path, virtual_page_chars=80)
    service.llm = MapReduceLLM(final_answer="Final deterministic map-reduce answer.")
    long_text = "\n".join(
        f"page {i}. SWCNT sample {i} has conductivity and structural stability." for i in range(1, 15)
    )
    doc = tmp_path / "long_map_reduce.txt"
    doc.write_text(long_text, encoding="utf-8")
    uploaded = service.upload_file(doc)

    answer = service.chat(uploaded.source_id, "What is this document about?")

    assert answer.mode == "rlm"
    assert answer.answer == "Final deterministic map-reduce answer."
    assert answer.answer_type == "document_grounded"
    assert answer.debug is not None
    assert answer.debug.program_steps == ["rlm_map_reduce"]
    assert answer.debug.trajectory["strategy"] == "map_reduce"
    assert [request.system for request in service.llm.requests] == [
        "MAP_REDUCE_TRIAGE",
        "MAP_REDUCE_DEEP_ANSWER",
        "MAP_REDUCE_FINALIZER",
    ]


def test_long_document_article_overview_question_uses_rlm_evidence(tmp_path: Path) -> None:
    service = make_service(tmp_path, virtual_page_chars=80)
    service.llm = MapReduceLLM(final_answer="Final grounded article overview.")
    long_text = "\n".join(
        f"page {i}. This patent document introduces a SWCNT preparation method and catalyst system."
        for i in range(1, 15)
    )
    doc = tmp_path / "long_patent.txt"
    doc.write_text(long_text, encoding="utf-8")
    uploaded = service.upload_file(doc)

    answer = service.chat(uploaded.source_id, "What is this document about?")

    assert uploaded.status == "rlm_ready"
    assert answer.mode == "rlm"
    assert answer.answer_type == "document_grounded"
    assert answer.answer == "Final grounded article overview."
    assert answer.evidences
    assert answer.citation_policy == "pdf_page_citation"
    assert answer.debug is not None
    assert answer.debug.program_steps == ["rlm_map_reduce"]
    assert answer.debug.runtime_status == "rlm_success"


def test_page_count_question_uses_rlm_metadata_refs(tmp_path: Path) -> None:
    service = make_service(tmp_path, virtual_page_chars=80)
    service.llm = MapReduceLLM(final_answer="This document has 14 pages.")
    long_text = "x" * (80 * 13 + 1)
    doc = tmp_path / "page_count.txt"
    doc.write_text(long_text, encoding="utf-8")
    uploaded = service.upload_file(doc)

    answer = service.chat(uploaded.source_id, "How many pages does this document have?")

    assert uploaded.doc_profile.page_count == 14
    assert answer.mode == "rlm"
    assert answer.answer_type == "document_grounded"
    assert answer.answer == "This document has 14 pages."
    assert answer.citation_policy == "pdf_page_citation"
    assert answer.evidences
    assert answer.debug is not None
    assert answer.debug.program_steps == ["rlm_map_reduce"]
    assert answer.debug.metadata_refs == [f"doc_profile:{uploaded.source_id}:page_count"]


def test_multi_document_page_count_uses_rlm_metadata_refs(tmp_path: Path) -> None:
    service = make_service(tmp_path, virtual_page_chars=80)
    service.llm = MapReduceLLM(final_answer="first.txt has 3 pages; second.txt has 2 pages; total 5 pages.")
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a" * (80 * 2 + 1), encoding="utf-8")
    second.write_text("b" * (80 + 1), encoding="utf-8")
    first_upload = service.upload_file(first)
    second_upload = service.upload_file(second)

    answer = service.chat_many(
        [first_upload.source_id, second_upload.source_id],
        "How many pages do these documents have?",
    )

    assert answer.mode == "rlm"
    assert answer.answer_type == "document_grounded"
    assert answer.answer == "first.txt has 3 pages; second.txt has 2 pages; total 5 pages."
    assert answer.evidences
    assert answer.debug is not None
    assert answer.debug.metadata_refs == [
        f"doc_profile:{first_upload.source_id}:page_count",
        f"doc_profile:{second_upload.source_id}:page_count",
    ]


def test_multi_document_rlm_uses_packs_from_each_source(tmp_path: Path) -> None:
    service = make_service(tmp_path, virtual_page_chars=80)
    service.llm = MapReduceLLM(final_answer="Final multi document answer.")
    first = tmp_path / "first_long.txt"
    second = tmp_path / "second_long.txt"
    first.write_text("\n".join(f"page {i}. First long document discusses material synthesis." for i in range(1, 12)), encoding="utf-8")
    second.write_text("\n".join(f"page {i}. Second long document discusses performance testing." for i in range(1, 12)), encoding="utf-8")
    first_upload = service.upload_file(first)
    second_upload = service.upload_file(second)

    answer = service.chat_many([first_upload.source_id, second_upload.source_id], "Summarize these two documents.")

    assert answer.mode == "rlm"
    assert answer.answer_type == "document_grounded"
    assert answer.answer == "Final multi document answer."
    assert len(answer.evidences) >= 2
    assert {evidence.file_name for evidence in answer.evidences} == {"first_long.txt", "second_long.txt"}
    assert answer.debug is not None
    assert answer.debug.runtime_status == "rlm_success"
    assert answer.debug.program_steps == ["rlm_map_reduce"]
    assert answer.debug.trajectory["strategy"] == "map_reduce"
    assert answer.debug.trajectory["triage_count"] == 2
    assert answer.debug.trajectory["deep_count"] == 2


def test_document_answer_contract_clips_display_evidence_and_records_policy(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    refs = [
        EvidenceRef(f"ev_{index}", "src_a", f"chk_{index}", index + 1, f"quote {index}", "a.pdf")
        for index in range(8)
    ]
    document_answer = DocumentRLMAnswer(
        answer="A simple fact answer.",
        evidences=refs,
        metadata_refs=[],
        trajectory={"strategy": "map_reduce"},
        runtime_status="rlm_success",
    )

    answer = service._document_answer_contract(
        "run_test",
        "rlm",
        document_answer,
        question="What is the coating material?",
        profiles=[DocProfile("src_a", "a.pdf", 8, 800, 8, 8, False, True, "rlm_ready")],
    )

    assert len(answer.evidences) == 3
    assert answer.debug is not None
    assert answer.debug.trajectory["evidence_display"]["available_count"] == 8
    assert answer.debug.trajectory["evidence_display"]["displayed_count"] == 3


def test_document_rlm_merges_key_pool_trace_into_run_debug(tmp_path: Path) -> None:
    service = make_service(tmp_path, virtual_page_chars=80)
    service.llm = TraceableMapReduceLLM(final_answer="Final answer with trace.")
    doc = tmp_path / "trace_doc.txt"
    doc.write_text(
        "\n".join(f"page {i}. traceable material content. " + ("details " * 20) for i in range(1, 16)),
        encoding="utf-8",
    )
    uploaded = service.upload_file(doc)

    answer = service.chat(uploaded.source_id, "What is this document about?")

    assert answer.debug is not None
    trace = answer.debug.trajectory["key_pool_trace"]
    assert [event["system"] for event in trace] == [
        "MAP_REDUCE_TRIAGE",
        "MAP_REDUCE_DEEP_ANSWER",
        "MAP_REDUCE_FINALIZER",
    ]
    assert {event["run_id"] for event in trace} == {answer.run_id}
    assert "fake" in trace[0]["key_masked"]


def test_multi_document_bare_summary_uses_model_plan_and_never_omits_selected_ready_document(tmp_path: Path) -> None:
    service = make_service(tmp_path, virtual_page_chars=80)
    service.llm = AllSelectedSummaryLLM(final_answer="unused")
    uploads = []
    for name in ("one", "two", "three", "four"):
        path = tmp_path / f"{name}.txt"
        path.write_text("\n".join(f"page {index}. {name} document content." for index in range(1, 12)), encoding="utf-8")
        uploads.append(service.upload_file(path))

    answer = service.chat_many([upload.source_id for upload in uploads], "总结")

    assert all(f"## {name}.txt" in answer.answer for name in ("one", "two", "three", "four"))
    assert answer.debug is not None
    assert answer.debug.trajectory["coverage_contract"] == "all_selected"
    assert answer.debug.trajectory["required_source_ids"] == [upload.source_id for upload in uploads]
    assert answer.debug.trajectory["missing_source_ids"] == []


def test_multi_document_plain_qa_uses_deterministic_map_reduce_without_keyword_routing(tmp_path: Path) -> None:
    service = make_service(tmp_path, virtual_page_chars=80)
    service.llm = MapReduceLLM(final_answer="Final plain QA answer for both selected documents.")
    first = tmp_path / "alpha_long.txt"
    second = tmp_path / "beta_long.txt"
    first.write_text("\n".join(f"page {i}. Alpha adhesive film method." for i in range(1, 12)), encoding="utf-8")
    second.write_text("\n".join(f"page {i}. Beta color harmony visualization." for i in range(1, 12)), encoding="utf-8")
    first_upload = service.upload_file(first)
    second_upload = service.upload_file(second)

    answer = service.chat_many([first_upload.source_id, second_upload.source_id], "帮我整体过一下这批文件大意")

    assert answer.mode == "rlm"
    assert answer.answer == "Final plain QA answer for both selected documents."
    assert answer.answer_type == "document_grounded"
    assert {evidence.file_name for evidence in answer.evidences} == {"alpha_long.txt", "beta_long.txt"}
    assert answer.debug is not None
    assert answer.debug.program_steps == ["rlm_map_reduce"]
    assert answer.debug.runtime_status == "rlm_success"
    assert [request.system for request in service.llm.requests].count("MAP_REDUCE_TRIAGE") == 2
    assert [request.system for request in service.llm.requests].count("MAP_REDUCE_DEEP_ANSWER") == 2


def test_multi_document_chat_skips_ocr_failed_docs_instead_of_falling_back_for_whole_selection(tmp_path: Path) -> None:
    service = QAService(
        settings=Settings(
            runtime_dir=tmp_path / "runtime",
            virtual_page_chars=80,
            glm_ocr_api_keys=("fake-ocr-key",),
        ),
        llm=MapReduceLLM(final_answer="Final answer for ready documents."),
        pdf_ocr=FailingPDFOCR(),
    )
    ready = tmp_path / "ready_long.txt"
    failed = tmp_path / "failed.pdf"
    ready.write_text("\n".join(f"page {i}. Ready coating composition." for i in range(1, 12)), encoding="utf-8")
    failed.write_bytes(b"%PDF-1.7")
    ready_upload = service.upload_file(ready)
    failed_upload = service.upload_file(failed)
    assert service.process_next_ocr_job_for_test() is True
    failed_upload = service.get_document(failed_upload.source_id)
    assert failed_upload is not None

    answer = service.chat_many(
        [ready_upload.source_id, failed_upload.source_id],
        "这些文章说了什么？一篇一篇列出来给我",
    )

    assert failed_upload.status == "ocr_failed"
    assert answer.mode == "rlm"
    assert answer.answer_type == "document_grounded"
    assert answer.fallback_reason == ""
    assert answer.answer.startswith("Final answer for ready documents.")
    assert "failed.pdf" in answer.answer
    assert "ocr_failed" in answer.answer
    assert {evidence.file_name for evidence in answer.evidences} == {"ready_long.txt"}
    assert answer.debug is not None
    assert answer.debug.program_steps == ["rlm_map_reduce"]
    assert answer.debug.trajectory["skipped_doc_profiles"][0]["source_id"] == failed_upload.source_id


def test_long_document_rlm_without_executable_steps_returns_status_not_template(tmp_path: Path) -> None:
    service = make_service(tmp_path, virtual_page_chars=80)
    service.llm = MapReduceLLM(final_answer="Should not be used.", fail_deep=True)
    doc = tmp_path / "long_no_repl.txt"
    doc.write_text(
        "\n".join(f"page {i}. " + ("CNT method details. " * 20) for i in range(1, 15)),
        encoding="utf-8",
    )
    uploaded = service.upload_file(doc)

    answer = service.chat(uploaded.source_id, "What is this document about?")

    assert uploaded.status == "rlm_ready"
    assert answer.mode == "base_fallback"
    assert answer.answer_type == "status_answer"
    assert answer.fallback_reason == "rlm_runtime_failed"
    assert "{page_count}" not in answer.answer
    assert "文档问答流程本次没有生成可用答案" in answer.answer
    assert "RLM" not in answer.answer


def test_rlm_answer_without_refs_does_not_claim_pdf_citation(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    document_answer = DocumentRLMAnswer(
        answer="The model finalized the draft answer.",
        evidences=[],
        metadata_refs=[],
        trajectory={"iterations": []},
        runtime_status="rlm_success",
    )

    answer = service._document_answer_contract("run_test", "rlm", document_answer)

    assert answer.mode == "rlm"
    assert answer.answer_type == "document_assisted"
    assert answer.citation_policy == "no_pdf_citation"
    assert answer.evidences == []


def test_unrelated_question_uses_base_fallback_without_fake_pdf_citation(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    doc = tmp_path / "copper.txt"
    doc.write_text("This document only discusses thermal conductivity of copper materials.", encoding="utf-8")
    uploaded = service.upload_file(doc)

    answer = service.chat(uploaded.source_id, "How should a Moon base be built?")

    assert answer.mode == "base_fallback"
    assert answer.answer_type == "base_fallback"
    assert answer.citation_policy == "no_pdf_citation"
    assert answer.evidences == []
    assert "page " not in answer.answer.lower()
