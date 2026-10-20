from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import re
import sqlite3
import threading

import pytest

from rlm_qa.config import Settings
from rlm_qa.conversation_memory import ConversationMemory
from rlm_qa.llm import ChatRequest, ChatResponse, LLMClient, LLMStreamCancelled, ToolCall
from rlm_qa.platform_documents import PlatformDocumentError, PlatformDocumentLoader
from rlm_qa.platform_service import PlatformQAService


class RecordingLLM(LLMClient):
    def __init__(self, answer: str = "direct grounded answer", *, auto_markers: bool = True) -> None:
        self.answer = answer
        self.auto_markers = auto_markers
        self.requests: list[ChatRequest] = []
        self.verifier_requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.system.startswith("CITATION_VERIFIER"):
            self.verifier_requests.append(request)
            return _citation_verifier_response(request)
        self.requests.append(request)
        answer = self.answer
        if self.auto_markers and request.purpose == "base" and "[[ref:" not in answer:
            ref_ids = list(dict.fromkeys(re.findall(r"\b(ref_\d{3}):", request.user)))
            if ref_ids:
                answer = f"{answer} " + " ".join(f"[[ref:{ref_id}]]" for ref_id in ref_ids)
        return ChatResponse(answer=answer)


class RecordingStreamingLLM(RecordingLLM):
    def __init__(self) -> None:
        super().__init__("direct streamed answer")
        self.stream_requests: list[ChatRequest] = []

    def chat_stream(self, request, on_delta, cancelled) -> ChatResponse:
        self.stream_requests.append(request)
        assert cancelled() is False
        on_delta("direct ")
        on_delta("streamed ")
        on_delta("answer")
        return ChatResponse(answer="direct streamed answer")


class ToolAwarePlatformLLM(RecordingLLM):
    supports_tool_calls = True

    def __init__(self, decision: ChatResponse, *, stream_parts: list[str] | None = None) -> None:
        super().__init__()
        self.decision = decision
        self.stream_parts = stream_parts or []
        self.stream_requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.purpose == "orchestrator":
            self.requests.append(request)
            return self.decision
        return super().chat(request)

    def chat_stream(self, request, on_delta, cancelled) -> ChatResponse:
        if request.purpose != "orchestrator":
            return super().chat_stream(request, on_delta, cancelled)
        self.stream_requests.append(request)
        assert cancelled() is False
        for part in self.stream_parts:
            on_delta(part)
        return self.decision


class MarkerInjectingStreamingLLM(RecordingLLM):
    def chat_stream(self, request, on_delta, cancelled) -> ChatResponse:
        assert cancelled() is False
        on_delta("direct streamed ")
        on_delta("[[re")
        on_delta("f:made_up]] ")
        on_delta("answer")
        return ChatResponse(answer="direct streamed [[ref:made_up]] answer")


class InlineReferenceStreamingLLM(RecordingLLM):
    def chat_stream(self, request, on_delta, cancelled) -> ChatResponse:
        assert cancelled() is False
        on_delta("supported claim ")
        on_delta("[[ref:ref_")
        on_delta("001]]")
        on_delta(" conclusion")
        return ChatResponse(answer="supported claim [[ref:ref_001]] conclusion")


class FailingLLM(LLMClient):
    def chat(self, request: ChatRequest) -> ChatResponse:
        raise RuntimeError("model failed")


class CitationFailingLLM(RecordingLLM):
    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.system.startswith("CITATION_VERIFIER"):
            self.verifier_requests.append(request)
            raise RuntimeError("citation verifier failed")
        return super().chat(request)


class CancellingStreamingLLM(LLMClient):
    def chat(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("streaming path expected")

    def chat_stream(self, request, on_delta, cancelled) -> ChatResponse:
        on_delta("partial")
        raise LLMStreamCancelled("client cancelled")


class CancellingNonStreamingLLM(RecordingLLM):
    def __init__(self, cancelled: threading.Event) -> None:
        super().__init__("completed after cancellation")
        self.cancelled = cancelled

    def chat(self, request: ChatRequest) -> ChatResponse:
        response = super().chat(request)
        self.cancelled.set()
        return response


class ReadFailingMemoryStore:
    ready = False

    def read_messages(self, chat_id):
        raise sqlite3.OperationalError("read failed")

    def append_exchange(self, chat_id, bootstrap, question, answer, run_id, *, max_messages):
        return len(bootstrap) + 2

    def close(self):
        return None


class WriteFailingMemoryStore:
    ready = False

    def read_messages(self, chat_id):
        return []

    def append_exchange(self, chat_id, bootstrap, question, answer, run_id, *, max_messages):
        raise sqlite3.OperationalError("write failed")

    def close(self):
        return None


class FailingRunStore:
    def run_path(self, run_id):
        return Path(f"{run_id}.json")

    def write_json(self, path, payload):
        raise OSError("run persistence failed")


class CancellingRunStore:
    def __init__(self, delegate, cancelled: threading.Event) -> None:
        self.delegate = delegate
        self.cancelled = cancelled

    def run_path(self, run_id):
        return self.delegate.run_path(run_id)

    def write_json(self, path, payload):
        self.delegate.write_json(path, payload)
        self.cancelled.set()


class AllSelectedMapReduceLLM(LLMClient):
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []
        self.verifier_requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.system.startswith("CITATION_VERIFIER"):
            self.verifier_requests.append(request)
            return _citation_verifier_response(request)
        if request.system == "MAP_REDUCE_PLAN":
            return ChatResponse(answer=json.dumps({"coverage": "all_selected", "answer_shape": "per_document"}))
        if request.system == "MAP_REDUCE_TRIAGE":
            pack_id = re.search(r'"pack_id":\s*"([^"]+)"', request.user)
            evidence_ids = re.findall(r"\bev_[A-Za-z0-9_:-]+\b", request.user)
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": pack_id.group(1) if pack_id else "pack_unknown",
                        "relevance_score": 1.0,
                        "has_answer": True,
                        "evidence_ids": evidence_ids,
                        "short_reason": "selected document",
                    }
                )
            )
        if request.system == "MAP_REDUCE_DEEP_ANSWER":
            evidence_ids = re.findall(r"\bev_[A-Za-z0-9_:-]+\b", request.user)
            return ChatResponse(
                answer=json.dumps(
                    {
                        "answer": "document summary",
                        "evidence_ids": evidence_ids,
                        "metadata_refs": [],
                        "confidence": 0.9,
                    }
                )
            )
        if request.system == "MAP_REDUCE_FINALIZER":
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
        if request.system == "MAP_REDUCE_FINALIZER_REPAIR":
            raise AssertionError("repair should not be needed")
        if request.system == "MAP_REDUCE_FINAL_STREAM":
            match = re.search(r"Required documents in exact display order:\n(\[[^\n]+\])", request.user)
            assert match is not None
            documents = json.loads(match.group(1))
            sections = []
            for document in documents:
                file_name = document["file_name"]
                ref_match = re.search(
                    rf"(?m)^\[\[ref:(ref_\d{{3}})\]\]: {re.escape(file_name)}, page ",
                    request.user,
                )
                marker = f" [[ref:{ref_match.group(1)}]]" if ref_match else ""
                sections.append(f"## {file_name}\nsummary for {file_name}{marker}")
            return ChatResponse(answer="\n\n".join(sections))
        raise AssertionError(f"unexpected request: {request.system}")


class ToolAwareAllSelectedMapReduceLLM(AllSelectedMapReduceLLM):
    supports_tool_calls = True

    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.purpose == "orchestrator":
            self.requests.append(request)
            return ChatResponse(
                tool_calls=(
                    ToolCall(
                        id="call_all_documents",
                        name="inspect_documents",
                        arguments=json.dumps(
                            {
                                "query": "逐篇总结所选文档",
                                "scope": {"mode": "all_selected", "source_ids": []},
                                "coverage": "all_in_scope",
                                "answer_shape": "per_source",
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
            )
        return super().chat(request)


class ToolAwareNoEvidenceLLM(LLMClient):
    supports_tool_calls = True

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.purpose == "orchestrator":
            return ChatResponse(
                tool_calls=(
                    ToolCall(
                        id="call_absent_topic",
                        name="inspect_documents",
                        arguments=json.dumps(
                            {
                                "query": "所选文档是否讨论量子纠缠？",
                                "scope": {"mode": "all_selected", "source_ids": []},
                                "coverage": "relevant_in_scope",
                                "answer_shape": "synthesis",
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
            )
        if request.system == "MAP_REDUCE_TRIAGE":
            pack_id = re.search(r'"pack_id":\s*"([^"]+)"', request.user)
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": pack_id.group(1) if pack_id else "pack_unknown",
                        "relevance_score": 0.0,
                        "has_answer": False,
                        "evidence_ids": [],
                        "short_reason": "not present",
                    }
                )
            )
        if request.purpose == "base":
            return ChatResponse(answer="所选文档中没有找到量子纠缠相关内容。")
        raise AssertionError(f"unexpected request: {request.system}")


class ConfiguredOnePassPlatformLLM(AllSelectedMapReduceLLM):
    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.system != "MAP_REDUCE_ONE_PASS":
            return super().chat(request)
        self.requests.append(request)
        pack_id = re.search(r'"pack_id":\s*"([^"]+)"', request.user)
        evidence_ids = list(dict.fromkeys(re.findall(r"\bev_[A-Za-z0-9_:-]+\b", request.user)))
        return ChatResponse(
            answer=json.dumps(
                {
                    "pack_id": pack_id.group(1) if pack_id else "pack_unknown",
                    "relevance_score": 1.0,
                    "has_answer": True,
                    "answer": "one-pass document summary",
                    "claims": [],
                    "evidence_ids": evidence_ids,
                    "metadata_refs": [],
                    "confidence": 0.9,
                    "short_reason": "selected document",
                }
            )
        )


class RelevantSubsetMapReduceLLM(LLMClient):
    def __init__(self) -> None:
        self.verifier_candidates: list[dict[str, object]] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.system == "MAP_REDUCE_PLAN":
            return ChatResponse(answer=json.dumps({"coverage": "relevant_subset", "answer_shape": "synthesis"}))
        if request.system == "MAP_REDUCE_TRIAGE":
            pack_id = re.search(r'"pack_id":\s*"([^"]+)"', request.user)
            evidence_ids = re.findall(r"\bev_[A-Za-z0-9_:-]+\b", request.user)
            relevant = "first-only.pdf" in request.user
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": pack_id.group(1) if pack_id else "pack_unknown",
                        "relevance_score": 1.0 if relevant else 0.0,
                        "has_answer": relevant,
                        "evidence_ids": evidence_ids if relevant else [],
                        "short_reason": "selected" if relevant else "unrelated",
                    }
                )
            )
        if request.system == "MAP_REDUCE_DEEP_ANSWER":
            evidence_ids = re.findall(r"\bev_[A-Za-z0-9_:-]+\b", request.user)
            return ChatResponse(
                answer=json.dumps(
                    {
                        "answer": "selected document answer",
                        "evidence_ids": evidence_ids,
                        "metadata_refs": [],
                        "confidence": 0.9,
                    }
                )
            )
        if request.system == "MAP_REDUCE_FINALIZER":
            return ChatResponse(answer="answer from the selected document [[ref_001]]")
        if request.system.startswith("CITATION_VERIFIER"):
            payload = json.loads(request.user)
            self.verifier_candidates = payload["candidates"]
            candidate = self.verifier_candidates[0]
            return ChatResponse(
                answer=json.dumps(
                    {
                        "citations": [
                            {
                                "evidence_id": candidate["evidence_id"],
                                "score": 0.91,
                                "quote": candidate["text"],
                            }
                        ]
                    }
                )
            )
        raise AssertionError(f"unexpected request: {request.system}")


def _citation_verifier_response(request: ChatRequest) -> ChatResponse:
    payload = json.loads(request.user.rsplit("\n\n", 1)[-1])
    citations = []
    for candidate in payload["candidates"]:
        text = " ".join(str(candidate["text"]).split())
        citations.append(
            {
                "evidence_id": candidate["evidence_id"],
                "score": 0.9,
                "quote": text,
            }
        )
    return ChatResponse(answer=json.dumps({"citations": citations}))


def write_document(
    root: Path,
    dir_name: str,
    *,
    file_id: int,
    file_name: str,
    pages: list[str],
) -> None:
    target = root / dir_name
    target.mkdir(parents=True)
    (target / "status.json").write_text(
        json.dumps({"status": "ok", "pages": len(pages)}),
        encoding="utf-8",
    )
    (target / "metadata.json").write_text(
        json.dumps({"fileId": file_id, "fileName": file_name}),
        encoding="utf-8",
    )
    rows = [
        {"page": page_number, "index": 0, "label": "text", "content": text}
        for page_number, text in enumerate(pages, start=1)
    ]
    (target / "elements.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def make_settings(tmp_path: Path, *, direct_max_pages: int = 10) -> Settings:
    return Settings(
        runtime_dir=tmp_path / "runtime",
        platform_data_root=tmp_path / "glm",
        platform_cache_dir=tmp_path / "cache",
        direct_max_pages=direct_max_pages,
        direct_max_tokens=30_000,
        chunk_chars=400,
        chunk_overlap_chars=40,
        rlm_batch_max_workers=2,
    )


def make_service(tmp_path: Path, llm: LLMClient, *, direct_max_pages: int = 10) -> PlatformQAService:
    settings = make_settings(tmp_path, direct_max_pages=direct_max_pages)
    loader = PlatformDocumentLoader(settings.platform_data_root, settings.platform_cache_dir, settings)
    return PlatformQAService(settings=settings, llm=llm, loader=loader)


def test_short_platform_document_uses_full_context_without_keyword_gate(tmp_path: Path) -> None:
    dir_name = "a" * 64
    write_document(
        tmp_path / "glm",
        dir_name,
        file_id=501,
        file_name="swcnt.pdf",
        pages=[
            "Single-walled carbon nanotubes are cylindrical carbon materials.",
            "Their electronic properties depend on diameter and chirality.",
        ],
    )
    llm = RecordingLLM()
    service = make_service(tmp_path, llm)

    answer = service.answer([dir_name], ["501"], "这份材料讲了什么？", "chat-1")

    assert answer.mode == "direct"
    assert answer.answer == "direct grounded answer [[ref:ref_001]] [[ref:ref_002]]"
    assert {ref.page_number for ref in answer.evidences} == {1, 2}
    assert {ref.doc_id for ref in answer.evidences} == {"501"}
    assert [ref.ref_id for ref in answer.evidences] == ["ref_001", "ref_002"]
    assert [ref.score for ref in answer.evidences] == [None, None]
    assert "Single-walled carbon nanotubes" in llm.requests[0].user
    assert "diameter and chirality" in llm.requests[0].user
    assert llm.requests[0].purpose == "base"
    assert llm.requests[0].model == "glm-5.2"
    assert answer.debug is not None
    assert answer.debug.trajectory["dir_names"] == [dir_name]
    assert answer.debug.trajectory["doc_ids"] == ["501"]
    assert answer.debug.trajectory["cache_hits"] == 0
    assert answer.debug.trajectory["citation_annotations"]["status"] == "inline_local"
    assert len(llm.verifier_requests) == 0


def test_root_model_answers_greeting_without_loading_selected_document_text(tmp_path: Path) -> None:
    dir_name = "0" * 64
    write_document(
        tmp_path / "glm",
        dir_name,
        file_id=500,
        file_name="available-but-not-read.pdf",
        pages=["This text file will be removed before the request."],
    )
    (tmp_path / "glm" / dir_name / "elements.jsonl").unlink()
    llm = ToolAwarePlatformLLM(ChatResponse(answer="你好！有什么可以帮你？"))
    service = make_service(tmp_path, llm)

    answer = service.answer([dir_name], ["500"], "你好", "chat-greeting")

    assert answer.mode == "base"
    assert answer.answer_type == "conversation"
    assert answer.answer == "你好！有什么可以帮你？"
    assert answer.evidences == []
    assert [request.purpose for request in llm.requests] == ["orchestrator"]
    assert "Selected document context" not in llm.requests[0].user
    assert answer.debug is not None
    assert answer.debug.trajectory["execution_contract"]["action"] == "respond"


def test_execution_contract_feature_flag_restores_legacy_document_first_path(tmp_path: Path) -> None:
    dir_name = "0f" * 32
    write_document(
        tmp_path / "glm",
        dir_name,
        file_id=507,
        file_name="legacy-path.pdf",
        pages=["Legacy full document context."],
    )
    settings = replace(make_settings(tmp_path), turn_execution_contract_enabled=False)
    llm = ToolAwarePlatformLLM(ChatResponse(answer="orchestrator should not run"))
    loader = PlatformDocumentLoader(settings.platform_data_root, settings.platform_cache_dir, settings)
    service = PlatformQAService(settings=settings, llm=llm, loader=loader)

    answer = service.answer([dir_name], ["507"], "你好", "chat-legacy-switch")

    assert answer.mode == "direct"
    assert [request.purpose for request in llm.requests] == ["base"]
    assert "Legacy full document context." in llm.requests[0].user


def test_root_model_streams_direct_greeting_without_starting_document_pipeline(tmp_path: Path) -> None:
    dir_name = "6" * 64
    write_document(
        tmp_path / "glm",
        dir_name,
        file_id=506,
        file_name="not-read-during-greeting.pdf",
        pages=["Document text."],
    )
    llm = ToolAwarePlatformLLM(
        ChatResponse(answer="你好！"),
        stream_parts=["你", "好", "！"],
    )
    service = make_service(tmp_path, llm)
    deltas: list[str] = []

    answer = service.answer(
        [dir_name],
        ["506"],
        "你好",
        "chat-stream-greeting",
        on_final_delta=lambda _run_id, delta: deltas.append(delta),
        cancelled=lambda: False,
    )

    assert "".join(deltas) == "你好！"
    assert answer.answer == "你好！"
    assert answer.mode == "base"
    assert len(llm.stream_requests) == 1
    assert llm.stream_requests[0].purpose == "orchestrator"
    assert not [request for request in llm.requests if request.purpose in {"subagent", "finalizer"}]


def test_root_model_tool_call_is_the_only_entry_to_existing_document_qa(tmp_path: Path) -> None:
    dir_name = "5" * 64
    write_document(
        tmp_path / "glm",
        dir_name,
        file_id=505,
        file_name="tool-selected.pdf",
        pages=["The selected document explains single-walled carbon nanotubes."],
    )
    llm = ToolAwarePlatformLLM(
        ChatResponse(
            tool_calls=(
                ToolCall(
                    id="call_document",
                    name="inspect_documents",
                    arguments=json.dumps(
                        {
                            "query": "总结所选文档",
                            "scope": {"mode": "all_selected", "source_ids": []},
                            "coverage": "all_in_scope",
                            "answer_shape": "per_source",
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        )
    )
    service = make_service(tmp_path, llm)

    answer = service.answer([dir_name], ["505"], "这篇文章说了什么？", "chat-document-tool")

    assert answer.mode == "direct"
    assert answer.answer.startswith("direct grounded answer")
    assert [request.purpose for request in llm.requests] == ["orchestrator", "base"]
    assert llm.requests[0].tools[0]["function"]["name"] == "inspect_documents"
    assert "The selected document explains" in llm.requests[1].user
    assert answer.debug is not None
    assert answer.debug.trajectory["execution_contract"]["action"] == "inspect_documents"
    assert answer.debug.trajectory["execution_contract"]["scope_mode"] == "all_selected"


def test_root_document_contract_replaces_duplicate_rlm_coverage_planner(tmp_path: Path) -> None:
    first = "1a" * 32
    second = "2b" * 32
    write_document(tmp_path / "glm", first, file_id=511, file_name="first.pdf", pages=["First fact."])
    write_document(tmp_path / "glm", second, file_id=522, file_name="second.pdf", pages=["Second fact."])
    llm = ToolAwareAllSelectedMapReduceLLM()
    service = make_service(tmp_path, llm, direct_max_pages=1)

    answer = service.answer(
        [first, second],
        ["511", "522"],
        "分别说一下这两篇文档做了什么",
        "chat-contract-rlm",
    )

    systems = [request.system for request in llm.requests]
    assert "MAP_REDUCE_PLAN" not in systems
    assert answer.mode == "rlm"
    assert answer.debug is not None
    assert answer.debug.trajectory["coverage_contract"] == "all_selected"
    assert answer.debug.trajectory["answer_shape"] == "per_document"
    assert answer.debug.trajectory["coverage_plan_injected"] is True
    assert answer.debug.trajectory["execution_contract"]["action"] == "inspect_documents"


def test_zero_relevance_document_result_falls_back_to_natural_base_answer(tmp_path: Path) -> None:
    first = "3c" * 32
    second = "4d" * 32
    write_document(tmp_path / "glm", first, file_id=533, file_name="first.pdf", pages=["A coating fact."])
    write_document(tmp_path / "glm", second, file_id=544, file_name="second.pdf", pages=["A battery fact."])
    llm = ToolAwareNoEvidenceLLM()
    service = make_service(tmp_path, llm, direct_max_pages=1)

    answer = service.answer(
        [first, second],
        ["533", "544"],
        "这些文档提到量子纠缠了吗？",
        "chat-contract-no-evidence",
    )

    systems = [request.system for request in llm.requests]
    assert answer.mode == "base"
    assert answer.answer_type == "document_not_found"
    assert answer.answer == "所选文档中没有找到量子纠缠相关内容。"
    assert answer.evidences == []
    assert "MAP_REDUCE_DEEP_ANSWER" not in systems
    assert "MAP_REDUCE_FINALIZER" not in systems
    assert answer.debug is not None
    assert answer.debug.trajectory["execution_contract"]["document_result"] == "no_relevant_evidence"


def test_direct_answer_deduplicates_exact_page_only_within_each_paragraph(tmp_path: Path) -> None:
    dir_name = "8" * 64
    write_document(
        tmp_path / "glm",
        dir_name,
        file_id=508,
        file_name="paragraph.pdf",
        pages=["Paragraph citation source."],
    )
    llm = RecordingLLM(
        "First [[ref:ref_001]]. Duplicate [[ref:ref_001]].\n\n"
        "Second paragraph [[ref:ref_001]]."
    )
    service = make_service(tmp_path, llm)

    answer = service.answer([dir_name], ["508"], "summarize", "chat-paragraph-citation")

    assert answer.answer == (
        "First [[ref:ref_001]]. Duplicate.\n\n"
        "Second paragraph [[ref:ref_001]]."
    )
    assert [ref.ref_id for ref in answer.evidences] == ["ref_001"]
    assert answer.evidences[0].file_name == "paragraph.pdf"
    assert answer.evidences[0].page_number == 1


def test_marker_free_answer_keeps_valid_answer_without_fabricated_reference_or_verifier(tmp_path: Path) -> None:
    dir_name = "9" * 64
    write_document(
        tmp_path / "glm",
        dir_name,
        file_id=509,
        file_name="fail-open.pdf",
        pages=["The document still supports a valid model answer."],
    )
    llm = RecordingLLM(auto_markers=False)
    service = make_service(tmp_path, llm)

    answer = service.answer([dir_name], ["509"], "summarize", "chat-citation-failure")

    assert answer.answer == "direct grounded answer"
    assert answer.evidences == []
    assert answer.answer_type == "document_assisted"
    assert answer.citation_policy == "no_pdf_citation"
    assert answer.debug is not None
    assert answer.debug.trajectory["citation_annotations"]["status"] == "no_inline_references"
    assert llm.verifier_requests == []


def test_aggregate_threshold_sends_multiple_documents_to_map_reduce(tmp_path: Path) -> None:
    first = "b" * 64
    second = "c" * 64
    write_document(tmp_path / "glm", first, file_id=1, file_name="first.pdf", pages=["First page.", "First result."])
    write_document(tmp_path / "glm", second, file_id=2, file_name="second.pdf", pages=["Second page.", "Second result."])
    llm = AllSelectedMapReduceLLM()
    service = make_service(tmp_path, llm, direct_max_pages=3)

    answer = service.answer([first, second], ["1", "2"], "请逐篇总结", "chat-2")

    assert answer.mode == "rlm"
    assert answer.debug is not None
    assert answer.debug.trajectory["strategy"] == "map_reduce"
    assert answer.debug.trajectory["coverage_contract"] == "all_selected"
    assert answer.debug.trajectory["missing_source_ids"] == []
    assert "## first.pdf" in answer.answer
    assert "## second.pdf" in answer.answer
    assert {ref.file_name for ref in answer.evidences} == {"first.pdf", "second.pdf"}
    assert {ref.doc_id for ref in answer.evidences} == {"1", "2"}
    assert "[[ref:ref_001]]" in answer.answer
    assert llm.verifier_requests == []


def test_platform_service_passes_one_pass_execution_mode_to_map_reduce(tmp_path: Path) -> None:
    first = "4" * 64
    second = "5" * 64
    write_document(
        tmp_path / "glm",
        first,
        file_id=41,
        file_name="first-one-pass.pdf",
        pages=["First page.", "First result."],
    )
    write_document(
        tmp_path / "glm",
        second,
        file_id=52,
        file_name="second-one-pass.pdf",
        pages=["Second page.", "Second result."],
    )
    settings = Settings(
        runtime_dir=tmp_path / "runtime",
        platform_data_root=tmp_path / "glm",
        platform_cache_dir=tmp_path / "cache",
        direct_max_pages=3,
        direct_max_tokens=30_000,
        chunk_chars=400,
        chunk_overlap_chars=40,
        rlm_batch_max_workers=3,
        rlm_map_execution_mode="one_pass",
        rlm_reduce_context_mode="compact",
    )
    llm = ConfiguredOnePassPlatformLLM()
    loader = PlatformDocumentLoader(settings.platform_data_root, settings.platform_cache_dir, settings)
    service = PlatformQAService(settings=settings, llm=llm, loader=loader)

    answer = service.answer(
        [first, second],
        ["41", "52"],
        "summarize every document",
        "chat-one-pass-platform",
    )

    systems = [request.system for request in llm.requests]
    assert answer.mode == "rlm"
    assert systems.count("MAP_REDUCE_ONE_PASS") == 2
    assert "MAP_REDUCE_TRIAGE" not in systems
    assert "MAP_REDUCE_DEEP_ANSWER" not in systems
    assert answer.debug is not None
    assert answer.debug.trajectory["map_execution_mode"] == "one_pass"
    assert answer.debug.trajectory["reduce_context_mode"] == "compact"


def test_rlm_stream_appends_verified_marker_and_keeps_memory_marker_free(tmp_path: Path) -> None:
    first = "8" * 64
    second = "7" * 64
    write_document(tmp_path / "glm", first, file_id=81, file_name="first-stream.pdf", pages=["First fact."])
    write_document(tmp_path / "glm", second, file_id=72, file_name="second-stream.pdf", pages=["Second fact."])
    service = make_service(tmp_path, AllSelectedMapReduceLLM(), direct_max_pages=1)
    events: list[tuple[str, str]] = []

    answer = service.answer(
        [first, second],
        ["81", "72"],
        "summarize each document",
        "chat-rlm-stream",
        on_final_delta=lambda run_id, delta: events.append((run_id, delta)),
        cancelled=lambda: False,
    )

    assert answer.mode == "rlm"
    assert "".join(delta for _, delta in events) == answer.answer
    assert events[0][1].startswith("## first-stream.pdf")
    assert "[[ref:ref_001]]" in answer.answer
    assert {ref.doc_id for ref in answer.evidences} == {"81", "72"}
    assert all("[[ref:" not in message.content for message in service.memory.read_messages("chat-rlm-stream"))


def test_rlm_keeps_only_inline_evidence_selected_by_map_and_deep_answer(tmp_path: Path) -> None:
    first = "a1" * 32
    second = "b2" * 32
    write_document(
        tmp_path / "glm",
        first,
        file_id=11,
        file_name="first-only.pdf",
        pages=["The relevant coating fact is in this document."],
    )
    write_document(
        tmp_path / "glm",
        second,
        file_id=22,
        file_name="unselected.pdf",
        pages=["This unrelated document discusses a battery."],
    )
    llm = RelevantSubsetMapReduceLLM()
    service = make_service(tmp_path, llm, direct_max_pages=1)

    answer = service.answer([first, second], ["11", "22"], "What is the coating fact?", "chat-subset")

    assert answer.mode == "rlm"
    assert answer.answer == "answer from the selected document [[ref:ref_001]]"
    assert {ref.file_name for ref in answer.evidences} == {"first-only.pdf"}
    assert answer.evidences[0].doc_id == "11"
    assert answer.evidences[0].page_number == 1
    assert answer.evidences[0].quote == "The relevant coating fact is in this document."
    assert llm.verifier_candidates == []


def test_invalid_selected_directory_stops_before_any_model_call(tmp_path: Path) -> None:
    valid = "d" * 64
    missing = "e" * 64
    write_document(tmp_path / "glm", valid, file_id=1, file_name="valid.pdf", pages=["Valid document."])
    llm = RecordingLLM()
    service = make_service(tmp_path, llm)

    with pytest.raises(PlatformDocumentError) as captured:
        service.answer([valid, missing], ["1", "2"], "总结", "chat-3")

    assert captured.value.code == "DIR_NOT_FOUND"
    assert llm.requests == []


def test_second_request_reports_cache_hits_without_changing_source_ids(tmp_path: Path) -> None:
    dir_name = "f" * 64
    write_document(tmp_path / "glm", dir_name, file_id=501, file_name="cached.pdf", pages=["Cached document."])
    llm = RecordingLLM()
    service = make_service(tmp_path, llm)

    first = service.answer([dir_name], ["501"], "first", "chat-4")
    second = service.answer([dir_name], ["501"], "second", "chat-4")

    assert first.evidences[0].source_id == second.evidences[0].source_id
    assert first.debug is not None and first.debug.trajectory["cache_hits"] == 0
    assert second.debug is not None and second.debug.trajectory["cache_hits"] == 1


def test_same_chat_recalls_persisted_history_when_client_sends_only_current_message(tmp_path: Path) -> None:
    dir_name = "7" * 64
    write_document(tmp_path / "glm", dir_name, file_id=507, file_name="memory.pdf", pages=["Memory text."])
    llm = RecordingLLM()
    service = make_service(tmp_path, llm)

    first = service.answer(
        [dir_name],
        ["507"],
        "first question",
        "chat-memory",
        messages=[{"role": "user", "content": "first question"}],
    )
    second = service.answer(
        [dir_name],
        ["507"],
        "what did I ask before?",
        "chat-memory",
        messages=[{"role": "user", "content": "what did I ask before?"}],
    )

    assert "[user] first question" in llm.requests[1].user
    assert "[assistant] direct grounded answer" in llm.requests[1].user
    assert "[[ref:" not in llm.requests[1].user
    assert second.debug is not None
    assert second.debug.trajectory["conversation_memory"]["status"] == "hit"
    assert second.debug.trajectory["conversation_memory"]["committed"] is True


def test_different_chat_cannot_see_persisted_history(tmp_path: Path) -> None:
    dir_name = "6" * 64
    write_document(tmp_path / "glm", dir_name, file_id=506, file_name="isolated.pdf", pages=["Isolated text."])
    llm = RecordingLLM()
    service = make_service(tmp_path, llm)

    service.answer([dir_name], ["506"], "private question", "chat-a")
    service.answer([dir_name], ["506"], "new question", "chat-b")

    assert "private question" not in llm.requests[1].user


def test_full_request_history_bootstraps_once_without_duplication(tmp_path: Path) -> None:
    dir_name = "5" * 64
    write_document(tmp_path / "glm", dir_name, file_id=505, file_name="bootstrap.pdf", pages=["Bootstrap text."])
    llm = RecordingLLM()
    service = make_service(tmp_path, llm)

    service.answer(
        [dir_name],
        ["505"],
        "current question",
        "chat-bootstrap",
        messages=[
            {"role": "user", "content": "older question"},
            {"role": "assistant", "content": "older answer [[ref:ref_099]]"},
            {"role": "user", "content": "current question"},
        ],
    )
    service.answer(
        [dir_name],
        ["505"],
        "next question",
        "chat-bootstrap",
        messages=[
            {"role": "user", "content": "older question"},
            {"role": "assistant", "content": "older answer [[ref:ref_099]]"},
            {"role": "user", "content": "current question"},
            {"role": "assistant", "content": "direct grounded answer [[ref:ref_001]]"},
            {"role": "user", "content": "next question"},
        ],
    )

    assert [message.content for message in service.memory.read_messages("chat-bootstrap")] == [
        "older question",
        "older answer",
        "current question",
        "direct grounded answer",
        "next question",
        "direct grounded answer",
    ]
    assert "[[ref:" not in llm.requests[1].user


def test_model_failure_does_not_commit_conversation_exchange(tmp_path: Path) -> None:
    dir_name = "4" * 64
    write_document(tmp_path / "glm", dir_name, file_id=504, file_name="failure.pdf", pages=["Failure text."])
    memory = ConversationMemory(tmp_path / "memory.sqlite3")
    settings = make_settings(tmp_path)
    loader = PlatformDocumentLoader(settings.platform_data_root, settings.platform_cache_dir, settings)
    service = PlatformQAService(settings=settings, llm=FailingLLM(), loader=loader, memory=memory)

    with pytest.raises(RuntimeError, match="model failed"):
        service.answer([dir_name], ["504"], "failed question", "chat-failure")

    assert memory.read_messages("chat-failure") == []


def test_non_stream_cancellation_signal_prevents_memory_commit(tmp_path: Path) -> None:
    dir_name = "4" * 64
    write_document(tmp_path / "glm", dir_name, file_id=504, file_name="cancel.json.pdf", pages=["Cancel text."])
    cancelled = threading.Event()
    service = make_service(tmp_path, CancellingNonStreamingLLM(cancelled))

    with pytest.raises(LLMStreamCancelled, match="client cancelled"):
        service.answer(
            [dir_name],
            ["504"],
            "cancel non-stream",
            "chat-non-stream-cancel",
            cancelled=cancelled.is_set,
        )

    assert service.memory.read_messages("chat-non-stream-cancel") == []




def test_direct_stream_callback_uses_run_id_and_accumulates_answer(tmp_path: Path) -> None:
    dir_name = "1" * 64
    write_document(tmp_path / "glm", dir_name, file_id=501, file_name="stream.pdf", pages=["Stream text."])
    llm = RecordingStreamingLLM()
    service = make_service(tmp_path, llm)
    events: list[tuple[str, str]] = []

    answer = service.answer(
        [dir_name],
        ["501"],
        "summarize",
        "chat-stream",
        on_final_delta=lambda run_id, delta: events.append((run_id, delta)),
        cancelled=lambda: False,
    )

    assert events[-1][1] == " answer"
    assert "".join(delta for _, delta in events) == "direct streamed answer"
    assert {run_id for run_id, _ in events} == {answer.run_id}
    assert "".join(delta for _, delta in events) == answer.answer
    assert len(llm.stream_requests) == 1
    assert llm.stream_requests[0].purpose == "base"
    assert llm.requests == []
    assert [(message.role, message.content) for message in service.memory.read_messages("chat-stream")] == [
        ("user", "summarize"),
        ("assistant", "direct streamed answer"),
    ]


def test_direct_stream_filters_model_generated_reference_markers_before_delivery(tmp_path: Path) -> None:
    dir_name = "0" * 64
    write_document(tmp_path / "glm", dir_name, file_id=500, file_name="filtered.pdf", pages=["Filter text."])
    service = make_service(tmp_path, MarkerInjectingStreamingLLM())
    events: list[str] = []

    answer = service.answer(
        [dir_name],
        ["500"],
        "summarize",
        "chat-filtered-marker",
        on_final_delta=lambda _run_id, delta: events.append(delta),
        cancelled=lambda: False,
    )

    streamed = "".join(events)
    assert streamed == answer.answer == "direct streamed answer"
    assert "made_up" not in streamed
    assert answer.evidences == []
    assert service.memory.read_messages("chat-filtered-marker")[-1].content == "direct streamed answer"


def test_direct_stream_publishes_evidence_before_preserving_inline_reference(tmp_path: Path) -> None:
    dir_name = "6" * 64
    write_document(tmp_path / "glm", dir_name, file_id=506, file_name="inline.pdf", pages=["Supported claim."])
    service = make_service(tmp_path, InlineReferenceStreamingLLM())
    timeline: list[tuple[str, object]] = []

    answer = service.answer(
        [dir_name],
        ["506"],
        "summarize",
        "chat-inline-ref",
        on_stream_evidence=lambda _run_id, refs: timeline.append(("evidence", refs)),
        on_final_delta=lambda _run_id, delta: timeline.append(("delta", delta)),
        cancelled=lambda: False,
    )

    assert timeline[0][0] == "evidence"
    streamed = "".join(str(value) for kind, value in timeline if kind == "delta")
    assert streamed == answer.answer == "supported claim [[ref:ref_001]] conclusion"
    assert answer.evidences[0].file_name == "inline.pdf"
    assert answer.evidences[0].doc_id == "506"


def test_direct_stream_persists_completed_timing_metrics(tmp_path: Path) -> None:
    dir_name = "2" * 64
    write_document(tmp_path / "glm", dir_name, file_id=502, file_name="metrics.pdf", pages=["Metric text."])
    service = make_service(tmp_path, RecordingStreamingLLM())

    answer = service.answer(
        [dir_name],
        ["502"],
        "summarize",
        "chat-metrics",
        on_final_delta=lambda _run_id, _delta: None,
        cancelled=lambda: False,
    )

    saved = service.store.read_json(service.store.run_path(answer.run_id))
    trajectory = saved["answer"]["debug"]["trajectory"]
    assert trajectory["stream_status"] == "completed"
    assert trajectory["stream_chunk_count"] == 3
    assert trajectory["request_received_at"]
    assert trajectory["finalizer_started_at"]
    assert trajectory["first_token_at"]
    assert trajectory["completed_at"]
    assert trajectory["time_to_first_token_ms"] >= 0
    assert trajectory["stream_duration_ms"] >= 0


def test_stream_cancellation_does_not_commit_partial_exchange(tmp_path: Path) -> None:
    dir_name = "3" * 64
    write_document(tmp_path / "glm", dir_name, file_id=503, file_name="cancel.pdf", pages=["Cancel text."])
    memory = ConversationMemory(tmp_path / "memory.sqlite3")
    settings = make_settings(tmp_path)
    loader = PlatformDocumentLoader(settings.platform_data_root, settings.platform_cache_dir, settings)
    service = PlatformQAService(settings=settings, llm=CancellingStreamingLLM(), loader=loader, memory=memory)

    with pytest.raises(LLMStreamCancelled, match="client cancelled"):
        service.answer(
            [dir_name],
            ["503"],
            "cancel this",
            "chat-cancel",
            on_final_delta=lambda _run_id, _delta: None,
            cancelled=lambda: False,
        )

    assert memory.read_messages("chat-cancel") == []


def test_late_stream_cancellation_before_commit_does_not_persist_exchange(tmp_path: Path) -> None:
    dir_name = "3" * 64
    write_document(tmp_path / "glm", dir_name, file_id=503, file_name="late-cancel.pdf", pages=["Cancel text."])
    service = make_service(tmp_path, RecordingStreamingLLM())
    cancelled = threading.Event()

    with pytest.raises(LLMStreamCancelled, match="client cancelled"):
        service.answer(
            [dir_name],
            ["503"],
            "cancel after final delta",
            "chat-late-cancel",
            on_final_delta=lambda _run_id, _delta: cancelled.set(),
            cancelled=cancelled.is_set,
        )

    assert service.memory.read_messages("chat-late-cancel") == []


def test_run_persistence_failure_happens_before_memory_commit(tmp_path: Path) -> None:
    dir_name = "3" * 64
    write_document(tmp_path / "glm", dir_name, file_id=503, file_name="run-failure.pdf", pages=["Run text."])
    service = make_service(tmp_path, RecordingLLM())
    service.store = FailingRunStore()

    with pytest.raises(OSError, match="run persistence failed"):
        service.answer([dir_name], ["503"], "do not retain", "chat-run-failure")

    assert service.memory.read_messages("chat-run-failure") == []


def test_stream_cancellation_during_run_persistence_prevents_memory_commit(tmp_path: Path) -> None:
    dir_name = "3" * 64
    write_document(tmp_path / "glm", dir_name, file_id=503, file_name="run-cancel.pdf", pages=["Run text."])
    service = make_service(tmp_path, RecordingStreamingLLM())
    cancelled = threading.Event()
    service.store = CancellingRunStore(service.store, cancelled)

    with pytest.raises(LLMStreamCancelled, match="client cancelled"):
        service.answer(
            [dir_name],
            ["503"],
            "cancel during run write",
            "chat-run-cancel",
            on_final_delta=lambda _run_id, _delta: None,
            cancelled=cancelled.is_set,
        )

    assert service.memory.read_messages("chat-run-cancel") == []


def test_memory_read_failure_uses_request_history_without_blocking_answer(tmp_path: Path) -> None:
    dir_name = "2" * 64
    write_document(tmp_path / "glm", dir_name, file_id=502, file_name="read-fail.pdf", pages=["Read text."])
    settings = make_settings(tmp_path)
    loader = PlatformDocumentLoader(settings.platform_data_root, settings.platform_cache_dir, settings)
    llm = RecordingLLM()
    memory = ConversationMemory(tmp_path / "unused.sqlite3", store=ReadFailingMemoryStore())
    service = PlatformQAService(settings=settings, llm=llm, loader=loader, memory=memory)

    answer = service.answer(
        [dir_name],
        ["502"],
        "current",
        "chat-read-error",
        messages=[
            {"role": "user", "content": "request-carried history"},
            {"role": "user", "content": "current"},
        ],
    )

    assert answer.answer == "direct grounded answer [[ref:ref_001]]"
    assert "request-carried history" in llm.requests[0].user
    assert answer.debug is not None
    assert answer.debug.trajectory["conversation_memory"]["status"] == "read_error"


def test_memory_write_failure_does_not_block_completed_answer(tmp_path: Path) -> None:
    dir_name = "1" * 64
    write_document(tmp_path / "glm", dir_name, file_id=501, file_name="write-fail.pdf", pages=["Write text."])
    settings = make_settings(tmp_path)
    loader = PlatformDocumentLoader(settings.platform_data_root, settings.platform_cache_dir, settings)
    memory = ConversationMemory(tmp_path / "unused.sqlite3", store=WriteFailingMemoryStore())
    service = PlatformQAService(settings=settings, llm=RecordingLLM(), loader=loader, memory=memory)

    answer = service.answer([dir_name], ["501"], "write failure", "chat-write-error")

    assert answer.answer == "direct grounded answer [[ref:ref_001]]"
    assert answer.debug is not None
    memory_trace = answer.debug.trajectory["conversation_memory"]
    assert memory_trace["status"] == "write_error"
    assert memory_trace["committed"] is False

def test_readiness_requires_every_provider_role(tmp_path: Path) -> None:
    data_root = tmp_path / "glm"
    data_root.mkdir()
    cache_dir = tmp_path / "cache"
    configured = dict(
        platform_data_root=data_root,
        platform_cache_dir=cache_dir,
        glm_generation_api_keys=("glm-a", "glm-b", "glm-c"),
        deepseek_subagent_api_keys=("subagent",),
        deepseek_api_key="fallback",
    )

    ready = PlatformQAService(
        settings=Settings(runtime_dir=tmp_path / "runtime-ready", **configured),
        llm=RecordingLLM(),
    ).readiness()
    assert ready["ready"] is True
    assert ready["glm_generation_pool_configured"] is True
    assert ready["deepseek_subagent_pool_configured"] is True
    assert ready["deepseek_fallback_configured"] is True
    assert ready["base_model_configured"] is True
    assert ready["finalizer_configured"] is True

    for missing_field in (
        "glm_generation_api_keys",
        "deepseek_subagent_api_keys",
        "deepseek_api_key",
    ):
        missing = dict(configured)
        missing[missing_field] = () if missing_field.endswith("_keys") else ""
        readiness = PlatformQAService(
            settings=Settings(
                runtime_dir=tmp_path / f"runtime-missing-{missing_field}",
                **missing,
            ),
            llm=RecordingLLM(),
        ).readiness()
        assert readiness["ready"] is False
        assert readiness["model_configured"] is False


def test_memory_readiness_is_reported_without_blocking_core_qa_readiness(tmp_path: Path) -> None:
    data_root = tmp_path / "glm"
    data_root.mkdir()
    settings = Settings(
        runtime_dir=tmp_path / "runtime",
        platform_data_root=data_root,
        platform_cache_dir=tmp_path / "cache",
        glm_generation_api_keys=("glm-a", "glm-b", "glm-c"),
        deepseek_api_key="fallback",
        deepseek_subagent_api_keys=("subagent",),
    )
    unavailable_memory = ConversationMemory(tmp_path)
    service = PlatformQAService(settings=settings, llm=RecordingLLM(), memory=unavailable_memory)

    readiness = service.readiness()

    assert readiness["ready"] is True
    assert readiness["conversation_memory_enabled"] is True
    assert readiness["conversation_memory_ready"] is False


def test_service_close_releases_conversation_database(tmp_path: Path) -> None:
    data_root = tmp_path / "glm"
    data_root.mkdir()
    settings = Settings(
        runtime_dir=tmp_path / "runtime",
        platform_data_root=data_root,
        platform_cache_dir=tmp_path / "cache",
    )
    service = PlatformQAService(settings=settings, llm=RecordingLLM())
    database = settings.runtime_dir / "conversation_memory.sqlite3"

    assert database.exists()
    assert service.memory.ready is True

    service.close()

    assert service.memory.ready is False
    database.unlink()
    assert database.exists() is False
