from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
import time
from typing import Any, Callable

from .citation_annotations import (
    ReferenceMarkerStreamFilter,
    canonicalize_reference_markers,
    deduplicate_paragraph_references,
    prepare_stream_refs,
    references_for_content,
    strip_reference_markers,
)
from .config import Settings
from .conversation_memory import ConversationMemory
from .contracts import AnswerContract, AnswerDebug, Chunk, DocProfile, Evidence, EvidenceRef, NoEvidenceError, Page
from .document_map_reduce import CoveragePlan, DocumentMapReduceRunner
from .evidence_quotes import normalize_evidence_refs
from .ids import new_id
from .llm import ChatRequest, LLMClient, LLMStreamCancelled, ModelRouterLLM
from .platform_documents import PlatformDocument, PlatformDocumentLoader
from .storage import FileStore
from .turn_executor import (
    DirectTurnResponse,
    DocumentInspection,
    InvalidExecutionContract,
    build_turn_request,
    parse_turn_response,
)


class PlatformQAService:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        loader: PlatformDocumentLoader | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.llm = llm or ModelRouterLLM(self.settings)
        self.loader = loader or PlatformDocumentLoader(
            self.settings.platform_data_root,
            self.settings.platform_cache_dir,
            self.settings,
        )
        self.store = FileStore(self.settings.runtime_dir)
        self.store.init()
        self.memory = memory or ConversationMemory.from_settings(self.settings)

    def answer(
        self,
        dir_names: list[str],
        doc_ids: list[str],
        question: str,
        chat_id: str,
        *,
        messages: list[dict[str, str]] | None = None,
        on_final_delta: Callable[[str, str], None] | None = None,
        on_stream_evidence: Callable[[str, list[EvidenceRef]], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AnswerContract:
        run_id = new_id("run")
        request_started = time.perf_counter()
        stream_trace: dict[str, Any] = {}
        partial_answer: list[str] = []
        is_streaming = on_final_delta is not None
        if is_streaming:
            stream_trace["request_received_at"] = _utc_now()
            stream_trace["stream_chunk_count"] = 0

        def emit_final_delta(delta: str) -> None:
            if not delta or on_final_delta is None:
                return
            now = time.perf_counter()
            if "first_token_at" not in stream_trace:
                stream_trace["first_token_at"] = _utc_now()
                stream_trace["_first_token_monotonic"] = now
            stream_trace["stream_chunk_count"] += 1
            partial_answer.append(delta)
            on_final_delta(run_id, delta)

        def emit_stream_evidence(refs: list[EvidenceRef]) -> None:
            if on_stream_evidence is not None and refs:
                on_stream_evidence(run_id, refs)

        cancel_check = cancelled or (lambda: False)
        with self.memory.session(chat_id) as memory_session:
            prepared_memory = memory_session.prepare(question, _memory_safe_messages(messages or []))
            model_question = memory_session.model_question(question, prepared_memory)
            documents: list[PlatformDocument] = []
            try:
                if self.settings.turn_execution_contract_enabled and getattr(
                    self.llm, "supports_tool_calls", False
                ):
                    catalog = [asdict(handle) for handle in self.loader.catalog_many(dir_names, doc_ids)]
                    trace_marker = _key_pool_trace_marker(self.llm)
                    turn_request = build_turn_request(
                        question=question,
                        prepared_memory=prepared_memory,
                        documents=catalog,
                        run_id=run_id,
                        model=self.settings.glm_generation_model,
                    )
                    streamed_turn_answer = ""
                    with _trace_context(self.llm, run_id):
                        if is_streaming:
                            stream_trace["orchestrator_started_at"] = _utc_now()
                            stream_trace["_finalizer_started_monotonic"] = time.perf_counter()
                            safe_delta, finish_turn_stream = _marker_safe_delta_stream(emit_final_delta, [])
                            turn_response = self.llm.chat_stream(turn_request, safe_delta, cancel_check)
                            streamed_turn_answer = finish_turn_stream()
                        else:
                            turn_response = self.llm.chat(turn_request)
                    turn = parse_turn_response(turn_response, catalog)
                    if isinstance(turn, DirectTurnResponse):
                        answer_text = streamed_turn_answer or strip_reference_markers(turn.answer)
                        if is_streaming and not streamed_turn_answer and answer_text:
                            emit_final_delta(answer_text)
                        answer = self._root_direct_answer(
                            run_id,
                            answer_text,
                            catalog=catalog,
                            trace_marker=trace_marker,
                        )
                    else:
                        selected = _selected_catalog_entries(catalog, turn)
                        documents = self.loader.load_many(
                            [entry["dir_name"] for entry in selected],
                            [entry["doc_id"] for entry in selected],
                        )
                        answer_kwargs = {
                            "on_final_delta": emit_final_delta if is_streaming else None,
                            "on_stream_evidence": emit_stream_evidence if is_streaming else None,
                            "cancelled": cancel_check,
                            "stream_trace": stream_trace,
                        }
                        document_result = "answered"
                        if self._aggregate_direct_ready(documents):
                            answer = self._answer_direct(
                                run_id,
                                documents,
                                turn.query,
                                turn.query,
                                **answer_kwargs,
                            )
                        else:
                            try:
                                answer = self._answer_rlm(
                                    run_id,
                                    documents,
                                    turn.query,
                                    turn.query,
                                    coverage_plan=_coverage_plan_from_turn(turn),
                                    **answer_kwargs,
                                )
                            except NoEvidenceError:
                                answer = self._answer_after_no_evidence(
                                    run_id,
                                    turn.query,
                                    selected,
                                    on_final_delta=emit_final_delta if is_streaming else None,
                                    cancelled=cancel_check,
                                    stream_trace=stream_trace,
                                )
                                document_result = "no_relevant_evidence"
                        answer = _answer_with_execution_contract(
                            answer,
                            turn,
                            catalog_count=len(catalog),
                            key_pool_trace=_key_pool_trace_since(self.llm, trace_marker),
                            document_result=document_result,
                        )
                else:
                    documents = self.loader.load_many(dir_names, doc_ids)
                    answer_method = (
                        self._answer_direct if self._aggregate_direct_ready(documents) else self._answer_rlm
                    )
                    answer = answer_method(
                        run_id,
                        documents,
                        question,
                        model_question,
                        on_final_delta=emit_final_delta if is_streaming else None,
                        on_stream_evidence=emit_stream_evidence if is_streaming else None,
                        cancelled=cancel_check,
                        stream_trace=stream_trace,
                    )
                if is_streaming:
                    _complete_stream_trace(stream_trace, request_started, status="completed")
                    answer = _answer_with_stream_trace(answer, stream_trace)
                if cancel_check():
                    raise LLMStreamCancelled("client cancelled before conversation commit")
            except Exception as exc:
                if is_streaming:
                    status = "client_cancelled" if isinstance(exc, LLMStreamCancelled) else "stream_failed"
                    _complete_stream_trace(stream_trace, request_started, status=status)
                    self.store.write_json(
                        self.store.run_path(run_id),
                        {
                            "chat_id": chat_id,
                            "question": question,
                            "dir_names": dir_names,
                            "doc_ids": [document.doc_id for document in documents] or doc_ids,
                            "partial_answer": "".join(partial_answer),
                            "stream": _public_stream_trace(stream_trace),
                            "error_code": type(exc).__name__,
                        },
                    )
                raise
            run_path = self.store.run_path(run_id)
            run_payload = {
                "chat_id": chat_id,
                "question": question,
                "dir_names": dir_names,
                "doc_ids": [document.doc_id for document in documents] or doc_ids,
                "answer": answer,
            }
            self.store.write_json(run_path, run_payload)
            if cancel_check():
                raise LLMStreamCancelled("client cancelled before conversation commit")
            memory_trace = memory_session.commit(question, strip_reference_markers(answer.answer), run_id)
            answer = _answer_with_memory_trace(answer, memory_trace)
            try:
                self.store.write_json(run_path, {**run_payload, "answer": answer})
            except OSError:
                pass
            return answer

    def readiness(self) -> dict[str, Any]:
        data_root = self.settings.platform_data_root
        cache_dir = self.settings.platform_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        generation_ready = bool(self.settings.glm_generation_api_keys)
        subagent_ready = bool(self.settings.deepseek_subagent_api_keys)
        fallback_ready = bool(self.settings.deepseek_api_key)
        model_ready = generation_ready and subagent_ready and fallback_ready
        return {
            "ready": data_root.is_dir() and cache_dir.is_dir() and model_ready,
            "data_root_readable": data_root.is_dir(),
            "cache_dir_writable": cache_dir.is_dir(),
            "model_configured": model_ready,
            "base_model_configured": generation_ready,
            "subagent_pool_configured": subagent_ready,
            "finalizer_configured": generation_ready,
            "glm_generation_pool_configured": generation_ready,
            "deepseek_subagent_pool_configured": subagent_ready,
            "deepseek_fallback_configured": fallback_ready,
            "conversation_memory_enabled": self.settings.conversation_memory_enabled,
            "conversation_memory_ready": self.memory.ready,
            "turn_execution_contract_enabled": self.settings.turn_execution_contract_enabled,
        }

    def close(self) -> None:
        self.memory.close()

    def _aggregate_direct_ready(self, documents: list[PlatformDocument]) -> bool:
        return (
            bool(documents)
            and sum(document.profile.page_count for document in documents) <= self.settings.direct_max_pages
            and sum(document.profile.token_estimate for document in documents) <= self.settings.direct_max_tokens
            and all(document.evidences for document in documents)
        )

    def _answer_direct(
        self,
        run_id: str,
        documents: list[PlatformDocument],
        question: str,
        model_question: str,
        *,
        on_final_delta: Callable[[str], None] | None = None,
        on_stream_evidence: Callable[[list[EvidenceRef]], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        stream_trace: dict[str, Any] | None = None,
    ) -> AnswerContract:
        pages = _pages(documents)
        chunks = _chunks(documents)
        refs = prepare_stream_refs(
            normalize_evidence_refs(_evidence_refs(documents), chunks, pages, question=question, answer="")
        )
        context = _direct_context(documents)
        system = (
            "Answer in Chinese unless the user asks otherwise. "
            "Use the complete page-tagged selected-document context as the primary source for document claims. "
            "If the selected documents do not answer part of the question, state that boundary and then use general knowledge when useful. "
            "Do not invent PDF pages, titles, quotations, or facts. Do not mention internal ids or routing. "
            "For each document-backed claim, add the matching [[ref:ref_NNN]] marker immediately after the claim. "
            "Use only the citation markers listed in the citation guide."
        )
        trace_marker = _key_pool_trace_marker(self.llm)
        request = ChatRequest(
            system=system,
            user=(
                f"Question:\n{model_question}\n\nSelected document context:\n{context}\n\n"
                f"Citation guide:\n{_citation_guide(refs)}"
            ),
            model=self.settings.glm_generation_model,
            run_id=run_id,
            purpose="base",
        )
        with _trace_context(self.llm, run_id):
            if on_stream_evidence is not None:
                on_stream_evidence(refs)
            if on_final_delta is not None:
                trace = stream_trace if stream_trace is not None else {}
                trace["finalizer_started_at"] = _utc_now()
                trace["_finalizer_started_monotonic"] = time.perf_counter()
                safe_delta, finish_stream = _marker_safe_delta_stream(on_final_delta, refs)
                self.llm.chat_stream(
                    request,
                    safe_delta,
                    cancelled or (lambda: False),
                )
                answer_text = finish_stream()
            else:
                answer_text = self.llm.chat(request).answer
        answer_text, display_refs, citation_trace = self._attach_citations(
            answer=answer_text,
            refs=refs,
            cancelled=cancelled,
        )
        return AnswerContract(
            run_id=run_id,
            mode="direct",
            answer_type="document_grounded" if display_refs else "document_assisted",
            answer=answer_text,
            evidences=display_refs,
            citation_policy="pdf_page_citation" if display_refs else "no_pdf_citation",
            debug=AnswerDebug(
                run_id=run_id,
                program_steps=["direct_full_context"],
                trajectory={
                    **_platform_trace(documents),
                    "strategy": "direct_full_context",
                    "citation_annotations": citation_trace,
                    "key_pool_trace": _key_pool_trace_since(self.llm, trace_marker),
                },
                runtime_status="direct_success",
            ),
        )

    def _root_direct_answer(
        self,
        run_id: str,
        answer: str,
        *,
        catalog: list[dict[str, Any]],
        trace_marker: int,
    ) -> AnswerContract:
        return AnswerContract(
            run_id=run_id,
            mode="base",
            answer_type="conversation",
            answer=answer,
            evidences=[],
            citation_policy="no_pdf_citation",
            debug=AnswerDebug(
                run_id=run_id,
                program_steps=["turn_executor_direct"],
                trajectory={
                    "execution_contract": {
                        "action": "respond",
                        "catalog_count": len(catalog),
                        "available_document_count": sum(
                            1 for entry in catalog if entry.get("status") == "available"
                        ),
                        "document_accessed": False,
                    },
                    "key_pool_trace": _key_pool_trace_since(self.llm, trace_marker),
                },
                runtime_status="conversation_success",
            ),
        )

    def _answer_after_no_evidence(
        self,
        run_id: str,
        query: str,
        selected_catalog: list[dict[str, Any]],
        *,
        on_final_delta: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        stream_trace: dict[str, Any] | None = None,
    ) -> AnswerContract:
        system = (
            "The selected-document inspection found no relevant evidence for the request. "
            "Answer naturally in Chinese unless the user asks otherwise. If the request asks "
            "whether the selected documents contain a topic, clearly state that no matching "
            "content was found. General knowledge may be added only when useful and must not be "
            "presented as coming from the documents. Do not add PDF citations or expose routing."
        )
        document_metadata = [
            {
                "file_name": entry.get("file_name"),
                "page_count": entry.get("page_count"),
            }
            for entry in selected_catalog
        ]
        request = ChatRequest(
            system=system,
            user=(
                f"Current request:\n{query}\n\n"
                f"Inspected document metadata:\n{document_metadata}"
            ),
            model=self.settings.glm_generation_model,
            run_id=run_id,
            purpose="base",
        )
        trace_marker = _key_pool_trace_marker(self.llm)
        with _trace_context(self.llm, run_id):
            if on_final_delta is not None:
                trace = stream_trace if stream_trace is not None else {}
                trace["finalizer_started_at"] = _utc_now()
                trace["_finalizer_started_monotonic"] = time.perf_counter()
                safe_delta, finish_stream = _marker_safe_delta_stream(on_final_delta, [])
                self.llm.chat_stream(request, safe_delta, cancelled or (lambda: False))
                answer_text = finish_stream()
            else:
                answer_text = strip_reference_markers(self.llm.chat(request).answer)
        return AnswerContract(
            run_id=run_id,
            mode="base",
            answer_type="document_not_found",
            answer=answer_text,
            evidences=[],
            citation_policy="no_pdf_citation",
            debug=AnswerDebug(
                run_id=run_id,
                program_steps=["document_no_evidence_base_answer"],
                trajectory={
                    "strategy": "base_after_no_relevant_evidence",
                    "key_pool_trace": _key_pool_trace_since(self.llm, trace_marker),
                },
                runtime_status="no_relevant_evidence",
            ),
        )

    def _answer_rlm(
        self,
        run_id: str,
        documents: list[PlatformDocument],
        question: str,
        model_question: str,
        *,
        on_final_delta: Callable[[str], None] | None = None,
        on_stream_evidence: Callable[[list[EvidenceRef]], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        stream_trace: dict[str, Any] | None = None,
        coverage_plan: CoveragePlan | None = None,
    ) -> AnswerContract:
        profiles = _profiles(documents)
        pages = _pages(documents)
        chunks = _chunks(documents)
        evidences = _evidences(documents)
        trace_marker = _key_pool_trace_marker(self.llm)
        runner_delta = on_final_delta
        with _trace_context(self.llm, run_id):
            document_answer = DocumentMapReduceRunner(
                llm=self.llm,
                subagent_model=self.settings.deepseek_subagent_model,
                max_workers=self.settings.rlm_batch_max_workers,
                max_deep_packs=self.settings.rlm_max_deep_packs,
                map_execution_mode=self.settings.rlm_map_execution_mode,
                reduce_context_mode=self.settings.rlm_reduce_context_mode,
                fallback_evidence_per_pack=self.settings.rlm_fallback_evidence_per_pack,
                run_id=run_id,
                max_timeout_seconds=self.settings.rlm_max_timeout_seconds,
                max_tokens=self.settings.rlm_max_tokens,
                max_errors=self.settings.rlm_max_errors,
                max_budget=self.settings.rlm_max_budget,
                cost_per_1k_tokens=self.settings.rlm_cost_per_1k_tokens,
                on_final_delta=runner_delta,
                on_stream_evidence=(
                    (lambda refs: on_stream_evidence(_bind_platform_refs(refs, documents)))
                    if on_stream_evidence is not None
                    else None
                ),
                cancelled=cancelled,
                stream_trace=stream_trace,
                coverage_plan=coverage_plan,
            ).answer_question_many(model_question, profiles, pages, chunks, evidences)
        normalized = _bind_platform_refs(document_answer.evidences, documents)
        final_answer, display_refs, citation_trace = self._attach_citations(
            answer=document_answer.answer,
            refs=normalized,
            cancelled=cancelled,
        )
        trajectory = {
            **dict(document_answer.trajectory or {}),
            **_platform_trace(documents),
            "citation_annotations": citation_trace,
            "key_pool_trace": _key_pool_trace_since(self.llm, trace_marker),
        }
        metadata_only = bool(document_answer.metadata_refs) and not normalized
        if metadata_only:
            answer_type = "metadata_answer"
            citation_policy = "document_metadata"
        elif display_refs:
            answer_type = "document_grounded" if document_answer.runtime_status == "rlm_success" else "document_recovered"
            citation_policy = "pdf_page_citation"
        else:
            answer_type = "document_assisted"
            citation_policy = "no_pdf_citation"
        return AnswerContract(
            run_id=run_id,
            mode="rlm",
            answer_type=answer_type,
            answer=final_answer,
            evidences=display_refs,
            citation_policy=citation_policy,
            debug=AnswerDebug(
                run_id=run_id,
                program_steps=["rlm_map_reduce"],
                trajectory=trajectory,
                runtime_status=document_answer.runtime_status,
                metadata_refs=document_answer.metadata_refs,
            ),
        )

    def _attach_citations(
        self,
        *,
        answer: str,
        refs: list[EvidenceRef],
        cancelled: Callable[[], bool] | None,
    ) -> tuple[str, list[EvidenceRef], dict[str, Any]]:
        cancel_check = cancelled or (lambda: False)
        answer = canonicalize_reference_markers(answer, refs)
        answer = deduplicate_paragraph_references(answer, refs)
        if cancel_check():
            raise LLMStreamCancelled("client cancelled before citation completion")
        inline_refs = references_for_content(refs, answer)
        if inline_refs:
            return answer, inline_refs, {
                "status": "inline_local",
                "mode": "inline",
                "candidate_count": len(refs),
                "selected_count": len(inline_refs),
                "verifier_called": False,
            }
        return answer, [], {
            "status": "no_inline_references",
            "mode": "none",
            "candidate_count": len(refs),
            "selected_count": 0,
            "verifier_called": False,
        }


def _marker_safe_delta_stream(
    on_delta: Callable[[str], None],
    refs: list[EvidenceRef] | None = None,
) -> tuple[Callable[[str], None], Callable[[], str]]:
    stream_filter = ReferenceMarkerStreamFilter(refs)
    parts: list[str] = []

    def feed(delta: str) -> None:
        safe_delta = stream_filter.feed(delta)
        if safe_delta:
            parts.append(safe_delta)
            on_delta(safe_delta)

    def finish() -> str:
        tail = stream_filter.finish()
        if tail:
            parts.append(tail)
            on_delta(tail)
        return "".join(parts)

    return feed, finish



def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _complete_stream_trace(trace: dict[str, Any], request_started: float, *, status: str) -> None:
    completed = time.perf_counter()
    trace["completed_at"] = _utc_now()
    trace["stream_status"] = status
    first_token = trace.get("_first_token_monotonic")
    finalizer_started = trace.get("_finalizer_started_monotonic")
    trace["time_to_first_token_ms"] = (
        max(0, int((first_token - request_started) * 1000)) if isinstance(first_token, (int, float)) else None
    )
    trace["pre_finalizer_ms"] = (
        max(0, int((finalizer_started - request_started) * 1000))
        if isinstance(finalizer_started, (int, float))
        else None
    )
    trace["stream_duration_ms"] = (
        max(0, int((completed - first_token) * 1000)) if isinstance(first_token, (int, float)) else 0
    )


def _public_stream_trace(trace: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in trace.items() if not key.startswith("_")}


def _answer_with_stream_trace(answer: AnswerContract, stream_trace: dict[str, Any]) -> AnswerContract:
    if answer.debug is None:
        return answer
    debug = replace(answer.debug, trajectory={**answer.debug.trajectory, **_public_stream_trace(stream_trace)})
    return replace(answer, debug=debug)


def _answer_with_memory_trace(answer: AnswerContract, memory_trace: dict[str, Any]) -> AnswerContract:
    if answer.debug is None:
        return answer
    debug = replace(
        answer.debug,
        trajectory={**answer.debug.trajectory, "conversation_memory": dict(memory_trace)},
    )
    return replace(answer, debug=debug)


def _selected_catalog_entries(
    catalog: list[dict[str, Any]],
    turn: DocumentInspection,
) -> list[dict[str, Any]]:
    by_source = {
        str(entry.get("source_id") or ""): entry
        for entry in catalog
        if entry.get("status") == "available"
    }
    selected = [by_source[source_id] for source_id in turn.source_ids if source_id in by_source]
    if not selected:
        raise InvalidExecutionContract("document tool selected no available documents")
    return selected


def _coverage_plan_from_turn(turn: DocumentInspection) -> CoveragePlan:
    return CoveragePlan(
        coverage="all_selected" if turn.coverage == "all_in_scope" else "relevant_subset",
        answer_shape="per_document" if turn.answer_shape == "per_source" else "synthesis",
        reason="root execution contract",
    )


def _answer_with_execution_contract(
    answer: AnswerContract,
    turn: DocumentInspection,
    *,
    catalog_count: int,
    key_pool_trace: list[dict[str, Any]],
    document_result: str,
) -> AnswerContract:
    if answer.debug is None:
        return answer
    execution_contract = {
        "action": "inspect_documents",
        "call_id": turn.call_id,
        "scope_mode": turn.scope_mode,
        "coverage": turn.coverage,
        "answer_shape": turn.answer_shape,
        "selected_source_ids": list(turn.source_ids),
        "catalog_count": catalog_count,
        "document_accessed": True,
        "document_result": document_result,
    }
    debug = replace(
        answer.debug,
        program_steps=["turn_executor", *answer.debug.program_steps],
        trajectory={
            **answer.debug.trajectory,
            "execution_contract": execution_contract,
            "key_pool_trace": key_pool_trace,
        },
    )
    return replace(answer, debug=debug)


def _profiles(documents: list[PlatformDocument]) -> list[DocProfile]:
    return [document.profile for document in documents]


def _memory_safe_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            **message,
            "content": strip_reference_markers(str(message.get("content") or "")),
        }
        if str(message.get("role") or "").strip() == "assistant"
        else dict(message)
        for message in messages
    ]


def _pages(documents: list[PlatformDocument]) -> list[Page]:
    return [page for document in documents for page in document.pages]


def _chunks(documents: list[PlatformDocument]) -> list[Chunk]:
    return [chunk for document in documents for chunk in document.chunks]


def _evidences(documents: list[PlatformDocument]) -> list[Evidence]:
    return [evidence for document in documents for evidence in document.evidences]


def _evidence_refs(documents: list[PlatformDocument]) -> list[EvidenceRef]:
    return [
        EvidenceRef(
            evidence_id=evidence.evidence_id,
            source_id=evidence.source_id,
            chunk_id=evidence.chunk_id,
            page_number=evidence.page_number,
            quote=evidence.quote,
            file_name=document.profile.file_name,
            doc_id=document.doc_id,
            page_end=_chunk_page_end(document.chunks, evidence.chunk_id, evidence.page_number),
        )
        for document in documents
        for evidence in document.evidences
    ]


def _bind_platform_refs(refs: list[EvidenceRef], documents: list[PlatformDocument]) -> list[EvidenceRef]:
    documents_by_source = {document.profile.source_id: document for document in documents}
    bound: list[EvidenceRef] = []
    for ref in refs:
        document = documents_by_source.get(ref.source_id)
        if document is None:
            continue
        bound.append(
            replace(
                ref,
                file_name=document.profile.file_name,
                doc_id=document.doc_id,
                page_end=_chunk_page_end(document.chunks, ref.chunk_id, ref.page_number),
            )
        )
    return bound


def _chunk_page_end(chunks: list[Chunk], chunk_id: str, fallback: int) -> int:
    return next((chunk.page_end for chunk in chunks if chunk.chunk_id == chunk_id), fallback)


def _direct_context(documents: list[PlatformDocument]) -> str:
    parts: list[str] = []
    for document in documents:
        parts.append(
            f"[DOCUMENT file={document.profile.file_name} source={document.profile.source_id} "
            f"dir_name={document.dir_name} doc_id={document.doc_id} pages={document.profile.page_count}]"
        )
        for page in document.pages:
            if page.text.strip():
                parts.append(f"[{document.profile.file_name} | page {page.page_number}]\n{page.text}")
    return "\n\n".join(parts)


def _citation_guide(refs: list[EvidenceRef]) -> str:
    return "\n".join(
        f"{ref.ref_id}: {ref.file_name}, page {ref.page_number}, evidence {ref.evidence_id}, "
        f"quote={ref.quote[:400]}"
        for ref in refs
        if ref.ref_id
    ) or "(none)"


def _platform_trace(documents: list[PlatformDocument]) -> dict[str, Any]:
    return {
        "dir_names": [document.dir_name for document in documents],
        "doc_ids": [document.doc_id for document in documents],
        "source_ids": [document.profile.source_id for document in documents],
        "fingerprints": [document.fingerprint for document in documents],
        "cache_hits": sum(1 for document in documents if document.cache_hit),
    }


def _trace_context(llm: LLMClient, run_id: str):
    factory = getattr(llm, "trace_run", None)
    return factory(run_id) if callable(factory) else nullcontext()


def _key_pool_trace_marker(llm: LLMClient) -> int:
    reader = getattr(llm, "key_pool_trace", None)
    if not callable(reader):
        return 0
    return len(reader())


def _key_pool_trace_since(llm: LLMClient, marker: int) -> list[dict[str, Any]]:
    reader = getattr(llm, "key_pool_trace", None)
    if not callable(reader):
        return []
    return list(reader())[marker:]
