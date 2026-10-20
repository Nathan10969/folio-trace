from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import time
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .chat_admission import ChatAdmission, ChatLease, ChatQueueFull, GlobalQueueFull
from .citation_annotations import (
    ParagraphReferenceDeduplicator,
    ReferenceMarkerStreamFilter,
    public_annotations,
    references_for_content,
)
from .contracts import AnswerContract, EvidenceRef, RLMQAError
from .ids import new_id
from .llm import LLMStreamCancelled
from .platform_documents import PlatformDocumentError
from .platform_service import PlatformQAService
from .request_workers import RequestWorkerRegistry


class Message(BaseModel):
    role: str
    content: str


class PlatformMetadata(BaseModel):
    chat_id: str = Field(default_factory=lambda: new_id("chat"))
    dir_names: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)


class PlatformChatBody(BaseModel):
    messages: list[Message]
    metadata: PlatformMetadata
    model: str = "precise"
    stream: bool = True


def create_app(service: PlatformQAService | None = None) -> FastAPI:
    qa_service = service or PlatformQAService()
    settings = getattr(qa_service, "settings", None)
    max_pending = getattr(settings, "conversation_memory_max_pending_per_chat", 8)
    max_global_requests = getattr(settings, "platform_max_inflight_requests", 64)
    admission = ChatAdmission(
        max_pending_per_chat=max_pending,
        max_global_requests=max_global_requests,
    )
    request_workers = RequestWorkerRegistry()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await request_workers.close()
            close = getattr(qa_service, "close", None)
            if callable(close):
                await asyncio.to_thread(close)

    app = FastAPI(title="FolioTrace — Platform Document QA", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        readiness = qa_service.readiness()
        if not readiness.get("ready"):
            raise HTTPException(status_code=503, detail=readiness)
        return readiness

    @app.post("/chat")
    async def chat(body: PlatformChatBody):
        request = _validated_request(body)
        if body.stream:
            return await _stream_response(
                qa_service,
                body,
                request,
                admission=admission,
                worker_registry=request_workers,
            )
        return await _run_admitted(
            admission,
            request_workers,
            request["chat_id"],
            _enhanced_answer,
            qa_service,
            request,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(body: PlatformChatBody):
        request = _validated_request(body)
        if body.stream:
            return await _stream_response(
                qa_service,
                body,
                request,
                admission=admission,
                worker_registry=request_workers,
            )
        answer = await _run_admitted(
            admission,
            request_workers,
            request["chat_id"],
            _answer_or_http_error,
            qa_service,
            request,
        )
        return _openai_response(answer, body.model)

    return app


app = create_app()


def _validated_request(body: PlatformChatBody) -> dict[str, Any]:
    chat_id = body.metadata.chat_id.strip()
    if not chat_id:
        raise HTTPException(
            status_code=400,
            detail={"code": "CHAT_ID_INVALID", "message": "metadata.chat_id must not be blank"},
        )
    if not body.metadata.dir_names:
        raise HTTPException(
            status_code=400,
            detail={"code": "NO_DOCUMENTS", "message": "metadata.dir_names must not be empty"},
        )
    if body.metadata.doc_ids and len(body.metadata.doc_ids) != len(body.metadata.dir_names):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "DOC_ID_COUNT_MISMATCH",
                "message": "metadata.doc_ids must align with metadata.dir_names",
            },
        )
    question = ""
    for message in reversed(body.messages):
        if message.role == "user" and message.content.strip():
            question = message.content.strip()
            break
    if not question:
        raise HTTPException(
            status_code=400,
            detail={"code": "QUESTION_MISSING", "message": "a non-empty user message is required"},
        )
    return {
        "dir_names": list(body.metadata.dir_names),
        "doc_ids": list(body.metadata.doc_ids),
        "question": question,
        "chat_id": chat_id,
        "messages": [message.model_dump() for message in body.messages],
    }


def _call_service(
    service: PlatformQAService,
    request: dict[str, Any],
    cancelled=None,
) -> AnswerContract:
    return service.answer(
        request["dir_names"],
        request["doc_ids"],
        request["question"],
        request["chat_id"],
        messages=request["messages"],
        cancelled=cancelled,
    )


def _call_service_streaming(
    service: PlatformQAService,
    request: dict[str, Any],
    on_final_delta,
    on_stream_evidence,
    cancelled,
) -> AnswerContract:
    return service.answer(
        request["dir_names"],
        request["doc_ids"],
        request["question"],
        request["chat_id"],
        messages=request["messages"],
        on_final_delta=on_final_delta,
        on_stream_evidence=on_stream_evidence,
        cancelled=cancelled,
    )


def _enhanced_answer(service: PlatformQAService, request: dict[str, Any], cancelled=None) -> dict[str, Any]:
    return _answer_or_http_error(service, request, cancelled=cancelled).to_dict()


def _answer_or_http_error(service: PlatformQAService, request: dict[str, Any], cancelled=None) -> AnswerContract:
    try:
        return _call_service(service, request, cancelled=cancelled)
    except PlatformDocumentError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "dir_name": exc.dir_name, "message": str(exc)},
        ) from None
    except RLMQAError:
        raise HTTPException(
            status_code=502,
            detail={"code": "DOCUMENT_QA_FAILED", "message": "document question answering failed"},
        ) from None


async def _stream_response(
    service: PlatformQAService,
    body: PlatformChatBody,
    request: dict[str, Any],
    *,
    admission: ChatAdmission | None = None,
    worker_registry: RequestWorkerRegistry | None = None,
) -> StreamingResponse:
    lease = await _acquire_or_429(admission, request["chat_id"])
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    cancel_event = threading.Event()

    def enqueue(kind: str, payload: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    def on_final_delta(run_id: str, delta: str) -> None:
        if not delta:
            return
        delivered = threading.Event()
        enqueue("delta", (run_id, delta, delivered))
        while not delivered.wait(timeout=0.05):
            if cancel_event.is_set():
                raise LLMStreamCancelled("client cancelled before streamed delta was delivered")
        if cancel_event.is_set():
            raise LLMStreamCancelled("client cancelled before streamed delta was delivered")

    def on_stream_evidence(run_id: str, refs: list[EvidenceRef]) -> None:
        delivered = threading.Event()
        enqueue("evidence", (run_id, refs, delivered))
        while not delivered.wait(timeout=0.05):
            if cancel_event.is_set():
                raise LLMStreamCancelled("client cancelled before stream evidence was delivered")

    def worker() -> None:
        try:
            answer = _call_service_streaming(
                service,
                request,
                on_final_delta,
                on_stream_evidence,
                cancel_event.is_set,
            )
        except PlatformDocumentError as exc:
            enqueue(
                "error",
                {"code": exc.code, "dir_name": exc.dir_name, "message": str(exc)},
            )
        except Exception:
            enqueue(
                "error",
                {"code": "DOCUMENT_QA_FAILED", "message": "document question answering failed"},
            )
        else:
            enqueue("result", answer)

    worker_awaitable = asyncio.to_thread(worker)
    worker_task = (
        worker_registry.start(worker_awaitable, cancel_event)
        if worker_registry is not None
        else asyncio.create_task(worker_awaitable)
    )
    try:
        first_event = await queue.get()
    except asyncio.CancelledError as exc:
        cancel_event.set()
        cancellation = await _drain_stream_worker(worker_task)
        if lease is not None:
            await lease.release()
        raise cancellation or exc

    async def events():
        event = first_event
        first_delta = True
        created = int(time.time())
        pending_delivery: threading.Event | None = None
        stream_refs: list[EvidenceRef] = []
        marker_filter = ReferenceMarkerStreamFilter()
        paragraph_filter: ParagraphReferenceDeduplicator | None = None
        try:
            while True:
                kind, payload = event
                if kind == "evidence":
                    _run_id, stream_refs, pending_delivery = payload
                    marker_filter = ReferenceMarkerStreamFilter(stream_refs)
                    paragraph_filter = ParagraphReferenceDeduplicator(stream_refs)
                    pending_delivery.set()
                    pending_delivery = None
                elif kind == "delta":
                    run_id, delta, pending_delivery = payload
                    if marker_filter is not None:
                        delta = marker_filter.feed(delta)
                    if paragraph_filter is not None:
                        delta = paragraph_filter.feed(delta)
                    if not delta:
                        pending_delivery.set()
                        pending_delivery = None
                        event = await queue.get()
                        continue
                    content = {"content": delta}
                    annotations = public_annotations(references_for_content(stream_refs, delta))
                    if annotations:
                        content["annotations"] = annotations
                    if first_delta:
                        content = {"role": "assistant", **content}
                        first_delta = False
                    yield _sse_data(_openai_chunk(run_id, body.model, created, content, None))
                    pending_delivery.set()
                    pending_delivery = None
                elif kind == "error":
                    yield _sse_data({"error": payload})
                    yield "data: [DONE]\n\n"
                    return
                elif kind == "result":
                    answer: AnswerContract = payload
                    if marker_filter is not None:
                        tail = marker_filter.finish()
                        if paragraph_filter is not None:
                            tail = paragraph_filter.feed(tail)
                        if tail:
                            content = {"content": tail}
                            annotations = public_annotations(references_for_content(stream_refs, tail))
                            if annotations:
                                content["annotations"] = annotations
                            if first_delta:
                                content = {"role": "assistant", **content}
                                first_delta = False
                            yield _sse_data(_openai_chunk(answer.run_id, body.model, created, content, None))
                    if first_delta:
                        yield _sse_data(
                            {"error": {"code": "EMPTY_STREAM", "message": "model returned no streamed content"}}
                        )
                        yield "data: [DONE]\n\n"
                        return
                    final = _openai_chunk(
                        answer.run_id,
                        body.model,
                        created,
                        {},
                        "stop",
                    )
                    final["rlm"] = _rlm_summary(answer)
                    yield _sse_data(final)
                    yield "data: [DONE]\n\n"
                    return
                event = await queue.get()
        finally:
            cancel_event.set()
            if pending_delivery is not None:
                pending_delivery.set()
            cancellation = await _drain_stream_worker(worker_task)
            if lease is not None:
                await lease.release()
            if cancellation is not None:
                raise cancellation

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _run_admitted(
    admission: ChatAdmission,
    worker_registry: RequestWorkerRegistry,
    chat_id: str,
    function,
    *args,
):
    lease = await _acquire_or_429(admission, chat_id)
    cancel_event = threading.Event()
    worker_task = worker_registry.start(
        asyncio.to_thread(function, *args, cancelled=cancel_event.is_set),
        cancel_event,
    )
    cancellation: asyncio.CancelledError | None = None
    try:
        while not worker_task.done():
            try:
                await asyncio.shield(worker_task)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
                cancel_event.set()
        if cancellation is not None:
            try:
                worker_task.result()
            except BaseException:
                pass
            raise cancellation
        return worker_task.result()
    finally:
        if lease is not None:
            await lease.release()


async def _acquire_or_429(admission: ChatAdmission | None, chat_id: str) -> ChatLease | None:
    if admission is None:
        return None
    try:
        return await admission.acquire(chat_id)
    except ChatQueueFull:
        raise HTTPException(
            status_code=429,
            detail={"code": "CHAT_QUEUE_FULL", "message": "too many pending requests for this chat"},
        ) from None
    except GlobalQueueFull:
        raise HTTPException(
            status_code=429,
            detail={"code": "REQUEST_QUEUE_FULL", "message": "too many active or pending requests"},
        ) from None


async def _drain_stream_worker(worker_task: asyncio.Task[Any]) -> asyncio.CancelledError | None:
    cancellation: asyncio.CancelledError | None = None
    while not worker_task.done():
        try:
            await asyncio.shield(worker_task)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
        except Exception:
            break
    if worker_task.done():
        try:
            worker_task.result()
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
        except Exception:
            pass
    return cancellation


def _openai_response(answer: AnswerContract, model: str) -> dict[str, Any]:
    return {
        "id": answer.run_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer.answer,
                    "annotations": public_annotations(answer.evidences),
                },
                "finish_reason": "stop",
            }
        ],
        "rlm": _rlm_summary(answer),
    }


def _openai_chunk(
    run_id: str,
    model: str,
    created: int,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": run_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _rlm_summary(answer: AnswerContract) -> dict[str, Any]:
    return {
        "run_id": answer.run_id,
        "mode": answer.mode,
        "answer_type": answer.answer_type,
        "citation_policy": answer.citation_policy,
        "evidences": [
            {
                "evidence_id": ref.evidence_id,
                "source_id": ref.source_id,
                "file_name": ref.file_name,
                "page_number": ref.page_number,
                "quote": ref.quote,
                "quote_source": ref.quote_source,
                "doc_id": ref.doc_id,
                "page_end": ref.page_end or ref.page_number,
                "ref_id": ref.ref_id,
                "score": ref.score,
            }
            for ref in answer.evidences
        ],
    }


def _sse_data(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
