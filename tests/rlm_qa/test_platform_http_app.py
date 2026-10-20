from __future__ import annotations

import asyncio
import json
import re
import threading

from starlette.testclient import TestClient
import pytest

from rlm_qa.contracts import AnswerContract, AnswerDebug, EvidenceRef
from rlm_qa.platform_documents import PlatformDocumentError
from rlm_qa.platform_http_app import PlatformChatBody, _stream_response, _validated_request, create_app


DIR_NAME = "a" * 64
ANSWER_TEXT = "单壁碳纳米管是由单层石墨烯卷曲形成的圆柱状碳材料。"
PAYLOAD = {
    "messages": [{"role": "user", "content": "什么是单壁碳纳米管？"}],
    "metadata": {"chat_id": "chat-1", "dir_names": [DIR_NAME], "doc_ids": ["501"]},
    "model": "precise",
    "stream": False,
}
EXPECTED_ANNOTATION = {
    "id": "ref_001",
    "sourceType": "pdf",
    "docId": "501",
    "fileName": "swcnt.pdf",
    "page": 1,
    "endPage": 1,
    "displayText": "swcnt.pdf[1]",
    "quote": "Single-walled carbon nanotubes are cylindrical carbon materials.",
    "score": 0.94,
}


class FakePlatformService:
    def __init__(
        self,
        *,
        error: PlatformDocumentError | None = None,
        ready: bool = True,
        stream_deltas: list[str] | None = None,
    ) -> None:
        self.error = error
        self.ready = ready
        self.stream_deltas = stream_deltas or [
            "单壁",
            "碳纳米管是由单层石墨烯卷曲形成的圆柱状碳材料。",
            " [[ref:ev_test]]",
        ]
        self.calls: list[dict] = []
        self.cancel_checks: list[object] = []
        self.closed = False

    def answer(
        self,
        dir_names,
        doc_ids,
        question,
        chat_id,
        *,
        messages=None,
        on_final_delta=None,
        on_stream_evidence=None,
        cancelled=None,
    ) -> AnswerContract:
        self.calls.append(
            {
                "dir_names": dir_names,
                "doc_ids": doc_ids,
                "question": question,
                "chat_id": chat_id,
                "messages": messages,
            }
        )
        if self.error is not None:
            raise self.error
        if on_stream_evidence is not None:
            on_stream_evidence(
                "run_test",
                [
                    EvidenceRef(
                        evidence_id="ev_test",
                        source_id="src_test",
                        chunk_id="chk_test",
                        page_number=1,
                        quote="Single-walled carbon nanotubes are cylindrical carbon materials.",
                        file_name="swcnt.pdf",
                        quote_source="exact_span",
                        doc_id="501",
                        page_end=1,
                        ref_id="ref_001",
                        score=0.94,
                    )
                ],
            )
        if on_final_delta is not None:
            self.cancel_checks.append(cancelled)
            for delta in self.stream_deltas:
                on_final_delta("run_test", delta)
        return AnswerContract(
            run_id="run_test",
            mode="direct",
            answer_type="document_grounded",
            answer=ANSWER_TEXT + " [[ref:ref_001]]",
            evidences=[
                EvidenceRef(
                    evidence_id="ev_test",
                    source_id="src_test",
                    chunk_id="chk_test",
                    page_number=1,
                    quote="Single-walled carbon nanotubes are cylindrical carbon materials.",
                    file_name="swcnt.pdf",
                    quote_source="exact_span",
                    doc_id="501",
                    page_end=1,
                    ref_id="ref_001",
                    score=0.94,
                )
            ],
            citation_policy="pdf_page_citation",
            debug=AnswerDebug(run_id="run_test", program_steps=["direct_full_context"]),
        )

    def readiness(self) -> dict:
        return {
            "ready": self.ready,
            "data_root_readable": self.ready,
            "cache_dir_writable": True,
            "model_configured": True,
        }

    def close(self) -> None:
        self.closed = True


class ParagraphCitationStreamService(FakePlatformService):
    def __init__(self) -> None:
        super().__init__(
            stream_deltas=[
                "First [[ref:ref_001]]. Duplicate [[ref:ref_002]]. ",
                "Other page [[ref:ref_003]]. Other document [[ref:ref_004]].\n",
                "\nSecond paragraph [[ref:ref_",
                "002]].",
            ]
        )
        self.refs = [
            self._ref(1, doc_id="A", page=1),
            self._ref(2, doc_id="A", page=1),
            self._ref(3, doc_id="A", page=2),
            self._ref(4, doc_id="B", page=1),
        ]
        self.answer_text = (
            "First [[ref:ref_001]]. Duplicate. "
            "Other page [[ref:ref_003]]. Other document [[ref:ref_004]].\n\n"
            "Second paragraph [[ref:ref_002]]."
        )

    @staticmethod
    def _ref(index: int, *, doc_id: str, page: int) -> EvidenceRef:
        return EvidenceRef(
            evidence_id=f"ev_{index}",
            source_id=f"src_{doc_id}",
            chunk_id=f"chk_{index}",
            page_number=page,
            quote=f"Quote {index}.",
            file_name=f"{doc_id}.pdf",
            quote_source="exact_span",
            doc_id=doc_id,
            page_end=page,
            ref_id=f"ref_{index:03d}",
            score=0.9,
        )

    def answer(
        self,
        dir_names,
        doc_ids,
        question,
        chat_id,
        *,
        messages=None,
        on_final_delta=None,
        on_stream_evidence=None,
        cancelled=None,
    ) -> AnswerContract:
        if on_stream_evidence is not None:
            on_stream_evidence("run_paragraph", self.refs)
        if on_final_delta is not None:
            for delta in self.stream_deltas:
                on_final_delta("run_paragraph", delta)
        return AnswerContract(
            run_id="run_paragraph",
            mode="direct",
            answer_type="document_grounded",
            answer=self.answer_text,
            evidences=self.refs,
            citation_policy="pdf_page_citation",
            debug=AnswerDebug(run_id="run_paragraph", program_steps=["direct_full_context"]),
        )


class NoEvidenceMarkerStreamService(FakePlatformService):
    def answer(
        self,
        dir_names,
        doc_ids,
        question,
        chat_id,
        *,
        messages=None,
        on_final_delta=None,
        on_stream_evidence=None,
        cancelled=None,
    ) -> AnswerContract:
        if on_final_delta is not None:
            on_final_delta("run_no_evidence", "Answer [[ref_")
            on_final_delta("run_no_evidence", "999]] without evidence.")
        return AnswerContract(
            run_id="run_no_evidence",
            mode="rlm",
            answer_type="document_assisted",
            answer="Answer without evidence.",
            evidences=[],
            citation_policy="no_pdf_citation",
            debug=AnswerDebug(run_id="run_no_evidence", program_steps=["rlm_map_reduce"]),
        )


class CommitTrackingStreamService(FakePlatformService):
    def __init__(self) -> None:
        super().__init__(stream_deltas=["first", "second"])
        self.committed = False
        self.callback_returned = threading.Event()
        self.allow_commit = threading.Event()
        self.finished = threading.Event()

    def answer(self, *args, on_final_delta=None, cancelled=None, **kwargs) -> AnswerContract:
        try:
            on_final_delta("run_test", "first")
            self.callback_returned.set()
            assert self.allow_commit.wait(timeout=1)
            if cancelled():
                raise RuntimeError("stream cancelled")
            self.committed = True
            return super().answer(*args, on_final_delta=None, cancelled=cancelled, **kwargs)
        finally:
            self.finished.set()


class AdmissionTrackingService(FakePlatformService):
    def __init__(self) -> None:
        super().__init__()
        self.entered: list[str] = []
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def answer(self, *args, **kwargs) -> AnswerContract:
        chat_id = args[3]
        self.entered.append(chat_id)
        if chat_id == "chat-a" and self.entered.count(chat_id) == 1:
            self.first_started.set()
            assert self.release_first.wait(timeout=1)
        return super().answer(*args, **kwargs)


class ShutdownTrackingService(FakePlatformService):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.closed_after_worker = False

    def answer(self, *args, **kwargs) -> AnswerContract:
        self.started.set()
        assert self.release.wait(timeout=1)
        result = super().answer(*args, **kwargs)
        self.finished.set()
        return result

    def close(self) -> None:
        self.closed_after_worker = self.finished.is_set()
        super().close()


def test_platform_chat_accepts_backend_payload_and_returns_enhanced_json() -> None:
    service = FakePlatformService()
    client = TestClient(create_app(service))

    response = client.post("/chat", json=PAYLOAD)

    assert response.status_code == 200
    assert response.json()["answer"].startswith("单壁碳纳米管")
    assert response.json()["evidences"][0]["page_number"] == 1
    assert service.calls[0]["dir_names"] == [DIR_NAME]
    assert service.calls[0]["doc_ids"] == ["501"]
    assert service.calls[0]["question"] == "什么是单壁碳纳米管？"
    assert service.calls[0]["messages"] == PAYLOAD["messages"]


def test_openai_non_stream_accepts_missing_chat_id_and_generates_one() -> None:
    service = FakePlatformService()
    payload = {
        **PAYLOAD,
        "metadata": {"dir_names": [DIR_NAME], "scene": "intelligent"},
    }

    response = TestClient(create_app(service)).post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert service.calls[0]["chat_id"].startswith("chat_")
    assert service.calls[0]["doc_ids"] == []


def test_openai_non_stream_response_preserves_rlm_metadata() -> None:
    client = TestClient(create_app(FakePlatformService()))

    response = client.post("/v1/chat/completions", json=PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"].startswith("单壁碳纳米管")
    assert body["choices"][0]["message"]["annotations"] == [EXPECTED_ANNOTATION]
    assert body["rlm"]["mode"] == "direct"
    assert body["rlm"]["evidences"][0]["file_name"] == "swcnt.pdf"


def test_openai_stream_starts_with_first_content_delta_and_ends_with_metadata() -> None:
    payload = {**PAYLOAD, "stream": True}
    service = FakePlatformService()
    client = TestClient(create_app(service))

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert ": connected" not in response.text
    assert ": heartbeat" not in response.text
    lines = [line for line in response.text.splitlines() if line]
    first = json.loads(lines[0].removeprefix("data: "))
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": "单壁"}
    marker = json.loads(lines[-3].removeprefix("data: "))
    assert marker["choices"][0]["delta"] == {
        "content": " [[ref:ref_001]]",
        "annotations": [EXPECTED_ANNOTATION],
    }
    stop = json.loads(lines[-2].removeprefix("data: "))
    assert stop["choices"][0]["finish_reason"] == "stop"
    assert stop["choices"][0]["delta"] == {}
    assert stop["rlm"]["run_id"] == "run_test"
    assert stop["rlm"]["evidences"][0]["page_number"] == 1
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    accumulated = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
    marker_ids = {
        marker_id
        for marker_id in re.findall(r"\[\[ref:([^\]]+)\]\]", accumulated)
    }
    annotation_ids = {
        annotation["id"]
        for event in events
        for annotation in event["choices"][0]["delta"].get("annotations", [])
    }
    assert accumulated == ANSWER_TEXT + " [[ref:ref_001]]"
    assert marker_ids == annotation_ids == {"ref_001"}
    annotation_frames = [
        event
        for line in lines[:-1]
        if "annotations" in (event := json.loads(line.removeprefix("data: ")))["choices"][0]["delta"]
    ]
    assert len(annotation_frames) == 1
    assert service.cancel_checks and callable(service.cancel_checks[0])


def test_openai_stream_normalizes_split_shorthand_marker_and_attaches_pdf_annotation() -> None:
    payload = {**PAYLOAD, "stream": True}
    service = FakePlatformService(stream_deltas=["Supported [[re", "f_00", "1]] claim."])

    response = TestClient(create_app(service)).post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    events = [json.loads(line.removeprefix("data: ")) for line in lines if line != "data: [DONE]"]
    accumulated = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
    annotations = [
        annotation
        for event in events
        for annotation in event["choices"][0]["delta"].get("annotations", [])
    ]

    assert accumulated == "Supported [[ref:ref_001]] claim."
    assert annotations == [EXPECTED_ANNOTATION]
    assert "[[ref_001]]" not in response.text


def test_openai_stream_removes_unknown_shorthand_when_no_evidence_exists() -> None:
    payload = {**PAYLOAD, "stream": True}

    response = TestClient(create_app(NoEvidenceMarkerStreamService())).post(
        "/v1/chat/completions",
        json=payload,
    )

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    events = [json.loads(line.removeprefix("data: ")) for line in lines if line != "data: [DONE]"]
    accumulated = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
    assert accumulated == "Answer without evidence."
    assert "ref_999" not in response.text


def test_openai_stream_deduplicates_exact_page_only_within_each_paragraph() -> None:
    payload = {**PAYLOAD, "stream": True}
    response = TestClient(create_app(ParagraphCitationStreamService())).post(
        "/v1/chat/completions",
        json=payload,
    )

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    events = [json.loads(line.removeprefix("data: ")) for line in lines if line != "data: [DONE]"]
    accumulated = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
    annotations = [
        annotation
        for event in events
        for annotation in event["choices"][0]["delta"].get("annotations", [])
    ]

    assert accumulated == (
        "First [[ref:ref_001]]. Duplicate. "
        "Other page [[ref:ref_003]]. Other document [[ref:ref_004]].\n\n"
        "Second paragraph [[ref:ref_002]]."
    )
    assert [annotation["displayText"] for annotation in annotations] == [
        "A.pdf[1]",
        "A.pdf[2]",
        "B.pdf[1]",
        "A.pdf[1]",
    ]


def test_chat_stream_uses_the_same_sse_contract() -> None:
    payload = {**PAYLOAD, "stream": True}
    response = TestClient(create_app(FakePlatformService())).post("/chat", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text


def test_stream_disconnect_after_first_delta_prevents_worker_completion() -> None:
    service = CommitTrackingStreamService()

    async def disconnect_after_first_delta() -> str:
        body = PlatformChatBody.model_validate({**PAYLOAD, "stream": True})
        response = await _stream_response(service, body, _validated_request(body))
        service.allow_commit.set()
        if await asyncio.to_thread(service.callback_returned.wait, 0.1):
            assert await asyncio.to_thread(service.finished.wait, 1)
        iterator = response.body_iterator
        first = await iterator.__anext__()
        await iterator.aclose()
        assert await asyncio.to_thread(service.finished.wait, 1)
        return first

    first = asyncio.run(disconnect_after_first_delta())

    assert '"content": "first"' in first
    assert service.committed is False


def test_http_admission_queues_same_chat_before_entering_worker_thread() -> None:
    service = AdmissionTrackingService()
    app = create_app(service)
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/chat")

    async def scenario() -> None:
        body_a = PlatformChatBody.model_validate(
            {**PAYLOAD, "metadata": {**PAYLOAD["metadata"], "chat_id": "chat-a"}}
        )
        body_b = PlatformChatBody.model_validate(
            {**PAYLOAD, "metadata": {**PAYLOAD["metadata"], "chat_id": "chat-b"}}
        )
        first = asyncio.create_task(endpoint(body_a))
        assert await asyncio.to_thread(service.first_started.wait, 1)
        second = asyncio.create_task(endpoint(body_a))
        await asyncio.sleep(0.05)

        other = await asyncio.wait_for(endpoint(body_b), timeout=0.5)

        assert other["answer"]
        assert service.entered.count("chat-a") == 1
        service.release_first.set()
        await asyncio.gather(first, second)

    asyncio.run(scenario())


def test_cancelled_non_stream_request_keeps_chat_lease_until_worker_exits() -> None:
    service = AdmissionTrackingService()
    app = create_app(service)
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/chat")

    async def scenario() -> None:
        body = PlatformChatBody.model_validate(
            {**PAYLOAD, "metadata": {**PAYLOAD["metadata"], "chat_id": "chat-a"}}
        )
        first = asyncio.create_task(endpoint(body))
        assert await asyncio.to_thread(service.first_started.wait, 1)
        first.cancel()
        second = asyncio.create_task(endpoint(body))
        await asyncio.sleep(0.05)

        assert service.entered.count("chat-a") == 1
        service.release_first.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await asyncio.wait_for(second, timeout=0.5)

    asyncio.run(scenario())


def test_app_shutdown_drains_non_stream_worker_before_service_close() -> None:
    service = ShutdownTrackingService()
    app = create_app(service)
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/chat")

    async def scenario() -> None:
        body = PlatformChatBody.model_validate(PAYLOAD)
        async with app.router.lifespan_context(app):
            request_task = asyncio.create_task(endpoint(body))
            assert await asyncio.to_thread(service.started.wait, 1)
            asyncio.get_running_loop().call_later(0.05, service.release.set)
        await request_task

    asyncio.run(scenario())

    assert service.closed is True
    assert service.closed_after_worker is True


def test_payload_validation_rejects_missing_question_empty_dirs_and_mismatched_ids() -> None:
    client = TestClient(create_app(FakePlatformService()))
    missing_question = {**PAYLOAD, "messages": [{"role": "assistant", "content": "hello"}]}
    empty_dirs = {**PAYLOAD, "metadata": {"chat_id": "chat-1", "dir_names": [], "doc_ids": []}}
    mismatch = {
        **PAYLOAD,
        "metadata": {"chat_id": "chat-1", "dir_names": [DIR_NAME, "b" * 64], "doc_ids": ["501"]},
    }

    assert client.post("/chat", json=missing_question).status_code == 400
    assert client.post("/chat", json=empty_dirs).status_code == 400
    assert client.post("/chat", json=mismatch).status_code == 400


def test_payload_validation_rejects_blank_chat_id() -> None:
    payload = {
        **PAYLOAD,
        "metadata": {"chat_id": "   ", "dir_names": [DIR_NAME], "doc_ids": ["501"]},
    }
    client = TestClient(create_app(FakePlatformService()), raise_server_exceptions=False)

    response = client.post("/chat", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "CHAT_ID_INVALID"


def test_non_stream_loader_error_maps_to_public_http_error() -> None:
    error = PlatformDocumentError("DIR_NOT_FOUND", DIR_NAME, f"selected document directory was not found: {DIR_NAME}", 404)
    client = TestClient(create_app(FakePlatformService(error=error)))

    response = client.post("/chat", json=PAYLOAD)

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "DIR_NOT_FOUND"
    assert response.json()["detail"]["dir_name"] == DIR_NAME
    assert "/opt/me_data" not in response.text


def test_stream_loader_error_is_an_sse_error_without_traceback() -> None:
    error = PlatformDocumentError("DIR_NOT_READY", DIR_NAME, "document is not ready", 422)
    payload = {**PAYLOAD, "stream": True}
    response = TestClient(create_app(FakePlatformService(error=error))).post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert '"code": "DIR_NOT_READY"' in response.text
    assert "Traceback" not in response.text
    assert "data: [DONE]" in response.text


def test_health_and_readiness_are_separate() -> None:
    client = TestClient(create_app(FakePlatformService(ready=False)))

    assert client.get("/healthz").status_code == 200
    readiness = client.get("/readyz")
    assert readiness.status_code == 503
    assert readiness.json()["detail"]["ready"] is False


def test_app_shutdown_closes_platform_service() -> None:
    service = FakePlatformService()

    with TestClient(create_app(service)) as client:
        assert client.get("/healthz").status_code == 200
        assert service.closed is False

    assert service.closed is True
