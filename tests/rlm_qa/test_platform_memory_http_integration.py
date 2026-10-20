from __future__ import annotations

import json
from pathlib import Path

from starlette.testclient import TestClient

from rlm_qa.config import Settings
from rlm_qa.llm import ChatRequest, ChatResponse, LLMClient
from rlm_qa.platform_documents import PlatformDocumentLoader
from rlm_qa.platform_http_app import create_app
from rlm_qa.platform_service import PlatformQAService


class MemoryHTTPFakeLLM(LLMClient):
    def __init__(self) -> None:
        self.answer_requests: list[ChatRequest] = []
        self.verifier_requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.system.startswith("CITATION_VERIFIER"):
            self.verifier_requests.append(request)
            payload = json.loads(request.user.rsplit("\n\n", 1)[-1])
            candidate = payload["candidates"][0]
            return ChatResponse(
                answer=json.dumps(
                    {
                        "citations": [
                            {
                                "evidence_id": candidate["evidence_id"],
                                "score": 0.9,
                                "quote": candidate["text"],
                            }
                        ]
                    }
                )
            )
        self.answer_requests.append(request)
        return ChatResponse(answer=f"answer-{len(self.answer_requests)}")

    def chat_stream(self, request, on_delta, cancelled) -> ChatResponse:
        self.answer_requests.append(request)
        assert cancelled() is False
        on_delta("stream ")
        on_delta("answer ")
        on_delta("[[ref:ref_001]]")
        return ChatResponse(answer="stream answer [[ref:ref_001]]")


def _write_platform_document(root: Path, dir_name: str) -> None:
    target = root / dir_name
    target.mkdir(parents=True)
    (target / "status.json").write_text(json.dumps({"status": "ok", "pages": 1}), encoding="utf-8")
    (target / "metadata.json").write_text(
        json.dumps({"fileId": 501, "fileName": "memory.pdf"}),
        encoding="utf-8",
    )
    (target / "elements.jsonl").write_text(
        json.dumps({"page": 1, "index": 0, "label": "text", "content": "Document memory text."}) + "\n",
        encoding="utf-8",
    )


def _payload(chat_id: str, question: str, *, stream: bool = False) -> dict:
    return {
        "messages": [{"role": "user", "content": question}],
        "metadata": {"chat_id": chat_id, "dir_names": ["d" * 64], "doc_ids": ["501"]},
        "model": "precise",
        "stream": stream,
    }


def test_http_memory_persists_same_chat_isolates_other_chat_and_streams(tmp_path: Path) -> None:
    dir_name = "d" * 64
    data_root = tmp_path / "glm"
    _write_platform_document(data_root, dir_name)
    settings = Settings(
        runtime_dir=tmp_path / "runtime",
        platform_data_root=data_root,
        platform_cache_dir=tmp_path / "cache",
        chunk_chars=400,
        chunk_overlap_chars=40,
    )
    llm = MemoryHTTPFakeLLM()
    loader = PlatformDocumentLoader(data_root, settings.platform_cache_dir, settings)
    service = PlatformQAService(settings=settings, llm=llm, loader=loader)
    database = settings.runtime_dir / "conversation_memory.sqlite3"

    with TestClient(create_app(service)) as client:
        first = client.post("/v1/chat/completions", json=_payload("chat-a", "first question"))
        second = client.post("/v1/chat/completions", json=_payload("chat-a", "what did I ask?"))
        other = client.post("/v1/chat/completions", json=_payload("chat-b", "other question"))
        streamed = client.post(
            "/v1/chat/completions",
            json=_payload("chat-a", "stream this", stream=True),
        )

        assert [first.status_code, second.status_code, other.status_code, streamed.status_code] == [200, 200, 200, 200]
        assert "[user] first question" in llm.answer_requests[1].user
        assert "[assistant] answer-1" in llm.answer_requests[1].user
        assert "first question" not in llm.answer_requests[2].user
        lines = [line for line in streamed.text.splitlines() if line]
        events = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
        accumulated = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
        assert accumulated == "stream answer [[ref:ref_001]]"
        citation_event = next(
            event
            for event in events
            if "[[ref:ref_001]]" in event["choices"][0]["delta"].get("content", "")
        )
        assert citation_event["choices"][0]["delta"]["annotations"][0]["id"] == "ref_001"
        assert events[-1]["choices"][0]["delta"] == {}
        assert lines[-1] == "data: [DONE]"
        assert len(service.memory.read_messages("chat-a")) == 6
        assert len(service.memory.read_messages("chat-b")) == 2
        assert all("[[ref:" not in message.content for message in service.memory.read_messages("chat-a"))
        assert llm.verifier_requests == []

    assert service.memory.ready is False
    database.unlink()
    assert database.exists() is False
