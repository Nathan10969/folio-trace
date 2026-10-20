from __future__ import annotations

import json
import re
import time
from pathlib import Path

from starlette.testclient import TestClient

from rlm_qa.app_service import QAService
from rlm_qa.config import Settings
from rlm_qa.contracts import Page
from rlm_qa.http_app import create_app
from rlm_qa.llm import ChatRequest, ChatResponse, FakeLLM, LLMClient


class ScriptedLLM(LLMClient):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("model called more times than expected")
        return ChatResponse(answer=self.responses.pop(0))


def test_web_assets_are_available_outside_checkout_working_directory(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    service = QAService(
        settings=Settings(runtime_dir=tmp_path / "runtime"),
        llm=FakeLLM(answer="unused"),
    )
    with TestClient(create_app(service)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200


class MapReduceLLM(LLMClient):
    def __init__(self, final_answer: str) -> None:
        self.final_answer = final_answer

    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.system == "MAP_REDUCE_PLAN":
            return ChatResponse(answer=json.dumps({"coverage": "relevant_subset", "answer_shape": "synthesis"}))
        if request.system == "MAP_REDUCE_TRIAGE":
            pack_id = re.search(r"PACK_ID=([^\s]+)", request.user)
            assert pack_id is not None
            evidence_ids = re.findall(r"\bev_[A-Za-z0-9_:-]+\b", request.user)
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": pack_id.group(1),
                        "relevance_score": 1.0,
                        "has_answer": True,
                        "evidence_ids": evidence_ids,
                        "short_reason": "fixture relevant",
                    }
                )
            )
        if request.system == "MAP_REDUCE_DEEP_ANSWER":
            return ChatResponse(
                answer=json.dumps(
                    {
                        "answer": "fixture deep answer",
                        "evidence_ids": re.findall(r"\bev_[A-Za-z0-9_:-]+\b", request.user),
                    }
                )
            )
        if request.system == "MAP_REDUCE_FINALIZER":
            return ChatResponse(answer=self.final_answer)
        raise AssertionError(f"unexpected request: {request.system}")


def rlm_metadata_answer() -> str:
    return (
        "```repl\n"
        "profiles = context.get('doc_profiles') or [context['doc_profile']]\n"
        "refs = [f\"doc_profile:{profile['source_id']}:page_count\" for profile in profiles]\n"
        "answer['content'] = {'answer': 'metadata read', 'evidence_ids': [], 'metadata_refs': refs}\n"
        "answer['ready'] = True\n"
        "```"
    )


def test_http_upload_and_chat_roundtrip(tmp_path: Path) -> None:
    service = QAService(
        settings=Settings(runtime_dir=tmp_path / "runtime"),
        llm=FakeLLM(answer="SWCNTs are one-dimensional nanomaterials."),
    )
    client = TestClient(create_app(service))

    upload = client.post(
        "/upload",
        files={
            "file": (
                "swcnt.txt",
                "Single-walled carbon nanotubes are one-dimensional nanomaterials.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 200
    source_id = upload.json()["source_id"]

    chat = client.post(
        "/chat",
        json={
            "source_id": source_id,
            "session_id": "local_session",
            "question": "nanotubes",
        },
    )

    assert chat.status_code == 200
    body = chat.json()
    assert body["mode"] == "direct"
    assert body["evidences"][0]["page_number"] == 1


def test_http_chat_accepts_multiple_source_ids_for_metadata_question(tmp_path: Path) -> None:
    service = QAService(
        settings=Settings(runtime_dir=tmp_path / "runtime", virtual_page_chars=80),
        llm=MapReduceLLM("first.txt has 2 pages; second.txt has 3 pages; total 5 pages."),
    )
    client = TestClient(create_app(service))

    first = client.post(
        "/upload",
        files={"file": ("first.txt", "a" * 81, "text/plain")},
    )
    second = client.post(
        "/upload",
        files={"file": ("second.txt", "b" * 161, "text/plain")},
    )

    chat = client.post(
        "/chat",
        json={
            "source_ids": [first.json()["source_id"], second.json()["source_id"]],
            "session_id": "local_session",
            "question": "How many pages do these documents have?",
        },
    )

    assert chat.status_code == 200
    body = chat.json()
    assert body["mode"] == "rlm"
    assert body["answer_type"] == "document_grounded"
    assert body["citation_policy"] == "pdf_page_citation"
    assert body["answer"] == "first.txt has 2 pages; second.txt has 3 pages; total 5 pages."
    assert body["debug"]["metadata_refs"] == [
        f"doc_profile:{first.json()['source_id']}:page_count",
        f"doc_profile:{second.json()['source_id']}:page_count",
    ]


def test_http_document_status_returns_latest_queued_pdf_profile(tmp_path: Path) -> None:
    service = QAService(
        settings=Settings(runtime_dir=tmp_path / "runtime", glm_ocr_api_keys=("fake-ocr-key",)),
        llm=FakeLLM(answer="unused"),
    )
    client = TestClient(create_app(service))

    upload = client.post(
        "/upload",
        files={"file": ("scan.pdf", b"%PDF-1.7", "application/pdf")},
    )
    source_id = upload.json()["source_id"]
    status = client.get(f"/documents/{source_id}")

    assert upload.status_code == 200
    assert upload.json()["status"] == "ocr_queued"
    assert status.status_code == 200
    assert status.json()["source_id"] == source_id
    assert status.json()["status"] == "ocr_queued"


def test_fastapi_lifecycle_runs_queued_pdf_ocr_in_background(tmp_path: Path) -> None:
    class FakePDFOCR:
        def extract_pdf_pages(self, path: Path, source_id: str, *, api_key: str) -> list[Page]:
            return [Page(source_id=source_id, page_number=1, text="OCR page", char_start=0, char_end=8)]

    service = QAService(
        settings=Settings(runtime_dir=tmp_path / "runtime", glm_ocr_api_keys=("fake-ocr-key",)),
        llm=FakeLLM(answer="unused"),
        pdf_ocr=FakePDFOCR(),
    )
    with TestClient(create_app(service)) as client:
        upload = client.post(
            "/upload",
            files={"file": ("scan.pdf", b"%PDF-1.7", "application/pdf")},
        )
        source_id = upload.json()["source_id"]
        status = upload.json()["status"]
        for _ in range(50):
            if status in {"direct_ready", "rlm_ready"}:
                break
            time.sleep(0.01)
            status = client.get(f"/documents/{source_id}").json()["status"]

    assert status == "direct_ready"
