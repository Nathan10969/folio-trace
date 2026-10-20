from __future__ import annotations

import json

import pytest

from rlm_qa.conversation_memory import ConversationMessage, PreparedMemory
from rlm_qa.llm import ChatResponse, ToolCall
from rlm_qa.turn_executor import (
    DirectTurnResponse,
    DocumentInspection,
    InvalidExecutionContract,
    build_turn_request,
    parse_turn_response,
)


DOCUMENTS = [
    {
        "position": 1,
        "source_id": "src_first",
        "doc_id": "501",
        "file_name": "first.pdf",
        "page_count": 12,
        "status": "available",
    },
    {
        "position": 2,
        "source_id": "src_second",
        "doc_id": "502",
        "file_name": "second.pdf",
        "page_count": 30,
        "status": "available",
    },
]


def _memory(*contents: str) -> PreparedMemory:
    messages = tuple(
        ConversationMessage(role="user" if index % 2 == 0 else "assistant", content=content)
        for index, content in enumerate(contents)
    )
    return PreparedMemory(
        messages=messages,
        status="hit",
        history_messages=len(messages),
        context_chars=sum(len(message.content) for message in messages),
        chat_id_hash="hash",
    )


def test_turn_request_prioritizes_current_request_and_bounds_recent_history() -> None:
    prepared = _memory("old-1", "old-2", "recent-1", "recent-2", "recent-3", "recent-4")

    request = build_turn_request(
        question="你好",
        prepared_memory=prepared,
        documents=DOCUMENTS,
        run_id="run_turn",
        model="glm-5.2",
    )

    payload = json.loads(request.user)
    assert payload["current_request"] == "你好"
    assert [item["content"] for item in payload["recent_conversation"]] == [
        "recent-1",
        "recent-2",
        "recent-3",
        "recent-4",
    ]
    assert payload["available_documents"] == [
        {"position": 1, "source_id": "src_first", "status": "available"},
        {"position": 2, "source_id": "src_second", "status": "available"},
    ]
    assert "first.pdf" not in request.user
    assert "page_count" not in request.user
    assert request.purpose == "orchestrator"
    assert request.tool_choice == "auto"
    assert [tool["function"]["name"] for tool in request.tools] == ["inspect_documents"]
    assert "Available documents are optional capabilities" in request.system
    assert "Do not mention available documents unless the current request asks about them" in request.system


def test_turn_response_uses_plain_content_without_document_execution() -> None:
    decision = parse_turn_response(ChatResponse(answer="你好，有什么可以帮你？"), DOCUMENTS)

    assert decision == DirectTurnResponse(answer="你好，有什么可以帮你？")


def test_turn_response_accepts_one_valid_document_tool_call() -> None:
    response = ChatResponse(
        tool_calls=(
            ToolCall(
                id="call_1",
                name="inspect_documents",
                arguments=json.dumps(
                    {
                        "query": "分别总结两份文档",
                        "scope": {"mode": "all_selected", "source_ids": []},
                        "coverage": "all_in_scope",
                        "answer_shape": "per_source",
                    },
                    ensure_ascii=False,
                ),
            ),
        )
    )

    decision = parse_turn_response(response, DOCUMENTS)

    assert decision == DocumentInspection(
        call_id="call_1",
        query="分别总结两份文档",
        scope_mode="all_selected",
        source_ids=("src_first", "src_second"),
        coverage="all_in_scope",
        answer_shape="per_source",
    )


def test_all_selected_accepts_redundant_complete_allowlisted_source_ids() -> None:
    response = ChatResponse(
        tool_calls=(
            ToolCall(
                id="call_all_redundant",
                name="inspect_documents",
                arguments=json.dumps(
                    {
                        "query": "summarize both documents",
                        "scope": {
                            "mode": "all_selected",
                            "source_ids": ["src_first", "src_second"],
                        },
                        "coverage": "all_in_scope",
                        "answer_shape": "per_source",
                    }
                ),
            ),
        )
    )

    decision = parse_turn_response(response, DOCUMENTS)

    assert isinstance(decision, DocumentInspection)
    assert decision.source_ids == ("src_first", "src_second")


def test_all_selected_rejects_a_partial_source_list() -> None:
    response = ChatResponse(
        tool_calls=(
            ToolCall(
                id="call_all_partial",
                name="inspect_documents",
                arguments=json.dumps(
                    {
                        "query": "summarize all documents",
                        "scope": {"mode": "all_selected", "source_ids": ["src_first"]},
                        "coverage": "all_in_scope",
                        "answer_shape": "per_source",
                    }
                ),
            ),
        )
    )

    with pytest.raises(InvalidExecutionContract, match="must enumerate every available source_id"):
        parse_turn_response(response, DOCUMENTS)


def test_turn_response_rejects_unknown_document_source_without_execution() -> None:
    response = ChatResponse(
        tool_calls=(
            ToolCall(
                id="call_bad",
                name="inspect_documents",
                arguments=json.dumps(
                    {
                        "query": "总结未知文档",
                        "scope": {"mode": "selected_subset", "source_ids": ["src_unknown"]},
                        "coverage": "relevant_in_scope",
                        "answer_shape": "synthesis",
                    }
                ),
            ),
        )
    )

    with pytest.raises(InvalidExecutionContract, match="unknown source_id"):
        parse_turn_response(response, DOCUMENTS)
