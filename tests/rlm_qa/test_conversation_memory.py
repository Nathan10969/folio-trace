from __future__ import annotations

import threading
import sqlite3

import pytest

from rlm_qa.conversation_memory import ConversationMemory, ConversationMemoryStore


class SchemaFailingConnection:
    def __init__(self) -> None:
        self.row_factory = None
        self.closed = False

    def execute(self, statement):
        return self

    def executescript(self, statement):
        raise sqlite3.OperationalError("schema failed")

    def close(self) -> None:
        self.closed = True


def test_store_closes_connection_when_schema_initialization_fails(tmp_path, monkeypatch) -> None:
    connection = SchemaFailingConnection()
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: connection)

    with pytest.raises(sqlite3.OperationalError, match="schema failed"):
        ConversationMemoryStore(tmp_path / "conversation.sqlite3")

    assert connection.closed is True


def test_memory_persists_bootstrap_and_exchange_across_reopen(tmp_path) -> None:
    database = tmp_path / "conversation.sqlite3"
    memory = ConversationMemory(database, max_context_chars=1_000, max_messages=20)

    with memory.session("chat-a") as session:
        prepared = session.prepare(
            "follow up",
            [
                {"role": "user", "content": "original question"},
                {"role": "assistant", "content": "original answer"},
                {"role": "user", "content": "follow up"},
            ],
        )

        assert prepared.status == "bootstrap"
        assert [message.content for message in prepared.messages] == [
            "original question",
            "original answer",
        ]
        assert "[user] original question" in session.model_question("follow up", prepared)
        committed = session.commit("follow up", "new answer", "run-1")

    assert committed["committed"] is True
    assert committed["retained_messages"] == 4
    memory.close()

    reopened = ConversationMemory(database, max_context_chars=1_000, max_messages=20)
    assert [(message.role, message.content) for message in reopened.read_messages("chat-a")] == [
        ("user", "original question"),
        ("assistant", "original answer"),
        ("user", "follow up"),
        ("assistant", "new answer"),
    ]
    assert reopened.read_messages("chat-b") == []
    reopened.close()


def test_persisted_history_wins_over_repeated_request_history(tmp_path) -> None:
    memory = ConversationMemory(tmp_path / "conversation.sqlite3")
    with memory.session("chat-a") as first:
        prepared = first.prepare("first", [{"role": "user", "content": "first"}])
        assert prepared.status == "empty"
        first.commit("first", "stored answer", "run-1")

    with memory.session("chat-a") as second:
        prepared = second.prepare(
            "second",
            [
                {"role": "user", "content": "bogus request history"},
                {"role": "assistant", "content": "bogus request answer"},
                {"role": "user", "content": "second"},
            ],
        )
        assert prepared.status == "hit"
        assert [message.content for message in prepared.messages] == ["first", "stored answer"]
        second.commit("second", "second answer", "run-2")

    assert [message.content for message in memory.read_messages("chat-a")] == [
        "first",
        "stored answer",
        "second",
        "second answer",
    ]
    memory.close()


def test_memory_prunes_old_messages_and_keeps_recent_complete_context(tmp_path) -> None:
    memory = ConversationMemory(
        tmp_path / "conversation.sqlite3",
        max_context_chars=44,
        max_messages=4,
    )
    for number in range(1, 4):
        with memory.session("chat-a") as session:
            prepared = session.prepare(f"question-{number}", [{"role": "user", "content": f"question-{number}"}])
            session.commit(f"question-{number}", f"answer-{number}", f"run-{number}")

    assert [message.content for message in memory.read_messages("chat-a")] == [
        "question-2",
        "answer-2",
        "question-3",
        "answer-3",
    ]

    with memory.session("chat-a") as session:
        prepared = session.prepare("question-4", [{"role": "user", "content": "question-4"}])
        rendered = session.model_question("question-4", prepared)

    assert "answer-3" in rendered
    assert "question-2" not in rendered
    assert all(message.content in rendered for message in prepared.messages)
    memory.close()


def test_odd_retention_limit_does_not_leave_orphan_assistant_message(tmp_path) -> None:
    memory = ConversationMemory(
        tmp_path / "conversation.sqlite3",
        max_context_chars=1_000,
        max_messages=3,
    )
    for number in range(1, 3):
        with memory.session("chat-a") as session:
            session.prepare(f"question-{number}", [])
            session.commit(f"question-{number}", f"answer-{number}", f"run-{number}")

    assert [(message.role, message.content) for message in memory.read_messages("chat-a")] == [
        ("user", "question-2"),
        ("assistant", "answer-2"),
    ]
    memory.close()


def test_large_bootstrap_is_bounded_before_persistence(tmp_path) -> None:
    memory = ConversationMemory(
        tmp_path / "conversation.sqlite3",
        max_context_chars=1_000,
        max_messages=4,
    )
    bootstrap = [
        {"role": "user" if number % 2 == 0 else "assistant", "content": f"history-{number}"}
        for number in range(20)
    ]
    bootstrap.append({"role": "user", "content": "current"})

    with memory.session("chat-a") as session:
        session.prepare("current", bootstrap)
        session.commit("current", "answer", "run-current")

    messages = memory.read_messages("chat-a")
    assert len(messages) <= 4
    assert messages[0].role == "user"
    assert [(message.role, message.content) for message in messages[-2:]] == [
        ("user", "current"),
        ("assistant", "answer"),
    ]
    memory.close()


def test_two_message_retention_does_not_insert_bootstrap_before_pruning(tmp_path, monkeypatch) -> None:
    store = ConversationMemoryStore(tmp_path / "conversation.sqlite3")
    inserted: list[str] = []
    original_insert = store._insert

    def record_insert(chat_id, role, content, run_id, created_at):
        inserted.append(content)
        original_insert(chat_id, role, content, run_id, created_at)

    monkeypatch.setattr(store, "_insert", record_insert)
    memory = ConversationMemory(
        tmp_path / "unused.sqlite3",
        max_context_chars=1_000,
        max_messages=2,
        store=store,
    )
    request = [
        {"role": "user" if number % 2 == 0 else "assistant", "content": f"history-{number}"}
        for number in range(100)
    ]
    request.append({"role": "user", "content": "current"})

    with memory.session("chat-a") as session:
        session.prepare("current", request)
        session.commit("current", "answer", "run-current")

    assert inserted == ["current", "answer"]
    memory.close()


def test_pruning_removes_all_leading_orphan_assistant_messages(tmp_path) -> None:
    store = ConversationMemoryStore(tmp_path / "conversation.sqlite3")
    for role, content in [
        ("user", "old-question"),
        ("assistant", "old-answer"),
        ("assistant", "orphan-one"),
        ("assistant", "orphan-two"),
    ]:
        store._insert("chat-a", role, content, "legacy", "2026-07-14T00:00:00+00:00")
    memory = ConversationMemory(
        tmp_path / "unused.sqlite3",
        max_context_chars=1_000,
        max_messages=4,
        store=store,
    )

    with memory.session("chat-a") as session:
        session.prepare("current", [])
        session.commit("current", "answer", "run-current")

    assert [(message.role, message.content) for message in memory.read_messages("chat-a")] == [
        ("user", "current"),
        ("assistant", "answer"),
    ]
    memory.close()


def test_same_chat_lock_serializes_while_other_chat_remains_available(tmp_path) -> None:
    memory = ConversationMemory(tmp_path / "conversation.sqlite3")
    same_chat_entered = threading.Event()
    other_chat_entered = threading.Event()

    def enter(chat_id: str, entered: threading.Event) -> None:
        with memory.session(chat_id):
            entered.set()

    with memory.session("chat-a"):
        same_thread = threading.Thread(target=enter, args=("chat-a", same_chat_entered))
        other_thread = threading.Thread(target=enter, args=("chat-b", other_chat_entered))
        same_thread.start()
        other_thread.start()

        assert other_chat_entered.wait(timeout=1)
        assert same_chat_entered.wait(timeout=0.1) is False

    assert same_chat_entered.wait(timeout=1)
    same_thread.join(timeout=1)
    other_thread.join(timeout=1)
    memory.close()
