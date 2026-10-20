from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    run_id     TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_turns_chat
    ON conversation_turns(chat_id, id);
"""


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str
    run_id: str = ""


@dataclass(frozen=True)
class PreparedMemory:
    messages: tuple[ConversationMessage, ...]
    status: str
    history_messages: int
    context_chars: int
    chat_id_hash: str


@dataclass
class _LockEntry:
    lock: threading.RLock
    users: int = 0


class ConversationMemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        try:
            self._conn.row_factory = sqlite3.Row
            with self._lock:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.executescript(_SCHEMA)
            if os.name != "nt":
                os.chmod(self.db_path, 0o600)
        except BaseException:
            self._conn.close()
            self._closed = True
            raise

    @property
    def ready(self) -> bool:
        if self._closed:
            return False
        try:
            with self._lock:
                self._conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def read_messages(self, chat_id: str) -> list[ConversationMessage]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, run_id FROM conversation_turns "
                "WHERE chat_id=? ORDER BY id",
                (chat_id,),
            ).fetchall()
        return [
            ConversationMessage(role=row["role"], content=row["content"], run_id=row["run_id"])
            for row in rows
        ]

    def append_exchange(
        self,
        chat_id: str,
        bootstrap: Sequence[ConversationMessage],
        question: str,
        answer: str,
        run_id: str,
        *,
        max_messages: int,
    ) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self._conn.execute(
                    "SELECT COUNT(*) AS count FROM conversation_turns WHERE chat_id=?",
                    (chat_id,),
                ).fetchone()["count"]
                if existing == 0:
                    bootstrap_limit = max_messages - 2
                    retained_bootstrap = list(bootstrap[-bootstrap_limit:]) if bootstrap_limit else []
                    while retained_bootstrap and retained_bootstrap[0].role != "user":
                        retained_bootstrap.pop(0)
                    for message in retained_bootstrap:
                        self._insert(chat_id, message.role, message.content, message.run_id, created_at)
                self._insert(chat_id, "user", question, run_id, created_at)
                self._insert(chat_id, "assistant", answer, run_id, created_at)
                self._conn.execute(
                    "DELETE FROM conversation_turns WHERE chat_id=? AND id NOT IN ("
                    "SELECT id FROM conversation_turns WHERE chat_id=? ORDER BY id DESC LIMIT ?"
                    ")",
                    (chat_id, chat_id, max_messages),
                )
                while True:
                    oldest = self._conn.execute(
                        "SELECT id, role FROM conversation_turns WHERE chat_id=? ORDER BY id LIMIT 1",
                        (chat_id,),
                    ).fetchone()
                    if oldest is None or oldest["role"] != "assistant":
                        break
                    self._conn.execute("DELETE FROM conversation_turns WHERE id=?", (oldest["id"],))
                retained = self._conn.execute(
                    "SELECT COUNT(*) AS count FROM conversation_turns WHERE chat_id=?",
                    (chat_id,),
                ).fetchone()["count"]
                self._conn.execute("COMMIT")
                return int(retained)
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _insert(self, chat_id: str, role: str, content: str, run_id: str, created_at: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported conversation role: {role}")
        self._conn.execute(
            "INSERT INTO conversation_turns(chat_id, role, content, run_id, created_at) "
            "VALUES (?,?,?,?,?)",
            (chat_id, role, content, run_id, created_at),
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True


class ConversationMemory:
    def __init__(
        self,
        db_path: str | Path,
        *,
        max_context_chars: int = 12_000,
        max_messages: int = 100,
        enabled: bool = True,
        store: ConversationMemoryStore | None = None,
    ) -> None:
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        self.enabled = enabled
        self.max_context_chars = max_context_chars
        self.max_messages = max_messages
        self._locks_guard = threading.RLock()
        self._locks: dict[str, _LockEntry] = {}
        self.store: ConversationMemoryStore | None = None
        if enabled:
            try:
                self.store = store or ConversationMemoryStore(db_path)
            except (OSError, sqlite3.Error):
                self.store = None

    @classmethod
    def from_settings(cls, settings: Settings) -> ConversationMemory:
        db_path = settings.conversation_memory_db_path or (
            settings.runtime_dir / "conversation_memory.sqlite3"
        )
        return cls(
            db_path,
            max_context_chars=settings.conversation_memory_max_context_chars,
            max_messages=settings.conversation_memory_max_messages,
            enabled=settings.conversation_memory_enabled,
        )

    @property
    def ready(self) -> bool:
        if not self.enabled:
            return True
        return self.store is not None and self.store.ready

    @contextmanager
    def session(self, chat_id: str) -> Iterator[ConversationMemorySession]:
        normalized_chat_id = str(chat_id).strip()
        if not normalized_chat_id:
            raise ValueError("chat_id must not be empty")
        entry = self._acquire_entry(normalized_chat_id)
        entry.lock.acquire()
        try:
            yield ConversationMemorySession(self, normalized_chat_id)
        finally:
            entry.lock.release()
            self._release_entry(normalized_chat_id, entry)

    def read_messages(self, chat_id: str) -> list[ConversationMessage]:
        if self.store is None:
            return []
        return self.store.read_messages(chat_id)

    def close(self) -> None:
        if self.store is not None:
            self.store.close()

    def _acquire_entry(self, chat_id: str) -> _LockEntry:
        with self._locks_guard:
            entry = self._locks.get(chat_id)
            if entry is None:
                entry = _LockEntry(lock=threading.RLock())
                self._locks[chat_id] = entry
            entry.users += 1
            return entry

    def _release_entry(self, chat_id: str, entry: _LockEntry) -> None:
        with self._locks_guard:
            entry.users -= 1
            if entry.users == 0 and self._locks.get(chat_id) is entry:
                del self._locks[chat_id]


class ConversationMemorySession:
    def __init__(self, memory: ConversationMemory, chat_id: str) -> None:
        self.memory = memory
        self.chat_id = chat_id
        self._prepared: PreparedMemory | None = None
        self._bootstrap: tuple[ConversationMessage, ...] = ()

    def prepare(self, question: str, request_messages: Sequence[dict[str, Any]]) -> PreparedMemory:
        request_history = _request_history(question, request_messages)
        self._bootstrap = tuple(request_history)
        if not self.memory.enabled:
            prepared = self._prepared_from(request_history, "disabled")
        elif self.memory.store is None:
            prepared = self._prepared_from(request_history, "read_error")
        else:
            try:
                stored = self.memory.store.read_messages(self.chat_id)
            except (OSError, sqlite3.Error):
                prepared = self._prepared_from(request_history, "read_error")
            else:
                if stored:
                    self._bootstrap = ()
                    prepared = self._prepared_from(stored, "hit")
                else:
                    status = "bootstrap" if request_history else "empty"
                    prepared = self._prepared_from(request_history, status)
        self._prepared = prepared
        return prepared

    def model_question(self, question: str, prepared: PreparedMemory | None = None) -> str:
        context = prepared or self._prepared
        if context is None:
            raise RuntimeError("prepare must be called before model_question")
        if not context.messages:
            return question
        history = "\n".join(_render_message(message) for message in context.messages)
        return f"Conversation history:\n{history}\n\nCurrent question:\n{question}"

    def commit(self, question: str, answer: str, run_id: str) -> dict[str, Any]:
        if self._prepared is None:
            raise RuntimeError("prepare must be called before commit")
        trace = self._trace(self._prepared.status, committed=False, retained_messages=0)
        if not self.memory.enabled:
            return trace
        if self.memory.store is None:
            return {**trace, "status": "write_error"}
        try:
            retained = self.memory.store.append_exchange(
                self.chat_id,
                self._bootstrap,
                question,
                answer,
                run_id,
                max_messages=self.memory.max_messages,
            )
        except (OSError, sqlite3.Error):
            return {**trace, "status": "write_error"}
        return self._trace(
            self._prepared.status,
            committed=True,
            retained_messages=retained,
        )

    def _prepared_from(self, messages: Sequence[ConversationMessage], status: str) -> PreparedMemory:
        bounded = _bounded_messages(messages, self.memory.max_context_chars)
        rendered = "\n".join(_render_message(message) for message in bounded)
        return PreparedMemory(
            messages=tuple(bounded),
            status=status,
            history_messages=len(bounded),
            context_chars=len(rendered),
            chat_id_hash=_chat_id_hash(self.chat_id),
        )

    def _trace(self, status: str, *, committed: bool, retained_messages: int) -> dict[str, Any]:
        prepared = self._prepared
        return {
            "status": status,
            "history_messages": prepared.history_messages if prepared else 0,
            "context_chars": prepared.context_chars if prepared else 0,
            "chat_id_hash": _chat_id_hash(self.chat_id),
            "committed": committed,
            "retained_messages": retained_messages,
        }


def _request_history(
    question: str,
    request_messages: Sequence[dict[str, Any]],
) -> list[ConversationMessage]:
    current_index = len(request_messages)
    normalized_question = question.strip()
    for index in range(len(request_messages) - 1, -1, -1):
        message = request_messages[index]
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role == "user" and content and (content == normalized_question or current_index == len(request_messages)):
            current_index = index
            if content == normalized_question:
                break
    history: list[ConversationMessage] = []
    for message in request_messages[:current_index]:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            history.append(ConversationMessage(role=role, content=content))
    return history


def _bounded_messages(
    messages: Sequence[ConversationMessage],
    max_context_chars: int,
) -> list[ConversationMessage]:
    selected: list[ConversationMessage] = []
    used = 0
    for message in reversed(messages):
        rendered_length = len(_render_message(message))
        added = rendered_length + (1 if selected else 0)
        if used + added > max_context_chars:
            break
        selected.append(message)
        used += added
    selected.reverse()
    return selected


def _render_message(message: ConversationMessage) -> str:
    return f"[{message.role}] {message.content}"


def _chat_id_hash(chat_id: str) -> str:
    return hashlib.sha256(chat_id.encode("utf-8")).hexdigest()[:12]
