from __future__ import annotations

import asyncio
from dataclasses import dataclass


class ChatQueueFull(RuntimeError):
    pass


class GlobalQueueFull(RuntimeError):
    pass


@dataclass
class _Entry:
    lock: asyncio.Lock
    users: int = 0


class ChatLease:
    def __init__(self, admission: ChatAdmission, chat_id: str, entry: _Entry) -> None:
        self._admission = admission
        self._chat_id = chat_id
        self._entry = entry
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._admission._release(self._chat_id, self._entry)


class ChatAdmission:
    def __init__(self, *, max_pending_per_chat: int = 8, max_global_requests: int = 64) -> None:
        if max_pending_per_chat < 0:
            raise ValueError("max_pending_per_chat must not be negative")
        if max_global_requests <= 0:
            raise ValueError("max_global_requests must be positive")
        self.max_pending_per_chat = max_pending_per_chat
        self.max_global_requests = max_global_requests
        self._guard = asyncio.Lock()
        self._entries: dict[str, _Entry] = {}
        self._total_users = 0

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    async def acquire(self, chat_id: str) -> ChatLease:
        normalized_chat_id = str(chat_id).strip()
        if not normalized_chat_id:
            raise ValueError("chat_id must not be empty")
        async with self._guard:
            if self._total_users >= self.max_global_requests:
                raise GlobalQueueFull("too many active or pending requests")
            entry = self._entries.get(normalized_chat_id)
            if entry is None:
                entry = _Entry(lock=asyncio.Lock())
                self._entries[normalized_chat_id] = entry
            if entry.users >= self.max_pending_per_chat + 1:
                raise ChatQueueFull(f"too many pending requests for chat_id={normalized_chat_id!r}")
            entry.users += 1
            self._total_users += 1
        try:
            await entry.lock.acquire()
        except BaseException:
            await self._drop_user(normalized_chat_id, entry)
            raise
        return ChatLease(self, normalized_chat_id, entry)

    async def _release(self, chat_id: str, entry: _Entry) -> None:
        async with self._guard:
            entry.lock.release()
            entry.users -= 1
            self._total_users -= 1
            if entry.users == 0 and self._entries.get(chat_id) is entry:
                del self._entries[chat_id]

    async def _drop_user(self, chat_id: str, entry: _Entry) -> None:
        async with self._guard:
            entry.users -= 1
            self._total_users -= 1
            if entry.users == 0 and self._entries.get(chat_id) is entry:
                del self._entries[chat_id]
