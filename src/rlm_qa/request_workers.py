from __future__ import annotations

import asyncio
from collections.abc import Awaitable
import threading
from typing import Any


class RequestWorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[asyncio.Task[Any], threading.Event | None] = {}
        self._closing = False

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    def start(
        self,
        awaitable: Awaitable[Any],
        cancel_event: threading.Event | None = None,
    ) -> asyncio.Task[Any]:
        if self._closing:
            if cancel_event is not None:
                cancel_event.set()
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise RuntimeError("request worker registry is closing")
        task = asyncio.create_task(awaitable)
        self._workers[task] = cancel_event
        task.add_done_callback(self._worker_done)
        return task

    async def close(self) -> None:
        self._closing = True
        workers = list(self._workers.items())
        for _task, cancel_event in workers:
            if cancel_event is not None:
                cancel_event.set()
        if workers:
            await asyncio.gather(*(task for task, _event in workers), return_exceptions=True)
        for task, _event in workers:
            self._workers.pop(task, None)

    def _worker_done(self, task: asyncio.Task[Any]) -> None:
        self._workers.pop(task, None)
