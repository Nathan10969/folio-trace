from __future__ import annotations

import asyncio
import threading

from rlm_qa.request_workers import RequestWorkerRegistry


def test_registry_cancels_and_drains_workers_before_shutdown_returns() -> None:
    async def scenario() -> None:
        registry = RequestWorkerRegistry()
        cancelled = threading.Event()
        stopped = threading.Event()

        def worker() -> None:
            cancelled.wait(timeout=1)
            stopped.set()

        task = registry.start(asyncio.to_thread(worker), cancelled)
        await asyncio.sleep(0)

        await registry.close()

        assert cancelled.is_set()
        assert stopped.is_set()
        assert task.done()
        assert registry.worker_count == 0

    asyncio.run(scenario())
