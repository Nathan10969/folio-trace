from __future__ import annotations

import asyncio

import pytest

from rlm_qa.chat_admission import ChatAdmission, ChatQueueFull, GlobalQueueFull


def test_same_chat_waiters_are_bounded_without_blocking_other_chats() -> None:
    async def scenario() -> None:
        admission = ChatAdmission(max_pending_per_chat=1)
        active = await admission.acquire("chat-a")
        queued_task = asyncio.create_task(admission.acquire("chat-a"))
        await asyncio.sleep(0)

        with pytest.raises(ChatQueueFull):
            await admission.acquire("chat-a")

        other = await asyncio.wait_for(admission.acquire("chat-b"), timeout=0.2)
        await other.release()
        await active.release()
        queued = await asyncio.wait_for(queued_task, timeout=0.2)
        await queued.release()

        assert admission.entry_count == 0

    asyncio.run(scenario())


def test_global_request_limit_cannot_be_bypassed_with_distinct_chat_ids() -> None:
    async def scenario() -> None:
        admission = ChatAdmission(max_pending_per_chat=1, max_global_requests=2)
        first = await admission.acquire("chat-a")
        second = await admission.acquire("chat-b")

        with pytest.raises(GlobalQueueFull):
            await admission.acquire("chat-c")

        await first.release()
        third = await admission.acquire("chat-c")
        await second.release()
        await third.release()

        assert admission.entry_count == 0

    asyncio.run(scenario())


def test_cancelled_waiter_releases_its_queue_slot() -> None:
    async def scenario() -> None:
        admission = ChatAdmission(max_pending_per_chat=1)
        active = await admission.acquire("chat-a")
        cancelled_task = asyncio.create_task(admission.acquire("chat-a"))
        await asyncio.sleep(0)
        cancelled_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await cancelled_task

        replacement_task = asyncio.create_task(admission.acquire("chat-a"))
        await asyncio.sleep(0)
        await active.release()
        replacement = await asyncio.wait_for(replacement_task, timeout=0.2)
        await replacement.release()

        assert admission.entry_count == 0

    asyncio.run(scenario())
