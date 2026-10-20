from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar


ItemT = TypeVar("ItemT")
ControlT = TypeVar("ControlT")
FirstT = TypeVar("FirstT")
FollowupT = TypeVar("FollowupT")


@dataclass(frozen=True)
class ParallelControlResult(Generic[ControlT, FirstT]):
    control_result: ControlT
    item_results: list[FirstT]


@dataclass(frozen=True)
class FollowupPipelineResult(Generic[FirstT, FollowupT]):
    first_results: list[FirstT]
    followup_results: list[FollowupT | None]


def run_parallel_with_control(
    control_fn: Callable[[], ControlT],
    items: Iterable[ItemT],
    item_fn: Callable[[ItemT], FirstT],
    *,
    max_workers: int,
) -> ParallelControlResult[ControlT, FirstT]:
    item_list = list(items)
    worker_count = max(1, min(int(max_workers), len(item_list) + 1))
    item_results: list[FirstT | None] = [None] * len(item_list)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        control_future = executor.submit(control_fn)
        futures = {executor.submit(item_fn, item): index for index, item in enumerate(item_list)}
        control_result = control_future.result()
        for future in as_completed(futures):
            item_results[futures[future]] = future.result()
    return ParallelControlResult(
        control_result=control_result,
        item_results=[result for result in item_results if result is not None],
    )


def run_followup_pipeline(
    items: Iterable[ItemT],
    first_fn: Callable[[ItemT], FirstT],
    should_follow: Callable[[FirstT], bool],
    followup_fn: Callable[[ItemT, FirstT], FollowupT],
    *,
    max_workers: int,
) -> FollowupPipelineResult[FirstT, FollowupT]:
    item_list = list(items)
    if not item_list:
        return FollowupPipelineResult(first_results=[], followup_results=[])

    worker_count = max(1, min(int(max_workers), len(item_list)))
    pending_first = deque(range(len(item_list)))
    first_results: list[FirstT | None] = [None] * len(item_list)
    followup_results: list[FollowupT | None] = [None] * len(item_list)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        active: dict[Future, tuple[str, int]] = {}

        def submit_first() -> None:
            index = pending_first.popleft()
            active[executor.submit(first_fn, item_list[index])] = ("first", index)

        while pending_first and len(active) < worker_count:
            submit_first()

        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            followups: list[int] = []
            for future in completed:
                stage, index = active.pop(future)
                if stage == "first":
                    first_result = future.result()
                    first_results[index] = first_result
                    if should_follow(first_result):
                        followups.append(index)
                else:
                    followup_results[index] = future.result()

            for index in followups:
                first_result = first_results[index]
                if first_result is not None:
                    active[executor.submit(followup_fn, item_list[index], first_result)] = ("followup", index)

            while pending_first and len(active) < worker_count:
                submit_first()

    return FollowupPipelineResult(
        first_results=[result for result in first_results if result is not None],
        followup_results=followup_results,
    )
