from __future__ import annotations

from threading import Event

from rlm_qa.map_scheduler import run_followup_pipeline, run_parallel_with_control


def test_followup_pipeline_starts_followup_before_slow_first_stage_finishes() -> None:
    followup_started = Event()

    def first(item: str) -> str:
        if item == "relevant":
            return "answerable"
        assert followup_started.wait(timeout=1), "followup was blocked behind a first-stage barrier"
        return "irrelevant"

    def followup(item: str, result: str) -> str:
        assert item == "relevant"
        assert result == "answerable"
        followup_started.set()
        return "deep answer"

    result = run_followup_pipeline(
        ["relevant", "slow"],
        first,
        lambda value: value == "answerable",
        followup,
        max_workers=2,
    )

    assert result.first_results == ["answerable", "irrelevant"]
    assert result.followup_results == ["deep answer", None]


def test_followup_pipeline_preserves_input_order() -> None:
    result = run_followup_pipeline(
        [3, 1, 2],
        lambda item: item * 10,
        lambda value: value % 2 == 0,
        lambda item, value: item + value,
        max_workers=3,
    )

    assert result.first_results == [30, 10, 20]
    assert result.followup_results == [33, 11, 22]


def test_parallel_control_overlaps_control_and_item_work() -> None:
    item_started = Event()

    def control() -> str:
        assert item_started.wait(timeout=1), "item work did not overlap the control task"
        return "plan"

    def item_work(item: int) -> int:
        item_started.set()
        return item * 2

    result = run_parallel_with_control(control, [1, 2], item_work, max_workers=3)

    assert result.control_result == "plan"
    assert result.item_results == [2, 4]
