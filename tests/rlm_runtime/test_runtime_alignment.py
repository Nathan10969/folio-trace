from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rlm_runtime.core.rlm import RLM


class ScriptedModel:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(self, system: str, user: str, model: str | None = None) -> str:
        self.calls.append({"system": system, "user": user, "model": model})
        if not self.responses:
            raise AssertionError("model called more times than expected")
        return self.responses.pop(0)


def test_completion_returns_metadata_usage_and_jsonl_trajectory(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.jsonl"
    model = ScriptedModel(
        [
            "Finish.\n```repl\nanswer['content'] = 'ok'\nanswer['ready'] = True\n```",
        ]
    )
    rlm = RLM(llm=model, max_iterations=1, run_id="run_root", trajectory_path=trajectory_path)

    result = rlm.completion("answer directly", context={})

    assert result.response == "ok"
    assert result.ready is True
    assert result.answer_type == "final"
    assert result.metadata.run_id == "run_root"
    assert result.metadata.parent_run_id is None
    assert result.metadata.depth == 1
    assert result.usage.total_calls == 1
    assert result.error_info is None
    assert result.trajectory["run_metadata"]["run_id"] == "run_root"

    lines = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()]
    assert [line["event"] for line in lines] == ["run_started", "llm_call_finished", "iteration_finished", "run_finished"]
    assert lines[0]["run_id"] == "run_root"
    assert lines[-1]["ready"] is True


class BatchFailureModel:
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, system: str, user: str, model: str | None = None) -> str:
        self.calls.append(user)
        if "root batch" in user and user not in {"ok", "bad", "later"}:
            return (
                "Use batch calls.\n```repl\n"
                "items = llm_query_batched(['ok', 'bad', 'later'])\n"
                "answer['content'] = items\n"
                "answer['ready'] = True\n"
                "```"
            )
        if user == "bad":
            raise RuntimeError("provider boom")
        return user.upper()


def test_llm_query_batched_keeps_order_and_returns_slot_error() -> None:
    rlm = RLM(llm=BatchFailureModel(), max_iterations=1, batch_max_workers=3, run_id="run_batch")

    result = rlm.completion("root batch", context={})

    assert result.error == ""
    assert result.response[0] == "OK"
    assert result.response[2] == "LATER"
    assert result.response[1].startswith("Error: provider_error")
    assert "provider boom" in result.response[1]
    events = result.trajectory["events"]
    failed_slots = [
        event
        for event in events
        if event["event"] == "llm_call_finished" and event.get("prompt_index") == 1
    ]
    assert failed_slots
    assert failed_slots[0]["error"]["code"] == "provider_error"


class ChildBatchModel:
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, system: str, user: str, model: str | None = None) -> str:
        self.calls.append(user)
        if "delegate children" in user:
            return (
                "Fan out.\n```repl\n"
                "children = rlm_query_batched(['child one', 'child two'])\n"
                "answer['content'] = children\n"
                "answer['ready'] = True\n"
                "```"
            )
        if "child one" in user:
            return "```repl\nanswer['content'] = 'one'\nanswer['ready'] = True\n```"
        if "child two" in user:
            return "```repl\nanswer['content'] = 'two'\nanswer['ready'] = True\n```"
        raise AssertionError(f"unexpected prompt: {user}")


def test_rlm_query_batched_records_child_metadata_parent_depth_and_prompt_index() -> None:
    rlm = RLM(
        llm=ChildBatchModel(),
        max_iterations=1,
        max_depth=2,
        max_concurrent_subcalls=2,
        run_id="run_parent",
    )

    result = rlm.completion("delegate children", context={})

    assert result.response == ["one", "two"]
    children = result.trajectory["children"]
    assert len(children) == 2
    assert {child["parent_run_id"] for child in children} == {"run_parent"}
    assert {child["depth"] for child in children} == {2}
    assert {child["prompt_index"] for child in children} == {0, 1}
    assert {child["response"] for child in children} == {"one", "two"}


class AlwaysFailModel:
    def __call__(self, system: str, user: str, model: str | None = None) -> str:
        raise RuntimeError("provider down")


def test_max_errors_returns_structured_error_instead_of_raising() -> None:
    rlm = RLM(llm=AlwaysFailModel(), max_iterations=3, max_errors=1, run_id="run_errors")

    result = rlm.completion("will fail", context={})

    assert result.ready is False
    assert result.error == "max_errors_exceeded"
    assert result.answer_type == "failed"
    assert result.error_info is not None
    assert result.error_info.code == "max_errors_exceeded"
    assert "provider down" in result.error_info.message
    assert result.trajectory["run_metadata"]["run_id"] == "run_errors"
    assert result.trajectory["events"][-1]["event"] == "run_finished"


class SlowModel:
    def __call__(self, system: str, user: str, model: str | None = None) -> str:
        time.sleep(0.03)
        return "```repl\nanswer['content'] = 'too slow'\nanswer['ready'] = True\n```"


def test_max_timeout_marks_result_as_timed_out_with_trajectory() -> None:
    rlm = RLM(llm=SlowModel(), max_iterations=1, max_timeout_seconds=0.001, run_id="run_timeout")

    result = rlm.completion("slow", context={})

    assert result.ready is False
    assert result.error == "max_timeout_exceeded"
    assert result.answer_type == "failed"
    assert result.error_info is not None
    assert result.error_info.code == "max_timeout_exceeded"
    assert result.trajectory["events"][-1]["error"]["code"] == "max_timeout_exceeded"


def test_max_tokens_stops_with_structured_error() -> None:
    model = ScriptedModel(
        [
            "Finish.\n```repl\nanswer['content'] = 'token gated'\nanswer['ready'] = True\n```",
        ]
    )
    rlm = RLM(llm=model, max_iterations=1, max_tokens=1, run_id="run_tokens")

    result = rlm.completion("this prompt should exceed one estimated token", context={})

    assert result.ready is False
    assert result.error == "max_tokens_exceeded"
    assert result.error_info is not None
    assert result.error_info.code == "max_tokens_exceeded"


def test_max_budget_uses_optional_cost_tracking_backend() -> None:
    model = ScriptedModel(
        [
            "Finish.\n```repl\nanswer['content'] = 'budget gated'\nanswer['ready'] = True\n```",
        ]
    )
    rlm = RLM(
        llm=model,
        max_iterations=1,
        max_budget=0.000001,
        cost_per_1k_tokens=1.0,
        run_id="run_budget",
    )

    result = rlm.completion("this prompt should exceed the tiny budget", context={})

    assert result.ready is False
    assert result.error == "max_budget_exceeded"
    assert result.error_info is not None
    assert result.error_info.code == "max_budget_exceeded"
    assert result.usage.estimated_cost is not None
