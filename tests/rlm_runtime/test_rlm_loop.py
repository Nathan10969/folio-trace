from __future__ import annotations

import time

from rlm_runtime.core.rlm import RLM, find_repl_code_blocks


class ScriptedModel:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("model called more times than expected")
        return self.responses.pop(0)


class RepairsMissingReplModel:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if len(self.calls) == 1:
            return "```python\nprint('not a repl block')\n```"
        assert "No executable REPL block was found" in user
        return "```repl\nanswer['content'] = 'repaired'\nanswer['ready'] = True\n```"


def test_find_repl_code_blocks_extracts_only_repl_fences() -> None:
    text = "```python\nprint('no')\n```\n```repl\nprint('yes')\n```\n``` repl\nprint('also')\n```"

    assert find_repl_code_blocks(text) == ["print('yes')", "print('also')"]


def test_find_repl_code_blocks_accepts_python_fence_with_repl_sentinel() -> None:
    text = "```python\nrepl\nprint('yes')\n```\n```python\nprint('no')\n```"

    assert find_repl_code_blocks(text) == ["print('yes')"]


def test_rlm_loop_feedback_reports_missing_executable_repl_block() -> None:
    model = RepairsMissingReplModel()
    rlm = RLM(llm=model, max_iterations=2)

    result = rlm.completion("repair missing repl", context={})

    assert result.response == "repaired"
    assert len(model.calls) == 2


def test_rlm_loop_executes_repl_and_feeds_stdout_back_to_model() -> None:
    model = ScriptedModel(
        [
            "I will inspect the context.\n```repl\nprint('value=' + str(context['value']))\n```",
            "Now I can answer.\n```repl\nanswer['content'] = 'final ' + str(context['value'])\nanswer['ready'] = True\n```",
        ]
    )
    rlm = RLM(llm=model, max_iterations=3)

    result = rlm.completion("What is the value?", context={"value": 2})

    assert result.response == "final 2"
    assert len(result.iterations) == 2
    assert "value=2" in model.calls[1][1]
    assert result.trajectory["iterations"][0]["repl_results"][0]["stdout"].strip() == "value=2"


def test_rlm_loop_exposes_llm_query_inside_repl() -> None:
    model = ScriptedModel(
        [
            "Use a sub call.\n```repl\nchild = llm_query('child prompt')\nanswer['content'] = child\nanswer['ready'] = True\n```",
            "child answer",
        ]
    )
    rlm = RLM(llm=model, max_iterations=2)

    result = rlm.completion("delegate", context={})

    assert result.response == "child answer"
    assert len(model.calls) == 2
    assert "plain sub-LLM" in model.calls[1][0]


def test_rlm_loop_exposes_llm_query_batched_inside_repl() -> None:
    model = ScriptedModel(
        [
            "Use batch sub calls.\n```repl\nitems = llm_query_batched(['a', 'b'])\nanswer['content'] = '|'.join(items)\nanswer['ready'] = True\n```",
            "A",
            "B",
        ]
    )
    rlm = RLM(llm=model, max_iterations=2)

    result = rlm.completion("batch", context={})

    assert result.response == "A|B"
    assert [call[1] for call in model.calls[1:]] == ["a", "b"]


def test_rlm_query_uses_child_rlm_when_depth_is_available() -> None:
    model = ScriptedModel(
        [
            "Delegate to a child RLM.\n```repl\nchild = rlm_query('solve child task')\nanswer['content'] = child\nanswer['ready'] = True\n```",
            "Child inspects and answers.\n```repl\nanswer['content'] = 'child final answer'\nanswer['ready'] = True\n```",
        ]
    )
    rlm = RLM(llm=model, max_iterations=2, max_depth=2)

    result = rlm.completion("delegate recursively", context={})

    assert result.response == "child final answer"
    assert len(model.calls) == 2
    assert "solve child task" in model.calls[1][1]


class SleepyBatchModel:
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append(user)
        if "batch concurrently" in user and not user.startswith("p"):
            return (
                "Use concurrent batch calls.\n```repl\n"
                "items = llm_query_batched(['p1', 'p2', 'p3', 'p4'])\n"
                "answer['content'] = ','.join(items)\n"
                "answer['ready'] = True\n"
                "```"
            )
        if user.startswith("p"):
            time.sleep(0.15)
            return user.upper()
        raise AssertionError(f"unexpected prompt: {user}")


def test_llm_query_batched_runs_concurrently_and_preserves_order() -> None:
    model = SleepyBatchModel()
    rlm = RLM(llm=model, max_iterations=2, batch_max_workers=4)

    start = time.perf_counter()
    result = rlm.completion("batch concurrently", context={})
    elapsed = time.perf_counter() - start

    assert result.response == "P1,P2,P3,P4"
    assert elapsed < 0.45
