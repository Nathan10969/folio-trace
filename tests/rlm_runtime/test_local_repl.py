from __future__ import annotations

from rlm_runtime.envs.local_repl import LocalREPL


def test_local_repl_persists_variables_between_executions() -> None:
    repl = LocalREPL(context={"value": 3})

    first = repl.execute_code("x = context['value'] + 2\nprint(x)")
    second = repl.execute_code("print(x)")

    assert first.stdout.strip() == "5"
    assert second.stdout.strip() == "5"


def test_local_repl_captures_final_answer_when_ready() -> None:
    repl = LocalREPL(context={})

    result = repl.execute_code("answer['content'] = 'done'\nanswer['ready'] = True")

    assert result.final_answer == "done"


def test_local_repl_injects_custom_tools_and_values() -> None:
    repl = LocalREPL(
        context={},
        custom_tools={
            "triple": {"tool": lambda value: value * 3, "description": "Triple a value."},
            "CONFIG": {"tool": {"limit": 7}, "description": "Static config."},
        },
    )

    result = repl.execute_code("print(triple(CONFIG['limit']))")

    assert result.stdout.strip() == "21"


def test_local_repl_restores_reserved_runtime_names() -> None:
    repl = LocalREPL(context={"value": 4}, llm_query=lambda prompt, model=None: f"llm:{prompt}")

    repl.execute_code("llm_query = 'broken'\ncontext = None\nanswer = {'content': 'bad', 'ready': False}")
    result = repl.execute_code(
        "print(callable(llm_query))\n"
        "print(context['value'])\n"
        "answer['content'] = llm_query('ping')\n"
        "answer['ready'] = True"
    )

    lines = result.stdout.strip().splitlines()
    assert lines == ["True", "4"]
    assert result.final_answer == "llm:ping"


def test_local_repl_captures_exceptions_as_stderr() -> None:
    repl = LocalREPL(context={})

    result = repl.execute_code("raise ValueError('boom')")

    assert "ValueError: boom" in result.stderr
    assert result.final_answer is None


def test_local_repl_truncates_large_local_snapshots() -> None:
    repl = LocalREPL(context={})

    result = repl.execute_code("large_text = 'x' * 5000")

    assert len(result.locals_snapshot["large_text"]) < 1200
    assert result.locals_snapshot["large_text"].endswith("...<truncated>")


def test_local_repl_records_llm_query_calls_with_model_prompt_size_and_pack_ids() -> None:
    repl = LocalREPL(
        context={},
        llm_query=lambda prompt, model=None: "sub-answer",
    )

    result = repl.execute_code(
        "value = llm_query('PACK_ID=pack_src_test_0001\\nfull pack text', model='glm-5.1')\n"
        "print(value)"
    )

    assert result.stdout.strip() == "sub-answer"
    assert result.rlm_calls == [
        {
            "kind": "llm_query",
            "model": "glm-5.1",
            "prompt_len": len("PACK_ID=pack_src_test_0001\nfull pack text"),
            "pack_ids": ["pack_src_test_0001"],
            "batched": False,
        }
    ]


def test_local_repl_records_batched_llm_query_items() -> None:
    repl = LocalREPL(
        context={},
        llm_query_batched=lambda prompts, model=None: [prompt.upper() for prompt in prompts],
    )

    result = repl.execute_code(
        "items = llm_query_batched(['PACK_ID=pack_a\\nalpha', 'PACK_ID=pack_b\\nbeta'], model='glm-5.1')\n"
        "print('|'.join(items))"
    )

    assert result.stdout.strip() == "PACK_ID=PACK_A\nALPHA|PACK_ID=PACK_B\nBETA"
    assert [call["kind"] for call in result.rlm_calls] == ["llm_query_batched_item", "llm_query_batched_item"]
    assert [call["pack_ids"] for call in result.rlm_calls] == [["pack_a"], ["pack_b"]]
