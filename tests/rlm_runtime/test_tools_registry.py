from __future__ import annotations

import pytest

from rlm_runtime.core.types import RLMResult, REPLResult
from rlm_runtime.tools.registry import RESERVED_TOOL_NAMES, ToolDefinition, validate_custom_tools


def test_validate_custom_tools_rejects_reserved_names() -> None:
    with pytest.raises(ValueError, match="reserved"):
        validate_custom_tools({"llm_query": lambda prompt: prompt})


def test_validate_custom_tools_accepts_callable_and_value_tools() -> None:
    tools = validate_custom_tools(
        {
            "double": {"tool": lambda value: value * 2, "description": "Double a number."},
            "CONFIG": {"tool": {"limit": 3}, "description": "Static config."},
        }
    )

    assert [tool.name for tool in tools] == ["double", "CONFIG"]
    assert all(isinstance(tool, ToolDefinition) for tool in tools)
    assert tools[0].description == "Double a number."


def test_reserved_names_cover_runtime_scaffold() -> None:
    assert {"context", "answer", "llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched"} <= RESERVED_TOOL_NAMES


def test_runtime_result_dataclasses_expose_expected_fields() -> None:
    repl_result = REPLResult(stdout="ok", stderr="", final_answer="done")
    result = RLMResult(response="done", iterations=[], trajectory={"iterations": []})

    assert repl_result.stdout == "ok"
    assert repl_result.final_answer == "done"
    assert result.response == "done"
    assert result.trajectory["iterations"] == []
