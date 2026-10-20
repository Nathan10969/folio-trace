from __future__ import annotations

from typing import Any

from rlm_runtime.tools.registry import validate_custom_tools


def build_system_prompt(custom_tools: dict[str, Any] | None = None) -> str:
    tool_lines = []
    for tool in validate_custom_tools(custom_tools):
        if tool.description:
            tool_lines.append(f"- `{tool.name}`: {tool.description}")
        else:
            tool_lines.append(f"- `{tool.name}`")

    tools_section = "\n".join(tool_lines) if tool_lines else "- No custom tools."
    return (
        "You are running inside a Recursive Language Model runtime. "
        "Solve the task by writing Python code inside ```repl``` blocks.\n\n"
        "Runtime variables and functions:\n"
        "- `context`: task context object.\n"
        "- `answer`: dict with `content` and `ready`; set `answer['ready'] = True` to finish.\n"
        "- `llm_query(prompt, model=None)`: a plain sub-LLM call.\n"
        "- `llm_query_batched(prompts, model=None)`: plain sub-LLM calls over a list.\n"
        "- `rlm_query(prompt, model=None)`: recursive RLM call when enabled; otherwise plain sub-LLM.\n"
        "- `rlm_query_batched(prompts, model=None)`: recursive batched RLM calls when enabled.\n"
        "- `SHOW_VARS()`: list variables stored in the REPL.\n\n"
        "Custom tools:\n"
        f"{tools_section}\n\n"
        "Inspect context before answering. Print only compact observations. "
        "Return the final result by setting `answer['content']` and `answer['ready'] = True`."
    )


def build_user_prompt(prompt: str, iteration: int, max_iterations: int, transcript: list[str]) -> str:
    prior = "\n\n".join(transcript)
    if prior:
        prior = f"\n\nPrevious runtime feedback:\n{prior}"
    return f"Task: {prompt}\n\nTurn {iteration + 1}/{max_iterations}.{prior}"
