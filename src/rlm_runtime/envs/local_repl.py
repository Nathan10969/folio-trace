from __future__ import annotations

import contextlib
import io
import re
import traceback
from typing import Any, Callable

from rlm_runtime.core.types import REPLResult
from rlm_runtime.tools.registry import ToolDefinition, validate_custom_tools


LLMQueryFn = Callable[[str, str | None], str]
LLMBatchedQueryFn = Callable[[list[str], str | None], list[str]]
MAX_STDIO_CHARS = 20_000
MAX_SNAPSHOT_REPR_CHARS = 1_000


def _default_llm_query(prompt: str, model: str | None = None) -> str:
    raise RuntimeError("llm_query is not configured")


class LocalREPL:
    def __init__(
        self,
        *,
        context: Any,
        custom_tools: dict[str, Any] | None = None,
        llm_query: LLMQueryFn | None = None,
        llm_query_batched: LLMBatchedQueryFn | None = None,
        rlm_query: LLMQueryFn | None = None,
        rlm_query_batched: LLMBatchedQueryFn | None = None,
    ) -> None:
        self.context = context
        self.answer: dict[str, Any] = {"content": "", "ready": False}
        self.custom_tool_defs = validate_custom_tools(custom_tools)
        self._llm_query_fn = llm_query or _default_llm_query
        self._llm_query_batched_fn = llm_query_batched or self._default_llm_query_batched
        self._rlm_query_fn = rlm_query or self._llm_query
        self._rlm_query_batched_fn = rlm_query_batched or self._llm_query_batched
        self._runtime_calls: list[dict[str, Any]] = []
        self.globals: dict[str, Any] = {"__builtins__": _safe_builtins()}
        self.locals: dict[str, Any] = {}
        self._install_scaffold()

    def execute_code(self, code: str) -> REPLResult:
        stdout_buffer = io.StringIO()
        stderr = ""
        self._runtime_calls = []
        combined = {**self.globals, **self.locals}
        with contextlib.redirect_stdout(stdout_buffer):
            try:
                exec(code, combined, combined)
            except Exception:
                stderr = traceback.format_exc()

        self.locals.update({key: value for key, value in combined.items() if key not in self.globals})
        self._restore_scaffold()
        final_answer = self.answer.get("content", "") if self.answer.get("ready") else None
        return REPLResult(
            stdout=_truncate(stdout_buffer.getvalue(), MAX_STDIO_CHARS),
            stderr=_truncate(stderr, MAX_STDIO_CHARS),
            final_answer=final_answer,
            locals_snapshot=self._locals_snapshot(),
            rlm_calls=list(self._runtime_calls),
        )

    def _install_scaffold(self) -> None:
        self.globals.update(
            {
                "llm_query": self._llm_query,
                "llm_query_batched": self._llm_query_batched,
                "rlm_query": self._rlm_query,
                "rlm_query_batched": self._rlm_query_batched,
                "SHOW_VARS": self._show_vars,
            }
        )
        for definition in self.custom_tool_defs:
            self._install_tool(definition)
        self._restore_scaffold()

    def _install_tool(self, definition: ToolDefinition) -> None:
        if callable(definition.tool):
            self.globals[definition.name] = definition.tool
        else:
            self.locals[definition.name] = definition.tool

    def _restore_scaffold(self) -> None:
        self.locals["context"] = self.context
        self.locals["answer"] = self.answer
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
        self.globals["rlm_query"] = self._rlm_query
        self.globals["rlm_query_batched"] = self._rlm_query_batched
        self.globals["SHOW_VARS"] = self._show_vars

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        self._runtime_calls.append(_query_log("llm_query", prompt, model=model, batched=False))
        return self._llm_query_fn(prompt, model)

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        for prompt in prompts:
            self._runtime_calls.append(_query_log("llm_query_batched_item", prompt, model=model, batched=True))
        return self._llm_query_batched_fn(prompts, model)

    def _rlm_query(self, prompt: str, model: str | None = None) -> str:
        self._runtime_calls.append(_query_log("rlm_query", prompt, model=model, batched=False))
        return self._rlm_query_fn(prompt, model)

    def _rlm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        for prompt in prompts:
            self._runtime_calls.append(_query_log("rlm_query_batched_item", prompt, model=model, batched=True))
        return self._rlm_query_batched_fn(prompts, model)

    def _default_llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        return [self._llm_query(prompt, model) for prompt in prompts]

    def _show_vars(self) -> list[str]:
        hidden = {"context", "answer"}
        return sorted(key for key in self.locals if key not in hidden and not key.startswith("__"))

    def _locals_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for key, value in self.locals.items():
            if key.startswith("__"):
                continue
            if key in {"context", "answer"}:
                continue
            snapshot[key] = _truncate(repr(value), MAX_SNAPSHOT_REPR_CHARS)
        return snapshot


def _safe_builtins() -> dict[str, Any]:
    blocked = {"compile", "eval", "exec", "globals", "input", "locals"}
    return {key: value for key, value in __builtins__.items() if key not in blocked}  # type: ignore[union-attr]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    suffix = "...<truncated>"
    return text[: max(0, limit - len(suffix))] + suffix


def _query_log(kind: str, prompt: str, *, model: str | None, batched: bool) -> dict[str, Any]:
    return {
        "kind": kind,
        "model": model,
        "prompt_len": len(prompt),
        "pack_ids": re.findall(r"PACK_ID=([A-Za-z0-9_:-]+)", prompt),
        "batched": batched,
    }
