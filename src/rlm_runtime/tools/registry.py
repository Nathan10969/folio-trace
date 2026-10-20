from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RESERVED_TOOL_NAMES = {
    "context",
    "answer",
    "history",
    "SHOW_VARS",
    "llm_query",
    "llm_query_batched",
    "rlm_query",
    "rlm_query_batched",
}


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    tool: Any
    description: str = ""


def validate_custom_tools(custom_tools: dict[str, Any] | None) -> list[ToolDefinition]:
    if not custom_tools:
        return []

    conflicts = set(custom_tools) & RESERVED_TOOL_NAMES
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"custom tool names are reserved: {names}")

    definitions: list[ToolDefinition] = []
    for name, entry in custom_tools.items():
        if isinstance(entry, dict) and "tool" in entry:
            definitions.append(ToolDefinition(name=name, tool=entry["tool"], description=str(entry.get("description", ""))))
        else:
            definitions.append(ToolDefinition(name=name, tool=entry, description=""))
    return definitions
