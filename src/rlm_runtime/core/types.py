from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class REPLResult:
    stdout: str
    stderr: str
    final_answer: Any | None = None
    locals_snapshot: dict[str, Any] = field(default_factory=dict)
    rlm_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RLMIteration:
    index: int
    model_response: str
    code_blocks: list[str]
    repl_results: list[REPLResult] = field(default_factory=list)
    final_answer: Any | None = None


@dataclass(frozen=True)
class UsageSummary:
    total_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class RuntimeMetadata:
    run_id: str
    parent_run_id: str | None = None
    depth: int = 1
    prompt_index: int | None = None
    model: str | None = None
    started_at: str = ""
    ended_at: str = ""
    elapsed_ms: int = 0


@dataclass(frozen=True)
class RuntimeErrorInfo:
    code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class RLMResult:
    response: Any
    iterations: list[RLMIteration]
    trajectory: dict[str, Any]
    error: str = ""
    ready: bool = False
    answer_type: str = "failed"
    usage: UsageSummary = field(default_factory=UsageSummary)
    metadata: RuntimeMetadata = field(
        default_factory=lambda: RuntimeMetadata(run_id="run_unknown")
    )
    error_info: RuntimeErrorInfo | None = None
