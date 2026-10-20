from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from rlm_runtime.core.types import RLMIteration


class TrajectoryLogger:
    def __init__(
        self,
        *,
        run_metadata: dict[str, Any] | None = None,
        trajectory_path: str | Path | None = None,
    ) -> None:
        self.run_metadata = run_metadata or {}
        self.iterations: list[RLMIteration] = []
        self.events: list[dict[str, Any]] = []
        self.children: list[dict[str, Any]] = []
        self.trajectory_path = Path(trajectory_path) if trajectory_path else None
        if self.trajectory_path is not None:
            self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    def add_iteration(self, iteration: RLMIteration) -> None:
        self.iterations.append(iteration)

    def add_event(self, event: dict[str, Any]) -> None:
        clean = _jsonable(event)
        self.events.append(clean)
        self._append_jsonl(clean)

    def add_child(self, child: dict[str, Any]) -> None:
        clean = _jsonable(child)
        self.children.append(clean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_metadata": _jsonable(self.run_metadata),
            "iterations": [_jsonable(asdict(iteration)) for iteration in self.iterations],
            "events": list(self.events),
            "children": list(self.children),
        }

    def _append_jsonl(self, entry: dict[str, Any]) -> None:
        if self.trajectory_path is None:
            return
        with self.trajectory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"
