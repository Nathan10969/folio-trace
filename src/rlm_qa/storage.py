from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, TypeVar
from uuid import uuid4

T = TypeVar("T")


class FileStore:
    def __init__(self, runtime_dir: Path):
        self.runtime_dir = runtime_dir

    def init(self) -> None:
        for name in ("uploads", "docs", "runs", "sessions"):
            (self.runtime_dir / name).mkdir(parents=True, exist_ok=True)

    def upload_dir(self, local_doc_id: str) -> Path:
        return self._safe_path("uploads", local_doc_id)

    def doc_dir(self, source_id: str) -> Path:
        return self._safe_path("docs", source_id)

    def run_path(self, run_id: str) -> Path:
        return self._safe_path("runs", f"{run_id}.json")

    def ocr_key_health_path(self) -> Path:
        return self._safe_path("ocr_key_health.json")

    def original_upload_path(self, local_doc_id: str) -> Path:
        candidates = sorted(self.upload_dir(local_doc_id).glob("original.*"))
        if not candidates:
            raise FileNotFoundError(f"original upload is missing for {local_doc_id}")
        return candidates[0]

    def iter_doc_dirs(self) -> list[Path]:
        docs_dir = self.runtime_dir / "docs"
        if not docs_dir.exists():
            return []
        return [path for path in docs_dir.iterdir() if path.is_dir()]

    def save_upload_path(self, source: Path, local_doc_id: str) -> Path:
        target_dir = self.upload_dir(local_doc_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"original{source.suffix.lower()}"
        shutil.copyfile(source, target)
        return target

    def save_upload_bytes(self, filename: str, data: bytes, local_doc_id: str) -> Path:
        target_dir = self.upload_dir(local_doc_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix.lower() or ".bin"
        target = target_dir / f"original{suffix}"
        target.write_bytes(data)
        return target

    def write_json(self, path: Path, data: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def read_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def write_jsonl(self, path: Path, rows: Iterable[object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(_jsonable(row), ensure_ascii=False))
                fh.write("\n")

    def read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def _safe_path(self, *parts: str) -> Path:
        root = self.runtime_dir.resolve()
        path = root.joinpath(*parts).resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"path escapes runtime directory: {path}")
        return path


def _jsonable(data: object) -> object:
    if is_dataclass(data):
        return asdict(data)
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, list):
        return [_jsonable(item) for item in data]
    if isinstance(data, dict):
        return {key: _jsonable(value) for key, value in data.items()}
    return data
