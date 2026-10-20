from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from .chunking import build_chunks, build_evidence, build_profile
from .config import Settings
from .contracts import Chunk, DocProfile, Evidence, Page, RLMQAError


_SAFE_DIR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CACHE_SCHEMA_VERSION = 1
_IMAGE_LABELS = {"image", "figure", "picture"}


@dataclass(frozen=True)
class PlatformDocument:
    dir_name: str
    doc_id: str
    fingerprint: str
    profile: DocProfile
    pages: list[Page]
    chunks: list[Chunk]
    evidences: list[Evidence]
    cache_hit: bool = False


@dataclass(frozen=True)
class PlatformDocumentHandle:
    position: int
    dir_name: str
    doc_id: str
    source_id: str
    file_name: str
    page_count: int
    status: str


class PlatformDocumentError(RLMQAError):
    def __init__(
        self,
        code: str,
        dir_name: str,
        message: str,
        status_code: int,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.dir_name = dir_name
        self.status_code = status_code


class PlatformDocumentLoader:
    def __init__(self, data_root: Path, cache_dir: Path, settings: Settings) -> None:
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir)
        self.settings = settings

    def load_many(self, dir_names: list[str], doc_ids: list[str] | None = None) -> list[PlatformDocument]:
        if not dir_names:
            raise PlatformDocumentError("NO_DOCUMENTS", "", "metadata.dir_names must not be empty", 400)
        normalized_doc_ids = list(doc_ids or [])
        if normalized_doc_ids and len(normalized_doc_ids) != len(dir_names):
            raise PlatformDocumentError(
                "DOC_ID_COUNT_MISMATCH",
                "",
                "metadata.doc_ids must align with metadata.dir_names",
                400,
            )
        if not normalized_doc_ids:
            normalized_doc_ids = [""] * len(dir_names)
        return [self.load(dir_name, doc_id or None) for dir_name, doc_id in zip(dir_names, normalized_doc_ids)]

    def catalog_many(
        self,
        dir_names: list[str],
        doc_ids: list[str] | None = None,
    ) -> list[PlatformDocumentHandle]:
        if not dir_names:
            raise PlatformDocumentError("NO_DOCUMENTS", "", "metadata.dir_names must not be empty", 400)
        normalized_doc_ids = list(doc_ids or [])
        if normalized_doc_ids and len(normalized_doc_ids) != len(dir_names):
            raise PlatformDocumentError(
                "DOC_ID_COUNT_MISMATCH",
                "",
                "metadata.doc_ids must align with metadata.dir_names",
                400,
            )
        if not normalized_doc_ids:
            normalized_doc_ids = [""] * len(dir_names)
        return [
            self._catalog_entry(position, dir_name, doc_id)
            for position, (dir_name, doc_id) in enumerate(
                zip(dir_names, normalized_doc_ids),
                start=1,
            )
        ]

    def load(self, dir_name: str, doc_id: str | None = None) -> PlatformDocument:
        resolved = self._resolve_dir(dir_name)
        status = self._read_json(resolved / "status.json", dir_name)
        metadata = self._read_json(resolved / "metadata.json", dir_name)
        if str(status.get("status") or "").lower() != "ok":
            raise PlatformDocumentError(
                "DIR_NOT_READY",
                dir_name,
                f"selected document directory is not ready: {dir_name}",
                422,
            )
        page_count = self._page_count(status, dir_name)
        fingerprint = self._fingerprint(resolved, dir_name)
        selected_doc_id = str(doc_id or metadata.get("fileId") or "")
        cache_path = self.cache_dir / dir_name / f"{fingerprint}.json"
        cached = self._read_cache(cache_path)
        if cached is not None:
            return self._from_cache(cached, selected_doc_id, cache_hit=True)

        pages = self._load_pages(resolved, dir_name, page_count)
        chunks = build_chunks(pages, self.settings)
        chunks, evidences = build_evidence(chunks)
        if not chunks or not evidences:
            raise PlatformDocumentError(
                "DIR_NO_TEXT",
                dir_name,
                f"selected document directory has no readable text: {dir_name}",
                422,
            )
        source_id = _source_id(dir_name)
        file_name = str(metadata.get("fileName") or f"{dir_name}.pdf")
        profile = build_profile(file_name, source_id, pages, chunks, evidences, self.settings)
        document = PlatformDocument(
            dir_name=dir_name,
            doc_id=selected_doc_id,
            fingerprint=fingerprint,
            profile=profile,
            pages=pages,
            chunks=chunks,
            evidences=evidences,
            cache_hit=False,
        )
        self._write_cache(cache_path, document)
        return document

    def _catalog_entry(
        self,
        position: int,
        dir_name: str,
        doc_id: str,
    ) -> PlatformDocumentHandle:
        source_id = _source_id(str(dir_name))
        fallback_name = f"document-{position}.pdf"
        try:
            resolved = self._resolve_dir(dir_name)
            status = self._read_json(resolved / "status.json", dir_name)
            metadata = self._read_json(resolved / "metadata.json", dir_name)
            if str(status.get("status") or "").lower() != "ok":
                raise PlatformDocumentError(
                    "DIR_NOT_READY",
                    dir_name,
                    f"selected document directory is not ready: {dir_name}",
                    422,
                )
            page_count = self._page_count(status, dir_name)
        except PlatformDocumentError:
            return PlatformDocumentHandle(
                position=position,
                dir_name=str(dir_name),
                doc_id=str(doc_id or ""),
                source_id=source_id,
                file_name=fallback_name,
                page_count=0,
                status="unavailable",
            )
        return PlatformDocumentHandle(
            position=position,
            dir_name=dir_name,
            doc_id=str(doc_id or metadata.get("fileId") or ""),
            source_id=source_id,
            file_name=str(metadata.get("fileName") or fallback_name),
            page_count=page_count,
            status="available",
        )

    def _resolve_dir(self, dir_name: str) -> Path:
        if not isinstance(dir_name, str) or dir_name in {"", ".", ".."} or not _SAFE_DIR_NAME.fullmatch(dir_name):
            raise PlatformDocumentError("DIR_NAME_INVALID", str(dir_name), "invalid document directory name", 400)
        root = self.data_root.resolve()
        try:
            resolved = (root / dir_name).resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError):
            raise PlatformDocumentError(
                "DIR_NOT_FOUND",
                dir_name,
                f"selected document directory was not found: {dir_name}",
                404,
            ) from None
        if resolved.parent != root:
            raise PlatformDocumentError("DIR_PATH_ESCAPE", dir_name, "document directory escapes configured root", 400)
        if not resolved.is_dir():
            raise PlatformDocumentError("DIR_NOT_FOUND", dir_name, "selected document path is not a directory", 404)
        return resolved

    def _read_json(self, path: Path, dir_name: str) -> dict[str, Any]:
        self._ensure_artifact_contained(path, dir_name)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PlatformDocumentError(
                "DIR_INVALID_ARTIFACT",
                dir_name,
                f"selected document has an invalid {path.name}",
                422,
            ) from exc
        if not isinstance(data, dict):
            raise PlatformDocumentError(
                "DIR_INVALID_ARTIFACT",
                dir_name,
                f"selected document has an invalid {path.name}",
                422,
            )
        return data

    def _page_count(self, status: dict[str, Any], dir_name: str) -> int:
        try:
            page_count = int(status.get("pages") or 0)
        except (TypeError, ValueError):
            page_count = 0
        if page_count <= 0:
            raise PlatformDocumentError("DIR_INVALID_ARTIFACT", dir_name, "document page count is invalid", 422)
        return page_count

    def _load_pages(self, directory: Path, dir_name: str, page_count: int) -> list[Page]:
        rows = self._read_jsonl_if_present(directory / "elements.jsonl", dir_name)
        if not _has_readable_rows(rows):
            rows = self._read_jsonl_if_present(directory / "text_blocks.jsonl", dir_name)
        grouped: dict[int, list[tuple[int, str]]] = {page: [] for page in range(1, page_count + 1)}
        for position, row in enumerate(rows):
            content = str(row.get("content") or "").strip()
            label = str(row.get("label") or "text").strip().lower()
            if not _include_content(label, content):
                continue
            try:
                page_number = int(row.get("page"))
                index = int(row.get("index", position))
            except (TypeError, ValueError):
                continue
            if page_number not in grouped:
                continue
            grouped[page_number].append((index, content))
        pages = []
        for page_number in range(1, page_count + 1):
            text = "\n\n".join(content for _, content in sorted(grouped[page_number]))
            pages.append(
                Page(
                    source_id=_source_id(dir_name),
                    page_number=page_number,
                    text=text,
                    char_start=0,
                    char_end=len(text),
                )
            )
        if not any(page.text.strip() for page in pages):
            raise PlatformDocumentError(
                "DIR_NO_TEXT",
                dir_name,
                f"selected document directory has no readable text: {dir_name}",
                422,
            )
        return pages

    def _read_jsonl_if_present(self, path: Path, dir_name: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        self._ensure_artifact_contained(path, dir_name)
        rows: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PlatformDocumentError(
                "DIR_INVALID_ARTIFACT",
                dir_name,
                f"selected document has an invalid {path.name}",
                422,
            ) from exc
        return rows

    def _fingerprint(self, directory: Path, dir_name: str) -> str:
        digest = hashlib.sha256()
        for name in ("status.json", "metadata.json", "elements.jsonl", "text_blocks.jsonl"):
            path = directory / name
            if not path.exists():
                continue
            self._ensure_artifact_contained(path, dir_name)
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _ensure_artifact_contained(self, path: Path, dir_name: str) -> None:
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError):
            return
        directory = path.parent.resolve()
        if path.is_symlink() or resolved.parent != directory:
            raise PlatformDocumentError(
                "DIR_PATH_ESCAPE",
                dir_name,
                f"selected document artifact escapes its directory: {path.name}",
                400,
            )

    def _read_cache(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or data.get("schema_version") != _CACHE_SCHEMA_VERSION:
            return None
        return data

    def _write_cache(self, path: Path, document: PlatformDocument) -> None:
        payload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "dir_name": document.dir_name,
            "metadata_doc_id": document.doc_id,
            "fingerprint": document.fingerprint,
            "profile": asdict(document.profile),
            "pages": [asdict(page) for page in document.pages],
            "chunks": [asdict(chunk) for chunk in document.chunks],
            "evidences": [asdict(evidence) for evidence in document.evidences],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _from_cache(self, data: dict[str, Any], doc_id: str, *, cache_hit: bool) -> PlatformDocument:
        return PlatformDocument(
            dir_name=str(data["dir_name"]),
            doc_id=str(doc_id or data.get("metadata_doc_id") or ""),
            fingerprint=str(data["fingerprint"]),
            profile=DocProfile(**data["profile"]),
            pages=[Page(**row) for row in data.get("pages") or []],
            chunks=[Chunk(**row) for row in data.get("chunks") or []],
            evidences=[Evidence(**row) for row in data.get("evidences") or []],
            cache_hit=cache_hit,
        )


def _source_id(dir_name: str) -> str:
    suffix = hashlib.sha256(dir_name.encode("utf-8")).hexdigest()[:20]
    return f"src_platform_{suffix}"


def _include_content(label: str, content: str) -> bool:
    if not content:
        return False
    lowered = content.lstrip().lower()
    if label in _IMAGE_LABELS and (lowered.startswith("http://") or lowered.startswith("https://") or lowered.startswith("<img")):
        return False
    return True


def _has_readable_rows(rows: list[dict[str, Any]]) -> bool:
    return any(_include_content(str(row.get("label") or "text").lower(), str(row.get("content") or "").strip()) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one normalized platform document directory")
    parser.add_argument("dir_name")
    parser.add_argument("doc_id", nargs="?", default="")
    args = parser.parse_args()
    settings = Settings.from_env()
    document = PlatformDocumentLoader(
        settings.platform_data_root,
        settings.platform_cache_dir,
        settings,
    ).load(args.dir_name, args.doc_id or None)
    print(
        json.dumps(
            {
                "dir_name": document.dir_name,
                "doc_id": document.doc_id,
                "source_id": document.profile.source_id,
                "file_name": document.profile.file_name,
                "page_count": document.profile.page_count,
                "chunk_count": len(document.chunks),
                "evidence_count": len(document.evidences),
                "fingerprint": document.fingerprint,
                "cache_hit": document.cache_hit,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
