from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class RLMQAError(Exception):
    """Base error for the local RLM QA service."""


class NoTextError(RLMQAError):
    """Raised when a document has no readable local text."""


class NoEvidenceError(RLMQAError):
    """Raised when no grounded document evidence can be found."""


class DocNotReadyError(RLMQAError):
    """Raised when a document cannot be used for grounded QA."""


@dataclass(frozen=True)
class Page:
    source_id: str
    page_number: int
    text: str
    char_start: int = 0
    char_end: int = 0


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    page_start: int
    page_end: int
    text: str
    evidence_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_id: str
    chunk_id: str
    page_number: int
    quote: str
    char_start: int = 0
    char_end: int = 0


@dataclass(frozen=True)
class DocumentPack:
    pack_id: str
    source_id: str
    file_name: str
    page_start: int
    page_end: int
    token_estimate: int
    chunk_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    text: str = ""


@dataclass(frozen=True)
class DocProfile:
    source_id: str
    file_name: str
    page_count: int
    token_estimate: int
    chunk_count: int
    evidence_count: int
    direct_ready: bool
    rlm_ready: bool
    status: str
    error_code: str = ""


@dataclass(frozen=True)
class UploadResult:
    local_doc_id: str
    source_id: str
    status: str
    doc_profile: DocProfile
    manifest: dict[str, Any]


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    source_id: str
    chunk_id: str
    page_number: int
    quote: str
    file_name: str = ""
    quote_source: str = ""
    char_start: int = 0
    char_end: int = 0
    doc_id: str = ""
    page_end: int = 0
    ref_id: str = ""
    score: float | None = None


@dataclass(frozen=True)
class AnswerDebug:
    run_id: str
    fallback_reason: str = ""
    program_steps: list[str] = field(default_factory=list)
    trajectory: dict[str, Any] = field(default_factory=dict)
    runtime_status: str = ""
    metadata_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnswerContract:
    run_id: str
    mode: str
    answer_type: str
    answer: str
    evidences: list[EvidenceRef]
    citation_policy: str
    fallback_reason: str = ""
    debug: AnswerDebug | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
