from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    runtime_dir: Path = Path("runtime/rlm_qa")
    platform_data_root: Path = Path("data/platform")
    platform_cache_dir: Path = Path("runtime/platform_docs")
    platform_max_inflight_requests: int = 64
    turn_execution_contract_enabled: bool = True
    conversation_memory_enabled: bool = True
    conversation_memory_db_path: Path | None = None
    conversation_memory_max_context_chars: int = 12_000
    conversation_memory_max_messages: int = 100
    conversation_memory_max_pending_per_chat: int = 8
    virtual_page_chars: int = 5000
    chunk_chars: int = 1600
    chunk_overlap_chars: int = 200
    direct_max_pages: int = 10
    direct_max_tokens: int = 30_000
    pdftotext_cmd: str = "pdftotext"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_api_key: str = ""
    deepseek_subagent_model: str = "deepseek-v4-pro"
    deepseek_subagent_api_keys: tuple[str, ...] = ()
    deepseek_finalizer_model: str = "deepseek-v4-pro"
    deepseek_finalizer_api_keys: tuple[str, ...] = ()
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_generation_model: str = "glm-5.2"
    glm_generation_api_keys: tuple[str, ...] = ()
    glm_generation_max_concurrency: int = 3
    glm_generation_max_attempts: int = 3
    glm_ocr_api_key: str = ""
    glm_ocr_api_keys: tuple[str, ...] = ()
    glm_ocr_model: str = "glm-ocr"
    glm_ocr_max_workers: int = 3
    glm_ocr_max_attempts: int = 3
    glm_ocr_request_timeout_seconds: float = 270.0
    glm_ocr_lease_timeout_seconds: float = 300.0
    glm_ocr_transient_cooldown_seconds: float = 30.0
    glm_ocr_rate_limit_cooldown_seconds: float = 60.0
    glm_subagent_model: str = "glm-5.1"
    glm_subagent_api_keys: tuple[str, ...] = ()
    rlm_subagent_pack_target_tokens: int = 80_000
    rlm_subagent_pack_hard_max_tokens: int = 100_000
    rlm_batch_max_workers: int = 4
    rlm_max_depth: int = 1
    rlm_max_timeout_seconds: float | None = None
    rlm_max_tokens: int | None = None
    rlm_max_errors: int | None = None
    rlm_max_budget: float | None = None
    rlm_cost_per_1k_tokens: float | None = None
    rlm_max_deep_packs: int = 8
    rlm_fallback_evidence_per_pack: int = 5
    rlm_map_execution_mode: str = "staged"
    rlm_reduce_context_mode: str = "expanded"
    rlm_subagent_max_retries: int = 1
    rlm_key_cooldown_seconds: float = 30.0
    rlm_key_lease_timeout_seconds: float | None = 120.0
    rlm_key_max_failures_before_dead: int = 3
    rlm_evidence_simple_limit: int = 3
    rlm_evidence_summary_limit: int = 10
    rlm_evidence_default_limit: int = 10
    rlm_evidence_detail_limit: int = 20
    rlm_evidence_multi_per_doc: int = 3
    rlm_evidence_multi_global: int = 20
    request_timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        local_env = _read_dotenv(Path(".env"))

        def value(key: str, default: str) -> str:
            return os.getenv(key) or local_env.get(key) or default

        return cls(
            runtime_dir=Path(value("RLMQA_RUNTIME_DIR", "runtime/rlm_qa")),
            platform_data_root=Path(value("RLM_PLATFORM_DATA_ROOT", "data/platform")),
            platform_cache_dir=Path(value("RLM_PLATFORM_CACHE_DIR", "runtime/platform_docs")),
            platform_max_inflight_requests=int(value("RLM_PLATFORM_MAX_INFLIGHT_REQUESTS", "64")),
            turn_execution_contract_enabled=_bool_value(
                "RLM_TURN_EXECUTION_CONTRACT_ENABLED",
                value("RLM_TURN_EXECUTION_CONTRACT_ENABLED", "true"),
            ),
            conversation_memory_enabled=_bool_value(
                "RLM_CONVERSATION_MEMORY_ENABLED",
                value("RLM_CONVERSATION_MEMORY_ENABLED", "true"),
            ),
            conversation_memory_db_path=_optional_path(value("RLM_CONVERSATION_MEMORY_DB", "")),
            conversation_memory_max_context_chars=int(
                value("RLM_CONVERSATION_MEMORY_MAX_CONTEXT_CHARS", "12000")
            ),
            conversation_memory_max_messages=int(value("RLM_CONVERSATION_MEMORY_MAX_MESSAGES", "100")),
            conversation_memory_max_pending_per_chat=int(
                value("RLM_CONVERSATION_MEMORY_MAX_PENDING_PER_CHAT", "8")
            ),
            pdftotext_cmd=value("PDFTOTEXT_CMD", "pdftotext"),
            deepseek_base_url=value("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=value("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            deepseek_api_key=value("DEEPSEEK_API_KEY", ""),
            deepseek_subagent_model=value("DEEPSEEK_SUBAGENT_MODEL", "deepseek-v4-pro"),
            deepseek_subagent_api_keys=_split_csv(value("DEEPSEEK_SUBAGENT_API_KEYS", "")),
            deepseek_finalizer_model=value("DEEPSEEK_FINALIZER_MODEL", "deepseek-v4-pro"),
            deepseek_finalizer_api_keys=_split_csv(value("DEEPSEEK_FINALIZER_API_KEYS", "")),
            glm_base_url=value("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            glm_generation_model=value("GLM_GENERATION_MODEL", "glm-5.2"),
            glm_generation_api_keys=_split_csv(value("GLM_GENERATION_API_KEYS", "")),
            glm_generation_max_concurrency=int(value("GLM_GENERATION_MAX_CONCURRENCY", "3")),
            glm_generation_max_attempts=int(value("GLM_GENERATION_MAX_ATTEMPTS", "3")),
            glm_ocr_api_key=_first_non_empty(
                value("GLM_OCR_API_KEY", ""),
                _split_csv(value("GLM_OCR_API_KEYS", "")),
            ),
            glm_ocr_api_keys=_split_csv(value("GLM_OCR_API_KEYS", "")),
            glm_ocr_model=value("GLM_OCR_MODEL", "glm-ocr"),
            glm_ocr_max_workers=int(value("GLM_OCR_MAX_WORKERS", "3")),
            glm_ocr_max_attempts=int(value("GLM_OCR_MAX_ATTEMPTS", "3")),
            glm_ocr_request_timeout_seconds=float(value("GLM_OCR_REQUEST_TIMEOUT_SECONDS", "270")),
            glm_ocr_lease_timeout_seconds=float(value("GLM_OCR_LEASE_TIMEOUT_SECONDS", "300")),
            glm_ocr_transient_cooldown_seconds=float(value("GLM_OCR_TRANSIENT_COOLDOWN_SECONDS", "30")),
            glm_ocr_rate_limit_cooldown_seconds=float(value("GLM_OCR_RATE_LIMIT_COOLDOWN_SECONDS", "60")),
            glm_subagent_model=value("GLM_SUBAGENT_MODEL", "glm-5.1"),
            glm_subagent_api_keys=_split_csv(value("GLM_SUBAGENT_API_KEYS", "")),
            rlm_batch_max_workers=int(value("RLM_BATCH_MAX_WORKERS", "4")),
            rlm_max_depth=int(value("RLM_MAX_DEPTH", "1")),
            rlm_max_timeout_seconds=_optional_float(value("RLM_MAX_TIMEOUT_SECONDS", "")),
            rlm_max_tokens=_optional_int(value("RLM_MAX_TOKENS", "")),
            rlm_max_errors=_optional_int(value("RLM_MAX_ERRORS", "")),
            rlm_max_budget=_optional_float(value("RLM_MAX_BUDGET", "")),
            rlm_cost_per_1k_tokens=_optional_float(value("RLM_COST_PER_1K_TOKENS", "")),
            rlm_max_deep_packs=int(value("RLM_MAX_DEEP_PACKS", "8")),
            rlm_fallback_evidence_per_pack=int(value("RLM_FALLBACK_EVIDENCE_PER_PACK", "5")),
            rlm_map_execution_mode=_choice_value(
                "RLM_MAP_EXECUTION_MODE",
                value("RLM_MAP_EXECUTION_MODE", "staged"),
                {"staged", "pipelined", "one_pass"},
            ),
            rlm_reduce_context_mode=_choice_value(
                "RLM_REDUCE_CONTEXT_MODE",
                value("RLM_REDUCE_CONTEXT_MODE", "expanded"),
                {"expanded", "compact"},
            ),
            rlm_subagent_max_retries=int(value("RLM_SUBAGENT_MAX_RETRIES", "1")),
            rlm_key_cooldown_seconds=float(value("RLM_KEY_COOLDOWN_SECONDS", "30")),
            rlm_key_lease_timeout_seconds=_optional_float(value("RLM_KEY_LEASE_TIMEOUT_SECONDS", "120")),
            rlm_key_max_failures_before_dead=int(value("RLM_KEY_MAX_FAILURES_BEFORE_DEAD", "3")),
            rlm_evidence_simple_limit=int(value("RLM_EVIDENCE_SIMPLE_LIMIT", "3")),
            rlm_evidence_summary_limit=int(value("RLM_EVIDENCE_SUMMARY_LIMIT", "10")),
            rlm_evidence_default_limit=int(value("RLM_EVIDENCE_DEFAULT_LIMIT", "10")),
            rlm_evidence_detail_limit=int(value("RLM_EVIDENCE_DETAIL_LIMIT", "20")),
            rlm_evidence_multi_per_doc=int(value("RLM_EVIDENCE_MULTI_PER_DOC", "3")),
            rlm_evidence_multi_global=int(value("RLM_EVIDENCE_MULTI_GLOBAL", "20")),
            request_timeout_seconds=float(value("RLM_REQUEST_TIMEOUT_SECONDS", "60")),
        )


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.lstrip("\ufeff").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _first_non_empty(primary: str, fallback_items: tuple[str, ...]) -> str:
    return primary or (fallback_items[0] if fallback_items else "")


def _optional_int(value: str) -> int | None:
    return int(value) if value.strip() else None


def _optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def _optional_path(value: str) -> Path | None:
    return Path(value.strip()) if value.strip() else None


def _bool_value(key: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be true or false")


def _choice_value(key: str, value: str, allowed: set[str]) -> str:
    normalized = value.strip().lower()
    if normalized in allowed:
        return normalized
    choices = ", ".join(sorted(allowed))
    raise ValueError(f"{key} must be one of: {choices}")
