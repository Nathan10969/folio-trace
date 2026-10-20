from __future__ import annotations

from pathlib import Path

from rlm_qa.config import Settings


def test_settings_loads_local_env_file_without_overriding_process_env(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=from-file",
                "DEEPSEEK_BASE_URL=https://example.invalid",
                "DEEPSEEK_MODEL=deepseek-v4-pro",
                "DEEPSEEK_SUBAGENT_MODEL=deepseek-v4-pro",
                "DEEPSEEK_SUBAGENT_API_KEYS=ds-a,ds-b",
                "RLMQA_RUNTIME_DIR=custom_runtime",
                "GLM_BASE_URL=https://glm.example.invalid/api/paas/v4",
                "GLM_GENERATION_MODEL=glm-5.2",
                "GLM_GENERATION_API_KEYS=glm-gen-a,glm-gen-b,glm-gen-c",
                "GLM_GENERATION_MAX_CONCURRENCY=3",
                "GLM_GENERATION_MAX_ATTEMPTS=3",
                "GLM_OCR_API_KEYS=ocr-a,ocr-b",
                "GLM_SUBAGENT_MODEL=glm-5.1",
                "GLM_SUBAGENT_API_KEYS=key-a,key-b",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-process")

    settings = Settings.from_env()

    assert settings.deepseek_api_key == "from-process"
    assert settings.deepseek_base_url == "https://example.invalid"
    assert settings.deepseek_model == "deepseek-v4-pro"
    assert settings.deepseek_subagent_model == "deepseek-v4-pro"
    assert settings.deepseek_subagent_api_keys == ("ds-a", "ds-b")
    assert settings.runtime_dir == Path("custom_runtime")
    assert settings.glm_base_url == "https://glm.example.invalid/api/paas/v4"
    assert settings.glm_generation_model == "glm-5.2"
    assert settings.glm_generation_api_keys == ("glm-gen-a", "glm-gen-b", "glm-gen-c")
    assert settings.glm_generation_max_concurrency == 3
    assert settings.glm_generation_max_attempts == 3
    assert settings.glm_ocr_api_keys == ("ocr-a", "ocr-b")
    assert settings.glm_ocr_api_key == "ocr-a"
    assert settings.glm_subagent_model == "glm-5.1"
    assert settings.glm_subagent_api_keys == ("key-a", "key-b")


def test_settings_reads_non_default_provider_role_overrides_from_env_file(
    tmp_path: Path, monkeypatch
) -> None:
    overrides = {
        "GLM_GENERATION_MODEL": "glm-generation-test",
        "GLM_GENERATION_MAX_CONCURRENCY": "2",
        "GLM_GENERATION_MAX_ATTEMPTS": "5",
        "DEEPSEEK_MODEL": "deepseek-fallback-test",
        "DEEPSEEK_SUBAGENT_MODEL": "deepseek-subagent-test",
    }
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(f"{key}={value}" for key, value in overrides.items()),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    for key in overrides:
        monkeypatch.delenv(key, raising=False)

    settings = Settings.from_env()

    assert settings.glm_generation_model == "glm-generation-test"
    assert settings.glm_generation_max_concurrency == 2
    assert settings.glm_generation_max_attempts == 5
    assert settings.deepseek_model == "deepseek-fallback-test"
    assert settings.deepseek_subagent_model == "deepseek-subagent-test"


def test_settings_loads_utf8_bom_env_file(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("\ufeffDEEPSEEK_API_KEY=from-bom-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = Settings.from_env()

    assert settings.deepseek_api_key == "from-bom-file"


def test_settings_reads_rlm_runtime_budget_and_key_pool_knobs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RLM_MAX_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("RLM_MAX_TOKENS", "12345")
    monkeypatch.setenv("RLM_MAX_ERRORS", "2")
    monkeypatch.setenv("RLM_MAX_BUDGET", "0.42")
    monkeypatch.setenv("RLM_COST_PER_1K_TOKENS", "0.01")
    monkeypatch.setenv("RLM_MAX_DEEP_PACKS", "7")
    monkeypatch.setenv("RLM_FALLBACK_EVIDENCE_PER_PACK", "4")
    monkeypatch.setenv("RLM_SUBAGENT_MAX_RETRIES", "3")
    monkeypatch.setenv("RLM_KEY_COOLDOWN_SECONDS", "9")
    monkeypatch.setenv("RLM_KEY_LEASE_TIMEOUT_SECONDS", "30")

    settings = Settings.from_env()

    assert settings.rlm_max_timeout_seconds == 12.5
    assert settings.rlm_max_tokens == 12345
    assert settings.rlm_max_errors == 2
    assert settings.rlm_max_budget == 0.42
    assert settings.rlm_cost_per_1k_tokens == 0.01
    assert settings.rlm_max_deep_packs == 7
    assert settings.rlm_fallback_evidence_per_pack == 4
    assert settings.rlm_subagent_max_retries == 3
    assert settings.rlm_key_cooldown_seconds == 9
    assert settings.rlm_key_lease_timeout_seconds == 30


def test_settings_reads_glm_ocr_queue_knobs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GLM_OCR_API_KEYS", "ocr-a,ocr-b,ocr-c")
    monkeypatch.setenv("GLM_OCR_MAX_WORKERS", "3")
    monkeypatch.setenv("GLM_OCR_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("GLM_OCR_REQUEST_TIMEOUT_SECONDS", "270")
    monkeypatch.setenv("GLM_OCR_LEASE_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("GLM_OCR_TRANSIENT_COOLDOWN_SECONDS", "21")
    monkeypatch.setenv("GLM_OCR_RATE_LIMIT_COOLDOWN_SECONDS", "51")

    settings = Settings.from_env()

    assert settings.glm_ocr_api_keys == ("ocr-a", "ocr-b", "ocr-c")
    assert settings.glm_ocr_max_workers == 3
    assert settings.glm_ocr_max_attempts == 4
    assert settings.glm_ocr_request_timeout_seconds == 270
    assert settings.glm_ocr_lease_timeout_seconds == 300
    assert settings.glm_ocr_transient_cooldown_seconds == 21
    assert settings.glm_ocr_rate_limit_cooldown_seconds == 51

def test_settings_reads_platform_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RLM_PLATFORM_DATA_ROOT", str(tmp_path / "glm"))
    monkeypatch.setenv("RLM_PLATFORM_CACHE_DIR", str(tmp_path / "cache"))

    settings = Settings.from_env()

    assert settings.platform_data_root == tmp_path / "glm"
    assert settings.platform_cache_dir == tmp_path / "cache"


def test_settings_reads_conversation_memory_knobs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RLM_CONVERSATION_MEMORY_ENABLED", "false")
    monkeypatch.setenv("RLM_CONVERSATION_MEMORY_DB", "custom/memory.sqlite3")
    monkeypatch.setenv("RLM_CONVERSATION_MEMORY_MAX_CONTEXT_CHARS", "9000")
    monkeypatch.setenv("RLM_CONVERSATION_MEMORY_MAX_MESSAGES", "80")
    monkeypatch.setenv("RLM_CONVERSATION_MEMORY_MAX_PENDING_PER_CHAT", "3")
    monkeypatch.setenv("RLM_PLATFORM_MAX_INFLIGHT_REQUESTS", "48")
    monkeypatch.setenv("RLM_REQUEST_TIMEOUT_SECONDS", "45")

    settings = Settings.from_env()

    assert settings.conversation_memory_enabled is False
    assert settings.conversation_memory_db_path == Path("custom/memory.sqlite3")
    assert settings.conversation_memory_max_context_chars == 9000
    assert settings.conversation_memory_max_messages == 80
    assert settings.conversation_memory_max_pending_per_chat == 3
    assert settings.platform_max_inflight_requests == 48
    assert settings.request_timeout_seconds == 45


def test_settings_can_disable_turn_execution_contract_for_rollback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RLM_TURN_EXECUTION_CONTRACT_ENABLED", "false")

    settings = Settings.from_env()

    assert settings.turn_execution_contract_enabled is False


def test_settings_rejects_ambiguous_conversation_memory_boolean(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RLM_CONVERSATION_MEMORY_ENABLED", "sometimes")

    try:
        Settings.from_env()
    except ValueError as exc:
        assert "RLM_CONVERSATION_MEMORY_ENABLED" in str(exc)
    else:
        raise AssertionError("ambiguous boolean must be rejected")


def test_settings_reads_map_execution_and_reduce_context_modes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RLM_MAP_EXECUTION_MODE", "one_pass")
    monkeypatch.setenv("RLM_REDUCE_CONTEXT_MODE", "compact")

    settings = Settings.from_env()

    assert settings.rlm_map_execution_mode == "one_pass"
    assert settings.rlm_reduce_context_mode == "compact"


def test_settings_rejects_unknown_map_execution_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RLM_MAP_EXECUTION_MODE", "magic")

    try:
        Settings.from_env()
    except ValueError as exc:
        assert "RLM_MAP_EXECUTION_MODE" in str(exc)
    else:
        raise AssertionError("unknown map execution mode must be rejected")


def test_settings_rejects_unknown_reduce_context_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RLM_REDUCE_CONTEXT_MODE", "huge")

    try:
        Settings.from_env()
    except ValueError as exc:
        assert "RLM_REDUCE_CONTEXT_MODE" in str(exc)
    else:
        raise AssertionError("unknown reduce context mode must be rejected")
