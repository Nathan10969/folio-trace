from __future__ import annotations

import threading

import httpx
import pytest

import rlm_qa.llm as llm_module

from rlm_qa.config import Settings
from rlm_qa.contracts import RLMQAError
from rlm_qa.llm import (
    ChatRequest,
    ChatResponse,
    LLMStreamCancelled,
    ModelRouterLLM,
)


def test_model_router_uses_deepseek_subagent_pool_and_fixed_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, str]] = []

    def fake_chat_openai_compatible(**kwargs):
        calls.append(
            {
                "base_url": kwargs["base_url"],
                "api_key": kwargs["api_key"],
                "model": kwargs["model"],
            }
        )
        return ChatResponse(answer="ok")

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible", fake_chat_openai_compatible)
    settings = Settings(
        deepseek_base_url="https://ds.example",
        deepseek_subagent_model="deepseek-v4-pro",
        deepseek_subagent_api_keys=("ds-a", "ds-b"),
        glm_subagent_model="glm-5.1",
        glm_subagent_api_keys=("glm-a",),
    )
    router = ModelRouterLLM(settings)

    first = router.chat(ChatRequest(system="s", user="u1", model="deepseek-v4-pro"))
    second = router.chat(ChatRequest(system="s", user="u2", model="deepseek-v4-pro"))

    assert first.answer == "ok"
    assert second.answer == "ok"
    assert [call["api_key"] for call in calls] == ["ds-a", "ds-b"]
    assert {call["base_url"] for call in calls} == {"https://ds.example"}
    assert {call["model"] for call in calls} == {"deepseek-v4-pro"}
    routes = [event for event in router.key_pool_trace() if event["event"] == "model_route"]
    assert [event["purpose"] for event in routes] == ["subagent", "subagent"]
    routes = [event for event in router.key_pool_trace() if event["event"] == "model_route"]
    assert {event["provider"] for event in routes} == {"deepseek"}
    assert {event["model"] for event in routes} == {"deepseek-v4-pro"}


def test_model_router_rejects_glm_subagent_model_after_ds_only_switch() -> None:
    settings = Settings(glm_subagent_model="glm-5.1", glm_subagent_api_keys=("glm-a",))
    router = ModelRouterLLM(settings)

    with pytest.raises(RLMQAError, match="GLM subagent routing is disabled"):
        router.chat(ChatRequest(system="s", user="u", model="glm-5.1"))


def test_model_router_retries_retryable_subagent_status_with_next_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, str]] = []

    def fake_chat_openai_compatible(**kwargs):
        calls.append({key: kwargs[key] for key in ("api_key", "model")})
        if len(calls) == 1:
            response = httpx.Response(
                429,
                request=httpx.Request("POST", "https://ds.example/chat/completions"),
            )
            raise httpx.HTTPStatusError(
                "rate limited",
                request=response.request,
                response=response,
            )
        return ChatResponse(answer="ok after retry")

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible", fake_chat_openai_compatible)
    settings = Settings(
        deepseek_base_url="https://ds.example",
        deepseek_subagent_model="deepseek-v4-pro",
        deepseek_subagent_api_keys=("ds-a", "ds-b"),
        rlm_subagent_max_retries=1,
        rlm_key_cooldown_seconds=0.01,
    )
    router = ModelRouterLLM(settings)

    response = router.chat(
        ChatRequest(system="s", user="u", model="caller-model", purpose="subagent")
    )

    assert response.answer == "ok after retry"
    assert calls == [
        {"api_key": "ds-a", "model": "deepseek-v4-pro"},
        {"api_key": "ds-b", "model": "deepseek-v4-pro"},
    ]
    releases = [event for event in router.key_pool_trace() if event["event"] == "release"]
    assert [event["success"] for event in releases] == [False, True]
    assert releases[0]["error"] == "429"
    assert releases[0]["failure_kind"] == "rate_limited"
    assert releases[0]["status"] == "cooldown"


def test_base_and_finalizer_share_glm_generation_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, str]] = []

    def fake_chat_openai_compatible(**kwargs):
        calls.append(
            {key: kwargs[key] for key in ("base_url", "api_key", "model")}
        )
        return ChatResponse(answer="ok")

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible", fake_chat_openai_compatible)
    router = ModelRouterLLM(
        Settings(
            glm_base_url="https://glm.example",
            glm_generation_model="glm-5.2",
            glm_generation_api_keys=("glm-a", "glm-b", "glm-c"),
            glm_generation_max_concurrency=3,
            deepseek_api_key="deepseek-secret",
            deepseek_subagent_api_keys=("ds-a",),
            deepseek_finalizer_api_keys=("legacy-finalizer",),
        )
    )

    router.chat(ChatRequest(system="DIRECT", user="one", purpose="base"))
    router.chat(ChatRequest(system="FINAL", user="two", purpose="finalizer"))
    router.chat(ChatRequest(system="DIRECT", user="three", purpose="base"))

    assert [call["api_key"] for call in calls] == ["glm-a", "glm-b", "glm-c"]
    assert {call["base_url"] for call in calls} == {"https://glm.example"}
    assert {call["model"] for call in calls} == {"glm-5.2"}
    trace = router.key_pool_trace()
    assert [event["event"] for event in trace] == [
        "acquire",
        "model_route",
        "release",
        "acquire",
        "model_route",
        "release",
        "acquire",
        "model_route",
        "release",
    ]
    routes = [event for event in trace if event["event"] == "model_route"]
    assert [event["purpose"] for event in routes] == ["base", "finalizer", "base"]
    assert {event["provider"] for event in routes} == {"glm"}
    assert {event["model"] for event in routes} == {"glm-5.2"}
    assert all(event["fallback_used"] is False for event in routes)
    assert all(event["fallback_reason"] == "" for event in routes)
    assert not any(
        raw_key in repr(trace)
        for raw_key in ("glm-a", "glm-b", "glm-c", "deepseek-secret", "ds-a")
    )


def test_model_router_initializes_only_role_pools_with_required_limits() -> None:
    router = ModelRouterLLM(
        Settings(
            glm_generation_api_keys=("glm-a", "glm-b"),
            glm_generation_max_concurrency=2,
            deepseek_subagent_api_keys=("ds-a", "ds-b"),
            deepseek_api_key="fallback",
            deepseek_finalizer_api_keys=("legacy-finalizer",),
            rlm_batch_max_workers=5,
        )
    )

    assert router._glm_generation_pool is not None
    assert router._glm_generation_pool.role == "generation"
    assert router._glm_generation_pool.config.max_global_concurrency == 2
    assert router._glm_generation_pool.config.max_per_key_concurrency == 1
    assert router._glm_generation_pool.config.max_per_run_concurrency == 1

    assert router._subagent_pool is not None
    assert router._subagent_pool.role == "subagent"
    assert router._subagent_pool.config.max_global_concurrency == 5
    assert router._subagent_pool.config.max_per_key_concurrency == 1
    assert router._subagent_pool.config.max_per_run_concurrency == 5

    assert router._generation_fallback_pool is not None
    assert router._generation_fallback_pool.role == "generation_fallback"
    assert router._generation_fallback_pool.config.max_global_concurrency == 1
    assert router._generation_fallback_pool.config.max_per_key_concurrency == 1
    assert router._generation_fallback_pool.config.max_per_run_concurrency == 1


def test_generation_calls_without_run_ids_do_not_share_per_run_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered: list[str] = []
    errors: list[BaseException] = []
    entered_lock = threading.Lock()
    both_entered = threading.Event()
    release_calls = threading.Event()

    def fake_chat_openai_compatible(**kwargs):
        with entered_lock:
            entered.append(kwargs["api_key"])
            if len(entered) == 2:
                both_entered.set()
        release_calls.wait(timeout=2)
        return ChatResponse(answer="ok")

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible", fake_chat_openai_compatible)
    router = ModelRouterLLM(
        Settings(
            glm_generation_api_keys=("glm-a", "glm-b"),
            glm_generation_max_concurrency=2,
            glm_generation_max_attempts=1,
            deepseek_api_key="fallback",
            deepseek_subagent_api_keys=("ds-a",),
        )
    )

    def call(user: str) -> None:
        try:
            router.chat(ChatRequest(system="DIRECT", user=user, purpose="base"))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(user,)) for user in ("one", "two")]
    for thread in threads:
        thread.start()
    try:
        assert both_entered.wait(timeout=1)
    finally:
        release_calls.set()
        for thread in threads:
            thread.join(timeout=2)

    assert errors == []
    assert set(entered) == {"glm-a", "glm-b"}
    routes = [event for event in router.key_pool_trace() if event["event"] == "model_route"]
    assert len({event["run_id"] for event in routes}) == 2
    assert not hasattr(router, "_finalizer_pool")


def test_model_router_does_not_dispatch_to_initialized_generation_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_chat_openai_compatible(**kwargs):
        calls.append(kwargs["api_key"])
        return ChatResponse(answer="unexpected")

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible", fake_chat_openai_compatible)
    router = ModelRouterLLM(Settings(deepseek_api_key="fallback"))

    with pytest.raises(RLMQAError, match="GLM_GENERATION_API_KEYS is not configured"):
        router.chat(ChatRequest(system="DIRECT", user="u", purpose="base"))

    assert calls == []
    assert router._generation_fallback_pool is not None
    assert router._generation_fallback_pool.trace == []


def test_model_router_streams_with_glm_generation_pool_and_releases_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, str]] = []

    def fake_stream(**kwargs):
        calls.append(
            {key: kwargs[key] for key in ("base_url", "api_key", "model")}
        )
        kwargs["on_delta"]("a")
        kwargs["on_delta"]("b")
        return ChatResponse(answer="ab")

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible_stream", fake_stream)
    settings = Settings(
        glm_base_url="https://glm.example",
        glm_generation_model="glm-5.2",
        glm_generation_api_keys=("glm-key",),
        deepseek_finalizer_api_keys=("legacy-finalizer",),
    )
    router = ModelRouterLLM(settings)
    deltas: list[str] = []

    response = router.chat_stream(
        ChatRequest(system="FINAL", user="u", purpose="finalizer"),
        deltas.append,
        lambda: False,
    )

    assert response.answer == "ab"
    assert deltas == ["a", "b"]
    assert calls == [
        {
            "base_url": "https://glm.example",
            "api_key": "glm-key",
            "model": "glm-5.2",
        }
    ]
    release = [event for event in router.key_pool_trace() if event["event"] == "release"][-1]
    assert release["success"] is True
    assert release["role"] == "generation"
    assert release["provider"] == "glm"
    assert release["model"] == "glm-5.2"


def test_model_router_stream_cancellation_releases_generation_key_as_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_stream(**kwargs):
        calls.append(kwargs["api_key"])
        raise LLMStreamCancelled("client cancelled")

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible_stream", fake_stream)
    router = ModelRouterLLM(
        Settings(
            glm_generation_api_keys=("glm-key",),
            deepseek_api_key="fallback-key",
        )
    )

    with pytest.raises(LLMStreamCancelled):
        router.chat_stream(
            ChatRequest(system="FINAL", user="u", purpose="finalizer"),
            lambda _: None,
            lambda: True,
        )

    release = [event for event in router.key_pool_trace() if event["event"] == "release"][-1]
    assert release["success"] is False
    assert release["error"] == "client_cancelled"
    assert release["failure_kind"] == "cancelled"
    assert release["status"] == "idle"
    assert router._glm_generation_pool is not None
    assert all(snapshot.in_flight == 0 for snapshot in router._glm_generation_pool.snapshot())
    assert all(
        snapshot.status.value == "idle"
        for snapshot in router._glm_generation_pool.snapshot()
    )
    assert calls == ["glm-key"]
    assert not any(
        event["event"] == "model_fallback" for event in router.key_pool_trace()
    )


def test_model_router_generation_stream_retries_only_before_first_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, str]] = []

    def fake_stream(**kwargs):
        calls.append({key: kwargs[key] for key in ("api_key", "model")})
        if len(calls) == 1:
            response = httpx.Response(
                429,
                request=httpx.Request("POST", "https://glm.example/chat/completions"),
            )
            raise httpx.HTTPStatusError(
                "rate limited",
                request=response.request,
                response=response,
            )
        kwargs["on_delta"]("ok")
        return ChatResponse(answer="ok")

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible_stream", fake_stream)
    settings = Settings(
        glm_base_url="https://glm.example",
        glm_generation_api_keys=("glm-a", "glm-b"),
        glm_generation_max_attempts=2,
        rlm_subagent_max_retries=0,
        rlm_key_cooldown_seconds=0.01,
    )
    router = ModelRouterLLM(settings)
    deltas: list[str] = []

    response = router.chat_stream(
        ChatRequest(system="FINAL", user="u", purpose="finalizer"),
        deltas.append,
        lambda: False,
    )

    assert response.answer == "ok"
    assert calls == [
        {"api_key": "glm-a", "model": "glm-5.2"},
        {"api_key": "glm-b", "model": "glm-5.2"},
    ]
    assert deltas == ["ok"]
    releases = [event for event in router.key_pool_trace() if event["event"] == "release"]
    assert releases[0]["failure_kind"] == "rate_limited"
    assert releases[0]["status"] == "cooldown"


def test_model_router_generation_stream_does_not_retry_after_first_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    deltas: list[str] = []

    def fake_stream(**kwargs):
        calls.append(kwargs["api_key"])
        kwargs["on_delta"]("partial")
        response = httpx.Response(
            503,
            request=httpx.Request("POST", "https://glm.example/chat/completions"),
        )
        raise httpx.HTTPStatusError(
            "failed",
            request=response.request,
            response=response,
        )

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible_stream", fake_stream)
    settings = Settings(
        glm_generation_api_keys=("glm-a", "glm-b"),
        glm_generation_max_attempts=2,
        deepseek_api_key="fallback-key",
        rlm_key_cooldown_seconds=0.01,
    )
    router = ModelRouterLLM(settings)

    with pytest.raises(llm_module.PartialStreamFailure) as exc_info:
        router.chat_stream(
            ChatRequest(system="FINAL", user="u", purpose="finalizer"),
            deltas.append,
            lambda: False,
        )

    assert calls == ["glm-a"]
    assert deltas == ["partial"]
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)
    assert exc_info.value.__cause__.response.status_code == 503
    assert not any(
        event["event"] == "model_fallback" for event in router.key_pool_trace()
    )


def test_model_router_generation_uses_glm_attempt_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_chat_openai_compatible(**kwargs):
        calls.append(kwargs["api_key"])
        if len(calls) < 3:
            response = httpx.Response(
                500,
                request=httpx.Request("POST", "https://glm.example/chat/completions"),
            )
            raise httpx.HTTPStatusError(
                "failed",
                request=response.request,
                response=response,
            )
        return ChatResponse(answer="ok")

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible", fake_chat_openai_compatible)
    router = ModelRouterLLM(
        Settings(
            glm_generation_api_keys=("glm-a", "glm-b", "glm-c"),
            glm_generation_max_attempts=3,
            rlm_subagent_max_retries=0,
            rlm_key_cooldown_seconds=0.01,
        )
    )

    response = router.chat(ChatRequest(system="DIRECT", user="u", purpose="base"))

    assert response.answer == "ok"
    assert calls == ["glm-a", "glm-b", "glm-c"]


def test_generation_falls_back_after_glm_pool_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str, str]] = []

    def fake_chat(**kwargs):
        calls.append(
            (
                kwargs["base_url"],
                kwargs["api_key"],
                kwargs["model"],
                kwargs["request"].run_id,
            )
        )
        if kwargs["model"] == "glm-5.2":
            response = httpx.Response(
                429,
                request=httpx.Request("POST", "https://glm.example/chat/completions"),
            )
            raise httpx.HTTPStatusError(
                "rate limited",
                request=response.request,
                response=response,
            )
        return ChatResponse(answer="deepseek fallback")

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible", fake_chat)
    router = ModelRouterLLM(
        Settings(
            glm_base_url="https://glm.example",
            glm_generation_api_keys=("glm-a", "glm-b", "glm-c"),
            glm_generation_max_attempts=3,
            deepseek_base_url="https://ds.example",
            deepseek_api_key="fallback-key",
            deepseek_model="deepseek-v4-pro",
            rlm_key_cooldown_seconds=0.01,
        )
    )

    response = router.chat(
        ChatRequest(system="DIRECT", user="u", purpose="base", run_id="run-1")
    )

    assert response.answer == "deepseek fallback"
    assert calls == [
        ("https://glm.example", "glm-a", "glm-5.2", "run-1"),
        ("https://glm.example", "glm-b", "glm-5.2", "run-1"),
        ("https://glm.example", "glm-c", "glm-5.2", "run-1"),
        ("https://ds.example", "fallback-key", "deepseek-v4-pro", "run-1"),
    ]
    trace = router.key_pool_trace()
    assert [event["event"] for event in trace] == [
        "acquire",
        "model_route",
        "release",
        "acquire",
        "model_route",
        "release",
        "acquire",
        "model_route",
        "release",
        "model_fallback",
        "acquire",
        "model_route",
        "release",
    ]
    fallback = [event for event in trace if event["event"] == "model_fallback"]
    assert fallback == [
        {
            "event": "model_fallback",
            "run_id": "run-1",
            "purpose": "base",
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "fallback_used": True,
            "fallback_reason": "429",
        }
    ]
    routes = [event for event in trace if event["event"] == "model_route"]
    assert [
        (
            event["run_id"],
            event["provider"],
            event["model"],
            event["fallback_used"],
            event["fallback_reason"],
        )
        for event in routes
    ] == [
        ("run-1", "glm", "glm-5.2", False, ""),
        ("run-1", "glm", "glm-5.2", False, ""),
        ("run-1", "glm", "glm-5.2", False, ""),
        ("run-1", "deepseek", "deepseek-v4-pro", True, "429"),
    ]


def test_generation_does_not_retry_or_fallback_on_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_chat(**kwargs):
        calls.append((kwargs["api_key"], kwargs["model"]))
        response = httpx.Response(
            400,
            request=httpx.Request("POST", "https://glm.example/chat/completions"),
        )
        raise httpx.HTTPStatusError(
            "bad request",
            request=response.request,
            response=response,
        )

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible", fake_chat)
    router = ModelRouterLLM(
        Settings(
            glm_generation_api_keys=("glm-a", "glm-b", "glm-c"),
            glm_generation_max_attempts=3,
            deepseek_api_key="fallback-key",
        )
    )

    with pytest.raises(httpx.HTTPStatusError):
        router.chat(ChatRequest(system="DIRECT", user="u", purpose="base"))

    assert calls == [("glm-a", "glm-5.2")]
    assert not any(
        event["event"] == "model_fallback" for event in router.key_pool_trace()
    )
    assert router._generation_fallback_pool is not None
    assert router._generation_fallback_pool.trace == []


def test_generation_fallback_is_attempted_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_chat(**kwargs):
        calls.append(kwargs["model"])
        response = httpx.Response(
            503,
            request=httpx.Request("POST", "https://provider.example/chat/completions"),
        )
        raise httpx.HTTPStatusError(
            "unavailable",
            request=response.request,
            response=response,
        )

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible", fake_chat)
    router = ModelRouterLLM(
        Settings(
            glm_generation_api_keys=("glm-a",),
            glm_generation_max_attempts=1,
            deepseek_api_key="fallback-key",
            deepseek_model="deepseek-v4-pro",
        )
    )

    with pytest.raises(httpx.HTTPStatusError):
        router.chat(ChatRequest(system="DIRECT", user="u", purpose="base"))

    assert calls == ["glm-5.2", "deepseek-v4-pro"]
    assert len(
        [
            event
            for event in router.key_pool_trace()
            if event["event"] == "model_fallback"
        ]
    ) == 1


def test_stream_falls_back_before_first_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_stream(**kwargs):
        calls.append((kwargs["base_url"], kwargs["api_key"], kwargs["model"]))
        if kwargs["model"] == "glm-5.2":
            response = httpx.Response(
                503,
                request=httpx.Request("POST", "https://glm.example/chat/completions"),
            )
            raise httpx.HTTPStatusError(
                "unavailable",
                request=response.request,
                response=response,
            )
        kwargs["on_delta"]("fallback")
        return ChatResponse(answer="fallback")

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible_stream", fake_stream)
    router = ModelRouterLLM(
        Settings(
            glm_base_url="https://glm.example",
            glm_generation_api_keys=("glm-a",),
            glm_generation_max_attempts=1,
            deepseek_base_url="https://ds.example",
            deepseek_api_key="fallback-key",
            deepseek_model="deepseek-v4-pro",
        )
    )
    deltas: list[str] = []

    response = router.chat_stream(
        ChatRequest(
            system="FINAL",
            user="u",
            purpose="finalizer",
            run_id="stream-run",
        ),
        deltas.append,
        lambda: False,
    )

    assert response.answer == "fallback"
    assert deltas == ["fallback"]
    assert calls == [
        ("https://glm.example", "glm-a", "glm-5.2"),
        ("https://ds.example", "fallback-key", "deepseek-v4-pro"),
    ]
    trace = router.key_pool_trace()
    fallback = [event for event in trace if event["event"] == "model_fallback"]
    assert fallback[0]["fallback_reason"] == "503"
    routes = [event for event in trace if event["event"] == "model_route"]
    assert [
        (
            event["run_id"],
            event["provider"],
            event["model"],
            event["fallback_used"],
            event["fallback_reason"],
        )
        for event in routes
    ] == [
        ("stream-run", "glm", "glm-5.2", False, ""),
        ("stream-run", "deepseek", "deepseek-v4-pro", True, "503"),
    ]


def test_stream_callback_error_is_neutral_and_never_retried_or_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    callback_error = RuntimeError("downstream callback failed")

    def fake_stream(**kwargs):
        calls.append(kwargs["api_key"])
        kwargs["on_delta"]("content")
        return ChatResponse(answer="content")

    def failing_callback(_delta: str) -> None:
        raise callback_error

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible_stream", fake_stream)
    router = ModelRouterLLM(
        Settings(
            glm_generation_api_keys=("glm-a", "glm-b"),
            glm_generation_max_attempts=2,
            deepseek_api_key="fallback-key",
        )
    )

    with pytest.raises(RuntimeError) as exc_info:
        router.chat_stream(
            ChatRequest(system="FINAL", user="u", purpose="finalizer"),
            failing_callback,
            lambda: False,
        )

    assert exc_info.value is callback_error
    assert calls == ["glm-a"]
    releases = [
        event for event in router.key_pool_trace() if event["event"] == "release"
    ]
    assert len(releases) == 1
    assert releases[0]["success"] is False
    assert releases[0]["failure_kind"] == "cancelled"
    assert releases[0]["status"] == "idle"
    assert not any(
        event["event"] == "model_fallback" for event in router.key_pool_trace()
    )
    assert router._glm_generation_pool is not None
    snapshots = router._glm_generation_pool.snapshot()
    assert all(snapshot.in_flight == 0 for snapshot in snapshots)
    assert all(snapshot.failure_count == 0 for snapshot in snapshots)


def test_model_router_stream_base_purpose_uses_glm_when_model_names_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, str]] = []

    def fake_stream(**kwargs):
        calls.append(
            {key: kwargs[key] for key in ("base_url", "api_key", "model")}
        )
        kwargs["on_delta"]("base")
        return ChatResponse(answer="base")

    monkeypatch.setattr("rlm_qa.llm._chat_openai_compatible_stream", fake_stream)
    settings = Settings(
        glm_base_url="https://glm.example",
        glm_generation_model="glm-5.2",
        glm_generation_api_keys=("glm-key",),
        deepseek_subagent_model="deepseek-v4-pro",
        deepseek_subagent_api_keys=("subagent-key",),
    )
    router = ModelRouterLLM(settings)

    router.chat_stream(
        ChatRequest(
            system="DIRECT",
            user="u",
            model="deepseek-v4-pro",
            purpose="base",
        ),
        lambda _delta: None,
        lambda: False,
    )

    assert calls == [
        {
            "base_url": "https://glm.example",
            "api_key": "glm-key",
            "model": "glm-5.2",
        }
    ]


def test_model_router_trace_preserves_cross_pool_call_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rlm_qa.llm._chat_openai_compatible",
        lambda **_kwargs: ChatResponse(answer="ok"),
    )
    router = ModelRouterLLM(
        Settings(
            glm_generation_api_keys=("glm-key",),
            deepseek_subagent_api_keys=("ds-key",),
        )
    )

    router.chat(ChatRequest(system="SUB", user="one", purpose="subagent"))
    router.chat(ChatRequest(system="DIRECT", user="two", purpose="base"))

    trace = router.key_pool_trace()
    routes = [event for event in trace if event["event"] == "model_route"]
    assert [event["provider"] for event in routes] == ["deepseek", "glm"]
    acquires = [event for event in trace if event["event"] == "acquire"]
    assert [event["role"] for event in acquires] == ["subagent", "generation"]
    assert [event["provider"] for event in acquires] == ["deepseek", "glm"]
    assert [event["model"] for event in acquires] == ["deepseek-v4-pro", "glm-5.2"]


@pytest.mark.parametrize("streaming", [False, True])
def test_model_router_rejects_unsupported_purpose(streaming: bool) -> None:
    router = ModelRouterLLM(Settings(glm_generation_api_keys=("glm-key",)))
    request = ChatRequest(system="s", user="u", purpose="reranker")

    with pytest.raises(RLMQAError, match="unsupported model call purpose: reranker"):
        if streaming:
            router.chat_stream(request, lambda _delta: None, lambda: False)
        else:
            router.chat(request)
