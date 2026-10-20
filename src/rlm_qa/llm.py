from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import threading
import time
from typing import Any, Callable, Iterator
from uuid import uuid4

import httpx

from .config import Settings
from .contracts import RLMQAError
from .key_pool import KeyFailureKind, KeyPool, KeyPoolAcquireTimeout, KeyPoolConfig


@dataclass(frozen=True)
class ChatRequest:
    system: str
    user: str
    model: str | None = None
    run_id: str = ""
    purpose: str = ""
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: str | dict[str, Any] | None = None
    tool_stream: bool = False


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatResponse:
    answer: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


class LLMStreamCancelled(RLMQAError):
    """Raised when the downstream client abandons an active model stream."""


class EmptyModelResponse(RLMQAError):
    """Raised when a provider returns no usable assistant content."""


class PartialStreamFailure(RLMQAError):
    """Raised when a provider fails after user-visible content was emitted."""


@dataclass(frozen=True)
class ProviderFailure:
    code: str
    retryable: bool
    fallback_eligible: bool
    failure_kind: KeyFailureKind


class LLMClient:
    supports_tool_calls = False

    def chat(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    def chat_stream(
        self,
        request: ChatRequest,
        on_delta: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> ChatResponse:
        response = self.chat(request)
        if cancelled():
            raise LLMStreamCancelled("client cancelled")
        if response.answer:
            on_delta(response.answer)
        return response


class FakeLLM(LLMClient):
    def __init__(self, answer: str):
        self.answer = answer

    def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(answer=self.answer)


class DeepSeekLLM(LLMClient):
    supports_tool_calls = True

    def __init__(self, settings: Settings):
        self.settings = settings

    def chat_stream(
        self,
        request: ChatRequest,
        on_delta: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> ChatResponse:
        if not self.settings.deepseek_api_key:
            raise RLMQAError("DEEPSEEK_API_KEY is not configured")
        return _chat_openai_compatible_stream(
            base_url=self.settings.deepseek_base_url,
            api_key=self.settings.deepseek_api_key,
            model=request.model or self.settings.deepseek_model,
            request=request,
            timeout=self.settings.request_timeout_seconds,
            on_delta=on_delta,
            cancelled=cancelled,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        if not self.settings.deepseek_api_key:
            raise RLMQAError("DEEPSEEK_API_KEY is not configured")
        return _chat_openai_compatible(
            base_url=self.settings.deepseek_base_url,
            api_key=self.settings.deepseek_api_key,
            model=request.model or self.settings.deepseek_model,
            request=request,
            timeout=self.settings.request_timeout_seconds,
        )


class ModelRouterLLM(LLMClient):
    supports_tool_calls = True

    def __init__(self, settings: Settings):
        self.settings = settings
        self._thread_state = threading.local()
        self._task_counter = 0
        self._task_lock = threading.Lock()
        self._trace: list[dict] = []
        self._trace_lock = threading.Lock()
        self._glm_generation_pool = (
            KeyPool(
                settings.glm_generation_api_keys,
                role="generation",
                config=KeyPoolConfig(
                    max_global_concurrency=settings.glm_generation_max_concurrency,
                    max_per_key_concurrency=1,
                    max_per_run_concurrency=1,
                    cooldown_seconds=settings.rlm_key_cooldown_seconds,
                    max_failures_before_dead=settings.rlm_key_max_failures_before_dead,
                    lease_timeout_seconds=settings.rlm_key_lease_timeout_seconds,
                ),
                trace_sink=self._pool_trace_sink(
                    provider="glm",
                    model=settings.glm_generation_model,
                ),
            )
            if settings.glm_generation_api_keys
            else None
        )
        self._subagent_pool = (
            KeyPool(
                settings.deepseek_subagent_api_keys,
                role="subagent",
                config=KeyPoolConfig(
                    max_global_concurrency=settings.rlm_batch_max_workers,
                    max_per_key_concurrency=1,
                    max_per_run_concurrency=settings.rlm_batch_max_workers,
                    cooldown_seconds=settings.rlm_key_cooldown_seconds,
                    max_failures_before_dead=settings.rlm_key_max_failures_before_dead,
                    lease_timeout_seconds=settings.rlm_key_lease_timeout_seconds,
                ),
                trace_sink=self._pool_trace_sink(
                    provider="deepseek",
                    model=settings.deepseek_subagent_model,
                ),
            )
            if settings.deepseek_subagent_api_keys
            else None
        )
        self._generation_fallback_pool = (
            KeyPool(
                (settings.deepseek_api_key,),
                role="generation_fallback",
                config=KeyPoolConfig(
                    max_global_concurrency=1,
                    max_per_key_concurrency=1,
                    max_per_run_concurrency=1,
                    cooldown_seconds=settings.rlm_key_cooldown_seconds,
                    max_failures_before_dead=settings.rlm_key_max_failures_before_dead,
                    lease_timeout_seconds=settings.rlm_key_lease_timeout_seconds,
                ),
                trace_sink=self._pool_trace_sink(
                    provider="deepseek",
                    model=settings.deepseek_model,
                ),
            )
            if settings.deepseek_api_key
            else None
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        request = self._bind_run_id(request)
        if request.model == self.settings.glm_subagent_model and self.settings.glm_subagent_model:
            raise RLMQAError("GLM subagent routing is disabled; use DeepSeek subagent model")
        if request.purpose == "subagent" or (
            not request.purpose and request.model == self.settings.deepseek_subagent_model
        ):
            return self._chat_subagent(request)
        if request.purpose in {"", "base", "finalizer", "orchestrator"}:
            return self._chat_generation(request)
        raise RLMQAError(f"unsupported model call purpose: {request.purpose}")

    def chat_stream(
        self,
        request: ChatRequest,
        on_delta: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> ChatResponse:
        request = self._bind_run_id(request)
        if request.model == self.settings.glm_subagent_model and self.settings.glm_subagent_model:
            raise RLMQAError("GLM subagent routing is disabled; use DeepSeek subagent model")
        if request.purpose == "subagent" or (
            not request.purpose and request.model == self.settings.deepseek_subagent_model
        ):
            return self._chat_subagent_stream(request, on_delta, cancelled)
        if request.purpose in {"", "base", "finalizer", "orchestrator"}:
            return self._chat_generation_stream(request, on_delta, cancelled)
        raise RLMQAError(f"unsupported model call purpose: {request.purpose}")

    def _chat_generation(self, request: ChatRequest) -> ChatResponse:
        try:
            return self._chat_pool(
                request,
                pool=self._glm_generation_pool,
                provider="glm",
                base_url=self.settings.glm_base_url,
                model=self.settings.glm_generation_model,
                max_attempts=self.settings.glm_generation_max_attempts,
                missing_message="GLM_GENERATION_API_KEYS is not configured",
            )
        except Exception as exc:
            failure = _classify_provider_error(exc)
            if not failure.fallback_eligible or self._generation_fallback_pool is None:
                raise
        self._record_model_fallback(request, failure)
        return self._chat_pool(
            request,
            pool=self._generation_fallback_pool,
            provider="deepseek",
            base_url=self.settings.deepseek_base_url,
            model=self.settings.deepseek_model,
            max_attempts=1,
            missing_message="DEEPSEEK_API_KEY is not configured",
            fallback_used=True,
            fallback_reason=failure.code,
        )

    def _chat_subagent(self, request: ChatRequest) -> ChatResponse:
        return self._chat_pool(
            request,
            pool=self._subagent_pool,
            provider="deepseek",
            base_url=self.settings.deepseek_base_url,
            model=self.settings.deepseek_subagent_model,
            max_attempts=self.settings.rlm_subagent_max_retries + 1,
            missing_message="DEEPSEEK_SUBAGENT_API_KEYS is not configured",
            effective_purpose="subagent",
        )

    def _chat_generation_stream(
        self,
        request: ChatRequest,
        on_delta: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> ChatResponse:
        try:
            return self._chat_pool_stream(
                request,
                on_delta=on_delta,
                cancelled=cancelled,
                pool=self._glm_generation_pool,
                provider="glm",
                base_url=self.settings.glm_base_url,
                model=self.settings.glm_generation_model,
                max_attempts=self.settings.glm_generation_max_attempts,
                missing_message="GLM_GENERATION_API_KEYS is not configured",
            )
        except Exception as exc:
            failure = _classify_provider_error(exc)
            if not failure.fallback_eligible or self._generation_fallback_pool is None:
                raise
        self._record_model_fallback(request, failure)
        return self._chat_pool_stream(
            request,
            on_delta=on_delta,
            cancelled=cancelled,
            pool=self._generation_fallback_pool,
            provider="deepseek",
            base_url=self.settings.deepseek_base_url,
            model=self.settings.deepseek_model,
            max_attempts=1,
            missing_message="DEEPSEEK_API_KEY is not configured",
            fallback_used=True,
            fallback_reason=failure.code,
        )

    def _chat_subagent_stream(
        self,
        request: ChatRequest,
        on_delta: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> ChatResponse:
        return self._chat_pool_stream(
            request,
            on_delta=on_delta,
            cancelled=cancelled,
            pool=self._subagent_pool,
            provider="deepseek",
            base_url=self.settings.deepseek_base_url,
            model=self.settings.deepseek_subagent_model,
            max_attempts=self.settings.rlm_subagent_max_retries + 1,
            missing_message="DEEPSEEK_SUBAGENT_API_KEYS is not configured",
            effective_purpose="subagent",
        )

    def _chat_pool(
        self,
        request: ChatRequest,
        *,
        pool: KeyPool | None,
        provider: str,
        base_url: str,
        model: str,
        max_attempts: int,
        missing_message: str,
        fallback_used: bool = False,
        fallback_reason: str = "",
        effective_purpose: str | None = None,
    ) -> ChatResponse:
        if pool is None:
            raise RLMQAError(missing_message)
        last_error: Exception | None = None
        attempts = max(1, max_attempts)
        for attempt in range(attempts):
            task_id = self._next_task_id()
            try:
                lease = pool.acquire(
                    run_id=self._active_run_id(request),
                    task_id=task_id,
                    timeout_seconds=self.settings.request_timeout_seconds,
                )
            except Exception as exc:
                last_error = exc
                failure = _classify_provider_error(exc)
                if failure.retryable and attempt + 1 < attempts:
                    continue
                raise
            started = time.perf_counter()
            try:
                self._record_model_route(
                    request,
                    provider=provider,
                    model=model,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                    effective_purpose=effective_purpose,
                )
                response = _chat_openai_compatible(
                    base_url=base_url,
                    api_key=lease.api_key,
                    model=model,
                    request=request,
                    timeout=self.settings.request_timeout_seconds,
                )
            except Exception as exc:
                last_error = exc
                failure = _classify_provider_error(exc)
                pool.release(
                    lease,
                    success=False,
                    latency_ms=_elapsed_ms(started),
                    error_code=failure.code,
                    failure_kind=failure.failure_kind,
                )
                if failure.retryable and attempt + 1 < attempts:
                    continue
                raise
            pool.release(
                lease,
                success=True,
                latency_ms=_elapsed_ms(started),
                input_tokens=_estimate_tokens(request.system) + _estimate_tokens(request.user),
                output_tokens=_estimate_tokens(response.answer),
            )
            return response
        if last_error is not None:
            raise last_error
        raise RLMQAError("model call failed without provider response")

    def _chat_pool_stream(
        self,
        request: ChatRequest,
        *,
        on_delta: Callable[[str], None],
        cancelled: Callable[[], bool],
        pool: KeyPool | None,
        provider: str,
        base_url: str,
        model: str,
        max_attempts: int,
        missing_message: str,
        fallback_used: bool = False,
        fallback_reason: str = "",
        effective_purpose: str | None = None,
    ) -> ChatResponse:
        if pool is None:
            raise RLMQAError(missing_message)
        last_error: Exception | None = None
        attempts = max(1, max_attempts)
        for attempt in range(attempts):
            task_id = self._next_task_id()
            try:
                lease = pool.acquire(
                    run_id=self._active_run_id(request),
                    task_id=task_id,
                    timeout_seconds=self.settings.request_timeout_seconds,
                )
            except Exception as exc:
                last_error = exc
                failure = _classify_provider_error(exc)
                if failure.retryable and attempt + 1 < attempts:
                    continue
                raise
            started = time.perf_counter()
            emitted = False
            callback_error: BaseException | None = None

            def tracked_delta(delta: str) -> None:
                nonlocal callback_error, emitted
                try:
                    on_delta(delta)
                except BaseException as exc:
                    callback_error = exc
                    raise
                emitted = True

            try:
                self._record_model_route(
                    request,
                    provider=provider,
                    model=model,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                    effective_purpose=effective_purpose,
                )
                provider_request = replace(
                    request,
                    tool_stream=bool(request.tools and provider == "glm"),
                )
                response = _chat_openai_compatible_stream(
                    base_url=base_url,
                    api_key=lease.api_key,
                    model=model,
                    request=provider_request,
                    timeout=self.settings.request_timeout_seconds,
                    on_delta=tracked_delta,
                    cancelled=cancelled,
                )
            except BaseException as exc:
                if callback_error is exc or not isinstance(exc, Exception):
                    pool.release(
                        lease,
                        success=False,
                        latency_ms=_elapsed_ms(started),
                        error_code=(
                            "downstream_callback_error"
                            if callback_error is exc
                            else "interrupted"
                        ),
                        failure_kind=KeyFailureKind.CANCELLED,
                    )
                    raise
                last_error = exc
                failure = _classify_provider_error(exc)
                pool.release(
                    lease,
                    success=False,
                    latency_ms=_elapsed_ms(started),
                    error_code=failure.code,
                    failure_kind=failure.failure_kind,
                )
                if failure.failure_kind == KeyFailureKind.CANCELLED:
                    raise
                if emitted:
                    raise PartialStreamFailure(
                        f"provider stream failed after content was emitted: {failure.code}"
                    ) from exc
                if failure.retryable and attempt + 1 < attempts:
                    continue
                raise
            pool.release(
                lease,
                success=True,
                latency_ms=_elapsed_ms(started),
                input_tokens=_estimate_tokens(request.system) + _estimate_tokens(request.user),
                output_tokens=_estimate_tokens(response.answer),
            )
            return response
        if last_error is not None:
            raise last_error
        raise RLMQAError("streaming model call failed without provider response")

    def _pool_trace_sink(
        self,
        *,
        provider: str,
        model: str,
    ) -> Callable[[dict], None]:
        def record(event: dict) -> None:
            self._record_trace({**event, "provider": provider, "model": model})

        return record

    def _record_model_route(
        self,
        request: ChatRequest,
        *,
        provider: str,
        model: str,
        fallback_used: bool,
        fallback_reason: str,
        effective_purpose: str | None = None,
    ) -> None:
        self._record_trace(
            {
                "event": "model_route",
                "run_id": self._active_run_id(request),
                "purpose": effective_purpose or request.purpose or "base",
                "provider": provider,
                "model": model,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
            }
        )

    def _record_model_fallback(
        self,
        request: ChatRequest,
        failure: ProviderFailure,
    ) -> None:
        self._record_trace(
            {
                "event": "model_fallback",
                "run_id": self._active_run_id(request),
                "purpose": request.purpose or "base",
                "provider": "deepseek",
                "model": self.settings.deepseek_model,
                "fallback_used": True,
                "fallback_reason": failure.code,
            }
        )

    def _record_trace(self, event: dict) -> None:
        trace_event = dict(event)
        trace_event.pop("api_key", None)
        with self._trace_lock:
            self._trace.append(trace_event)

    @contextmanager
    def trace_run(self, run_id: str) -> Iterator[None]:
        previous = getattr(self._thread_state, "run_id", None)
        self._thread_state.run_id = run_id
        try:
            yield
        finally:
            self._thread_state.run_id = previous

    def key_pool_trace(self) -> list[dict]:
        with self._trace_lock:
            return [dict(event) for event in self._trace]

    def _active_run_id(self, request: ChatRequest | None = None) -> str:
        if request is not None and request.run_id:
            return request.run_id
        active_run_id = getattr(self._thread_state, "run_id", None)
        if active_run_id:
            return str(active_run_id)
        raise RLMQAError("model call run_id is not bound")

    def _bind_run_id(self, request: ChatRequest) -> ChatRequest:
        if request.run_id:
            return request
        active_run_id = getattr(self._thread_state, "run_id", None)
        if active_run_id:
            return replace(request, run_id=str(active_run_id))
        return replace(request, run_id=f"llm_{uuid4().hex}")

    def _next_task_id(self) -> str:
        with self._task_lock:
            self._task_counter += 1
            counter = self._task_counter
        return f"chat_{int(time.time() * 1000)}_{counter}"


def _chat_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    request: ChatRequest,
    timeout: float,
) -> ChatResponse:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ],
        "stream": False,
    }
    if request.tools:
        payload["tools"] = list(request.tools)
        payload["tool_choice"] = request.tool_choice or "auto"
    headers = {"Authorization": f"Bearer {api_key}"}
    url = base_url.rstrip("/") + "/chat/completions"
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
    data = response.json()
    content: object | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                content = message.get("content")
                tool_calls = _tool_calls_from_message(message.get("tool_calls"))
    answer = content if isinstance(content, str) and content.strip() else ""
    if not answer and not tool_calls:
        raise EmptyModelResponse("provider returned no usable assistant content")
    return ChatResponse(answer=answer, tool_calls=tool_calls)



def _chat_openai_compatible_stream(
    *,
    base_url: str,
    api_key: str,
    model: str,
    request: ChatRequest,
    timeout: float,
    on_delta: Callable[[str], None],
    cancelled: Callable[[], bool],
) -> ChatResponse:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ],
        "stream": True,
    }
    if request.tools:
        payload["tools"] = list(request.tools)
        payload["tool_choice"] = request.tool_choice or "auto"
        if request.tool_stream:
            payload["tool_stream"] = True
    headers = {"Authorization": f"Bearer {api_key}"}
    url = base_url.rstrip("/") + "/chat/completions"
    answer_parts: list[str] = []
    pending_leading_whitespace: list[str] = []
    has_usable_content = False
    tool_call_buffers: dict[int, dict[str, str]] = {}
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if cancelled():
                    raise LLMStreamCancelled("client cancelled")
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data_text = line[5:].strip()
                if data_text == "[DONE]":
                    break
                data = json.loads(data_text)
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                _merge_stream_tool_calls(tool_call_buffers, delta.get("tool_calls"))
                content = delta.get("content")
                if not isinstance(content, str) or not content:
                    continue
                if not has_usable_content and not content.strip():
                    pending_leading_whitespace.append(content)
                    continue
                if not has_usable_content:
                    for prefix in pending_leading_whitespace:
                        on_delta(prefix)
                        answer_parts.append(prefix)
                    pending_leading_whitespace.clear()
                on_delta(content)
                answer_parts.append(content)
                has_usable_content = True
    if cancelled():
        raise LLMStreamCancelled("client cancelled")
    tool_calls = _tool_calls_from_buffers(tool_call_buffers)
    if not has_usable_content and not tool_calls:
        raise EmptyModelResponse("provider returned no usable assistant content")
    return ChatResponse(answer="".join(answer_parts), tool_calls=tool_calls)


def _tool_calls_from_message(value: object) -> tuple[ToolCall, ...]:
    if not isinstance(value, list):
        return ()
    parsed: list[ToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        call_id = str(item.get("id") or "").strip()
        name = str(function.get("name") or "").strip()
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        if call_id and name and isinstance(arguments, str):
            parsed.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return tuple(parsed)


def _merge_stream_tool_calls(
    buffers: dict[int, dict[str, str]],
    value: object,
) -> None:
    if not isinstance(value, list):
        return
    for position, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index", position))
        except (TypeError, ValueError):
            continue
        buffer = buffers.setdefault(index, {"id": "", "name": "", "arguments": ""})
        call_id = item.get("id")
        if isinstance(call_id, str) and call_id:
            buffer["id"] = call_id
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            buffer["name"] = name
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        if isinstance(arguments, str):
            buffer["arguments"] += arguments


def _tool_calls_from_buffers(buffers: dict[int, dict[str, str]]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for index in sorted(buffers):
        buffer = buffers[index]
        if buffer["id"] and buffer["name"]:
            calls.append(
                ToolCall(
                    id=buffer["id"],
                    name=buffer["name"],
                    arguments=buffer["arguments"],
                )
            )
    return tuple(calls)

def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _classify_provider_error(exc: Exception) -> ProviderFailure:
    if isinstance(exc, LLMStreamCancelled):
        return ProviderFailure(
            code="client_cancelled",
            retryable=False,
            fallback_eligible=False,
            failure_kind=KeyFailureKind.CANCELLED,
        )
    if isinstance(exc, KeyPoolAcquireTimeout):
        return ProviderFailure(
            code="key_pool_acquire_timeout",
            retryable=False,
            fallback_eligible=True,
            failure_kind=KeyFailureKind.TRANSIENT,
        )
    if isinstance(exc, EmptyModelResponse):
        return ProviderFailure(
            code="empty_response",
            retryable=True,
            fallback_eligible=True,
            failure_kind=KeyFailureKind.TRANSIENT,
        )
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in {401, 403}:
            return ProviderFailure(
                code=str(status),
                retryable=True,
                fallback_eligible=True,
                failure_kind=KeyFailureKind.PERMANENT_KEY,
            )
        retryable = status in {408, 409, 425, 429} or 500 <= status < 600
        return ProviderFailure(
            code=str(status),
            retryable=retryable,
            fallback_eligible=retryable,
            failure_kind=(
                KeyFailureKind.RATE_LIMITED
                if status == 429
                else KeyFailureKind.TRANSIENT
            ),
        )
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return ProviderFailure(
            code="timeout",
            retryable=True,
            fallback_eligible=True,
            failure_kind=KeyFailureKind.TRANSIENT,
        )
    if isinstance(exc, httpx.RequestError):
        return ProviderFailure(
            code="network_error",
            retryable=True,
            fallback_eligible=True,
            failure_kind=KeyFailureKind.TRANSIENT,
        )
    return ProviderFailure(
        code="provider_error",
        retryable=False,
        fallback_eligible=False,
        failure_kind=KeyFailureKind.TRANSIENT,
    )
