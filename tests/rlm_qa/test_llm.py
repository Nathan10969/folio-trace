from __future__ import annotations

import json

import httpx
import pytest

import rlm_qa.llm as llm_module

from rlm_qa.config import Settings
from rlm_qa.key_pool import KeyFailureKind, KeyPoolAcquireTimeout
from rlm_qa.llm import ChatRequest, ModelRouterLLM


class DummyResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "ok"}}]}


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": " \t\n"}}]},
    ],
    ids=("empty", "missing", "whitespace"),
)
def test_openai_compatible_rejects_unusable_assistant_content(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict,
) -> None:
    class UnusableResponse(DummyResponse):
        def json(self) -> dict:
            return payload

    class UnusableClient:
        def __init__(self, timeout: float):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict, headers: dict):
            return UnusableResponse()

    monkeypatch.setattr(httpx, "Client", UnusableClient)

    with pytest.raises(llm_module.EmptyModelResponse):
        llm_module._chat_openai_compatible(
            base_url="https://provider.example",
            api_key="key",
            model="model",
            request=ChatRequest(system="s", user="u"),
            timeout=30,
        )


def test_openai_compatible_sends_and_parses_tool_call_contract(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class ToolResponse(DummyResponse):
        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_doc_1",
                                    "type": "function",
                                    "function": {
                                        "name": "inspect_documents",
                                        "arguments": '{"query":"总结所选文档"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

    class ToolClient:
        def __init__(self, timeout: float):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict, headers: dict):
            captured.update(url=url, json=json, headers=headers)
            return ToolResponse()

    monkeypatch.setattr(httpx, "Client", ToolClient)
    tool = {
        "type": "function",
        "function": {
            "name": "inspect_documents",
            "description": "Read selected documents only when needed.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    response = llm_module._chat_openai_compatible(
        base_url="https://provider.example",
        api_key="key",
        model="model",
        request=ChatRequest(
            system="s",
            user="u",
            tools=(tool,),
            tool_choice="auto",
        ),
        timeout=30,
    )

    assert response.answer == ""
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_doc_1"
    assert response.tool_calls[0].name == "inspect_documents"
    assert response.tool_calls[0].arguments == '{"query":"总结所选文档"}'
    assert captured["json"]["tools"] == [tool]
    assert captured["json"]["tool_choice"] == "auto"


def test_model_router_sends_deepseek_subagent_requests_to_deepseek_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyClient:
        def __init__(self, timeout: float):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict, headers: dict):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return DummyResponse()

    monkeypatch.setattr(httpx, "Client", DummyClient)
    settings = Settings(
        deepseek_api_key="deepseek-key",
        deepseek_base_url="https://deepseek.example.invalid",
        deepseek_model="deepseek-v4-flash",
        deepseek_subagent_model="deepseek-v4-flash",
        deepseek_subagent_api_keys=("deepseek-subagent-key",),
    )

    answer = ModelRouterLLM(settings).chat(ChatRequest(system="sys", user="user", model="deepseek-v4-flash"))

    assert answer.answer == "ok"
    assert captured["url"] == "https://deepseek.example.invalid/chat/completions"
    assert captured["json"] == {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ],
        "stream": False,
    }
    assert captured["headers"] == {"Authorization": "Bearer deepseek-subagent-key"}

class DummyStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.closed = True

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        yield from self.lines


def _install_stream_client(monkeypatch, response: DummyStreamResponse, captured: dict) -> None:
    class DummyClient:
        def __init__(self, timeout: float):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def stream(self, method: str, url: str, json: dict, headers: dict):
            captured.update(method=method, url=url, json=json, headers=headers)
            return response

    monkeypatch.setattr(httpx, "Client", DummyClient)


def test_deepseek_stream_forwards_non_empty_deltas_and_accumulates(monkeypatch) -> None:
    captured: dict[str, object] = {}
    provider = DummyStreamResponse(
        [
            'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
            'data: {"choices":[{"delta":{"content":"single"}}]}',
            ': keep-alive',
            'data: {"choices":[{"delta":{"content":" wall nanotube"}}]}',
            'data: {"choices":[],"usage":{"output_tokens":3}}',
            'data: [DONE]',
        ]
    )
    _install_stream_client(monkeypatch, provider, captured)
    deltas: list[str] = []

    result = llm_module._chat_openai_compatible_stream(
        base_url="https://ds.example",
        api_key="key",
        model="deepseek-v4-flash",
        request=ChatRequest(system="s", user="u"),
        timeout=30,
        on_delta=deltas.append,
        cancelled=lambda: False,
    )

    assert deltas == ["single", " wall nanotube"]
    assert result.answer == "single wall nanotube"
    assert captured["json"]["stream"] is True
    assert captured["url"] == "https://ds.example/chat/completions"
    assert provider.closed is True


def test_openai_compatible_stream_accumulates_split_tool_call(monkeypatch) -> None:
    captured: dict[str, object] = {}
    provider = DummyStreamResponse(
        [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_doc_2",
                                        "type": "function",
                                        "function": {
                                            "name": "inspect_documents",
                                            "arguments": '{"query":"总结',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "arguments": '文档","scope":{"mode":"all_selected"}}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "data: [DONE]",
        ]
    )
    _install_stream_client(monkeypatch, provider, captured)
    deltas: list[str] = []
    tool = {
        "type": "function",
        "function": {
            "name": "inspect_documents",
            "description": "Read selected documents only when needed.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    response = llm_module._chat_openai_compatible_stream(
        base_url="https://provider.example",
        api_key="key",
        model="model",
        request=ChatRequest(
            system="s",
            user="u",
            tools=(tool,),
            tool_choice="auto",
            tool_stream=True,
        ),
        timeout=30,
        on_delta=deltas.append,
        cancelled=lambda: False,
    )

    assert deltas == []
    assert response.answer == ""
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_doc_2"
    assert response.tool_calls[0].name == "inspect_documents"
    assert json.loads(response.tool_calls[0].arguments) == {
        "query": "总结文档",
        "scope": {"mode": "all_selected"},
    }
    assert captured["json"]["tools"] == [tool]
    assert captured["json"]["tool_choice"] == "auto"
    assert captured["json"]["tool_stream"] is True


def test_model_router_streams_orchestrator_tool_calls_via_glm(monkeypatch) -> None:
    captured: dict[str, object] = {}
    provider = DummyStreamResponse(
        [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_doc",'
            '"type":"function","function":{"name":"inspect_documents",'
            '"arguments":"{\\"query\\":\\"summarize\\"}"}}]}}]}',
            "data: [DONE]",
        ]
    )
    _install_stream_client(monkeypatch, provider, captured)
    settings = Settings(
        glm_generation_api_keys=("glm-generation-key",),
        glm_base_url="https://glm.example.invalid",
        glm_generation_model="glm-5.2",
        deepseek_api_key="deepseek-fallback-key",
    )
    router = ModelRouterLLM(settings)
    tool = {
        "type": "function",
        "function": {
            "name": "inspect_documents",
            "description": "Read selected documents only when needed.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    response = router.chat_stream(
        ChatRequest(
            system="s",
            user="u",
            purpose="orchestrator",
            tools=(tool,),
            tool_choice="auto",
        ),
        lambda _delta: None,
        lambda: False,
    )

    assert router.supports_tool_calls is True
    assert response.tool_calls[0].name == "inspect_documents"
    assert captured["url"] == "https://glm.example.invalid/chat/completions"
    assert captured["json"]["tools"] == [tool]
    assert captured["json"]["tool_stream"] is True


def test_deepseek_stream_stops_and_closes_when_cancelled(monkeypatch) -> None:
    captured: dict[str, object] = {}
    provider = DummyStreamResponse(
        [
            'data: {"choices":[{"delta":{"content":"first"}}]}',
            'data: {"choices":[{"delta":{"content":"second"}}]}',
            'data: [DONE]',
        ]
    )
    _install_stream_client(monkeypatch, provider, captured)
    deltas: list[str] = []

    with pytest.raises(llm_module.LLMStreamCancelled):
        llm_module._chat_openai_compatible_stream(
            base_url="https://ds.example",
            api_key="key",
            model="deepseek-v4-flash",
            request=ChatRequest(system="s", user="u"),
            timeout=30,
            on_delta=deltas.append,
            cancelled=lambda: bool(deltas),
        )

    assert deltas == ["first"]
    assert provider.closed is True


def test_openai_compatible_stream_rejects_stream_without_usable_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    provider = DummyStreamResponse(
        [
            'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
            'data: {"choices":[{"delta":{}}]}',
            'data: {"choices":[{"delta":{"content":"   "}}]}',
            'data: {"choices":[]}',
            "data: [DONE]",
        ]
    )
    _install_stream_client(monkeypatch, provider, captured)
    deltas: list[str] = []

    with pytest.raises(llm_module.EmptyModelResponse):
        llm_module._chat_openai_compatible_stream(
            base_url="https://provider.example",
            api_key="key",
            model="model",
            request=ChatRequest(system="s", user="u"),
            timeout=30,
            on_delta=deltas.append,
            cancelled=lambda: False,
        )

    assert deltas == []
    assert provider.closed is True


@pytest.mark.parametrize(
    ("status", "retryable", "fallback_eligible", "failure_kind"),
    [
        (400, False, False, KeyFailureKind.TRANSIENT),
        (401, True, True, KeyFailureKind.PERMANENT_KEY),
        (403, True, True, KeyFailureKind.PERMANENT_KEY),
        (404, False, False, KeyFailureKind.TRANSIENT),
        (408, True, True, KeyFailureKind.TRANSIENT),
        (409, True, True, KeyFailureKind.TRANSIENT),
        (422, False, False, KeyFailureKind.TRANSIENT),
        (425, True, True, KeyFailureKind.TRANSIENT),
        (429, True, True, KeyFailureKind.RATE_LIMITED),
        (500, True, True, KeyFailureKind.TRANSIENT),
        (503, True, True, KeyFailureKind.TRANSIENT),
    ],
)
def test_provider_failure_classifier_handles_http_statuses(
    status: int,
    retryable: bool,
    fallback_eligible: bool,
    failure_kind: KeyFailureKind,
) -> None:
    response = httpx.Response(
        status,
        request=httpx.Request("POST", "https://provider.example/chat/completions"),
    )
    error = httpx.HTTPStatusError(
        "provider status",
        request=response.request,
        response=response,
    )

    failure = llm_module._classify_provider_error(error)

    assert failure.code == str(status)
    assert failure.retryable is retryable
    assert failure.fallback_eligible is fallback_eligible
    assert failure.failure_kind == failure_kind


@pytest.mark.parametrize(
    ("error", "retryable", "fallback_eligible", "failure_kind"),
    [
        (
            llm_module.LLMStreamCancelled("cancelled"),
            False,
            False,
            KeyFailureKind.CANCELLED,
        ),
        (
            KeyPoolAcquireTimeout("pool busy"),
            False,
            True,
            KeyFailureKind.TRANSIENT,
        ),
        (
            httpx.ReadTimeout(
                "timed out",
                request=httpx.Request("POST", "https://provider.example"),
            ),
            True,
            True,
            KeyFailureKind.TRANSIENT,
        ),
        (
            httpx.ConnectError(
                "network failed",
                request=httpx.Request("POST", "https://provider.example"),
            ),
            True,
            True,
            KeyFailureKind.TRANSIENT,
        ),
        (TimeoutError("timed out"), True, True, KeyFailureKind.TRANSIENT),
        (ValueError("programming error"), False, False, KeyFailureKind.TRANSIENT),
    ],
)
def test_provider_failure_classifier_handles_non_http_failures(
    error: Exception,
    retryable: bool,
    fallback_eligible: bool,
    failure_kind: KeyFailureKind,
) -> None:
    failure = llm_module._classify_provider_error(error)

    assert failure.retryable is retryable
    assert failure.fallback_eligible is fallback_eligible
    assert failure.failure_kind == failure_kind


def test_provider_failure_classifier_retries_empty_model_response() -> None:
    failure = llm_module._classify_provider_error(
        llm_module.EmptyModelResponse("empty response")
    )

    assert failure.code == "empty_response"
    assert failure.retryable is True
    assert failure.fallback_eligible is True
    assert failure.failure_kind == KeyFailureKind.TRANSIENT
