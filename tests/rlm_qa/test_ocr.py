from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from rlm_qa.config import Settings
from rlm_qa.ocr import GLMOCRClient, OCRProviderError


class FakeHTTPClient:
    def __init__(self, response: dict):
        self.response = response
        self.requests: list[dict] = []

    def __enter__(self) -> "FakeHTTPClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, *, json: dict, headers: dict) -> httpx.Response:
        self.requests.append({"url": url, "json": json, "headers": headers})
        return httpx.Response(200, json=self.response, request=httpx.Request("POST", url))


def test_glm_ocr_client_calls_layout_parsing_and_maps_layout_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"pdf bytes")
    fake_http = FakeHTTPClient(
        {
            "id": "task_1",
            "model": "GLM-OCR",
            "layout_details": [
                [{"index": 1, "label": "text", "content": "First OCR page"}],
                [
                    {"index": 1, "label": "title", "content": "Second page title"},
                    {"index": 2, "label": "text", "content": "Second OCR page"},
                ],
            ],
            "data_info": {"num_pages": 2},
        }
    )
    settings = Settings(
        glm_ocr_api_key="ocr-key",
        glm_base_url="https://glm.example.invalid/api/paas/v4",
    )

    pages = GLMOCRClient(settings, client_factory=lambda **_: fake_http).extract_pdf_pages(
        pdf,
        "src_test",
        api_key="leased-ocr-key",
    )

    assert [page.page_number for page in pages] == [1, 2]
    assert pages[0].text == "First OCR page"
    assert pages[1].text == "Second page title\nSecond OCR page"
    assert fake_http.requests[0]["url"] == "https://glm.example.invalid/api/paas/v4/layout_parsing"
    assert fake_http.requests[0]["headers"] == {"Authorization": "Bearer leased-ocr-key"}
    assert fake_http.requests[0]["json"]["model"] == "glm-ocr"
    assert fake_http.requests[0]["json"]["file"].startswith("data:application/pdf;base64,")


def test_glm_ocr_client_exposes_provider_http_status_for_scheduler(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"pdf bytes")
    fake_http = FakeHTTPClient({"error": {"message": "too many requests"}})

    class RateLimitedClient(FakeHTTPClient):
        def post(self, url: str, *, json: dict, headers: dict) -> httpx.Response:
            self.requests.append({"url": url, "json": json, "headers": headers})
            return httpx.Response(429, json=self.response, request=httpx.Request("POST", url))

    client = RateLimitedClient(fake_http.response)
    settings = Settings(glm_base_url="https://glm.example.invalid/api/paas/v4")

    with pytest.raises(OCRProviderError) as raised:
        GLMOCRClient(settings, client_factory=lambda **_: client).extract_pdf_pages(
            pdf,
            "src_test",
            api_key="leased-ocr-key",
        )

    assert raised.value.status_code == 429
    assert raised.value.error_code == "http_429"
