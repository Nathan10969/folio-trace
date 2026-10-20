from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import Settings
from .contracts import NoTextError, Page, RLMQAError


class OCRProviderError(RLMQAError):
    def __init__(self, message: str, *, status_code: int | None, error_code: str):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class GLMOCRClient:
    def __init__(
        self,
        settings: Settings,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
    ):
        self.settings = settings
        self.client_factory = client_factory

    def extract_pdf_pages(self, path: Path, source_id: str, *, api_key: str) -> list[Page]:
        if not api_key:
            raise RLMQAError("GLM OCR attempt has no assigned API key")

        encoded_pdf = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "model": self.settings.glm_ocr_model,
            "file": f"data:application/pdf;base64,{encoded_pdf}",
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        url = self.settings.glm_base_url.rstrip("/") + "/layout_parsing"

        try:
            with self.client_factory(timeout=self.settings.glm_ocr_request_timeout_seconds) as client:
                response = client.post(url, json=payload, headers=headers)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    raise OCRProviderError(
                        f"GLM OCR provider returned HTTP {status_code}",
                        status_code=status_code,
                        error_code=f"http_{status_code}",
                    ) from exc
        except httpx.TimeoutException as exc:
            raise OCRProviderError(
                "GLM OCR provider request timed out",
                status_code=None,
                error_code="network_timeout",
            ) from exc
        except httpx.HTTPError as exc:
            raise OCRProviderError(
                "GLM OCR provider transport failed",
                status_code=None,
                error_code="network_error",
            ) from exc

        try:
            response_data = response.json()
        except ValueError as exc:
            raise OCRProviderError(
                "GLM OCR provider returned invalid JSON",
                status_code=None,
                error_code="invalid_response",
            ) from exc
        return pages_from_glm_ocr_response(response_data, source_id)


def pages_from_glm_ocr_response(data: dict[str, Any], source_id: str) -> list[Page]:
    segments = _segments_from_layout_details(data.get("layout_details"))
    if not segments:
        segments = _segments_from_markdown(data.get("md_results"))
    if not segments:
        raise NoTextError("GLM OCR returned no readable text")

    pages: list[Page] = []
    cursor = 0
    for index, segment in enumerate(segments, start=1):
        text = segment.strip()
        if not text:
            cursor += len(segment)
            continue
        start = cursor
        end = cursor + len(segment)
        pages.append(Page(source_id=source_id, page_number=index, text=text, char_start=start, char_end=end))
        cursor = end
    if not pages:
        raise NoTextError("GLM OCR returned no readable text")
    return pages


def _segments_from_layout_details(layout_details: Any) -> list[str]:
    if not isinstance(layout_details, list):
        return []

    segments: list[str] = []
    for page in layout_details:
        if not isinstance(page, list):
            continue
        parts: list[str] = []
        for block in page:
            if not isinstance(block, dict):
                continue
            content = block.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        if parts:
            segments.append("\n".join(parts))
    return segments


def _segments_from_markdown(md_results: Any) -> list[str]:
    if isinstance(md_results, str) and md_results.strip():
        return [md_results.strip()]
    if isinstance(md_results, list):
        return [str(item).strip() for item in md_results if str(item).strip()]
    return []
