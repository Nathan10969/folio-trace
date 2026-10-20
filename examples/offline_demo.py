"""Run the actual local UI with a fixed response; no API calls or PDF OCR.

Upload sample-paper.md and ask about its synthetic adhesion measurements.
This fixture is for interface inspection, not model-quality evaluation.
"""
from pathlib import Path

import uvicorn

from rlm_qa.app_service import QAService
from rlm_qa.config import Settings
from rlm_qa.http_app import create_app
from rlm_qa.llm import FakeLLM


if __name__ == "__main__":
    settings = Settings(runtime_dir=Path("runtime/offline_demo"))
    service = QAService(
        settings=settings,
        llm=FakeLLM(
            answer="Formulation B retained 92% of its initial adhesion, compared with "
            "85% for A and 78% for C. Its binder-to-additive mass ratio was 8:1. "
            "B performed best among the three tested formulations in this synthetic "
            "document; the example does not establish a universal optimum."
        ),
    )
    print("FolioTrace offline fixture demo: fixed answer, no external model calls.")
    print("Upload examples/sample-paper.md; PDF OCR is not configured in this demo.")
    uvicorn.run(create_app(service), host="127.0.0.1", port=8787)
