from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .app_service import QAService


class ChatBody(BaseModel):
    source_id: str | None = None
    source_ids: list[str] | None = None
    session_id: str = "local_session"
    question: str
    stream: bool = False


def create_app(service: QAService | None = None) -> FastAPI:
    qa_service = service or QAService()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        qa_service.start_ocr_workers()
        try:
            yield
        finally:
            qa_service.stop_ocr_workers()

    app = FastAPI(title="FolioTrace — Local Document QA", lifespan=lifespan)
    static_dir = Path(__file__).with_name("static")
    if not static_dir.exists():
        static_dir = Path(__file__).resolve().parents[2] / "web" / "static"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    def index() -> FileResponse:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="web/static/index.html not found")
        return FileResponse(index_path)

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)) -> dict:
        data = await file.read()
        result = qa_service.upload_bytes(file.filename or "uploaded.bin", data)
        return asdict(result)

    @app.get("/documents/{source_id}")
    def document_status(source_id: str) -> dict:
        result = qa_service.get_document(source_id)
        if result is None:
            raise HTTPException(status_code=404, detail="document not found")
        return asdict(result)

    @app.post("/chat")
    def chat(body: ChatBody) -> dict:
        source_ids = body.source_ids or ([body.source_id] if body.source_id else [])
        if not source_ids:
            raise HTTPException(status_code=400, detail="source_id or source_ids is required")
        if len(source_ids) == 1:
            answer = qa_service.chat(source_ids[0], body.question, body.session_id)
        else:
            answer = qa_service.chat_many(source_ids, body.question, body.session_id)
        return answer.to_dict()

    return app


app = create_app()
