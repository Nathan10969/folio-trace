from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .chunking import build_chunks, build_evidence, build_profile, no_text_profile
from .config import Settings
from .contracts import (
    AnswerContract,
    AnswerDebug,
    Chunk,
    DocProfile,
    Evidence,
    EvidenceRef,
    NoTextError,
    Page,
    RLMQAError,
    UploadResult,
)
from .decision import decide_mode
from .document_adapter import DocumentRLMAnswer
from .document_map_reduce import DocumentMapReduceRunner
from .evidence_quotes import normalize_evidence_refs, select_display_evidence_refs
from .ids import new_id
from .ingest import PDFOCRClient, extract_pages
from .key_pool import KeyPool, KeyPoolConfig
from .llm import ChatRequest, LLMClient, ModelRouterLLM
from .ocr import GLMOCRClient
from .ocr_jobs import OCRJob, OCRJobRunner
from .storage import FileStore
from .tools import search_context


class QAService:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        pdf_ocr: PDFOCRClient | None = None,
    ):
        self.settings = settings or Settings.from_env()
        self.store = FileStore(self.settings.runtime_dir)
        self.store.init()
        self.llm = llm or ModelRouterLLM(self.settings)
        self.pdf_ocr = pdf_ocr or GLMOCRClient(self.settings)
        self._ocr_recovery_completed = False
        self.ocr_runner = self._build_ocr_runner()

    def upload_file(self, path: Path) -> UploadResult:
        data = path.read_bytes()
        return self.upload_bytes(path.name, data)

    def upload_bytes(self, filename: str, data: bytes) -> UploadResult:
        local_doc_id = new_id("local")
        source_id = new_id("src")
        original_path = self.store.save_upload_bytes(filename, data, local_doc_id)
        doc_dir = self.store.doc_dir(source_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        manifest = {"local_doc_id": local_doc_id, "source_id": source_id, "file_name": filename}
        if original_path.suffix.lower() == ".pdf":
            return self._queue_pdf_upload(local_doc_id, source_id, filename, manifest)

        try:
            pages = extract_pages(original_path, source_id, self.settings, self.pdf_ocr)
            chunks = build_chunks(pages, self.settings)
            chunks, evidences = build_evidence(chunks)
            profile = build_profile(filename, source_id, pages, chunks, evidences, self.settings)
            self.store.write_jsonl(doc_dir / "pages.jsonl", pages)
            self.store.write_jsonl(doc_dir / "chunks.jsonl", chunks)
            self.store.write_jsonl(doc_dir / "evidence.jsonl", evidences)
        except NoTextError:
            if original_path.suffix.lower() == ".pdf":
                profile = no_text_profile(filename, source_id, status="ocr_failed", error_code="GLM_OCR_FAILED")
            else:
                profile = no_text_profile(filename, source_id)
            self.store.write_jsonl(doc_dir / "pages.jsonl", [])
            self.store.write_jsonl(doc_dir / "chunks.jsonl", [])
            self.store.write_jsonl(doc_dir / "evidence.jsonl", [])

        self.store.write_json(doc_dir / "doc_profile.json", profile)
        self.store.write_json(doc_dir / "manifest.json", manifest)
        self.store.write_json(self.store.upload_dir(local_doc_id) / "upload_meta.json", manifest)
        return UploadResult(
            local_doc_id=local_doc_id,
            source_id=source_id,
            status=profile.status,
            doc_profile=profile,
            manifest=manifest,
        )

    def get_document(self, source_id: str) -> UploadResult | None:
        profile = self._load_profile(source_id)
        manifest_path = self.store.doc_dir(source_id) / "manifest.json"
        if profile is None or not manifest_path.exists():
            return None
        manifest = self.store.read_json(manifest_path)
        return UploadResult(
            local_doc_id=str(manifest.get("local_doc_id") or ""),
            source_id=source_id,
            status=profile.status,
            doc_profile=profile,
            manifest=manifest,
        )

    def start_ocr_workers(self) -> int:
        if self.ocr_runner is None:
            return 0
        jobs: list[OCRJob] = []
        for doc_dir in self.store.iter_doc_dirs():
            job_path = doc_dir / "ocr_job.json"
            if job_path.exists():
                jobs.append(OCRJob.from_dict(self.store.read_json(job_path)))
        recovered = self.ocr_runner.recover_pending_jobs(
            jobs,
            stale_after_seconds=self.settings.glm_ocr_lease_timeout_seconds,
            recover_running=not self._ocr_recovery_completed,
        )
        self._ocr_recovery_completed = True
        self.ocr_runner.start()
        return recovered

    def stop_ocr_workers(self) -> None:
        if self.ocr_runner is not None:
            self.ocr_runner.stop()

    def process_next_ocr_job_for_test(self) -> bool:
        if self.ocr_runner is None:
            return False
        return self.ocr_runner.process_next_for_test()

    def _build_ocr_runner(self) -> OCRJobRunner | None:
        api_keys = self.settings.glm_ocr_api_keys
        if not api_keys and self.settings.glm_ocr_api_key:
            api_keys = (self.settings.glm_ocr_api_key,)
        if not api_keys:
            return None
        key_pool = KeyPool(
            api_keys,
            role="ocr",
            config=KeyPoolConfig(
                max_global_concurrency=self.settings.glm_ocr_max_workers,
                max_per_key_concurrency=1,
                max_per_run_concurrency=1,
                cooldown_seconds=self.settings.glm_ocr_transient_cooldown_seconds,
                rate_limit_cooldown_seconds=self.settings.glm_ocr_rate_limit_cooldown_seconds,
                max_failures_before_dead=self.settings.rlm_key_max_failures_before_dead,
                lease_timeout_seconds=self.settings.glm_ocr_lease_timeout_seconds,
            ),
        )
        self._restore_ocr_key_health(key_pool)
        return OCRJobRunner(
            key_pool=key_pool,
            max_workers=self.settings.glm_ocr_max_workers,
            max_attempts=self.settings.glm_ocr_max_attempts,
            execute=self._execute_ocr_job,
            save_job=self._save_ocr_job,
            complete=self._complete_ocr_job,
            fail=self._fail_ocr_job,
            key_state_changed=self._persist_ocr_key_health,
        )

    def _restore_ocr_key_health(self, key_pool: KeyPool) -> None:
        health_path = self.store.ocr_key_health_path()
        if not health_path.exists():
            return
        health = self.store.read_json(health_path)
        key_pool.mark_dead_key_ids(health.get("dead_key_ids") or [])

    def _persist_ocr_key_health(self, key_pool: KeyPool) -> None:
        dead_key_ids = [snapshot.key_id for snapshot in key_pool.snapshot() if snapshot.status.value == "dead"]
        self.store.write_json(self.store.ocr_key_health_path(), {"dead_key_ids": dead_key_ids})

    def _queue_pdf_upload(
        self,
        local_doc_id: str,
        source_id: str,
        filename: str,
        manifest: dict[str, Any],
    ) -> UploadResult:
        job = OCRJob.create(local_doc_id=local_doc_id, source_id=source_id, file_name=filename)
        profile = no_text_profile(filename, source_id, status="ocr_queued", error_code="OCR_PENDING")
        self._write_empty_document_artifacts(source_id)
        self.store.write_json(self.store.doc_dir(source_id) / "doc_profile.json", profile)
        self.store.write_json(self.store.doc_dir(source_id) / "manifest.json", manifest)
        self.store.write_json(self.store.upload_dir(local_doc_id) / "upload_meta.json", manifest)

        if self.ocr_runner is None:
            job.state = "ocr_failed"
            job.last_error_code = "GLM_OCR_KEYS_NOT_CONFIGURED"
            self._save_ocr_job(job)
            self._fail_ocr_job(job, job.last_error_code)
            profile = self._load_profile(source_id) or profile
        else:
            self._save_ocr_job(job)
            self.ocr_runner.enqueue(job)
            profile = self._load_profile(source_id) or profile

        return UploadResult(
            local_doc_id=local_doc_id,
            source_id=source_id,
            status=profile.status,
            doc_profile=profile,
            manifest=manifest,
        )

    def _execute_ocr_job(self, job: OCRJob, api_key: str) -> list[Page]:
        path = self.store.original_upload_path(job.local_doc_id)
        return self.pdf_ocr.extract_pdf_pages(path, job.source_id, api_key=api_key)

    def _complete_ocr_job(self, job: OCRJob, pages: list[Page]) -> None:
        chunks = build_chunks(pages, self.settings)
        chunks, evidences = build_evidence(chunks)
        profile = build_profile(job.file_name, job.source_id, pages, chunks, evidences, self.settings)
        doc_dir = self.store.doc_dir(job.source_id)
        self.store.write_jsonl(doc_dir / "pages.jsonl", pages)
        self.store.write_jsonl(doc_dir / "chunks.jsonl", chunks)
        self.store.write_jsonl(doc_dir / "evidence.jsonl", evidences)
        self.store.write_json(doc_dir / "doc_profile.json", profile)

    def _fail_ocr_job(self, job: OCRJob, error_code: str) -> None:
        profile = no_text_profile(job.file_name, job.source_id, status="ocr_failed", error_code=error_code)
        self._write_empty_document_artifacts(job.source_id)
        self.store.write_json(self.store.doc_dir(job.source_id) / "doc_profile.json", profile)

    def _save_ocr_job(self, job: OCRJob) -> None:
        self.store.write_json(self.store.doc_dir(job.source_id) / "ocr_job.json", job)
        if job.state not in {"ocr_queued", "ocr_running"}:
            return
        profile = self._load_profile(job.source_id)
        if profile is None:
            return
        error_code = job.last_error_code or "OCR_PENDING"
        pending_profile = replace(profile, status=job.state, error_code=error_code)
        self.store.write_json(self.store.doc_dir(job.source_id) / "doc_profile.json", pending_profile)

    def _write_empty_document_artifacts(self, source_id: str) -> None:
        doc_dir = self.store.doc_dir(source_id)
        self.store.write_jsonl(doc_dir / "pages.jsonl", [])
        self.store.write_jsonl(doc_dir / "chunks.jsonl", [])
        self.store.write_jsonl(doc_dir / "evidence.jsonl", [])

    def chat(self, source_id: str, question: str, session_id: str = "local_session") -> AnswerContract:
        run_id = new_id("run")
        profile = self._load_profile(source_id)
        if profile is None:
            return self._base_fallback(run_id, question, "doc_not_found")

        mode = decide_mode(profile, question, self.settings)
        if mode == "base_fallback":
            return self._base_fallback(run_id, question, profile.error_code or "doc_not_ready")

        chunks = self._load_chunks(source_id)
        evidences = self._load_evidence(source_id)
        if mode == "rlm":
            pages = self._load_pages(source_id)
            try:
                document_answer = self._run_document_rlm(run_id, question, [profile], pages, chunks, evidences)
            except RLMQAError as exc:
                return self._status_answer(run_id, str(exc), reason="rlm_runtime_failed")
            answer = self._document_answer_contract(run_id, mode, document_answer, question=question, profiles=[profile])
            self._write_run(run_id, source_id, session_id, question, answer)
            return answer

        hits = search_context(chunks, question)
        if not hits and not _is_direct_context_question(question):
            return self._base_fallback(run_id, question, "no_verified_evidence")

        refs = _all_evidence_refs(evidences, profile)
        if not refs:
            return self._base_fallback(run_id, question, "no_verified_evidence")

        pages = self._load_pages(source_id)
        context = _direct_page_context(profile, pages, evidences)
        system = (
            "Answer the user in Chinese unless the user asks otherwise. "
            "Use the complete page-tagged document context for document claims. "
            "Do not invent PDF pages, titles, or facts."
        )
        user = f"Question: {question}\n\nDocument context:\n{context}"
        try:
            answer_text = self.llm.chat(ChatRequest(system=system, user=user)).answer
        except RLMQAError as exc:
            return self._status_answer(run_id, str(exc))

        refs = normalize_evidence_refs(refs, chunks, pages, question=question, answer=answer_text)
        display_refs, evidence_display = select_display_evidence_refs(
            refs,
            question=question,
            profiles=[profile],
            settings=self.settings,
        )
        answer = AnswerContract(
            run_id=run_id,
            mode=mode,
            answer_type="document_grounded",
            answer=answer_text,
            evidences=display_refs,
            citation_policy="pdf_page_citation",
            debug=AnswerDebug(
                run_id=run_id,
                program_steps=["direct_context"],
                trajectory={"evidence_display": evidence_display},
            ),
        )
        self._write_run(run_id, source_id, session_id, question, answer)
        return answer

    def chat_many(self, source_ids: list[str], question: str, session_id: str = "local_session") -> AnswerContract:
        if len(source_ids) == 1:
            return self.chat(source_ids[0], question, session_id)

        run_id = new_id("run")
        profiles: list[DocProfile] = []
        for source_id in source_ids:
            profile = self._load_profile(source_id)
            if profile is None:
                return self._base_fallback(run_id, question, "doc_not_found")
            profiles.append(profile)

        source_key = ",".join(source_ids)
        ready_profiles = [profile for profile in profiles if profile.rlm_ready or profile.direct_ready]
        skipped_profiles = [profile for profile in profiles if not (profile.rlm_ready or profile.direct_ready)]
        if not ready_profiles:
            return self._status_answer(
                run_id,
                _all_documents_not_ready_message(skipped_profiles),
                reason="doc_not_ready",
            )

        pages: list[Page] = []
        chunks: list[Chunk] = []
        evidences: list[Evidence] = []
        for profile in ready_profiles:
            pages.extend(self._load_pages(profile.source_id))
            chunks.extend(self._load_chunks(profile.source_id))
            evidences.extend(self._load_evidence(profile.source_id))

        try:
            document_answer = self._run_document_rlm(run_id, question, ready_profiles, pages, chunks, evidences)
        except RLMQAError as exc:
            return self._status_answer(run_id, str(exc), reason="rlm_runtime_failed")

        answer = self._document_answer_contract(run_id, "rlm", document_answer, question=question, profiles=ready_profiles)
        answer = _append_skipped_document_notice(answer, skipped_profiles)
        self._write_run(run_id, source_key, session_id, question, answer)
        return answer

    def _run_document_rlm(
        self,
        run_id: str,
        question: str,
        profiles: list[DocProfile],
        pages: list[Page],
        chunks: list[Chunk],
        evidences: list[Evidence],
    ) -> DocumentRLMAnswer:
        trace_marker = _key_pool_trace_marker(self.llm)
        with _llm_trace_context(self.llm, run_id):
            document_answer = DocumentMapReduceRunner(
                llm=self.llm,
                subagent_model=self.settings.deepseek_subagent_model,
                max_workers=self.settings.rlm_batch_max_workers,
                max_deep_packs=self.settings.rlm_max_deep_packs,
                map_execution_mode=self.settings.rlm_map_execution_mode,
                reduce_context_mode=self.settings.rlm_reduce_context_mode,
                fallback_evidence_per_pack=self.settings.rlm_fallback_evidence_per_pack,
                run_id=run_id,
                max_timeout_seconds=self.settings.rlm_max_timeout_seconds,
                max_tokens=self.settings.rlm_max_tokens,
                max_errors=self.settings.rlm_max_errors,
                max_budget=self.settings.rlm_max_budget,
                cost_per_1k_tokens=self.settings.rlm_cost_per_1k_tokens,
            ).answer_question_many(question, profiles, pages, chunks, evidences)
        refs = normalize_evidence_refs(
            document_answer.evidences,
            chunks,
            pages,
            question=question,
            answer=document_answer.answer,
        )
        trajectory = dict(document_answer.trajectory or {})
        trajectory["key_pool_trace"] = _key_pool_trace_since(self.llm, trace_marker)
        return replace(document_answer, evidences=refs, trajectory=trajectory)

    def _document_answer_contract(
        self,
        run_id: str,
        mode: str,
        document_answer: DocumentRLMAnswer,
        *,
        question: str = "",
        profiles: list[DocProfile] | None = None,
    ) -> AnswerContract:
        metadata_only = bool(document_answer.metadata_refs) and not document_answer.evidences
        profiles = profiles or []
        display_evidences, evidence_display = select_display_evidence_refs(
            document_answer.evidences,
            question=question,
            profiles=profiles,
            settings=self.settings,
        )
        trajectory = dict(document_answer.trajectory or {})
        trajectory["evidence_display"] = evidence_display
        program_step = "rlm_map_reduce" if trajectory.get("strategy") == "map_reduce" else "rlm_agentic"
        if metadata_only:
            answer_type = "metadata_answer"
            citation_policy = "document_metadata"
        elif not document_answer.evidences:
            answer_type = "document_assisted"
            citation_policy = "no_pdf_citation"
        else:
            answer_type = (
                "document_grounded"
                if document_answer.runtime_status == "rlm_success"
                else "document_recovered"
            )
            citation_policy = "pdf_page_citation"

        return AnswerContract(
            run_id=run_id,
            mode=mode,
            answer_type=answer_type,
            answer=document_answer.answer,
            evidences=display_evidences,
            citation_policy=citation_policy,
            debug=AnswerDebug(
                run_id=run_id,
                program_steps=[program_step],
                trajectory=trajectory,
                runtime_status=document_answer.runtime_status,
                metadata_refs=document_answer.metadata_refs,
            ),
        )

    def _base_fallback(self, run_id: str, question: str, reason: str) -> AnswerContract:
        system = "Answer in Chinese from general knowledge. Do not cite uploaded PDF pages."
        try:
            answer_text = self.llm.chat(ChatRequest(system=system, user=question)).answer
        except RLMQAError:
            answer_text = "The local base model is unavailable, and no verified document answer could be produced."
        answer = AnswerContract(
            run_id=run_id,
            mode="base_fallback",
            answer_type="base_fallback",
            answer=answer_text,
            evidences=[],
            citation_policy="no_pdf_citation",
            fallback_reason=reason,
            debug=AnswerDebug(run_id=run_id, fallback_reason=reason),
        )
        self.store.write_json(self.store.run_path(run_id), {"question": question, "answer": answer})
        return answer

    def _status_answer(self, run_id: str, message: str, *, reason: str = "llm_unavailable") -> AnswerContract:
        answer = AnswerContract(
            run_id=run_id,
            mode="base_fallback",
            answer_type="status_answer",
            answer=message,
            evidences=[],
            citation_policy="no_pdf_citation",
            fallback_reason=reason,
            debug=AnswerDebug(run_id=run_id, fallback_reason=reason),
        )
        self.store.write_json(self.store.run_path(run_id), {"answer": answer})
        return answer

    def _load_profile(self, source_id: str) -> DocProfile | None:
        path = self.store.doc_dir(source_id) / "doc_profile.json"
        if not path.exists():
            return None
        return DocProfile(**self.store.read_json(path))

    def _load_pages(self, source_id: str) -> list[Page]:
        rows = self.store.read_jsonl(self.store.doc_dir(source_id) / "pages.jsonl")
        return [Page(**row) for row in rows]

    def _load_chunks(self, source_id: str) -> list[Chunk]:
        rows = self.store.read_jsonl(self.store.doc_dir(source_id) / "chunks.jsonl")
        return [Chunk(**row) for row in rows]

    def _load_evidence(self, source_id: str) -> list[Evidence]:
        rows = self.store.read_jsonl(self.store.doc_dir(source_id) / "evidence.jsonl")
        return [Evidence(**row) for row in rows]

    def _write_run(self, run_id: str, source_id: str, session_id: str, question: str, answer: AnswerContract) -> None:
        self.store.write_json(
            self.store.run_path(run_id),
            {
                "run_id": run_id,
                "source_id": source_id,
                "session_id": session_id,
                "question": question,
                "answer": asdict(answer),
            },
        )


def _all_evidence_refs(evidences: list[Evidence], profile: DocProfile) -> list[EvidenceRef]:
    return [
        EvidenceRef(
            evidence_id=evidence.evidence_id,
            source_id=evidence.source_id,
            chunk_id=evidence.chunk_id,
            page_number=evidence.page_number,
            quote=evidence.quote,
            file_name=profile.file_name,
        )
        for evidence in evidences
        if evidence.source_id == profile.source_id
    ]


def _direct_page_context(profile: DocProfile, pages: list[Page], evidences: list[Evidence]) -> str:
    evidence_ids_by_page: dict[int, list[str]] = {}
    for evidence in evidences:
        if evidence.source_id != profile.source_id:
            continue
        evidence_ids_by_page.setdefault(evidence.page_number, []).append(evidence.evidence_id)

    parts: list[str] = []
    for page in sorted(pages, key=lambda item: item.page_number):
        evidence_ids = ", ".join(evidence_ids_by_page.get(page.page_number, [])) or "none"
        parts.append(
            f"[{profile.file_name} | page {page.page_number}]\n"
            f"Evidence IDs: {evidence_ids}\n"
            f"{page.text}"
        )
    return "\n\n".join(parts)


def _is_direct_context_question(question: str) -> bool:
    normalized = question.strip().lower()
    english_terms = (
        "what is this document about",
        "what is this paper about",
        "what is this article about",
        "what does this document say",
        "what does this paper say",
        "summarize",
        "summary",
        "overview",
        "main idea",
    )
    chinese_terms = (
        "这篇文章",
        "这篇论文",
        "这份文档",
        "这个文档",
        "这篇说了什么",
        "讲了什么",
        "说了什么",
        "总结",
        "概括",
        "主要内容",
    )
    return any(term in normalized for term in english_terms) or any(term in question for term in chinese_terms)


def _append_skipped_document_notice(answer: AnswerContract, skipped_profiles: list[DocProfile]) -> AnswerContract:
    if not skipped_profiles:
        return answer
    note = _skipped_documents_note(skipped_profiles)
    debug = answer.debug
    if debug is not None:
        trajectory = dict(debug.trajectory or {})
        trajectory["skipped_doc_profiles"] = [asdict(profile) for profile in skipped_profiles]
        debug = replace(debug, trajectory=trajectory)
    return replace(answer, answer=f"{answer.answer}\n\n{note}", debug=debug)


def _skipped_documents_note(skipped_profiles: list[DocProfile]) -> str:
    lines = ["以下文档没有可用文本，本次未进入 RLM 问答："]
    for profile in skipped_profiles:
        reason = profile.error_code or profile.status or "doc_not_ready"
        lines.append(
            f"- {profile.file_name}: {profile.status} ({reason}; "
            f"{profile.page_count} 页, {profile.chunk_count} chunks)"
        )
    return "\n".join(lines)


def _all_documents_not_ready_message(profiles: list[DocProfile]) -> str:
    if not profiles:
        return "所选文档不存在或尚未处理完成，无法进行文档问答。"
    return "所选文档都没有可用文本，无法进行文档问答。\n\n" + _skipped_documents_note(profiles)


def _key_pool_trace_marker(llm: LLMClient) -> int:
    trace_fn = getattr(llm, "key_pool_trace", None)
    if not callable(trace_fn):
        return 0
    return len(trace_fn())


def _key_pool_trace_since(llm: LLMClient, marker: int) -> list[dict[str, Any]]:
    trace_fn = getattr(llm, "key_pool_trace", None)
    if not callable(trace_fn):
        return []
    return list(trace_fn())[marker:]


def _llm_trace_context(llm: LLMClient, run_id: str):
    trace_run = getattr(llm, "trace_run", None)
    if callable(trace_run):
        return trace_run(run_id)
    return nullcontext()
