from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from queue import Empty, Queue
import threading
import time
from typing import Any, Callable, Iterable

from .contracts import NoTextError, RLMQAError
from .ids import new_id
from .key_pool import KeyFailureKind, KeyPool
from .ocr import OCRProviderError


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OCRJob:
    job_id: str
    local_doc_id: str
    source_id: str
    file_name: str
    state: str = "ocr_queued"
    attempt_count: int = 0
    created_at: str = field(default_factory=_timestamp)
    started_at: str | None = None
    completed_at: str | None = None
    not_before: str | None = None
    last_error_code: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, *, local_doc_id: str, source_id: str, file_name: str) -> "OCRJob":
        return cls(
            job_id=new_id("ocrjob"),
            local_doc_id=local_doc_id,
            source_id=source_id,
            file_name=file_name,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OCRJob":
        return cls(
            job_id=str(data["job_id"]),
            local_doc_id=str(data["local_doc_id"]),
            source_id=str(data["source_id"]),
            file_name=str(data["file_name"]),
            state=str(data.get("state") or "ocr_queued"),
            attempt_count=int(data.get("attempt_count") or 0),
            created_at=str(data.get("created_at") or _timestamp()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            not_before=data.get("not_before"),
            last_error_code=str(data.get("last_error_code") or ""),
            attempts=list(data.get("attempts") or []),
        )


class OCRJobRunner:
    def __init__(
        self,
        *,
        key_pool: KeyPool,
        max_workers: int,
        max_attempts: int,
        execute: Callable[[OCRJob, str], Any],
        save_job: Callable[[OCRJob], None],
        complete: Callable[[OCRJob, Any], None],
        fail: Callable[[OCRJob, str], None],
        key_state_changed: Callable[[KeyPool], None] | None = None,
        acquire_timeout_seconds: float = 1.0,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.key_pool = key_pool
        self.max_workers = max_workers
        self.max_attempts = max_attempts
        self.execute = execute
        self.save_job = save_job
        self.complete = complete
        self.fail = fail
        self.key_state_changed = key_state_changed
        self.acquire_timeout_seconds = acquire_timeout_seconds
        self._queue: Queue[OCRJob | None] = Queue()
        self._queued_job_ids: set[str] = set()
        self._scheduled_job_ids: set[str] = set()
        self._scheduled_timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []

    def start(self) -> None:
        with self._lock:
            if any(worker.is_alive() for worker in self._workers):
                return
            self._workers = [
                threading.Thread(target=self._worker, name=f"rlm-ocr-{index + 1}", daemon=True)
                for index in range(self.max_workers)
            ]
            for worker in self._workers:
                worker.start()

    def stop(self, *, join_timeout_seconds: float = 2.0) -> None:
        with self._lock:
            workers = list(self._workers)
            self._workers = []
            timers = list(self._scheduled_timers.values())
            self._scheduled_timers = {}
            self._scheduled_job_ids = set()
        for timer in timers:
            timer.cancel()
        for _ in workers:
            self._queue.put(None)
        for worker in workers:
            worker.join(timeout=join_timeout_seconds)

    def enqueue(self, job: OCRJob) -> None:
        if job.state in {"ocr_failed", "ocr_succeeded"}:
            return
        delay_seconds = _delay_until(job.not_before)
        if delay_seconds > 0:
            self._schedule(job, delay_seconds)
            return
        with self._lock:
            if job.job_id in self._queued_job_ids or job.job_id in self._scheduled_job_ids:
                return
            self._queued_job_ids.add(job.job_id)
        self._queue.put(job)

    def recover_pending_jobs(
        self,
        jobs: Iterable[OCRJob],
        *,
        stale_after_seconds: float | None = None,
        recover_running: bool = False,
    ) -> int:
        recovered = 0
        now = datetime.now(timezone.utc)
        for job in jobs:
            if job.state == "ocr_queued":
                self.enqueue(job)
                recovered += 1
                continue
            if job.state != "ocr_running":
                continue
            if not recover_running and not _is_stale(job.started_at, now, stale_after_seconds):
                continue
            job.state = "ocr_queued"
            job.last_error_code = "OCR_WORKER_RECOVERED"
            self.save_job(job)
            self.enqueue(job)
            recovered += 1
        return recovered

    def process_next_for_test(self) -> bool:
        try:
            job = self._queue.get_nowait()
        except Empty:
            return False
        try:
            if job is None:
                return False
            self._remove_queued_marker(job)
            self._process(job)
            return True
        finally:
            self._queue.task_done()

    def key_pool_trace(self) -> list[dict[str, Any]]:
        return list(self.key_pool.trace)

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._remove_queued_marker(job)
                self._process(job)
            finally:
                self._queue.task_done()

    def _process(self, job: OCRJob) -> None:
        try:
            lease = self.key_pool.acquire(
                run_id=job.source_id,
                task_id=f"{job.job_id}_attempt_{job.attempt_count + 1}",
                timeout_seconds=self.acquire_timeout_seconds,
            )
        except RLMQAError:
            if self.key_pool.has_usable_key():
                self._requeue(job, "GLM_OCR_KEY_WAIT")
            else:
                self._terminal_failure(job, "GLM_OCR_KEYS_UNAVAILABLE")
            return

        attempt_started_at = _timestamp()
        started = time.perf_counter()
        job.state = "ocr_running"
        job.attempt_count += 1
        job.started_at = attempt_started_at
        job.last_error_code = ""
        self.save_job(job)

        try:
            result = self.execute(job, lease.api_key)
        except OCRProviderError as exc:
            self._handle_provider_error(job, lease, exc, started, attempt_started_at)
        except NoTextError:
            self._release(lease, success=True, latency_ms=_elapsed_ms(started))
            self._append_attempt(job, lease.key_id, attempt_started_at, "document_error", "GLM_OCR_EMPTY_TEXT", started)
            self._terminal_failure(job, "GLM_OCR_EMPTY_TEXT")
        except Exception:
            self._release(lease, success=True, latency_ms=_elapsed_ms(started))
            self._append_attempt(job, lease.key_id, attempt_started_at, "processing_error", "GLM_OCR_PROCESSING_FAILED", started)
            self._terminal_failure(job, "GLM_OCR_PROCESSING_FAILED")
        else:
            self._release(lease, success=True, latency_ms=_elapsed_ms(started))
            self._append_attempt(job, lease.key_id, attempt_started_at, "success", "", started)
            try:
                self.complete(job, result)
            except Exception:
                self._terminal_failure(job, "GLM_OCR_PROCESSING_FAILED")
                return
            job.state = "ocr_succeeded"
            job.completed_at = _timestamp()
            job.not_before = None
            job.last_error_code = ""
            self.save_job(job)

    def _handle_provider_error(
        self,
        job: OCRJob,
        lease,
        error: OCRProviderError,
        started: float,
        attempt_started_at: str,
    ) -> None:
        failure_kind, retryable = _provider_failure_policy(error)
        if not retryable:
            self._release(lease, success=True, latency_ms=_elapsed_ms(started), error_code=error.error_code)
            self._append_attempt(job, lease.key_id, attempt_started_at, "document_error", error.error_code, started)
            self._terminal_failure(job, "GLM_OCR_DOCUMENT_REJECTED")
            return

        self._release(
            lease,
            success=False,
            latency_ms=_elapsed_ms(started),
            error_code=error.error_code,
            failure_kind=failure_kind,
        )
        self._append_attempt(job, lease.key_id, attempt_started_at, "retryable_error", error.error_code, started)
        if job.attempt_count >= self.max_attempts:
            self._terminal_failure(job, "GLM_OCR_RETRIES_EXHAUSTED")
        elif not self.key_pool.has_usable_key():
            self._terminal_failure(job, "GLM_OCR_KEYS_UNAVAILABLE")
        else:
            self._requeue(job, error.error_code)

    def _requeue(self, job: OCRJob, error_code: str) -> None:
        job.state = "ocr_queued"
        job.last_error_code = error_code
        delay_seconds = self.key_pool.next_available_delay_seconds(run_id=job.source_id)
        job.not_before = _timestamp_after(delay_seconds) if delay_seconds and delay_seconds > 0 else None
        self.save_job(job)
        self.enqueue(job)

    def _terminal_failure(self, job: OCRJob, error_code: str) -> None:
        job.state = "ocr_failed"
        job.completed_at = _timestamp()
        job.not_before = None
        job.last_error_code = error_code
        self.save_job(job)
        self.fail(job, error_code)

    def _append_attempt(
        self,
        job: OCRJob,
        key_id: str,
        started_at: str,
        outcome: str,
        error_code: str,
        started: float,
    ) -> None:
        job.attempts.append(
            {
                "attempt": job.attempt_count,
                "key_id": key_id,
                "started_at": started_at,
                "completed_at": _timestamp(),
                "outcome": outcome,
                "error_code": error_code,
                "latency_ms": _elapsed_ms(started),
            }
        )

    def _remove_queued_marker(self, job: OCRJob) -> None:
        with self._lock:
            self._queued_job_ids.discard(job.job_id)

    def _schedule(self, job: OCRJob, delay_seconds: float) -> None:
        with self._lock:
            if job.job_id in self._queued_job_ids or job.job_id in self._scheduled_job_ids:
                return
            self._scheduled_job_ids.add(job.job_id)
            timer = threading.Timer(delay_seconds, self._enqueue_scheduled_job, args=(job,))
            timer.daemon = True
            self._scheduled_timers[job.job_id] = timer
        timer.start()

    def _enqueue_scheduled_job(self, job: OCRJob) -> None:
        with self._lock:
            self._scheduled_job_ids.discard(job.job_id)
            self._scheduled_timers.pop(job.job_id, None)
        if job.state in {"ocr_failed", "ocr_succeeded"}:
            return
        job.not_before = None
        self.save_job(job)
        self.enqueue(job)

    def _release(self, lease, **kwargs: Any) -> None:
        self.key_pool.release(lease, **kwargs)
        if self.key_state_changed is not None:
            self.key_state_changed(self.key_pool)


def _provider_failure_policy(error: OCRProviderError) -> tuple[KeyFailureKind, bool]:
    if error.status_code == 429:
        return KeyFailureKind.RATE_LIMITED, True
    if error.status_code in {401, 403}:
        return KeyFailureKind.PERMANENT_KEY, True
    if error.status_code is None or error.status_code == 408 or error.status_code >= 500:
        return KeyFailureKind.TRANSIENT, True
    return KeyFailureKind.TRANSIENT, False


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _is_stale(started_at: str | None, now: datetime, stale_after_seconds: float | None) -> bool:
    if stale_after_seconds is None:
        return False
    if not started_at:
        return True
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return (now - started).total_seconds() >= stale_after_seconds


def _delay_until(timestamp: str | None) -> float:
    if not timestamp:
        return 0.0
    try:
        due = datetime.fromisoformat(timestamp)
    except ValueError:
        return 0.0
    if due.tzinfo is None:
        due = due.replace(tzinfo=timezone.utc)
    return max(0.0, (due - datetime.now(timezone.utc)).total_seconds())


def _timestamp_after(delay_seconds: float | None) -> str | None:
    if delay_seconds is None:
        return None
    return (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).isoformat()
