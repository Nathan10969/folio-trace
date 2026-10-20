from __future__ import annotations

from dataclasses import asdict
import threading
import time

from rlm_qa.contracts import NoTextError
from rlm_qa.key_pool import KeyPool, KeyPoolConfig
from rlm_qa.ocr import OCRProviderError
from rlm_qa.ocr_jobs import OCRJob, OCRJobRunner


def make_job() -> OCRJob:
    return OCRJob.create(
        local_doc_id="local_1",
        source_id="src_1",
        file_name="paper.pdf",
    )


def test_runner_marks_successful_job_completed_with_masked_key_trace() -> None:
    saved: list[dict] = []
    executed_keys: list[str] = []
    completed: list[str] = []
    pool = KeyPool(["sk-ocr-secret"], role="ocr")
    runner = OCRJobRunner(
        key_pool=pool,
        max_workers=1,
        max_attempts=3,
        execute=lambda job, api_key: executed_keys.append(api_key),
        save_job=lambda job: saved.append(asdict(job)),
        complete=lambda job, result: completed.append(job.source_id),
        fail=lambda job, error_code: (_ for _ in ()).throw(AssertionError(error_code)),
    )

    job = make_job()
    runner.enqueue(job)

    assert runner.process_next_for_test() is True
    assert job.state == "ocr_succeeded"
    assert job.attempt_count == 1
    assert executed_keys == ["sk-ocr-secret"]
    assert completed == ["src_1"]
    assert saved[-1]["state"] == "ocr_succeeded"
    assert "sk-ocr-secret" not in repr(saved)
    assert "sk-ocr-secret" not in repr(runner.key_pool_trace())


def test_runner_requeues_rate_limited_job_on_next_available_key() -> None:
    executed_keys: list[str] = []
    failures: list[str] = []
    calls = 0

    def execute(job: OCRJob, api_key: str) -> None:
        nonlocal calls
        calls += 1
        executed_keys.append(api_key)
        if calls == 1:
            raise OCRProviderError("rate limited", status_code=429, error_code="http_429")

    pool = KeyPool(
        ["sk-ocr-a", "sk-ocr-b"],
        role="ocr",
        config=KeyPoolConfig(cooldown_seconds=5, rate_limit_cooldown_seconds=5),
    )
    runner = OCRJobRunner(
        key_pool=pool,
        max_workers=1,
        max_attempts=3,
        execute=execute,
        save_job=lambda job: None,
        complete=lambda job, result: None,
        fail=lambda job, error_code: failures.append(error_code),
    )
    job = make_job()
    runner.enqueue(job)

    assert runner.process_next_for_test() is True
    assert job.state == "ocr_queued"
    assert runner.process_next_for_test() is True
    assert job.state == "ocr_succeeded"
    assert executed_keys == ["sk-ocr-a", "sk-ocr-b"]
    assert failures == []


def test_runner_marks_empty_ocr_text_as_document_failure_without_cooling_key() -> None:
    failures: list[str] = []
    pool = KeyPool(["sk-ocr-a"], role="ocr")
    runner = OCRJobRunner(
        key_pool=pool,
        max_workers=1,
        max_attempts=3,
        execute=lambda job, api_key: (_ for _ in ()).throw(NoTextError("empty OCR text")),
        save_job=lambda job: None,
        complete=lambda job, result: None,
        fail=lambda job, error_code: failures.append(error_code),
    )
    job = make_job()
    runner.enqueue(job)

    assert runner.process_next_for_test() is True
    assert job.state == "ocr_failed"
    assert failures == ["GLM_OCR_EMPTY_TEXT"]
    assert pool.snapshot()[0].failure_count == 0


def test_runner_marks_job_failed_when_authentication_kills_every_key() -> None:
    failures: list[str] = []
    pool = KeyPool(["sk-ocr-a"], role="ocr")
    runner = OCRJobRunner(
        key_pool=pool,
        max_workers=1,
        max_attempts=3,
        execute=lambda job, api_key: (_ for _ in ()).throw(
            OCRProviderError("unauthorized", status_code=401, error_code="http_401")
        ),
        save_job=lambda job: None,
        complete=lambda job, result: None,
        fail=lambda job, error_code: failures.append(error_code),
    )
    job = make_job()
    runner.enqueue(job)

    assert runner.process_next_for_test() is True
    assert job.state == "ocr_failed"
    assert failures == ["GLM_OCR_KEYS_UNAVAILABLE"]
    assert pool.has_usable_key() is False


def test_runner_recovers_stale_running_job_after_restart() -> None:
    executed: list[str] = []
    saved: list[str] = []
    pool = KeyPool(["sk-ocr-a"], role="ocr")
    runner = OCRJobRunner(
        key_pool=pool,
        max_workers=1,
        max_attempts=3,
        execute=lambda job, api_key: executed.append(api_key),
        save_job=lambda job: saved.append(job.state),
        complete=lambda job, result: None,
        fail=lambda job, error_code: (_ for _ in ()).throw(AssertionError(error_code)),
    )
    job = make_job()
    job.state = "ocr_running"
    job.started_at = "2000-01-01T00:00:00+00:00"

    assert runner.recover_pending_jobs([job], stale_after_seconds=300) == 1
    assert job.state == "ocr_queued"
    assert runner.process_next_for_test() is True
    assert job.state == "ocr_succeeded"
    assert executed == ["sk-ocr-a"]
    assert saved[0] == "ocr_queued"


def test_runner_uses_three_distinct_ocr_keys_for_three_concurrent_jobs() -> None:
    acquired_keys: list[str] = []
    acquired_lock = threading.Lock()
    all_started = threading.Event()
    release_workers = threading.Event()
    pool = KeyPool(
        ["sk-ocr-a", "sk-ocr-b", "sk-ocr-c"],
        role="ocr",
        config=KeyPoolConfig(max_global_concurrency=3, max_per_key_concurrency=1),
    )

    def execute(job: OCRJob, api_key: str) -> None:
        with acquired_lock:
            acquired_keys.append(api_key)
            if len(acquired_keys) == 3:
                all_started.set()
        release_workers.wait(timeout=1)

    runner = OCRJobRunner(
        key_pool=pool,
        max_workers=3,
        max_attempts=3,
        execute=execute,
        save_job=lambda job: None,
        complete=lambda job, result: None,
        fail=lambda job, error_code: (_ for _ in ()).throw(AssertionError(error_code)),
    )
    jobs = [
        OCRJob.create(local_doc_id=f"local_{index}", source_id=f"src_{index}", file_name=f"{index}.pdf")
        for index in range(3)
    ]
    runner.start()
    try:
        for job in jobs:
            runner.enqueue(job)
        assert all_started.wait(timeout=1)
        assert set(acquired_keys) == {"sk-ocr-a", "sk-ocr-b", "sk-ocr-c"}
        assert all(snapshot.in_flight == 1 for snapshot in pool.snapshot())
    finally:
        release_workers.set()
        deadline = time.monotonic() + 1
        while any(job.state != "ocr_succeeded" for job in jobs) and time.monotonic() < deadline:
            time.sleep(0.01)
        runner.stop()

    assert all(job.state == "ocr_succeeded" for job in jobs)


def test_runner_delays_single_rate_limited_job_until_key_cooldown_expires() -> None:
    calls = 0

    def execute(job: OCRJob, api_key: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OCRProviderError("rate limited", status_code=429, error_code="http_429")

    pool = KeyPool(
        ["sk-ocr-a"],
        role="ocr",
        config=KeyPoolConfig(cooldown_seconds=0.05, rate_limit_cooldown_seconds=0.05),
    )
    runner = OCRJobRunner(
        key_pool=pool,
        max_workers=1,
        max_attempts=3,
        execute=execute,
        save_job=lambda job: None,
        complete=lambda job, result: None,
        fail=lambda job, error_code: (_ for _ in ()).throw(AssertionError(error_code)),
        acquire_timeout_seconds=0.01,
    )
    job = make_job()
    runner.enqueue(job)

    assert runner.process_next_for_test() is True
    assert job.state == "ocr_queued"
    assert job.not_before is not None
    assert runner.process_next_for_test() is False
    deadline = time.monotonic() + 0.5
    processed = False
    while not processed and time.monotonic() < deadline:
        processed = runner.process_next_for_test()
        if processed:
            break
        time.sleep(0.01)
    assert processed is True
    assert job.state == "ocr_succeeded"
    assert job.attempt_count == 2
