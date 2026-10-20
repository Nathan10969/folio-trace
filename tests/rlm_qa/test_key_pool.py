from __future__ import annotations

import threading
import time

import pytest
import rlm_qa.key_pool as key_pool_module

from rlm_qa.contracts import RLMQAError
from rlm_qa.key_pool import KeyFailureKind, KeyPool, KeyPoolConfig, KeyStatus


def test_key_pool_acquire_release_and_masked_trace_never_exposes_raw_key() -> None:
    pool = KeyPool(["sk-alpha-secret"], role="subagent")

    lease = pool.acquire(run_id="run_1", task_id="task_1")
    assert lease.api_key == "sk-alpha-secret"
    assert lease.key_id.startswith("subagent_")
    assert lease.masked_key.endswith("cret")
    assert lease.masked_key != lease.api_key
    assert pool.snapshot()[0].status == KeyStatus.BUSY

    pool.release(lease, success=True, latency_ms=12, input_tokens=10, output_tokens=5)

    assert pool.snapshot()[0].status == KeyStatus.IDLE
    event = pool.trace[-1]
    assert event["event"] == "release"
    assert event["key_id"] == lease.key_id
    assert event["key_masked"] == lease.masked_key
    assert "sk-alpha-secret" not in repr(event)


def test_key_pool_lease_ids_are_unique_when_monotonic_time_collides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(key_pool_module.time, "monotonic", lambda: 123.456789)
    pool = KeyPool(
        ["sk-collision-safe"],
        role="subagent",
        config=KeyPoolConfig(
            max_global_concurrency=2,
            max_per_key_concurrency=2,
            max_per_run_concurrency=2,
        ),
    )

    first = pool.acquire(run_id="run", task_id="same-task")
    second = pool.acquire(run_id="run", task_id="same-task")

    assert first.lease_id != second.lease_id

    pool.release(first, success=True)
    pool.release(second, success=True)

    assert pool.snapshot()[0].in_flight == 0


def test_key_pool_redaction_does_not_reintroduce_another_raw_key() -> None:
    first_raw_key = "..."
    second_raw_key = "abcZZZZtail"
    forwarded: list[dict[str, object]] = []
    pool = KeyPool(
        [first_raw_key, second_raw_key],
        role="subagent",
        config=KeyPoolConfig(
            max_global_concurrency=2,
            max_per_key_concurrency=2,
            max_per_run_concurrency=2,
        ),
        trace_sink=forwarded.append,
    )

    first = pool.acquire(run_id="run", task_id="first")
    second = pool.acquire(run_id="run", task_id="second")
    pool.release(first, success=False, error_code=second.api_key)
    pool.release(second, success=True)

    assert first_raw_key not in repr(pool.trace)
    assert second_raw_key not in repr(pool.trace)
    assert first_raw_key not in repr(forwarded)
    assert second_raw_key not in repr(forwarded)


def test_key_pool_forwards_ordered_acquire_release_trace_to_sink() -> None:
    raw_key = "sk-forwarded-secret"
    forwarded: list[dict[str, object]] = []
    pool = KeyPool([raw_key], role="subagent", trace_sink=forwarded.append)

    lease = pool.acquire(run_id="run_sink", task_id="task_sink")
    pool.release(lease, success=True, latency_ms=37)

    assert [event["event"] for event in forwarded] == ["acquire", "release"]
    assert forwarded[1]["latency_ms"] == 37
    assert forwarded[0] is not pool.trace[0]
    assert all("api_key" not in event for event in forwarded)
    assert all("api_key" not in event for event in pool.trace)
    assert raw_key not in repr(forwarded)
    assert raw_key not in repr(pool.trace)


def test_trace_sink_exceptions_do_not_break_pool_accounting() -> None:
    def raising_sink(event: dict[str, object]) -> None:
        raise RuntimeError(f"sink failed for {event['event']}")

    pool = KeyPool(["sk-raising-sink"], role="subagent", trace_sink=raising_sink)

    first = pool.acquire(run_id="run_1", task_id="task_1")
    pool.release(first, success=True)

    snapshot = pool.snapshot()[0]
    assert snapshot.status == KeyStatus.IDLE
    assert snapshot.in_flight == 0

    second = pool.acquire(run_id="run_2", task_id="task_2", timeout_seconds=0.05)
    pool.release(second, success=True)
    assert [event["event"] for event in pool.trace] == ["acquire", "release", "acquire", "release"]


def test_blocking_trace_sink_does_not_hold_pool_condition_lock() -> None:
    sink_entered = threading.Event()
    release_sink = threading.Event()

    def blocking_sink(event: dict[str, object]) -> None:
        if event["event"] == "acquire":
            sink_entered.set()
            release_sink.wait(timeout=2)

    pool = KeyPool(["sk-blocking-sink"], role="subagent", trace_sink=blocking_sink)
    leases: list[key_pool_module.KeyLease] = []
    acquire_errors: list[BaseException] = []

    def acquire_key() -> None:
        try:
            leases.append(pool.acquire(run_id="run", task_id="task"))
        except BaseException as exc:
            acquire_errors.append(exc)

    acquire_thread = threading.Thread(target=acquire_key)
    snapshot_finished = threading.Event()
    snapshots: list[list[key_pool_module.KeySnapshot]] = []

    def take_snapshot() -> None:
        snapshots.append(pool.snapshot())
        snapshot_finished.set()

    snapshot_thread: threading.Thread | None = None
    snapshot_completed_while_sink_blocked = False
    acquire_thread.start()
    try:
        assert sink_entered.wait(timeout=1)
        snapshot_thread = threading.Thread(target=take_snapshot)
        snapshot_thread.start()
        snapshot_completed_while_sink_blocked = snapshot_finished.wait(timeout=0.5)
    finally:
        release_sink.set()
        acquire_thread.join(timeout=1)
        if snapshot_thread is not None:
            snapshot_thread.join(timeout=1)

    assert snapshot_completed_while_sink_blocked
    assert acquire_errors == []
    assert len(leases) == 1
    assert snapshots[0][0].status == KeyStatus.BUSY
    pool.release(leases[0], success=True)


def test_trace_sink_mutation_cannot_mutate_pool_trace() -> None:
    def mutating_sink(event: dict[str, object]) -> None:
        event.clear()
        event["event"] = "sink_mutated"

    pool = KeyPool(["sk-mutation-safe"], role="subagent", trace_sink=mutating_sink)

    lease = pool.acquire(run_id="run", task_id="task")
    pool.release(lease, success=True)

    assert [event["event"] for event in pool.trace] == ["acquire", "release"]
    assert all("run_id" in event for event in pool.trace)


@pytest.mark.parametrize("raw_key", ["********", "abc...hijk"])
def test_adversarial_raw_keys_are_masked_without_appearing_in_trace(raw_key: str) -> None:
    forwarded: list[dict[str, object]] = []
    pool = KeyPool([raw_key], role="subagent", trace_sink=forwarded.append)

    lease = pool.acquire(run_id="run", task_id="task")
    pool.release(lease, success=True)

    assert lease.masked_key != raw_key
    assert raw_key not in repr(pool.trace)
    assert raw_key not in repr(forwarded)


def test_key_pool_enforces_global_and_per_key_concurrency() -> None:
    pool = KeyPool(
        ["sk-a", "sk-b"],
        role="subagent",
        config=KeyPoolConfig(max_global_concurrency=2, max_per_key_concurrency=1),
    )
    first = pool.acquire(run_id="run", task_id="task-1")
    second = pool.acquire(run_id="run", task_id="task-2")
    assert {first.api_key, second.api_key} == {"sk-a", "sk-b"}

    acquired: list[str] = []

    def try_acquire() -> None:
        lease = pool.acquire(run_id="run", task_id="task-3", timeout_seconds=0.5)
        acquired.append(lease.api_key)
        pool.release(lease, success=True)

    thread = threading.Thread(target=try_acquire)
    thread.start()
    time.sleep(0.05)
    assert acquired == []
    pool.release(first, success=True)
    thread.join(timeout=1)

    assert acquired == ["sk-a"]
    pool.release(second, success=True)


def test_key_pool_raises_specific_timeout_when_second_acquire_is_blocked() -> None:
    pool = KeyPool(
        ["sk-only"],
        role="subagent",
        config=KeyPoolConfig(max_global_concurrency=1, acquire_poll_seconds=0.001),
    )
    first = pool.acquire(run_id="run_1", task_id="task_1")

    try:
        with pytest.raises(RLMQAError, match="subagent key pool acquire timed out") as exc_info:
            pool.acquire(run_id="run_2", task_id="task_2", timeout_seconds=0.01)
    finally:
        pool.release(first, success=True)

    timeout_type = getattr(key_pool_module, "KeyPoolAcquireTimeout", None)
    assert timeout_type is not None
    assert exc_info.type is timeout_type
    assert issubclass(timeout_type, RLMQAError)


def test_key_pool_cooldown_and_dead_after_failures() -> None:
    pool = KeyPool(
        ["sk-a"],
        role="subagent",
        config=KeyPoolConfig(cooldown_seconds=0.05, max_failures_before_dead=2),
    )

    first = pool.acquire(run_id="run", task_id="task-1")
    pool.release(first, success=False, error_code="429")
    assert pool.snapshot()[0].status == KeyStatus.COOLDOWN

    time.sleep(0.06)
    second = pool.acquire(run_id="run", task_id="task-2")
    pool.release(second, success=False, error_code="429")
    assert pool.snapshot()[0].status == KeyStatus.DEAD


def test_key_pool_watchdog_releases_expired_lease_and_ignores_late_release() -> None:
    pool = KeyPool(
        ["sk-alpha-secret"],
        role="subagent",
        config=KeyPoolConfig(
            max_global_concurrency=1,
            lease_timeout_seconds=0.01,
            cooldown_seconds=0.01,
        ),
    )

    lease = pool.acquire(run_id="run_watchdog", task_id="task_1")
    assert pool.snapshot()[0].status == KeyStatus.BUSY

    time.sleep(0.03)
    snapshot = pool.snapshot()

    assert snapshot[0].in_flight == 0
    assert snapshot[0].status in {KeyStatus.IDLE, KeyStatus.COOLDOWN}
    assert any(event["event"] == "watchdog_release" for event in pool.trace)
    assert "sk-alpha-secret" not in repr(pool.trace)

    pool.release(lease, success=True)

    assert pool.snapshot()[0].in_flight == 0
    assert pool.trace[-1]["event"] == "late_release_ignored"


def test_key_pool_rate_limit_cools_key_without_counting_permanent_failure() -> None:
    pool = KeyPool(
        ["sk-ocr-a"],
        role="ocr",
        config=KeyPoolConfig(cooldown_seconds=0.05, rate_limit_cooldown_seconds=0.05),
    )

    lease = pool.acquire(run_id="ocr_job", task_id="attempt_1")
    pool.release(lease, success=False, failure_kind=KeyFailureKind.RATE_LIMITED, error_code="http_429")

    snapshot = pool.snapshot()[0]
    assert snapshot.status == KeyStatus.COOLDOWN
    assert snapshot.failure_count == 0
    assert pool.trace[-1]["failure_kind"] == "rate_limited"


def test_key_pool_marks_permanently_invalid_key_dead_immediately() -> None:
    pool = KeyPool(["sk-ocr-a"], role="ocr")

    lease = pool.acquire(run_id="ocr_job", task_id="attempt_1")
    pool.release(lease, success=False, failure_kind=KeyFailureKind.PERMANENT_KEY, error_code="http_401")

    snapshot = pool.snapshot()[0]
    assert snapshot.status == KeyStatus.DEAD
    assert snapshot.failure_count == 1
    assert pool.has_usable_key() is False


def test_key_pool_deduplicates_repeated_key_without_breaking_release_accounting() -> None:
    pool = KeyPool(
        ["sk-duplicate", "sk-duplicate"],
        role="ocr",
        config=KeyPoolConfig(max_global_concurrency=2, max_per_key_concurrency=1),
    )

    assert len(pool.snapshot()) == 1
    first = pool.acquire(run_id="run_1", task_id="task_1")
    with pytest.raises(RLMQAError, match="acquire timed out"):
        pool.acquire(run_id="run_2", task_id="task_2", timeout_seconds=0.01)
    pool.release(first, success=True)

    second = pool.acquire(run_id="run_2", task_id="task_2")
    pool.release(second, success=True)
    assert pool.snapshot()[0].in_flight == 0
