from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Iterable

from .contracts import RLMQAError


class KeyPoolAcquireTimeout(RLMQAError):
    """Raised when a key cannot be acquired before its deadline."""


class KeyStatus(StrEnum):
    IDLE = "idle"
    BUSY = "busy"
    COOLDOWN = "cooldown"
    DEAD = "dead"


class KeyFailureKind(StrEnum):
    TRANSIENT = "transient"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"
    PERMANENT_KEY = "permanent_key"


@dataclass(frozen=True)
class KeyPoolConfig:
    max_global_concurrency: int = 4
    max_per_key_concurrency: int = 1
    max_per_run_concurrency: int = 4
    cooldown_seconds: float = 30.0
    rate_limit_cooldown_seconds: float | None = None
    max_failures_before_dead: int = 3
    acquire_poll_seconds: float = 0.02
    lease_timeout_seconds: float | None = None


@dataclass(frozen=True)
class KeyLease:
    api_key: str
    key_id: str
    masked_key: str
    role: str
    run_id: str
    task_id: str
    acquired_at: float
    lease_id: str


@dataclass(frozen=True)
class KeySnapshot:
    key_id: str
    masked_key: str
    role: str
    status: KeyStatus
    in_flight: int
    failure_count: int
    cooldown_until: float


@dataclass
class _KeyState:
    api_key: str
    key_id: str
    masked_key: str
    role: str
    status: KeyStatus = KeyStatus.IDLE
    in_flight: int = 0
    failure_count: int = 0
    cooldown_until: float = 0.0


_PendingTrace = tuple[int, dict[str, Any]]


class KeyPool:
    def __init__(
        self,
        api_keys: list[str] | tuple[str, ...],
        *,
        role: str,
        config: KeyPoolConfig | None = None,
        trace_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        clean_keys: list[str] = []
        seen_keys: set[str] = set()
        for raw_key in api_keys:
            key = raw_key.strip() if raw_key else ""
            if key and key not in seen_keys:
                clean_keys.append(key)
                seen_keys.add(key)
        if not clean_keys:
            raise RLMQAError(f"{role} key pool has no keys")
        self.role = role
        self.config = config or KeyPoolConfig()
        self._condition = threading.Condition()
        self._states = [
            _KeyState(
                api_key=key,
                key_id=f"{role}_{_short_hash(key)}",
                masked_key=_mask_key(key),
                role=role,
            )
            for key in clean_keys
        ]
        raw_keys = tuple(state.api_key for state in self._states)
        redaction_masks = {
            key: _safe_mask(key, raw_keys)
            for key in raw_keys
        }
        for state in self._states:
            state.masked_key = redaction_masks[state.api_key]
        self._redaction_masks = redaction_masks
        self._redaction_pattern = re.compile(
            "|".join(re.escape(key) for key in sorted(raw_keys, key=len, reverse=True))
        )
        self._next_index = 0
        self._lease_counter = 0
        self._run_inflight: dict[str, int] = {}
        self._active_leases: dict[str, KeyLease] = {}
        self.trace: list[dict[str, Any]] = []
        self._trace_sink = trace_sink
        self._trace_delivery_condition = threading.Condition()
        self._next_trace_sequence = 0
        self._next_trace_delivery = 0

    def acquire(self, *, run_id: str, task_id: str, timeout_seconds: float | None = None) -> KeyLease:
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while True:
            pending_traces: list[_PendingTrace] = []
            lease: KeyLease | None = None
            timed_out = False
            with self._condition:
                pending_traces.extend(self._release_expired_leases_locked())
                self._refresh_cooldowns_locked()
                state = self._available_state_locked(run_id)
                if state is not None:
                    now = time.monotonic()
                    self._lease_counter += 1
                    state.in_flight += 1
                    state.status = KeyStatus.BUSY
                    self._run_inflight[run_id] = self._run_inflight.get(run_id, 0) + 1
                    lease = KeyLease(
                        api_key=state.api_key,
                        key_id=state.key_id,
                        masked_key=state.masked_key,
                        role=state.role,
                        run_id=run_id,
                        task_id=task_id,
                        acquired_at=now,
                        lease_id=f"{state.key_id}:{task_id}:{int(now * 1000000)}:{self._lease_counter}",
                    )
                    self._active_leases[lease.lease_id] = lease
                    pending_trace = self._record_trace(
                        {
                            "event": "acquire",
                            "run_id": run_id,
                            "task_id": task_id,
                            "key_id": state.key_id,
                            "key_masked": state.masked_key,
                            "role": state.role,
                        }
                    )
                    if pending_trace is not None:
                        pending_traces.append(pending_trace)
                elif deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                    else:
                        wait_seconds = min(self.config.acquire_poll_seconds, remaining)
                else:
                    wait_seconds = self.config.acquire_poll_seconds

                if lease is None and not timed_out and not pending_traces:
                    self._condition.wait(wait_seconds)
                    continue

            self._deliver_traces(pending_traces)
            if lease is not None:
                return lease
            if timed_out:
                raise KeyPoolAcquireTimeout(f"{self.role} key pool acquire timed out")

    def release(
        self,
        lease: KeyLease,
        *,
        success: bool,
        latency_ms: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error_code: str = "",
        failure_kind: KeyFailureKind = KeyFailureKind.TRANSIENT,
    ) -> None:
        pending_trace: _PendingTrace | None = None
        with self._condition:
            if lease.lease_id not in self._active_leases:
                pending_trace = self._record_trace(
                    {
                        "event": "late_release_ignored",
                        "run_id": lease.run_id,
                        "task_id": lease.task_id,
                        "key_id": lease.key_id,
                        "key_masked": lease.masked_key,
                        "role": lease.role,
                    }
                )
                self._condition.notify_all()
            else:
                self._active_leases.pop(lease.lease_id, None)
                state = self._state_by_id(lease.key_id)
                if state is None:
                    return
                state.in_flight = max(0, state.in_flight - 1)
                self._run_inflight[lease.run_id] = max(0, self._run_inflight.get(lease.run_id, 0) - 1)
                if self._run_inflight.get(lease.run_id) == 0:
                    self._run_inflight.pop(lease.run_id, None)

                if success or failure_kind == KeyFailureKind.CANCELLED:
                    if success:
                        state.failure_count = 0
                    state.cooldown_until = 0.0
                    state.status = KeyStatus.IDLE if state.in_flight == 0 else KeyStatus.BUSY
                elif failure_kind == KeyFailureKind.PERMANENT_KEY:
                    state.failure_count += 1
                    state.cooldown_until = 0.0
                    state.status = KeyStatus.DEAD
                elif failure_kind == KeyFailureKind.RATE_LIMITED:
                    cooldown_seconds = self.config.rate_limit_cooldown_seconds or self.config.cooldown_seconds
                    state.cooldown_until = time.monotonic() + cooldown_seconds
                    state.status = KeyStatus.COOLDOWN
                else:
                    state.failure_count += 1
                    if state.failure_count >= self.config.max_failures_before_dead:
                        state.status = KeyStatus.DEAD
                    else:
                        state.cooldown_until = time.monotonic() + self.config.cooldown_seconds
                        state.status = KeyStatus.COOLDOWN

                pending_trace = self._record_trace(
                    {
                        "event": "release",
                        "run_id": lease.run_id,
                        "task_id": lease.task_id,
                        "key_id": lease.key_id,
                        "key_masked": lease.masked_key,
                        "role": lease.role,
                        "success": success,
                        "latency_ms": latency_ms,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "error": error_code,
                        "failure_kind": failure_kind.value if not success else "",
                        "status": state.status.value,
                    }
                )
                self._condition.notify_all()

        self._deliver_trace(pending_trace)

    def snapshot(self) -> list[KeySnapshot]:
        with self._condition:
            pending_traces = self._release_expired_leases_locked()
            self._refresh_cooldowns_locked()
            snapshots = [
                KeySnapshot(
                    key_id=state.key_id,
                    masked_key=state.masked_key,
                    role=state.role,
                    status=state.status,
                    in_flight=state.in_flight,
                    failure_count=state.failure_count,
                    cooldown_until=state.cooldown_until,
                )
                for state in self._states
            ]
        self._deliver_traces(pending_traces)
        return snapshots

    def has_usable_key(self) -> bool:
        with self._condition:
            pending_traces = self._release_expired_leases_locked()
            self._refresh_cooldowns_locked()
            has_usable_key = any(state.status != KeyStatus.DEAD for state in self._states)
        self._deliver_traces(pending_traces)
        return has_usable_key

    def next_available_delay_seconds(self, *, run_id: str) -> float | None:
        with self._condition:
            pending_traces = self._release_expired_leases_locked()
            self._refresh_cooldowns_locked()
            if not any(state.status != KeyStatus.DEAD for state in self._states):
                delay = None
            elif self._has_available_state_locked(run_id):
                delay = 0.0
            else:
                now = time.monotonic()
                cooldowns = [
                    state.cooldown_until
                    for state in self._states
                    if state.status == KeyStatus.COOLDOWN and state.cooldown_until > now
                ]
                if cooldowns:
                    delay = max(0.0, min(cooldowns) - now)
                else:
                    delay = max(self.config.acquire_poll_seconds, 0.1)
        self._deliver_traces(pending_traces)
        return delay

    def mark_dead_key_ids(self, key_ids: Iterable[str]) -> None:
        dead_ids = set(key_ids)
        if not dead_ids:
            return
        pending_traces: list[_PendingTrace] = []
        with self._condition:
            for state in self._states:
                if state.key_id not in dead_ids:
                    continue
                state.status = KeyStatus.DEAD
                state.cooldown_until = 0.0
                pending_trace = self._record_trace(
                    {
                        "event": "restore_dead",
                        "key_id": state.key_id,
                        "key_masked": state.masked_key,
                        "role": state.role,
                    }
                )
                if pending_trace is not None:
                    pending_traces.append(pending_trace)
            self._condition.notify_all()
        self._deliver_traces(pending_traces)

    def _available_state_locked(self, run_id: str) -> _KeyState | None:
        global_inflight = sum(state.in_flight for state in self._states)
        if global_inflight >= self.config.max_global_concurrency:
            return None
        if self._run_inflight.get(run_id, 0) >= self.config.max_per_run_concurrency:
            return None
        total = len(self._states)
        for offset in range(total):
            index = (self._next_index + offset) % total
            state = self._states[index]
            if state.status in {KeyStatus.COOLDOWN, KeyStatus.DEAD}:
                continue
            if state.in_flight >= self.config.max_per_key_concurrency:
                continue
            self._next_index = (index + 1) % total
            return state
        return None

    def _has_available_state_locked(self, run_id: str) -> bool:
        global_inflight = sum(state.in_flight for state in self._states)
        if global_inflight >= self.config.max_global_concurrency:
            return False
        if self._run_inflight.get(run_id, 0) >= self.config.max_per_run_concurrency:
            return False
        return any(
            state.status not in {KeyStatus.COOLDOWN, KeyStatus.DEAD}
            and state.in_flight < self.config.max_per_key_concurrency
            for state in self._states
        )

    def _refresh_cooldowns_locked(self) -> None:
        now = time.monotonic()
        for state in self._states:
            if state.status == KeyStatus.COOLDOWN and state.cooldown_until <= now:
                state.status = KeyStatus.IDLE if state.in_flight == 0 else KeyStatus.BUSY
                state.cooldown_until = 0.0

    def _release_expired_leases_locked(self) -> list[_PendingTrace]:
        if self.config.lease_timeout_seconds is None:
            return []
        pending_traces: list[_PendingTrace] = []
        now = time.monotonic()
        expired = [
            lease
            for lease in self._active_leases.values()
            if now - lease.acquired_at > self.config.lease_timeout_seconds
        ]
        for lease in expired:
            self._active_leases.pop(lease.lease_id, None)
            state = self._state_by_id(lease.key_id)
            if state is None:
                continue
            state.in_flight = max(0, state.in_flight - 1)
            self._run_inflight[lease.run_id] = max(0, self._run_inflight.get(lease.run_id, 0) - 1)
            if self._run_inflight.get(lease.run_id) == 0:
                self._run_inflight.pop(lease.run_id, None)
            state.failure_count += 1
            if state.failure_count >= self.config.max_failures_before_dead:
                state.status = KeyStatus.DEAD
            else:
                state.cooldown_until = now + self.config.cooldown_seconds
                state.status = KeyStatus.COOLDOWN
            pending_trace = self._record_trace(
                {
                    "event": "watchdog_release",
                    "run_id": lease.run_id,
                    "task_id": lease.task_id,
                    "key_id": lease.key_id,
                    "key_masked": lease.masked_key,
                    "role": lease.role,
                    "success": False,
                    "latency_ms": int((now - lease.acquired_at) * 1000),
                    "error": "lease_timeout",
                    "failure_kind": KeyFailureKind.TRANSIENT.value,
                    "status": state.status.value,
                }
            )
            if pending_trace is not None:
                pending_traces.append(pending_trace)
        return pending_traces

    def _record_trace(self, event: dict[str, Any]) -> _PendingTrace | None:
        trace_event = dict(event)
        trace_event.pop("api_key", None)
        for field, value in trace_event.items():
            if isinstance(value, str):
                trace_event[field] = self._redact_text(value)
        self.trace.append(trace_event)
        if self._trace_sink is None:
            return None
        sequence = self._next_trace_sequence
        self._next_trace_sequence += 1
        return sequence, dict(trace_event)

    def _redact_text(self, value: str) -> str:
        return self._redaction_pattern.sub(
            lambda match: self._redaction_masks[match.group(0)],
            value,
        )

    def _deliver_traces(self, pending_traces: Iterable[_PendingTrace]) -> None:
        for pending_trace in pending_traces:
            self._deliver_trace(pending_trace)

    def _deliver_trace(self, pending_trace: _PendingTrace | None) -> None:
        if pending_trace is None:
            return
        sequence, event = pending_trace
        with self._trace_delivery_condition:
            while sequence != self._next_trace_delivery:
                self._trace_delivery_condition.wait()
            try:
                if self._trace_sink is not None:
                    self._trace_sink(event)
            except Exception:
                pass
            finally:
                self._next_trace_delivery += 1
                self._trace_delivery_condition.notify_all()

    def _state_by_id(self, key_id: str) -> _KeyState | None:
        for state in self._states:
            if state.key_id == key_id:
                return state
        return None


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        masked_key = "*" * len(key)
    else:
        masked_key = f"{key[:3]}...{key[-4:]}"
    if key and key in masked_key:
        replacement = "#" if key[0] != "#" else "!"
        return replacement + ("*" * (len(key) - 1))
    return masked_key


def _safe_mask(key: str, raw_keys: tuple[str, ...]) -> str:
    conventional_mask = _mask_key(key)
    if conventional_mask != key and not any(raw_key in conventional_mask for raw_key in raw_keys):
        return conventional_mask

    mask_length = max(1, len(key))
    used_characters = set("".join(raw_keys))
    for codepoint in range(0x21, 0x7F):
        character = chr(codepoint)
        if character not in used_characters:
            return character * mask_length

    for codepoint in range(0x80, 0x110000):
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        character = chr(codepoint)
        if character not in used_characters:
            return character * mask_length

    return ""


def _short_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
