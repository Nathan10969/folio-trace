from __future__ import annotations

import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rlm_runtime.core.prompts import build_system_prompt, build_user_prompt
from rlm_runtime.core.types import (
    RLMIteration,
    RLMResult,
    REPLResult,
    RuntimeErrorInfo,
    RuntimeMetadata,
    UsageSummary,
)
from rlm_runtime.envs.local_repl import LocalREPL
from rlm_runtime.logger.trajectory import TrajectoryLogger


LLMFn = Callable[..., str]


@dataclass(frozen=True)
class _ConcurrentResult:
    index: int
    value: str = ""
    error: RuntimeErrorInfo | None = None
    elapsed_ms: int = 0


class RLM:
    def __init__(
        self,
        *,
        llm: LLMFn,
        model_name: str | None = None,
        max_iterations: int = 5,
        custom_tools: dict[str, Any] | None = None,
        max_depth: int = 1,
        batch_max_workers: int = 4,
        max_concurrent_subcalls: int = 4,
        max_timeout_seconds: float | None = None,
        max_timeout: float | None = None,
        max_tokens: int | None = None,
        max_errors: int | None = None,
        max_budget: float | None = None,
        cost_per_1k_tokens: float | None = None,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        trajectory_path: str | Path | None = None,
    ) -> None:
        self.llm = llm
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.custom_tools = custom_tools or {}
        self.max_depth = max(1, int(max_depth))
        self.batch_max_workers = max(1, int(batch_max_workers))
        self.max_concurrent_subcalls = max(1, int(max_concurrent_subcalls))
        self.max_timeout_seconds = max_timeout_seconds if max_timeout_seconds is not None else max_timeout
        self.max_tokens = max_tokens
        self.max_errors = max_errors
        self.max_budget = max_budget
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.run_id = run_id
        self.parent_run_id = parent_run_id
        self.trajectory_path = Path(trajectory_path) if trajectory_path else None
        self._thread_state = threading.local()
        self._usage_lock = threading.Lock()
        self._reset_usage()

    def completion(
        self,
        prompt: str,
        *,
        context: Any,
        custom_tools: dict[str, Any] | None = None,
        _depth: int = 1,
        _parent_run_id: str | None = None,
        _prompt_index: int | None = None,
        _run_id: str | None = None,
        _model: str | None = None,
    ) -> RLMResult:
        parent_run_id = _parent_run_id if _parent_run_id is not None else self.parent_run_id
        if parent_run_id is None:
            self._reset_usage()
        run_id = _run_id or (self.run_id if parent_run_id is None and self.run_id else _new_run_id())
        started_at = _utc_now()
        started_perf = time.perf_counter()
        metadata = RuntimeMetadata(
            run_id=run_id,
            parent_run_id=parent_run_id,
            depth=_depth,
            prompt_index=_prompt_index,
            model=_model or self.model_name,
            started_at=started_at,
        )
        logger = TrajectoryLogger(run_metadata=asdict(metadata), trajectory_path=self.trajectory_path)
        logger.add_event(
            {
                "event": "run_started",
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "depth": _depth,
                "prompt_index": _prompt_index,
                "model": _model or self.model_name,
                "started_at": started_at,
            }
        )
        previous_state = self._set_thread_state(
            logger=logger,
            run_id=run_id,
            depth=_depth,
            started_perf=started_perf,
        )
        tools = {**self.custom_tools, **(custom_tools or {})}
        transcript: list[str] = []
        repl = LocalREPL(
            context=context,
            custom_tools=tools,
            llm_query=self._llm_query,
            llm_query_batched=self._llm_query_batched,
            rlm_query=lambda child_prompt, model=None: self._rlm_query(child_prompt, model, _depth, None),
            rlm_query_batched=lambda child_prompts, model=None: self._rlm_query_batched(child_prompts, model, _depth),
        )
        system = build_system_prompt(tools)
        last_response = ""
        consecutive_errors = 0

        try:
            for index in range(self.max_iterations):
                limit_error = self._limit_error(started_perf)
                if limit_error is not None:
                    return self._finish_result(
                        response=last_response,
                        ready=False,
                        answer_type="failed",
                        error_info=limit_error,
                        logger=logger,
                        metadata=metadata,
                        started_perf=started_perf,
                    )

                user = build_user_prompt(prompt, index, self.max_iterations, transcript)
                call_started = time.perf_counter()
                try:
                    model_response = self._call_llm(system, user, model=_model)
                    consecutive_errors = 0
                    logger.add_event(
                        _llm_event(
                            run_id=run_id,
                            parent_run_id=parent_run_id,
                            depth=_depth,
                            call_type="root_llm",
                            model=_model or self.model_name,
                            prompt_index=index,
                            elapsed_ms=_elapsed_ms(call_started),
                            error=None,
                        )
                    )
                except Exception as exc:
                    consecutive_errors += 1
                    provider_error = RuntimeErrorInfo(
                        code="provider_error",
                        message=str(exc),
                        retryable=True,
                    )
                    logger.add_event(
                        _llm_event(
                            run_id=run_id,
                            parent_run_id=parent_run_id,
                            depth=_depth,
                            call_type="root_llm",
                            model=_model or self.model_name,
                            prompt_index=index,
                            elapsed_ms=_elapsed_ms(call_started),
                            error=provider_error,
                        )
                    )
                    if self.max_errors is not None and consecutive_errors >= self.max_errors:
                        return self._finish_result(
                            response=last_response,
                            ready=False,
                            answer_type="failed",
                            error_info=RuntimeErrorInfo(
                                code="max_errors_exceeded",
                                message=f"{consecutive_errors} consecutive errors; last error: {exc}",
                                retryable=True,
                            ),
                            logger=logger,
                            metadata=metadata,
                            started_perf=started_perf,
                        )
                    transcript.append(f"LLM call failed: {exc}")
                    continue

                last_response = model_response
                limit_error = self._limit_error(started_perf)
                if limit_error is not None:
                    return self._finish_result(
                        response=last_response,
                        ready=False,
                        answer_type="failed",
                        error_info=limit_error,
                        logger=logger,
                        metadata=metadata,
                        started_perf=started_perf,
                    )

                code_blocks = find_repl_code_blocks(model_response)
                repl_results: list[REPLResult] = []
                final_answer: Any | None = None

                for code in code_blocks:
                    repl_result = repl.execute_code(code)
                    repl_results.append(repl_result)
                    if repl_result.final_answer is not None:
                        final_answer = repl_result.final_answer
                        break

                iteration = RLMIteration(
                    index=index,
                    model_response=model_response,
                    code_blocks=code_blocks,
                    repl_results=repl_results,
                    final_answer=final_answer,
                )
                logger.add_iteration(iteration)
                logger.add_event(
                    {
                        "event": "iteration_finished",
                        "run_id": run_id,
                        "parent_run_id": parent_run_id,
                        "depth": _depth,
                        "iteration": index,
                        "code_block_count": len(code_blocks),
                        "ready": final_answer is not None,
                    }
                )
                transcript.append(_format_feedback(model_response, repl_results))

                if final_answer is not None:
                    return self._finish_result(
                        response=final_answer,
                        ready=True,
                        answer_type="final",
                        error_info=None,
                        logger=logger,
                        metadata=metadata,
                        started_perf=started_perf,
                    )

            return self._finish_result(
                response=last_response,
                ready=False,
                answer_type="failed",
                error_info=RuntimeErrorInfo(
                    code="max_iterations_exhausted",
                    message=f"RLM reached {self.max_iterations} iterations without a ready answer.",
                    retryable=True,
                ),
                logger=logger,
                metadata=metadata,
                started_perf=started_perf,
            )
        finally:
            self._restore_thread_state(previous_state)

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        system = "You are a plain sub-LLM call inside an RLM runtime. Answer directly and concisely."
        state = self._current_state()
        started = time.perf_counter()
        try:
            response = self._call_llm(system, prompt, model=model)
        except Exception as exc:
            error = RuntimeErrorInfo(code="provider_error", message=str(exc), retryable=True)
            self._log_runtime_event(
                _llm_event(
                    run_id=state.get("run_id"),
                    parent_run_id=None,
                    depth=state.get("depth"),
                    call_type="llm_query",
                    model=model or self.model_name,
                    prompt_index=None,
                    elapsed_ms=_elapsed_ms(started),
                    error=error,
                )
            )
            return _slot_error_message(error)
        self._log_runtime_event(
            _llm_event(
                run_id=state.get("run_id"),
                parent_run_id=None,
                depth=state.get("depth"),
                call_type="llm_query",
                model=model or self.model_name,
                prompt_index=None,
                elapsed_ms=_elapsed_ms(started),
                error=None,
            )
        )
        return response

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self._llm_query(prompts[0], model)]
        system = "You are a plain sub-LLM call inside an RLM runtime. Answer directly and concisely."
        state = self._current_state()
        slots = _run_ordered_concurrently(
            prompts,
            lambda prompt, _index: self._call_llm(system, prompt, model=model),
            max_workers=min(self.batch_max_workers, len(prompts)),
        )
        results: list[str] = []
        for slot in slots:
            self._log_runtime_event(
                _llm_event(
                    run_id=state.get("run_id"),
                    parent_run_id=None,
                    depth=state.get("depth"),
                    call_type="llm_query_batched",
                    model=model or self.model_name,
                    prompt_index=slot.index,
                    elapsed_ms=slot.elapsed_ms,
                    error=slot.error,
                )
            )
            results.append(_slot_error_message(slot.error) if slot.error else slot.value)
        return results

    def _rlm_query(self, prompt: str, model: str | None, depth: int, prompt_index: int | None) -> str:
        if depth >= self.max_depth:
            return self._llm_query(prompt, model)
        state = self._current_state()
        parent_run_id = str(state.get("run_id") or "")
        child = self.completion(
            prompt,
            context={},
            _depth=depth + 1,
            _parent_run_id=parent_run_id,
            _prompt_index=prompt_index,
            _model=model,
        )
        self._add_child_summary(child, prompt_index=prompt_index)
        if child.error_info is not None:
            return _slot_error_message(child.error_info)
        return child.response if isinstance(child.response, str) else str(child.response)

    def _rlm_query_batched(self, prompts: list[str], model: str | None, depth: int) -> list[str]:
        if depth >= self.max_depth:
            return self._llm_query_batched(prompts, model)
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self._rlm_query(prompts[0], model, depth, 0)]
        state = self._current_state()
        parent_run_id = str(state.get("run_id") or "")
        parent_logger = state.get("logger")
        child_results = self._run_child_rlms_batched(
            prompts,
            model=model,
            depth=depth,
            parent_run_id=parent_run_id,
            max_workers=min(self.max_concurrent_subcalls, len(prompts)),
        )
        results: list[str] = []
        for index, child in enumerate(child_results):
            if parent_logger is not None:
                parent_logger.add_child(
                    {
                        "run_id": child.metadata.run_id,
                        "parent_run_id": child.metadata.parent_run_id,
                        "depth": child.metadata.depth,
                        "prompt_index": index,
                        "response": child.response,
                        "ready": child.ready,
                        "error": asdict(child.error_info) if child.error_info else None,
                        "metadata": asdict(child.metadata),
                    }
                )
            if child.error_info is not None:
                results.append(_slot_error_message(child.error_info))
            else:
                results.append(child.response if isinstance(child.response, str) else str(child.response))
        return results

    def _call_llm(self, system: str, user: str, *, model: str | None = None) -> str:
        input_tokens = _estimate_tokens(system) + _estimate_tokens(user)
        if model is None:
            try:
                response = self.llm(system, user)
            except Exception:
                self._record_usage(model, input_tokens=input_tokens, output_tokens=0)
                raise
            self._record_usage(model, input_tokens=input_tokens, output_tokens=_estimate_tokens(response))
            return response
        try:
            response = self.llm(system, user, model)
        except TypeError:
            response = self.llm(system, user)
        except Exception:
            self._record_usage(model, input_tokens=input_tokens, output_tokens=0)
            raise
        self._record_usage(model, input_tokens=input_tokens, output_tokens=_estimate_tokens(response))
        return response

    def _run_child_rlms_batched(
        self,
        prompts: list[str],
        *,
        model: str | None,
        depth: int,
        parent_run_id: str,
        max_workers: int,
    ) -> list[RLMResult]:
        results: list[RLMResult | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.completion,
                    prompt,
                    context={},
                    _depth=depth + 1,
                    _parent_run_id=parent_run_id,
                    _prompt_index=index,
                    _model=model,
                ): index
                for index, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    error = RuntimeErrorInfo(code="provider_error", message=str(exc), retryable=True)
                    metadata = RuntimeMetadata(
                        run_id=_new_run_id(),
                        parent_run_id=parent_run_id,
                        depth=depth + 1,
                        prompt_index=index,
                        model=model or self.model_name,
                    )
                    results[index] = RLMResult(
                        response="",
                        iterations=[],
                        trajectory={"run_metadata": asdict(metadata), "iterations": [], "events": [], "children": []},
                        error=error.code,
                        ready=False,
                        answer_type="failed",
                        usage=self._usage_summary(),
                        metadata=metadata,
                        error_info=error,
                    )
        return [result or _missing_child_result(parent_run_id, depth + 1, index, model or self.model_name) for index, result in enumerate(results)]

    def _set_thread_state(
        self,
        *,
        logger: TrajectoryLogger,
        run_id: str,
        depth: int,
        started_perf: float,
    ) -> dict[str, Any]:
        previous = {
            "logger": getattr(self._thread_state, "logger", None),
            "run_id": getattr(self._thread_state, "run_id", None),
            "depth": getattr(self._thread_state, "depth", None),
            "started_perf": getattr(self._thread_state, "started_perf", None),
        }
        self._thread_state.logger = logger
        self._thread_state.run_id = run_id
        self._thread_state.depth = depth
        self._thread_state.started_perf = started_perf
        return previous

    def _restore_thread_state(self, previous: dict[str, Any]) -> None:
        self._thread_state.logger = previous["logger"]
        self._thread_state.run_id = previous["run_id"]
        self._thread_state.depth = previous["depth"]
        self._thread_state.started_perf = previous["started_perf"]

    def _current_state(self) -> dict[str, Any]:
        return {
            "logger": getattr(self._thread_state, "logger", None),
            "run_id": getattr(self._thread_state, "run_id", None),
            "depth": getattr(self._thread_state, "depth", None),
            "started_perf": getattr(self._thread_state, "started_perf", None),
        }

    def _log_runtime_event(self, event: dict[str, Any]) -> None:
        logger = getattr(self._thread_state, "logger", None)
        if logger is not None:
            logger.add_event(event)

    def _add_child_summary(self, child: RLMResult, *, prompt_index: int | None) -> None:
        logger = getattr(self._thread_state, "logger", None)
        if logger is None:
            return
        logger.add_child(
            {
                "run_id": child.metadata.run_id,
                "parent_run_id": child.metadata.parent_run_id,
                "depth": child.metadata.depth,
                "prompt_index": prompt_index,
                "response": child.response,
                "ready": child.ready,
                "error": asdict(child.error_info) if child.error_info else None,
                "metadata": asdict(child.metadata),
            }
        )

    def _finish_result(
        self,
        *,
        response: Any,
        ready: bool,
        answer_type: str,
        error_info: RuntimeErrorInfo | None,
        logger: TrajectoryLogger,
        metadata: RuntimeMetadata,
        started_perf: float,
    ) -> RLMResult:
        elapsed_ms = _elapsed_ms(started_perf)
        finished = RuntimeMetadata(
            run_id=metadata.run_id,
            parent_run_id=metadata.parent_run_id,
            depth=metadata.depth,
            prompt_index=metadata.prompt_index,
            model=metadata.model,
            started_at=metadata.started_at,
            ended_at=_utc_now(),
            elapsed_ms=elapsed_ms,
        )
        logger.run_metadata = asdict(finished)
        logger.add_event(
            {
                "event": "run_finished",
                "run_id": finished.run_id,
                "parent_run_id": finished.parent_run_id,
                "depth": finished.depth,
                "prompt_index": finished.prompt_index,
                "ready": ready,
                "answer_type": answer_type,
                "elapsed_ms": elapsed_ms,
                "error": asdict(error_info) if error_info else None,
            }
        )
        return RLMResult(
            response=response,
            iterations=logger.iterations,
            trajectory=logger.to_dict(),
            error=error_info.code if error_info else "",
            ready=ready,
            answer_type=answer_type,
            usage=self._usage_summary(),
            metadata=finished,
            error_info=error_info,
        )

    def _limit_error(self, started_perf: float) -> RuntimeErrorInfo | None:
        if self.max_timeout_seconds is not None and (time.perf_counter() - started_perf) > self.max_timeout_seconds:
            return RuntimeErrorInfo(
                code="max_timeout_exceeded",
                message=f"RLM run exceeded {self.max_timeout_seconds:.3f}s timeout.",
                retryable=True,
            )
        if self.max_tokens is not None and self._usage_summary().total_tokens > self.max_tokens:
            return RuntimeErrorInfo(
                code="max_tokens_exceeded",
                message=f"RLM run exceeded {self.max_tokens} token budget.",
                retryable=False,
            )
        if self.max_budget is not None and self._usage_summary().estimated_cost is not None:
            if self._usage_summary().estimated_cost > self.max_budget:
                return RuntimeErrorInfo(
                    code="max_budget_exceeded",
                    message=f"RLM run exceeded ${self.max_budget:.6f} budget.",
                    retryable=False,
                )
        return None

    def _reset_usage(self) -> None:
        with self._usage_lock:
            self._total_calls = 0
            self._input_tokens = 0
            self._output_tokens = 0
            self._estimated_cost: float | None = None

    def _record_usage(self, model: str | None, *, input_tokens: int, output_tokens: int) -> None:
        del model
        with self._usage_lock:
            self._total_calls += 1
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            if self.cost_per_1k_tokens is not None:
                total_tokens = self._input_tokens + self._output_tokens
                self._estimated_cost = (total_tokens / 1000.0) * self.cost_per_1k_tokens

    def _usage_summary(self) -> UsageSummary:
        with self._usage_lock:
            return UsageSummary(
                total_calls=self._total_calls,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                estimated_cost=self._estimated_cost,
            )


def find_repl_code_blocks(text: str) -> list[str]:
    pattern = re.compile(r"```[ \t]*(repl|python)[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    blocks: list[str] = []
    for match in pattern.finditer(text):
        language = match.group(1).lower()
        body = match.group(2).strip()
        if language == "repl":
            blocks.append(body)
            continue
        lines = body.splitlines()
        if lines and lines[0].strip().lower() in {"repl", "# repl"}:
            blocks.append("\n".join(lines[1:]).strip())
    return blocks


def _format_feedback(model_response: str, repl_results: list[REPLResult]) -> str:
    parts = [f"Model response:\n{model_response}"]
    if not repl_results:
        parts.append(
            "No executable REPL block was found. "
            "Write Python code inside an exact ```repl``` fenced block and set answer['ready'] = True."
        )
    for index, result in enumerate(repl_results, start=1):
        if result.stdout:
            parts.append(f"REPL #{index} stdout:\n{result.stdout.strip()}")
        if result.stderr:
            parts.append(f"REPL #{index} stderr:\n{result.stderr.strip()}")
    return "\n\n".join(parts)


def _run_ordered_concurrently(
    items: list[str],
    fn: Callable[[str, int], str],
    *,
    max_workers: int,
) -> list[_ConcurrentResult]:
    results: list[_ConcurrentResult | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one_concurrent, fn, item, index): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                results[index] = _ConcurrentResult(
                    index=index,
                    error=RuntimeErrorInfo(code="provider_error", message=str(exc), retryable=True),
                    elapsed_ms=0,
                )
    return [result or _ConcurrentResult(index=index) for index, result in enumerate(results)]


def _run_one_concurrent(fn: Callable[[str, int], str], item: str, index: int) -> _ConcurrentResult:
    started = time.perf_counter()
    try:
        return _ConcurrentResult(index=index, value=fn(item, index), elapsed_ms=_elapsed_ms(started))
    except Exception as exc:
        return _ConcurrentResult(
            index=index,
            error=RuntimeErrorInfo(code="provider_error", message=str(exc), retryable=True),
            elapsed_ms=_elapsed_ms(started),
        )


def _llm_event(
    *,
    run_id: str | None,
    parent_run_id: str | None,
    depth: int | None,
    call_type: str,
    model: str | None,
    prompt_index: int | None,
    elapsed_ms: int,
    error: RuntimeErrorInfo | None,
) -> dict[str, Any]:
    return {
        "event": "llm_call_finished",
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "depth": depth,
        "call_type": call_type,
        "model": model,
        "prompt_index": prompt_index,
        "elapsed_ms": elapsed_ms,
        "error": asdict(error) if error else None,
    }


def _slot_error_message(error: RuntimeErrorInfo | None) -> str:
    if error is None:
        return ""
    return f"Error: {error.code}: {error.message}"


def _new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:16]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(started_perf: float) -> int:
    return int((time.perf_counter() - started_perf) * 1000)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _missing_child_result(parent_run_id: str, depth: int, prompt_index: int, model: str | None) -> RLMResult:
    error = RuntimeErrorInfo(code="missing_child_result", message="Child RLM did not return a result.", retryable=True)
    metadata = RuntimeMetadata(
        run_id=_new_run_id(),
        parent_run_id=parent_run_id,
        depth=depth,
        prompt_index=prompt_index,
        model=model,
    )
    return RLMResult(
        response="",
        iterations=[],
        trajectory={"run_metadata": asdict(metadata), "iterations": [], "events": [], "children": []},
        error=error.code,
        ready=False,
        answer_type="failed",
        usage=UsageSummary(),
        metadata=metadata,
        error_info=error,
    )
