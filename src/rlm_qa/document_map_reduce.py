from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .citation_annotations import prepare_stream_refs
from .contracts import Chunk, DocProfile, Evidence, EvidenceRef, NoEvidenceError, Page, RLMQAError
from .document_adapter import (
    DocumentRLMAnswer,
    _evidence_refs,
    _expanded_evidence_text,
    _metadata_facts,
    _metadata_facts_text,
    _parse_document_response,
)
from .document_tools import make_multi_document_context
from .evidence_quotes import normalize_evidence_refs
from .llm import ChatRequest, LLMClient
from .map_scheduler import run_followup_pipeline, run_parallel_with_control


@dataclass(frozen=True)
class TriageResult:
    pack_id: str
    relevance_score: float
    has_answer: bool
    evidence_ids: list[str] = field(default_factory=list)
    short_reason: str = ""
    error: str = ""


@dataclass(frozen=True)
class CoveragePlan:
    coverage: str
    answer_shape: str
    reason: str = ""
    fallback_used: bool = False


@dataclass(frozen=True)
class DeepAnswerResult:
    pack_id: str
    file_name: str
    answer: str
    source_id: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    metadata_refs: list[str] = field(default_factory=list)
    confidence: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class OnePassMapResult:
    pack_id: str
    file_name: str
    source_id: str
    relevance_score: float
    has_answer: bool
    answer: str
    claims: list[dict[str, Any]] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    metadata_refs: list[str] = field(default_factory=list)
    confidence: float = 0.0
    short_reason: str = ""
    error: str = ""


@dataclass(frozen=True)
class SourceFallback:
    source_id: str
    answer: str
    evidence_ids: list[str] = field(default_factory=list)


class DocumentMapReduceRunner:
    def __init__(
        self,
        *,
        llm: LLMClient,
        subagent_model: str = "deepseek-v4-flash",
        max_workers: int = 4,
        max_deep_packs: int = 8,
        map_execution_mode: str = "staged",
        reduce_context_mode: str = "expanded",
        fallback_evidence_per_pack: int = 5,
        run_id: str = "",
        max_timeout_seconds: float | None = None,
        max_tokens: int | None = None,
        max_errors: int | None = None,
        max_budget: float | None = None,
        cost_per_1k_tokens: float | None = None,
        on_final_delta: Callable[[str], None] | None = None,
        on_stream_evidence: Callable[[list[EvidenceRef]], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        stream_trace: dict[str, Any] | None = None,
        coverage_plan: CoveragePlan | None = None,
    ) -> None:
        self.llm = llm
        self.subagent_model = subagent_model
        self.max_workers = max(1, int(max_workers))
        self.max_deep_packs = max(1, int(max_deep_packs))
        self.map_execution_mode = (
            map_execution_mode
            if map_execution_mode in {"staged", "pipelined", "one_pass"}
            else "staged"
        )
        self.reduce_context_mode = (
            reduce_context_mode if reduce_context_mode in {"expanded", "compact"} else "expanded"
        )
        self.fallback_evidence_per_pack = max(1, int(fallback_evidence_per_pack))
        self.run_id = run_id
        self.max_timeout_seconds = max_timeout_seconds
        self.max_tokens = max_tokens
        self.max_errors = max_errors
        self.max_budget = max_budget
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.on_final_delta = on_final_delta
        self.on_stream_evidence = on_stream_evidence
        self.cancelled = cancelled or (lambda: False)
        self.stream_trace = stream_trace if stream_trace is not None else {}
        self.coverage_plan = coverage_plan

    def answer_question_many(
        self,
        question: str,
        profiles: list[DocProfile],
        pages: list[Page],
        chunks: list[Chunk],
        evidences: list[Evidence],
    ) -> DocumentRLMAnswer:
        context = make_multi_document_context(profiles, pages, chunks, evidences, question=question)
        packs = list(context.get("packs") or [])
        if not packs:
            raise RLMQAError("文档问答流程本次没有生成可用答案，请重试或换一种问法。")

        planner_parallelized = False
        one_pass_results: list[OnePassMapResult] = []
        if self.map_execution_mode == "one_pass":
            if self.coverage_plan is not None:
                plan = self.coverage_plan
                one_pass_results = _run_parallel(
                    packs,
                    lambda pack: self._one_pass_map(question, pack),
                    max_workers=min(self.max_workers, len(packs)),
                )
            elif len(profiles) > 1:
                parallel_result = run_parallel_with_control(
                    lambda: self._plan_coverage(question, profiles),
                    packs,
                    lambda pack: self._one_pass_map(question, pack),
                    max_workers=min(self.max_workers, len(packs) + 1),
                )
                plan = parallel_result.control_result
                one_pass_results = parallel_result.item_results
                planner_parallelized = True
            else:
                plan = CoveragePlan("relevant_subset", "synthesis", "single document")
                one_pass_results = _run_parallel(
                    packs,
                    lambda pack: self._one_pass_map(question, pack),
                    max_workers=min(self.max_workers, len(packs)),
                )
        else:
            plan = (
                self.coverage_plan
                or (
                    self._plan_coverage(question, profiles)
                    if len(profiles) > 1
                    else CoveragePlan("relevant_subset", "synthesis", "single document")
                )
            )
        overview_mode = plan.coverage == "all_selected"
        pipeline_used = False
        pipeline_fallback_reason = ""
        pipelined_deep_by_pack: dict[str, DeepAnswerResult] = {}
        one_pass_repair_count = 0
        if self.map_execution_mode == "one_pass":
            failed_pack_ids = {result.pack_id for result in one_pass_results if result.error}
            failed_packs = [
                pack for pack in packs if str(pack["pack_id"]) in failed_pack_ids
            ]
            triage_results = (
                self._triage_packs(question, failed_packs)
                if failed_packs and not overview_mode
                else []
            )
        elif overview_mode:
            triage_results: list[TriageResult] = []
        elif self.map_execution_mode == "pipelined" and len(packs) <= self.max_deep_packs:
            triage_results, pipelined_deep_by_pack = self._triage_deep_pipeline(question, packs)
            pipeline_used = True
        else:
            triage_results = self._triage_packs(question, packs)
            if self.map_execution_mode == "pipelined":
                pipeline_fallback_reason = "pack_count_exceeds_max_deep_packs"

        selection_triage = list(triage_results)
        if self.map_execution_mode == "one_pass":
            selection_triage.extend(
                _one_pass_triage(result) for result in one_pass_results if not result.error
            )
        selected = self._select_packs(packs, selection_triage, plan)
        if not selected:
            raise NoEvidenceError("文档问答流程本次没有生成可用答案，请重试或换一种问法。")
        triage_by_pack = {result.pack_id: result for result in selection_triage}
        required_source_ids = [profile.source_id for profile in profiles] if plan.coverage == "all_selected" else []
        if self.map_execution_mode == "one_pass":
            one_pass_deep_by_pack = {
                result.pack_id: _one_pass_deep_result(
                    result,
                    evidence_cap_per_pack=(
                        self.fallback_evidence_per_pack if overview_mode else None
                    ),
                )
                for result in one_pass_results
                if not result.error and result.answer
            }
            repair_packs = [
                pack for pack in selected if str(pack["pack_id"]) not in one_pass_deep_by_pack
            ]
            if repair_packs:
                one_pass_repair_count = len(repair_packs)
                repaired = self._deep_answer_packs(
                    question,
                    repair_packs,
                    triage_by_pack,
                    evidence_cap_per_pack=self.fallback_evidence_per_pack if overview_mode else None,
                )
                for result in repaired:
                    one_pass_deep_by_pack[result.pack_id] = result
            deep_results = [
                one_pass_deep_by_pack[str(pack["pack_id"])]
                for pack in selected
                if str(pack["pack_id"]) in one_pass_deep_by_pack
            ]
        elif pipeline_used:
            missing_packs = [
                pack for pack in selected if str(pack["pack_id"]) not in pipelined_deep_by_pack
            ]
            if missing_packs:
                for result in self._deep_answer_packs(question, missing_packs, triage_by_pack):
                    pipelined_deep_by_pack[result.pack_id] = result
            deep_results = [
                pipelined_deep_by_pack[str(pack["pack_id"])]
                for pack in selected
                if str(pack["pack_id"]) in pipelined_deep_by_pack
            ]
        else:
            deep_results = self._deep_answer_packs(
                question,
                selected,
                triage_by_pack,
                evidence_cap_per_pack=self.fallback_evidence_per_pack if overview_mode else None,
            )
        deep_repair_attempted = one_pass_repair_count > 0
        if required_source_ids:
            initial_source_ids = {result.source_id for result in deep_results if result.source_id}
            repair_source_ids = [source_id for source_id in required_source_ids if source_id not in initial_source_ids]
            if repair_source_ids:
                repair_packs = [pack for pack in selected if str(pack.get("source_id") or "") in repair_source_ids]
                if repair_packs:
                    deep_repair_attempted = True
                    deep_results.extend(
                        self._deep_answer_packs(
                            question,
                            repair_packs,
                            triage_by_pack,
                            evidence_cap_per_pack=self.fallback_evidence_per_pack,
                        )
                    )
        if not deep_results:
            raise NoEvidenceError("文档问答流程本次没有生成可用答案，请重试或换一种问法。")

        deep_result_source_ids = _unique(result.source_id for result in deep_results if result.source_id)
        deep_missing_source_ids = [source_id for source_id in required_source_ids if source_id not in deep_result_source_ids]

        evidence_ids = _unique(
            evidence_id
            for result in deep_results
            for evidence_id in result.evidence_ids
        )
        metadata_refs = _unique(
            [f"doc_profile:{profile.source_id}:page_count" for profile in profiles]
            + [ref for result in deep_results for ref in result.metadata_refs]
        )
        quote_context = "\n\n".join(result.answer for result in deep_results if result.answer)
        refs = prepare_stream_refs(
            normalize_evidence_refs(
                _evidence_refs(evidence_ids, evidences, profiles),
                chunks,
                pages,
                question=question,
                answer=quote_context,
            )
        )
        metadata_facts = _metadata_facts(metadata_refs, context)
        if self.on_stream_evidence is not None and refs:
            self.on_stream_evidence(refs)
        final_covered_source_ids: list[str] = []
        missing_source_ids: list[str] = []
        finalizer_repair_attempted = False
        finalizer_local_fallback_used = False
        if plan.coverage == "all_selected":
            (
                final_answer,
                final_covered_source_ids,
                missing_source_ids,
                finalizer_repair_attempted,
                finalizer_local_fallback_used,
            ) = self._finalize_all_selected(
                question=question,
                profiles=profiles,
                deep_results=deep_results,
                refs=refs,
                chunks=chunks,
                metadata_facts=metadata_facts,
            )
        else:
            final_answer = self._finalize_answer(
                question=question,
                deep_results=deep_results,
                refs=refs,
                chunks=chunks,
                metadata_facts=metadata_facts,
            )
        runtime_status = "rlm_success" if refs or metadata_facts else "rlm_degraded_recovered"
        return DocumentRLMAnswer(
            answer=final_answer,
            evidences=refs,
            metadata_refs=[fact["ref"] for fact in metadata_facts],
            trajectory={
                "strategy": "map_reduce",
                "coverage_contract": plan.coverage,
                "answer_shape": plan.answer_shape,
                "coverage_plan_reason": plan.reason,
                "coverage_plan_fallback": plan.fallback_used,
                "coverage_plan_injected": self.coverage_plan is not None,
                "triage_skipped": overview_mode or (
                    self.map_execution_mode == "one_pass" and not triage_results
                ),
                "triage_skip_reason": (
                    "coverage_all_selected"
                    if overview_mode
                    else "one_pass_map"
                    if self.map_execution_mode == "one_pass" and not triage_results
                    else ""
                ),
                "map_execution_mode": self.map_execution_mode,
                "reduce_context_mode": self.reduce_context_mode,
                "planner_parallelized": planner_parallelized,
                "one_pass_count": len(one_pass_results),
                "one_pass_repair_count": one_pass_repair_count,
                "one_pass": [result.__dict__ for result in one_pass_results],
                "pipeline_used": pipeline_used,
                "pipeline_fallback_reason": pipeline_fallback_reason,
                "required_source_ids": required_source_ids,
                "selected_pack_ids_by_source": _pack_ids_by_source(selected),
                "deep_result_source_ids": deep_result_source_ids,
                "deep_missing_source_ids": deep_missing_source_ids,
                "deep_repair_attempted": deep_repair_attempted,
                "final_covered_source_ids": final_covered_source_ids,
                "missing_source_ids": missing_source_ids,
                "finalizer_repair_attempted": finalizer_repair_attempted,
                "finalizer_local_fallback_used": finalizer_local_fallback_used,
                "triage_count": len(triage_results),
                "deep_count": len(deep_results),
                "selected_pack_ids": [pack["pack_id"] for pack in selected],
                "triage": [result.__dict__ for result in triage_results],
                "deep": [result.__dict__ for result in deep_results],
                "runtime_limits": self._runtime_limits(),
            },
            runtime_status=runtime_status,
        )

    def _plan_coverage(self, question: str, profiles: list[DocProfile]) -> CoveragePlan:
        documents = [
            {
                "source_id": profile.source_id,
                "file_name": profile.file_name,
                "page_count": profile.page_count,
            }
            for profile in profiles
        ]
        user = (
            f"Question:\n{question}\n\n"
            f"Selected document metadata:\n{json.dumps(documents, ensure_ascii=False)}\n\n"
            "Return strict JSON: "
            '{"coverage": "all_selected | relevant_subset", '
            '"answer_shape": "per_document | synthesis", "reason": "..."}. '
            "Choose all_selected when the answer must account for every selected document. "
            "Choose relevant_subset only when the question can be answered from a subset."
        )
        for attempt in range(2):
            try:
                response = self.llm.chat(
                    ChatRequest(
                        system="MAP_REDUCE_PLAN",
                        user=user,
                        model=self.subagent_model,
                        run_id=self.run_id,
                        purpose="subagent",
                    )
                ).answer
            except Exception:
                continue
            plan = _coverage_plan_from_response(response)
            if plan is not None:
                return plan
        return _safe_multi_document_plan()

    def _one_pass_map(self, question: str, pack: dict[str, Any]) -> OnePassMapResult:
        pack_id = str(pack["pack_id"])
        file_name = str(pack.get("file_name") or "")
        source_id = str(pack.get("source_id") or "")
        user = (
            f"Question:\n{question}\n\n"
            f"Pack metadata:\n{json.dumps(_pack_without_text(pack), ensure_ascii=False)}\n\n"
            f"Pack text:\n{pack.get('text', '')}\n\n"
            "Analyze relevance and answer from this pack in one pass. Return strict JSON: "
            '{"pack_id": "...", "relevance_score": 0.0, "has_answer": false, '
            '"answer": "", "claims": [{"text": "...", "evidence_ids": ["actual evidence id"]}], '
            '"evidence_ids": [], "metadata_refs": [], "confidence": 0.0, '
            '"short_reason": "..."}. '
            "Use only evidence ids shown in the pack. Do not cap evidence_ids at 3."
        )
        try:
            response = self.llm.chat(
                ChatRequest(
                    system="MAP_REDUCE_ONE_PASS",
                    user=user,
                    model=self.subagent_model,
                    run_id=self.run_id,
                    purpose="subagent",
                )
            ).answer
            data = _json_object(response)
            if not data:
                raise ValueError("one-pass map did not return a JSON object")
            allowed_ids = _string_list(pack.get("evidence_ids"))
            claims = _validated_claims(data.get("claims"), allowed_ids)
            evidence_ids = _valid_ids(_string_list(data.get("evidence_ids")), allowed_ids)
            evidence_ids = _unique(
                evidence_ids
                + [
                    evidence_id
                    for claim in claims
                    for evidence_id in claim["evidence_ids"]
                ]
            )
            return OnePassMapResult(
                pack_id=pack_id,
                file_name=file_name,
                source_id=source_id,
                relevance_score=float(data.get("relevance_score") or 0.0),
                has_answer=bool(data.get("has_answer")),
                answer=str(data.get("answer") or ""),
                claims=claims,
                evidence_ids=evidence_ids,
                metadata_refs=_string_list(data.get("metadata_refs")),
                confidence=float(data.get("confidence") or 0.0),
                short_reason=str(data.get("short_reason") or ""),
            )
        except Exception as exc:
            return OnePassMapResult(
                pack_id=pack_id,
                file_name=file_name,
                source_id=source_id,
                relevance_score=0.0,
                has_answer=False,
                answer="",
                error=_safe_llm_error_code(exc),
            )

    def _triage_packs(self, question: str, packs: list[dict[str, Any]]) -> list[TriageResult]:
        return _run_parallel(
            packs,
            lambda pack: self._triage_pack(question, pack),
            max_workers=min(self.max_workers, len(packs)),
        )

    def _triage_pack(self, question: str, pack: dict[str, Any]) -> TriageResult:
        user = (
            f"Question:\n{question}\n\n"
            f"Pack metadata:\n{json.dumps(_pack_without_text(pack), ensure_ascii=False)}\n\n"
            f"Pack text:\n{pack.get('text', '')}\n\n"
            "Return strict JSON: "
            '{"pack_id": "...", "relevance_score": 0.0, "has_answer": false, '
            '"evidence_ids": [], "short_reason": "..."}'
        )
        try:
            response = self.llm.chat(
                ChatRequest(
                    system="MAP_REDUCE_TRIAGE",
                    user=user,
                    model=self.subagent_model,
                    run_id=self.run_id,
                    purpose="subagent",
                )
            ).answer
            data = _json_object(response)
            return TriageResult(
                pack_id=str(data.get("pack_id") or pack["pack_id"]),
                relevance_score=float(data.get("relevance_score") or 0.0),
                has_answer=bool(data.get("has_answer")),
                evidence_ids=_string_list(data.get("evidence_ids")),
                short_reason=str(data.get("short_reason") or ""),
            )
        except Exception as exc:
            return TriageResult(
                pack_id=str(pack["pack_id"]),
                relevance_score=0.0,
                has_answer=False,
                error=_safe_llm_error_code(exc),
            )

    def _triage_deep_pipeline(
        self,
        question: str,
        packs: list[dict[str, Any]],
    ) -> tuple[list[TriageResult], dict[str, DeepAnswerResult]]:
        result = run_followup_pipeline(
            packs,
            lambda pack: self._triage_pack(question, pack),
            lambda triage: triage.has_answer or triage.relevance_score > 0,
            lambda pack, triage: self._deep_answer_pack(
                question,
                pack,
                {triage.pack_id: triage},
            ),
            max_workers=min(self.max_workers, len(packs)),
        )
        deep_by_pack = {
            deep.pack_id: deep
            for deep in result.followup_results
            if deep is not None and _usable_deep_result(deep)
        }
        return result.first_results, deep_by_pack

    def _select_packs(
        self,
        packs: list[dict[str, Any]],
        triage_results: list[TriageResult],
        plan: CoveragePlan,
    ) -> list[dict[str, Any]]:
        if plan.coverage == "all_selected":
            return list(packs)

        packs_by_id = {str(pack["pack_id"]): pack for pack in packs}
        ranked = sorted(
            triage_results,
            key=lambda result: (not result.has_answer, -result.relevance_score, result.pack_id),
        )
        selected = [
            packs_by_id[result.pack_id]
            for result in ranked
            if result.pack_id in packs_by_id and (result.has_answer or result.relevance_score > 0)
        ]
        if (
            not selected
            and ranked
            and all(result.error for result in ranked)
            and ranked[0].pack_id in packs_by_id
        ):
            selected = [packs_by_id[ranked[0].pack_id]]
        return selected[: self.max_deep_packs]

    def _deep_answer_packs(
        self,
        question: str,
        packs: list[dict[str, Any]],
        triage_by_pack: dict[str, TriageResult],
        *,
        evidence_cap_per_pack: int | None = None,
    ) -> list[DeepAnswerResult]:
        return [
            result
            for result in _run_parallel(
                packs,
                lambda pack: self._deep_answer_pack(
                    question,
                    pack,
                    triage_by_pack,
                    evidence_cap_per_pack=evidence_cap_per_pack,
                ),
                max_workers=min(self.max_workers, len(packs)),
            )
            if _usable_deep_result(result)
        ]

    def _deep_answer_pack(
        self,
        question: str,
        pack: dict[str, Any],
        triage_by_pack: dict[str, TriageResult],
        *,
        evidence_cap_per_pack: int | None = None,
    ) -> DeepAnswerResult:
        pack_id = str(pack["pack_id"])
        user = (
            f"Question:\n{question}\n\n"
            f"Pack:\n{pack.get('text', '')}\n\n"
            "Return strict JSON: "
            '{"answer": "...", "claims": [{"text": "...", '
            '"evidence_ids": ["actual evidence id"]}], '
            '"evidence_ids": ["actual evidence id"], '
            '"metadata_refs": [], "confidence": 0.0}. '
            "Use only evidence ids shown in the pack. Do not cap evidence_ids at 3."
        )
        try:
            response = self.llm.chat(
                ChatRequest(
                    system="MAP_REDUCE_DEEP_ANSWER",
                    user=user,
                    model=self.subagent_model,
                    run_id=self.run_id,
                    purpose="subagent",
                )
            ).answer
            data = _json_object(response)
            if not data:
                raise ValueError("deep answer did not return a JSON object")
            answer, evidence_ids, metadata_refs = _parse_document_response(response)
            allowed_ids = _string_list(pack.get("evidence_ids"))
            claims = _validated_claims(data.get("claims"), allowed_ids)
            claim_evidence_ids = [
                evidence_id
                for claim in claims
                for evidence_id in claim["evidence_ids"]
            ]
            verified_ids = _valid_ids(
                _unique(evidence_ids + claim_evidence_ids),
                allowed_ids,
            )
            if not verified_ids:
                triage = triage_by_pack.get(pack_id)
                verified_ids = _valid_ids(triage.evidence_ids if triage else [], allowed_ids)
            if not verified_ids:
                verified_ids = allowed_ids[: self.fallback_evidence_per_pack]
            if evidence_cap_per_pack is not None:
                verified_ids = verified_ids[: max(1, int(evidence_cap_per_pack))]
            claims = [
                {
                    "text": claim["text"],
                    "evidence_ids": _valid_ids(claim["evidence_ids"], verified_ids),
                }
                for claim in claims
            ]
            return DeepAnswerResult(
                pack_id=pack_id,
                file_name=str(pack.get("file_name") or ""),
                answer=answer,
                source_id=str(pack.get("source_id") or ""),
                evidence_ids=verified_ids,
                claims=claims,
                metadata_refs=metadata_refs,
                confidence=float(data.get("confidence") or 0.0),
            )
        except Exception as exc:
            return DeepAnswerResult(
                pack_id=pack_id,
                file_name=str(pack.get("file_name") or ""),
                answer="",
                source_id=str(pack.get("source_id") or ""),
                error=_safe_llm_error_code(exc),
            )

    def _finalize_answer(
        self,
        *,
        question: str,
        deep_results: list[DeepAnswerResult],
        refs: list[EvidenceRef],
        chunks: list[Chunk],
        metadata_facts: list[dict[str, Any]],
    ) -> str:
        if self.reduce_context_mode == "compact":
            evidence_text = _compact_evidence_text(refs)
            drafts = _compact_deep_result_drafts(deep_results, refs)
        else:
            evidence_text = _expanded_evidence_text(refs, chunks) if refs else "(none)"
            drafts = "\n\n".join(
                f"[file {result.file_name} | pack {result.pack_id} | confidence {result.confidence}]\n{result.answer}"
                for result in deep_results
            )
        metadata_text = _metadata_facts_text(metadata_facts) if metadata_facts else "(none)"
        user = (
            f"Question: {question}\n\n"
            f"Pack-level answers:\n{drafts}\n\n"
            f"Verified evidence excerpts:\n{evidence_text}\n\n"
            f"Verified metadata facts:\n{metadata_text}\n\n"
            "Write the final answer in Chinese. Do not mention internal field names. "
            "For each document-backed claim, add the matching [[ref:ref_NNN]] marker immediately after the claim. "
            "Use only reference markers shown in the citation guide. "
            f"Citation guide:\n{_citation_guide(refs)}"
        )
        request = ChatRequest(
            system="MAP_REDUCE_FINALIZER",
            user=user,
            model=self.subagent_model,
            run_id=self.run_id,
            purpose="finalizer",
        )
        if self.on_final_delta is not None:
            self._mark_finalizer_started()
            return self.llm.chat_stream(request, self.on_final_delta, self.cancelled).answer
        return self.llm.chat(request).answer

    def _finalize_all_selected(
        self,
        *,
        question: str,
        profiles: list[DocProfile],
        deep_results: list[DeepAnswerResult],
        refs: list[EvidenceRef],
        chunks: list[Chunk],
        metadata_facts: list[dict[str, Any]],
    ) -> tuple[str, list[str], list[str], bool, bool]:
        if self.reduce_context_mode == "compact":
            evidence_text = _compact_evidence_text(refs)
            drafts = _compact_deep_result_drafts(deep_results, refs)
        else:
            evidence_text = _expanded_evidence_text(refs, chunks) if refs else "(none)"
            drafts = _deep_result_drafts(deep_results)
        metadata_text = _metadata_facts_text(metadata_facts) if metadata_facts else "(none)"
        required_documents = [
            {"source_id": profile.source_id, "file_name": profile.file_name}
            for profile in profiles
        ]
        request = ChatRequest(
            system="MAP_REDUCE_FINAL_STREAM",
            user=(
                f"Question: {question}\n\n"
                "Required documents in exact display order:\n"
                f"{json.dumps(required_documents, ensure_ascii=False)}\n\n"
                f"Pack-level answers with their allowed evidence ids:\n{drafts}\n\n"
                f"Verified evidence excerpts:\n{evidence_text}\n\n"
                f"Verified metadata facts:\n{metadata_text}\n\n"
                "Write the final answer in Chinese. Include exactly one Markdown section for every required document, "
                "using the exact heading `## <file_name>`. Keep the required order and do not omit a document. "
                "Do not expose source ids, evidence ids, or other internal field names. "
                "For each document-backed claim, add the matching [[ref:ref_NNN]] marker immediately after the claim. "
                "Use only reference markers shown in the citation guide. "
                f"Citation guide:\n{_citation_guide(refs)}"
            ),
            model=self.subagent_model,
            run_id=self.run_id,
            purpose="finalizer",
        )
        if self.on_final_delta is not None:
            self._mark_finalizer_started()
            response = self.llm.chat_stream(request, self.on_final_delta, self.cancelled).answer
        else:
            try:
                response = self.llm.chat(request).answer
            except Exception:
                response = ""
        covered_source_ids = _covered_source_ids(response, profiles)
        missing_profiles = [profile for profile in profiles if profile.source_id not in covered_source_ids]
        local_fallback = _render_missing_source_sections(missing_profiles, deep_results, refs)
        if local_fallback:
            separator = "\n\n" if response and not response.endswith("\n\n") else ""
            suffix = f"{separator}{local_fallback}"
            if self.on_final_delta is not None:
                self.on_final_delta(suffix)
            response += suffix
        covered_source_ids = _covered_source_ids(response, profiles)
        missing_source_ids = [
            profile.source_id for profile in profiles if profile.source_id not in covered_source_ids
        ]
        return response, covered_source_ids, missing_source_ids, False, bool(local_fallback)

    def _mark_finalizer_started(self) -> None:
        if "finalizer_started_at" in self.stream_trace:
            return
        self.stream_trace["finalizer_started_at"] = datetime.now(timezone.utc).isoformat()
        self.stream_trace["_finalizer_started_monotonic"] = time.perf_counter()

    def _runtime_limits(self) -> dict[str, Any]:
        return {
            "max_timeout_seconds": self.max_timeout_seconds,
            "max_tokens": self.max_tokens,
            "max_errors": self.max_errors,
            "max_budget": self.max_budget,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "max_workers": self.max_workers,
            "max_deep_packs": self.max_deep_packs,
            "map_execution_mode": self.map_execution_mode,
            "reduce_context_mode": self.reduce_context_mode,
            "fallback_evidence_per_pack": self.fallback_evidence_per_pack,
        }


def _run_parallel(items, fn, *, max_workers: int):
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fn, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [result for result in results if result is not None]


def _usable_deep_result(result: DeepAnswerResult) -> bool:
    return bool(result.answer or result.evidence_ids or result.metadata_refs)


def _pack_without_text(pack: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in pack.items() if key != "text"}


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _safe_llm_error_code(exc: Exception) -> str:
    del exc
    return "LLM_CALL_FAILED"


def _coverage_plan_from_response(text: str) -> CoveragePlan | None:
    data = _json_object(text)
    coverage = str(data.get("coverage") or "")
    answer_shape = str(data.get("answer_shape") or "")
    if coverage not in {"all_selected", "relevant_subset"}:
        return None
    if answer_shape not in {"per_document", "synthesis"}:
        return None
    return CoveragePlan(
        coverage=coverage,
        answer_shape=answer_shape,
        reason=str(data.get("reason") or ""),
    )


def _safe_multi_document_plan() -> CoveragePlan:
    return CoveragePlan(
        coverage="all_selected",
        answer_shape="per_document",
        reason="planner fallback",
        fallback_used=True,
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _validated_claims(value: Any, allowed_ids: list[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    claims: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        claims.append(
            {
                "text": text,
                "evidence_ids": _valid_ids(
                    _string_list(item.get("evidence_ids")),
                    allowed_ids,
                ),
            }
        )
    return claims


def _one_pass_triage(result: OnePassMapResult) -> TriageResult:
    return TriageResult(
        pack_id=result.pack_id,
        relevance_score=result.relevance_score,
        has_answer=result.has_answer,
        evidence_ids=result.evidence_ids,
        short_reason=result.short_reason,
        error=result.error,
    )


def _one_pass_deep_result(
    result: OnePassMapResult,
    *,
    evidence_cap_per_pack: int | None = None,
) -> DeepAnswerResult:
    evidence_ids = list(result.evidence_ids)
    if evidence_cap_per_pack is not None:
        evidence_ids = evidence_ids[: max(1, int(evidence_cap_per_pack))]
    claims = [
        {
            "text": claim["text"],
            "evidence_ids": _valid_ids(claim["evidence_ids"], evidence_ids),
        }
        for claim in result.claims
    ]
    return DeepAnswerResult(
        pack_id=result.pack_id,
        file_name=result.file_name,
        answer=result.answer,
        source_id=result.source_id,
        evidence_ids=evidence_ids,
        claims=claims,
        metadata_refs=result.metadata_refs,
        confidence=result.confidence,
        error=result.error,
    )


def _deep_result_drafts(results: list[DeepAnswerResult]) -> str:
    return "\n\n".join(
        f"[source {result.source_id} | file {result.file_name} | pack {result.pack_id} | confidence {result.confidence}]\n"
        f"Allowed evidence ids: {json.dumps(result.evidence_ids, ensure_ascii=False)}\n"
        f"{result.answer}"
        for result in results
        if result.answer
    ) or "(none)"


def _compact_deep_result_drafts(
    results: list[DeepAnswerResult],
    refs: list[EvidenceRef],
) -> str:
    ref_id_by_evidence = {
        ref.evidence_id: ref.ref_id
        for ref in refs
        if ref.evidence_id and ref.ref_id
    }
    items: list[dict[str, Any]] = []
    for result in results:
        claims = result.claims or (
            [{"text": result.answer, "evidence_ids": result.evidence_ids}]
            if result.answer
            else []
        )
        normalized_claims = []
        for claim in claims:
            evidence_ids = _string_list(claim.get("evidence_ids"))
            normalized_claims.append(
                {
                    "text": str(claim.get("text") or ""),
                    "evidence_ids": evidence_ids,
                    "reference_markers": [
                        f"[[ref:{ref_id_by_evidence[evidence_id]}]]"
                        for evidence_id in evidence_ids
                        if evidence_id in ref_id_by_evidence
                    ],
                }
            )
        items.append(
            {
                "source_id": result.source_id,
                "file_name": result.file_name,
                "pack_id": result.pack_id,
                "confidence": result.confidence,
                "answer": result.answer,
                "claims": normalized_claims,
            }
        )
    return json.dumps(items, ensure_ascii=False, indent=2) if items else "(none)"


def _compact_evidence_text(refs: list[EvidenceRef]) -> str:
    return "\n\n".join(
        f"[[ref:{ref.ref_id}]] | evidence {ref.evidence_id} | {ref.file_name} | "
        f"page {ref.page_number} | quote_source {ref.quote_source or 'unspecified'}\n"
        f"Quote: {ref.quote}"
        for ref in refs
        if ref.ref_id
    ) or "(none)"


def _fallback_item_for_source(
    source_id: str,
    deep_results: list[DeepAnswerResult],
) -> SourceFallback | None:
    source_results = [
        result for result in deep_results if result.source_id == source_id and result.answer.strip()
    ]
    if not source_results:
        return None
    answers = _unique(result.answer.strip() for result in source_results)
    evidence_ids = _unique(
        evidence_id for result in source_results for evidence_id in result.evidence_ids
    )
    return SourceFallback(
        source_id=source_id,
        answer="\n\n".join(answers),
        evidence_ids=evidence_ids,
    )


def _render_missing_source_sections(
    profiles: list[DocProfile],
    deep_results: list[DeepAnswerResult],
    refs: list[EvidenceRef],
) -> str:
    sections: list[str] = []
    for profile in profiles:
        item = _fallback_item_for_source(profile.source_id, deep_results)
        if item is None:
            continue
        body = _append_reference_markers(item.answer, item.evidence_ids, refs)
        sections.append(f"## {profile.file_name}\n{body}")
    return "\n\n".join(sections)


def _covered_source_ids(answer: str, profiles: list[DocProfile]) -> list[str]:
    return [
        profile.source_id
        for profile in profiles
        if re.search(rf"(?m)^##\s+{re.escape(profile.file_name)}\s*$", answer)
    ]


def _append_reference_markers(
    answer: str,
    evidence_ids: list[str],
    refs: list[EvidenceRef],
) -> str:
    ref_by_evidence_id = {
        ref.evidence_id: ref.ref_id
        for ref in refs
        if ref.evidence_id and ref.ref_id
    }
    markers = [
        f"[[ref:{ref_by_evidence_id[evidence_id]}]]"
        for evidence_id in _unique(evidence_ids)
        if evidence_id in ref_by_evidence_id
        and f"[[ref:{ref_by_evidence_id[evidence_id]}]]" not in answer
    ]
    return f"{answer.rstrip()} {' '.join(markers)}" if markers else answer.rstrip()


def _citation_guide(refs: list[EvidenceRef]) -> str:
    return "\n".join(
        f"[[ref:{ref.ref_id}]]: {ref.file_name}, page {ref.page_number}, evidence {ref.evidence_id}, "
        f"quote={ref.quote[:400]}"
        for ref in refs
        if ref.ref_id
    ) or "(none)"


def _valid_ids(ids: list[str], allowed: list[str]) -> list[str]:
    allowed_set = {str(item) for item in allowed}
    return _unique(item for item in ids if item in allowed_set)


def _pack_ids_by_source(packs: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for pack in packs:
        source_id = str(pack.get("source_id") or "")
        pack_id = str(pack.get("pack_id") or "")
        if source_id and pack_id:
            result.setdefault(source_id, []).append(pack_id)
    return result


def _unique(items) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = str(item)
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
