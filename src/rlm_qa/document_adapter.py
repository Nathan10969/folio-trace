from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from rlm_runtime import RLM

from .contracts import Chunk, DocProfile, Evidence, EvidenceRef, Page, RLMQAError
from .document_tools import make_multi_document_context, make_multi_document_tools
from .llm import ChatRequest, LLMClient


@dataclass(frozen=True)
class DocumentRLMAnswer:
    answer: str
    evidences: list[EvidenceRef]
    metadata_refs: list[str]
    trajectory: dict[str, Any]
    runtime_status: str


class DocumentRLMRunner:
    def __init__(
        self,
        *,
        llm: LLMClient,
        max_iterations: int = 5,
        subagent_model: str = "glm-5.1",
        max_depth: int = 1,
        batch_max_workers: int = 4,
        max_timeout_seconds: float | None = None,
        max_tokens: int | None = None,
        max_errors: int | None = None,
        max_budget: float | None = None,
        cost_per_1k_tokens: float | None = None,
    ) -> None:
        self.llm = llm
        self.max_iterations = max_iterations
        self.subagent_model = subagent_model
        self.max_depth = max_depth
        self.batch_max_workers = batch_max_workers
        self.max_timeout_seconds = max_timeout_seconds
        self.max_tokens = max_tokens
        self.max_errors = max_errors
        self.max_budget = max_budget
        self.cost_per_1k_tokens = cost_per_1k_tokens

    def answer_question(
        self,
        question: str,
        profile: DocProfile,
        pages: list[Page],
        chunks: list[Chunk],
        evidences: list[Evidence],
    ) -> DocumentRLMAnswer:
        return self.answer_question_many(question, [profile], pages, chunks, evidences)

    def answer_question_many(
        self,
        question: str,
        profiles: list[DocProfile],
        pages: list[Page],
        chunks: list[Chunk],
        evidences: list[Evidence],
    ) -> DocumentRLMAnswer:
        context = make_multi_document_context(profiles, pages, chunks, evidences, question=question)
        tools = make_multi_document_tools(profiles, pages, chunks, evidences)
        rlm = RLM(
            llm=self._chat,
            max_iterations=self.max_iterations,
            custom_tools=tools,
            max_depth=self.max_depth,
            batch_max_workers=self.batch_max_workers,
            max_timeout_seconds=self.max_timeout_seconds,
            max_tokens=self.max_tokens,
            max_errors=self.max_errors,
            max_budget=self.max_budget,
            cost_per_1k_tokens=self.cost_per_1k_tokens,
        )
        result = rlm.completion(_document_prompt(question, self.subagent_model), context=context)
        if result.error:
            raise RLMQAError("文档问答流程本次没有生成可用答案，请重试或换一种问法。")
        draft_answer, evidence_ids, metadata_refs = _parse_document_response(result.response)
        refs = _evidence_refs(evidence_ids, evidences, profiles)
        metadata_facts = _metadata_facts(metadata_refs, context)
        if not refs and not metadata_facts:
            recovered_evidence_ids = _representative_evidence_ids_from_runtime_pack_calls(result.trajectory, context)
            refs = _evidence_refs(recovered_evidence_ids, evidences, profiles)
        recovered = False
        runtime_status = _assess_runtime_status(
            result_error=result.error,
            has_verified_support=bool(refs or metadata_facts),
            recovered=recovered,
        )
        final_answer = self._finalize_answer(question, draft_answer, refs, chunks, metadata_facts)
        return DocumentRLMAnswer(
            answer=final_answer,
            evidences=refs,
            metadata_refs=[fact["ref"] for fact in metadata_facts],
            trajectory=result.trajectory,
            runtime_status=runtime_status,
        )

    def _chat(self, system: str, user: str, model: str | None = None) -> str:
        return self.llm.chat(ChatRequest(system=system, user=user, model=model)).answer

    def _finalize_answer(
        self,
        question: str,
        draft_answer: str,
        refs: list[EvidenceRef],
        chunks: list[Chunk],
        metadata_facts: list[dict[str, Any]],
    ) -> str:
        evidence_text = _expanded_evidence_text(refs, chunks) if refs else "(none)"
        metadata_text = _metadata_facts_text(metadata_facts) if metadata_facts else "(none)"
        system = (
            "Answer in Chinese. Treat the RLM draft answer as the primary answer. "
            "Use verified evidence excerpts and metadata facts as supporting context when available. "
            "Do not reject the draft merely because supporting context is absent. "
            "If supporting context conflicts with the draft, state the uncertainty briefly. "
            "Do not mention RLM draft, metadata_refs, evidence_ids, support context, or internal field names. "
            "Do not fabricate PDF citations, pages, titles, or facts."
        )
        user = (
            f"Question: {question}\n\n"
            f"RLM draft answer:\n{draft_answer}\n\n"
            f"Verified evidence excerpts:\n{evidence_text}\n\n"
            f"Verified metadata facts:\n{metadata_text}\n\n"
            "Write the final answer for the user."
        )
        return self.llm.chat(ChatRequest(system=system, user=user)).answer


def _document_prompt(question: str, subagent_model: str) -> str:
    return (
        "Answer the user's document question in Chinese unless the user asks otherwise. "
        "Use the document context and tools. Do not invent PDF pages or evidence ids. "
        "The original user question is also available as context['question']. "
        "Normal strategy: call list_packs() to inspect macro packs, call read_packs(pack_ids), "
        f"then send whole pack texts to llm_query or llm_query_batched with model='{subagent_model}' for semantic reading. "
        "If there is one pack, read that pack and ask one sub-LLM for a structured summary/answer. "
        "If there are multiple packs, use llm_query_batched over fat pack prompts and combine the short results. "
        "search_context(query, top_k=5) is only a cheap lexical helper for narrowing or citation recovery. "
        "Tool contracts: list_packs() returns pack metadata without full text; "
        "read_packs(ids) returns full macro-pack text; "
        "expand_evidence(ids) returns expanded chunk text for verified evidence ids; "
        "lookup_evidence(ids) returns a dict keyed by exact evidence id strings. "
        "Only use evidence ids returned by packs, context['evidence_by_id'], search_context, or lookup_evidence. "
        "Never use placeholder ids such as ev_1 unless that exact id exists. "
        "When you ask sub-LLMs to read packs, keep the pack ids aligned with their returned drafts. "
        "If a useful sub-LLM draft omits evidence_ids, use 1-5 representative ids from that pack's evidence_ids list. "
        "Set answer['content'] to either a dict or JSON string with this exact shape: "
        '{"answer": "...", "evidence_ids": ["actual evidence id"], "metadata_refs": ["doc_profile:source_id:page_count"]}. '
        "Use metadata_refs only for facts present in context['doc_profile'] or context['doc_profiles_by_source']; "
        "for example page_count, file_name, token_estimate, chunk_count, evidence_count, status, direct_ready, or rlm_ready. "
        "Each metadata ref must be exactly doc_profile:<source_id>:<field>; do not append values or multiple fields to one ref. "
        "Example code: import json; packs = list_packs(); full = read_packs([packs[0]['pack_id']]); "
        "pack = full[packs[0]['pack_id']]; "
        "prompt = 'Answer the question from this pack. Return JSON with answer and evidence_ids. Use only evidence ids shown in the pack.\\nQuestion: ' + context['question'] + '\\n\\nPack:\\n' + pack['text']; "
        f"summary = llm_query(prompt, model='{subagent_model}'); "
        "answer['content'] = json.loads(summary); "
        "answer['ready'] = True. "
        f"User question: {question}"
    )


def _parse_document_response(response: Any) -> tuple[str, list[str], list[str]]:
    if isinstance(response, dict):
        answer_value = response.get("answer") or response.get("content") or ""
        nested = _json_object_from_text(str(answer_value))
        if nested is not None:
            nested_answer, nested_ids, nested_metadata_refs = _parse_document_response(nested)
            answer = nested_answer
            raw_ids = response.get("evidence_ids") or nested_ids
            raw_metadata_refs = response.get("metadata_refs") or nested_metadata_refs
        else:
            answer = str(answer_value)
            raw_ids = response.get("evidence_ids") or []
            raw_metadata_refs = response.get("metadata_refs") or []
        metadata_refs = _coerce_metadata_refs(raw_metadata_refs)
        if isinstance(raw_ids, list):
            return answer, [str(item) for item in raw_ids], metadata_refs
        return answer, _extract_evidence_ids(str(raw_ids)), metadata_refs

    text = str(response).strip()
    data = _json_object_from_text(text)
    if data is None:
        return text, _extract_evidence_ids(text), _extract_metadata_refs(text)
    answer = str(data.get("answer") or data.get("content") or text)
    raw_ids = data.get("evidence_ids") or []
    raw_metadata_refs = data.get("metadata_refs") or []
    metadata_refs = _coerce_metadata_refs(raw_metadata_refs)
    if isinstance(raw_ids, list):
        return answer, [str(item) for item in raw_ids], metadata_refs
    return answer, _extract_evidence_ids(str(raw_ids)), metadata_refs


def _json_object_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fence_match = re.fullmatch(r"```[ \t]*(?:json)?[ \t]*\n(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        stripped = fence_match.group(1).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _extract_evidence_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for match in re.findall(r"\bev_[A-Za-z0-9_:-]+\b", text):
        if match not in seen:
            ids.append(match)
            seen.add(match)
    return ids


def _coerce_metadata_refs(raw_refs: Any) -> list[str]:
    if isinstance(raw_refs, list):
        refs = [str(item) for item in raw_refs]
    else:
        refs = _extract_metadata_refs(str(raw_refs))
    seen: set[str] = set()
    clean: list[str] = []
    for ref in refs:
        if ref not in seen:
            clean.append(ref)
            seen.add(ref)
    return clean


def _extract_metadata_refs(text: str) -> list[str]:
    seen: set[str] = set()
    refs: list[str] = []
    for match in re.findall(r"\bdoc_profile:[A-Za-z0-9_.-]+:[A-Za-z_][A-Za-z0-9_]*\b", text):
        if match not in seen:
            refs.append(match)
            seen.add(match)
    return refs


def _metadata_facts(metadata_refs: list[str], context: dict[str, Any]) -> list[dict[str, Any]]:
    profiles_by_source = context.get("doc_profiles_by_source") or {}
    allowed_fields = {
        "source_id",
        "file_name",
        "page_count",
        "token_estimate",
        "chunk_count",
        "evidence_count",
        "direct_ready",
        "rlm_ready",
        "status",
        "error_code",
    }
    facts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in metadata_refs:
        parts = str(ref).split(":")
        if len(parts) != 3 or parts[0] != "doc_profile":
            continue
        _, source_id, field = parts
        if field not in allowed_fields:
            continue
        profile = profiles_by_source.get(source_id)
        if not isinstance(profile, dict) or field not in profile:
            continue
        if ref in seen:
            continue
        seen.add(ref)
        facts.append(
            {
                "ref": ref,
                "source_id": source_id,
                "file_name": str(profile.get("file_name") or ""),
                "field": field,
                "value": profile.get(field),
            }
        )
    return facts


def _metadata_facts_text(metadata_facts: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for fact in metadata_facts:
        parts.append(
            f"[{fact['ref']} | {fact['file_name']} | source {fact['source_id']}]\n"
            f"{fact['field']}: {fact['value']}"
        )
    return "\n\n".join(parts)


def _evidence_refs(evidence_ids: list[str], evidences: list[Evidence], profiles: list[DocProfile]) -> list[EvidenceRef]:
    by_id = {evidence.evidence_id: evidence for evidence in evidences}
    file_names = {profile.source_id: profile.file_name for profile in profiles}
    refs: list[EvidenceRef] = []
    for evidence_id in evidence_ids:
        evidence = by_id.get(evidence_id)
        if evidence is None:
            continue
        refs.append(
            EvidenceRef(
                evidence_id=evidence.evidence_id,
                source_id=evidence.source_id,
                chunk_id=evidence.chunk_id,
                page_number=evidence.page_number,
                quote=evidence.quote,
                file_name=file_names.get(evidence.source_id, ""),
            )
        )
    return refs


def _representative_evidence_ids_from_runtime_pack_calls(
    trajectory: dict[str, Any],
    context: dict[str, Any],
    *,
    max_per_pack: int = 5,
) -> list[str]:
    packs_by_id = context.get("packs_by_id") or {}
    if not isinstance(packs_by_id, dict):
        return []
    seen_pack_ids: set[str] = set()
    evidence_ids: list[str] = []
    for call in _iter_runtime_calls(trajectory):
        for raw_pack_id in call.get("pack_ids") or []:
            pack_id = str(raw_pack_id)
            if pack_id in seen_pack_ids:
                continue
            pack = packs_by_id.get(pack_id)
            if not isinstance(pack, dict):
                continue
            seen_pack_ids.add(pack_id)
            added_for_pack = 0
            for evidence_id in pack.get("evidence_ids") or []:
                value = str(evidence_id)
                if value not in evidence_ids:
                    evidence_ids.append(value)
                    added_for_pack += 1
                if added_for_pack >= max(1, int(max_per_pack)):
                    break
    return evidence_ids


def _iter_runtime_calls(trajectory: dict[str, Any]):
    for iteration in trajectory.get("iterations") or []:
        for repl_result in iteration.get("repl_results") or []:
            calls = repl_result.get("rlm_calls") or []
            if isinstance(calls, dict):
                calls = [calls]
            for call in calls:
                if isinstance(call, dict):
                    yield call


def _expanded_evidence_text(refs: list[EvidenceRef], chunks: list[Chunk], max_chars_per_ref: int = 5000) -> str:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    parts: list[str] = []
    for ref in refs:
        chunk = chunks_by_id.get(ref.chunk_id)
        text = chunk.text if chunk is not None else ref.quote
        parts.append(
            f"[{ref.ref_id} | {ref.evidence_id} | {ref.file_name} | page {ref.page_number} | chunk {ref.chunk_id}]\n"
            f"Quote: {ref.quote}\n"
            f"Expanded text:\n{text[:max_chars_per_ref]}"
        )
    return "\n\n".join(parts)


def _assess_runtime_status(
    *,
    result_error: str,
    has_verified_support: bool,
    recovered: bool,
) -> str:
    if result_error or recovered or not has_verified_support:
        return "rlm_degraded_recovered"
    return "rlm_success"
