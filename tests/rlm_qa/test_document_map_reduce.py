from __future__ import annotations

import json
import re
from dataclasses import replace
from threading import Event

import pytest

from rlm_qa.contracts import Chunk, DocProfile, Evidence, NoEvidenceError, Page
from rlm_qa.document_map_reduce import CoveragePlan, DocumentMapReduceRunner
from rlm_qa.llm import ChatRequest, ChatResponse, LLMClient


class MapReduceLLM(LLMClient):
    def __init__(self):
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if "MAP_REDUCE_PLAN" in request.system:
            return ChatResponse(answer=json.dumps({"coverage": "relevant_subset", "answer_shape": "synthesis"}))
        if "MAP_REDUCE_TRIAGE" in request.system:
            if "relevant.pdf" in request.user:
                return ChatResponse(
                    answer=json.dumps(
                        {
                            "pack_id": "pack_src_relevant_0001",
                            "relevance_score": 0.95,
                            "has_answer": True,
                            "evidence_ids": [f"ev_rel_{index}" for index in range(5)],
                            "short_reason": "contains the RLM answer",
                        }
                    )
                )
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": "pack_src_other_0001",
                        "relevance_score": 0.0,
                        "has_answer": False,
                        "evidence_ids": [],
                        "short_reason": "unrelated",
                    }
                )
            )
        if "MAP_REDUCE_DEEP_ANSWER" in request.system:
            assert "relevant.pdf" in request.user
            assert "unrelated.pdf" not in request.user
            return ChatResponse(
                answer=json.dumps(
                    {
                        "answer": "The relevant document explains RLM with multiple grounded claims.",
                        "evidence_ids": [f"ev_rel_{index}" for index in range(5)],
                        "confidence": 0.91,
                    }
                )
            )
        if "MAP_REDUCE_FINALIZER" in request.system:
            return ChatResponse(answer="Final map-reduce answer.")
        raise AssertionError(f"unexpected request: {request.system}")


class AllIrrelevantLLM(LLMClient):
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.system == "MAP_REDUCE_PLAN":
            return ChatResponse(
                answer=json.dumps({"coverage": "relevant_subset", "answer_shape": "synthesis"})
            )
        if request.system == "MAP_REDUCE_TRIAGE":
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": _pack_id_from_request(request.user),
                        "relevance_score": 0.0,
                        "has_answer": False,
                        "evidence_ids": [],
                        "short_reason": "not relevant",
                    }
                )
            )
        raise AssertionError(f"unexpected request after zero-relevance triage: {request.system}")


class StreamingMapReduceLLM(MapReduceLLM):
    def __init__(self) -> None:
        super().__init__()
        self.stream_requests: list[ChatRequest] = []

    def chat_stream(self, request, on_delta, cancelled) -> ChatResponse:
        self.stream_requests.append(request)
        assert request.system == "MAP_REDUCE_FINALIZER"
        assert cancelled() is False
        on_delta("Final ")
        on_delta("streamed answer.")
        return ChatResponse(answer="Final streamed answer.")


class MultiDocOverviewLLM(LLMClient):
    def __init__(self):
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if "MAP_REDUCE_PLAN" in request.system:
            return ChatResponse(answer=json.dumps({"coverage": "all_selected", "answer_shape": "per_document"}))
        if "MAP_REDUCE_TRIAGE" in request.system:
            if "first.pdf" in request.user:
                return ChatResponse(
                    answer=json.dumps(
                        {
                            "pack_id": "pack_src_first_0001",
                            "relevance_score": 1.0,
                            "has_answer": True,
                            "evidence_ids": ["ev_first"],
                            "short_reason": "first summary",
                        }
                    )
                )
            if "second.pdf" in request.user:
                return ChatResponse(
                    answer=json.dumps(
                        {
                            "pack_id": "pack_src_second_0001",
                            "relevance_score": 0.0,
                            "has_answer": False,
                            "evidence_ids": [],
                            "short_reason": "second summary but model marked irrelevant",
                        }
                    )
                )
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": "pack_src_third_0001",
                        "relevance_score": 0.0,
                        "has_answer": False,
                        "evidence_ids": [],
                        "short_reason": "third summary but model marked irrelevant",
                    }
                )
            )
        if "MAP_REDUCE_DEEP_ANSWER" in request.system:
            if "first.pdf" in request.user:
                return ChatResponse(answer=json.dumps({"answer": "first article summary", "evidence_ids": []}))
            if "second.pdf" in request.user:
                return ChatResponse(answer=json.dumps({"answer": "second article summary", "evidence_ids": []}))
            if "third.pdf" in request.user:
                return ChatResponse(answer=json.dumps({"answer": "third article summary", "evidence_ids": []}))
        if request.system == "MAP_REDUCE_FINAL_STREAM":
            return ChatResponse(
                answer=(
                    "## first.pdf\nfirst article summary [[ref:ref_001]]\n\n"
                    "## second.pdf\nsecond article summary [[ref:ref_002]]\n\n"
                    "## third.pdf\nthird article summary [[ref:ref_003]]"
                )
            )
        raise AssertionError(f"unexpected request: {request.system}")


class StreamingAllSelectedLLM(MultiDocOverviewLLM):
    def __init__(self) -> None:
        super().__init__()
        self.stream_requests: list[ChatRequest] = []

    def chat_stream(self, request, on_delta, cancelled) -> ChatResponse:
        self.stream_requests.append(request)
        assert request.system == "MAP_REDUCE_FINAL_STREAM"
        assert "first.pdf" in request.user
        assert "second.pdf" in request.user
        assert "third.pdf" in request.user
        answer = (
            "## first.pdf\nfirst [[ref:ref_001]]\n"
            "## second.pdf\nsecond [[ref:ref_002]]\n"
            "## third.pdf\nthird [[ref:ref_003]]"
        )
        on_delta(answer[:20])
        on_delta(answer[20:])
        return ChatResponse(answer=answer)


class OmittingStreamingAllSelectedLLM(StreamingAllSelectedLLM):
    def chat_stream(self, request, on_delta, cancelled) -> ChatResponse:
        self.stream_requests.append(request)
        assert request.system == "MAP_REDUCE_FINAL_STREAM"
        answer = "## first.pdf\nfirst [[ref:ref_001]]\n## third.pdf\nthird [[ref:ref_003]]"
        on_delta(answer)
        return ChatResponse(answer=answer)


class EvidenceFloodOverviewLLM(LLMClient):
    def chat(self, request: ChatRequest) -> ChatResponse:
        if "MAP_REDUCE_PLAN" in request.system:
            return ChatResponse(answer=json.dumps({"coverage": "all_selected", "answer_shape": "per_document"}))
        if "MAP_REDUCE_TRIAGE" in request.system:
            pack_id = "pack_src_alpha_0001" if "alpha.pdf" in request.user else "pack_src_beta_0001"
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": pack_id,
                        "relevance_score": 1.0,
                        "has_answer": True,
                        "evidence_ids": [],
                        "short_reason": "overview",
                    }
                )
            )
        if "MAP_REDUCE_DEEP_ANSWER" in request.system:
            prefix = "alpha" if "alpha.pdf" in request.user else "beta"
            return ChatResponse(
                answer=json.dumps(
                    {
                        "answer": f"{prefix} summary",
                        "evidence_ids": [f"ev_{prefix}_{index}" for index in range(7)],
                    }
                )
            )
        if request.system == "MAP_REDUCE_FINAL_STREAM":
            return ChatResponse(
                answer=(
                    "## alpha.pdf\nalpha summary [[ref:ref_001]]\n\n"
                    "## beta.pdf\nbeta summary [[ref:ref_006]]"
                )
            )
        raise AssertionError(f"unexpected request: {request.system}")


class PlannedCoverageLLM(LLMClient):
    def __init__(
        self,
        *,
        coverage: str = "all_selected",
        answer_shape: str = "per_document",
        downrank_file_name: str = "",
    ) -> None:
        self.coverage = coverage
        self.answer_shape = answer_shape
        self.downrank_file_name = downrank_file_name
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.system == "MAP_REDUCE_PLAN":
            return ChatResponse(
                answer=json.dumps(
                    {
                        "coverage": self.coverage,
                        "answer_shape": self.answer_shape,
                        "reason": "fixture plan",
                    }
                )
            )
        if request.system == "MAP_REDUCE_TRIAGE":
            pack_id = _pack_id_from_request(request.user)
            is_downranked = self.downrank_file_name in request.user
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": pack_id,
                        "relevance_score": 0.0 if is_downranked else 1.0,
                        "has_answer": not is_downranked,
                        "evidence_ids": [],
                        "short_reason": "fixture relevant",
                    }
                )
            )
        if request.system == "MAP_REDUCE_DEEP_ANSWER":
            file_name = _file_name_from_request(request.user)
            return ChatResponse(answer=json.dumps({"answer": f"{file_name} summary", "evidence_ids": []}))
        if request.system == "MAP_REDUCE_FINAL_STREAM":
            return ChatResponse(
                answer=(
                    "## first.pdf\nfirst final summary [[ref:ref_001]]\n\n"
                    "## second.pdf\nsecond final summary [[ref:ref_002]]\n\n"
                    "## third.pdf\nthird final summary [[ref:ref_003]]"
                )
            )
        raise AssertionError(f"unexpected request: {request.system}")


class FinalizerDropsSecondThenRepairsLLM(PlannedCoverageLLM):
    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.system == "MAP_REDUCE_FINAL_STREAM":
            self.requests.append(request)
            return ChatResponse(
                answer=(
                    "## first.pdf\nfirst final summary [[ref:ref_001]]\n\n"
                    "## third.pdf\nthird final summary [[ref:ref_003]]"
                )
            )
        if request.system == "MAP_REDUCE_FINALIZER_REPAIR":
            self.requests.append(request)
            return ChatResponse(
                answer=json.dumps(
                    {
                        "items": [
                            {"source_id": "src_second", "answer": "second repaired summary", "evidence_ids": []}
                        ]
                    }
                )
            )
        return super().chat(request)


class DeepFailureThenRepairLLM(PlannedCoverageLLM):
    def __init__(self) -> None:
        super().__init__()
        self.second_attempt_count = 0

    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.system == "MAP_REDUCE_DEEP_ANSWER" and "second.pdf" in request.user:
            self.second_attempt_count += 1
            if self.second_attempt_count == 1:
                self.requests.append(request)
                return ChatResponse(answer="not json")
        return super().chat(request)


class SecretFailingTriageLLM(LLMClient):
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.system == "MAP_REDUCE_PLAN":
            return ChatResponse(answer=json.dumps({"coverage": "relevant_subset", "answer_shape": "synthesis"}))
        if request.system == "MAP_REDUCE_TRIAGE":
            raise RuntimeError("provider echoed sk-secret-must-not-reach-debug")
        if request.system == "MAP_REDUCE_DEEP_ANSWER":
            return ChatResponse(answer=json.dumps({"answer": "recovered answer", "evidence_ids": []}))
        if request.system == "MAP_REDUCE_FINALIZER":
            return ChatResponse(answer="safe final answer")
        raise AssertionError(f"unexpected request: {request.system}")


class PipelinedBarrierLLM(LLMClient):
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []
        self.deep_started = Event()

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.system == "MAP_REDUCE_PLAN":
            return ChatResponse(
                answer=json.dumps({"coverage": "relevant_subset", "answer_shape": "synthesis"})
            )
        if request.system == "MAP_REDUCE_TRIAGE" and "relevant.pdf" in request.user:
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": "pack_src_relevant_0001",
                        "relevance_score": 1.0,
                        "has_answer": True,
                        "evidence_ids": ["ev_rel_0"],
                        "short_reason": "answerable",
                    }
                )
            )
        if request.system == "MAP_REDUCE_TRIAGE":
            assert self.deep_started.wait(timeout=2), "deep answer was blocked behind the triage barrier"
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": "pack_src_other_0001",
                        "relevance_score": 0.0,
                        "has_answer": False,
                        "evidence_ids": [],
                        "short_reason": "irrelevant",
                    }
                )
            )
        if request.system == "MAP_REDUCE_DEEP_ANSWER":
            self.deep_started.set()
            return ChatResponse(
                answer=json.dumps(
                    {
                        "answer": "pipelined grounded answer",
                        "evidence_ids": ["ev_rel_0"],
                        "confidence": 0.9,
                    }
                )
            )
        if request.system == "MAP_REDUCE_FINALIZER":
            return ChatResponse(answer="pipelined final answer")
        raise AssertionError(f"unexpected request: {request.system}")


class OnePassOverlapLLM(LLMClient):
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []
        self.map_started = Event()

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.system == "MAP_REDUCE_PLAN":
            assert self.map_started.wait(timeout=1), "coverage planner did not overlap pack mapping"
            return ChatResponse(
                answer=json.dumps({"coverage": "relevant_subset", "answer_shape": "synthesis"})
            )
        if request.system == "MAP_REDUCE_ONE_PASS":
            self.map_started.set()
            if "relevant.pdf" in request.user:
                return ChatResponse(
                    answer=json.dumps(
                        {
                            "pack_id": "pack_src_relevant_0001",
                            "relevance_score": 0.98,
                            "has_answer": True,
                            "answer": "one-pass grounded answer",
                            "claims": [
                                {
                                    "text": "RLM is explained here.",
                                    "evidence_ids": ["ev_rel_0", "invented_evidence"],
                                }
                            ],
                            "evidence_ids": ["ev_rel_0", "invented_evidence"],
                            "metadata_refs": [],
                            "confidence": 0.93,
                            "short_reason": "directly relevant",
                        }
                    )
                )
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": "pack_src_other_0001",
                        "relevance_score": 0.0,
                        "has_answer": False,
                        "answer": "",
                        "claims": [],
                        "evidence_ids": [],
                        "metadata_refs": [],
                        "confidence": 0.1,
                        "short_reason": "unrelated",
                    }
                )
            )
        if request.system == "MAP_REDUCE_FINALIZER":
            return ChatResponse(answer="one-pass final answer")
        raise AssertionError(f"unexpected request: {request.system}")


class OnePassAllSelectedRepairLLM(LLMClient):
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.system == "MAP_REDUCE_PLAN":
            return ChatResponse(
                answer=json.dumps({"coverage": "all_selected", "answer_shape": "per_document"})
            )
        if request.system == "MAP_REDUCE_ONE_PASS":
            if "second.pdf" in request.user:
                return ChatResponse(answer="not json")
            file_name = "first.pdf" if "first.pdf" in request.user else "third.pdf"
            source = "first" if file_name == "first.pdf" else "third"
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": f"pack_src_{source}_0001",
                        "relevance_score": 1.0,
                        "has_answer": True,
                        "answer": f"{source} one-pass summary",
                        "claims": [],
                        "evidence_ids": [f"ev_{source}"],
                        "metadata_refs": [],
                        "confidence": 0.9,
                        "short_reason": "overview",
                    }
                )
            )
        if request.system == "MAP_REDUCE_DEEP_ANSWER":
            assert "second.pdf" in request.user
            assert "first.pdf" not in request.user
            assert "third.pdf" not in request.user
            return ChatResponse(
                answer=json.dumps(
                    {
                        "answer": "second repaired summary",
                        "evidence_ids": ["ev_second"],
                        "confidence": 0.8,
                    }
                )
            )
        if request.system == "MAP_REDUCE_FINAL_STREAM":
            return ChatResponse(
                answer=(
                    "## first.pdf\nfirst summary [[ref:ref_001]]\n\n"
                    "## second.pdf\nsecond summary [[ref:ref_002]]\n\n"
                    "## third.pdf\nthird summary [[ref:ref_003]]"
                )
            )
        raise AssertionError(f"unexpected request: {request.system}")


class OnePassRelevantRepairLLM(LLMClient):
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.system == "MAP_REDUCE_PLAN":
            return ChatResponse(
                answer=json.dumps({"coverage": "relevant_subset", "answer_shape": "synthesis"})
            )
        if request.system == "MAP_REDUCE_ONE_PASS" and "relevant.pdf" in request.user:
            return ChatResponse(answer="malformed")
        if request.system == "MAP_REDUCE_ONE_PASS":
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": "pack_src_other_0001",
                        "relevance_score": 0.0,
                        "has_answer": False,
                        "answer": "",
                        "claims": [],
                        "evidence_ids": [],
                        "metadata_refs": [],
                        "confidence": 0.0,
                        "short_reason": "unrelated",
                    }
                )
            )
        if request.system == "MAP_REDUCE_TRIAGE":
            assert "relevant.pdf" in request.user
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": "pack_src_relevant_0001",
                        "relevance_score": 0.9,
                        "has_answer": True,
                        "evidence_ids": ["ev_rel_1"],
                        "short_reason": "recovered relevance",
                    }
                )
            )
        if request.system == "MAP_REDUCE_DEEP_ANSWER":
            assert "relevant.pdf" in request.user
            return ChatResponse(
                answer=json.dumps(
                    {
                        "answer": "recovered relevant answer",
                        "evidence_ids": ["ev_rel_1"],
                        "confidence": 0.8,
                    }
                )
            )
        if request.system == "MAP_REDUCE_FINALIZER":
            return ChatResponse(answer="recovered one-pass final answer")
        raise AssertionError(f"unexpected request: {request.system}")


class ClaimsOnlyDeepLLM(LLMClient):
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.system == "MAP_REDUCE_PLAN":
            return ChatResponse(
                answer=json.dumps({"coverage": "relevant_subset", "answer_shape": "synthesis"})
            )
        if request.system == "MAP_REDUCE_TRIAGE":
            relevant = "relevant.pdf" in request.user
            return ChatResponse(
                answer=json.dumps(
                    {
                        "pack_id": (
                            "pack_src_relevant_0001" if relevant else "pack_src_other_0001"
                        ),
                        "relevance_score": 1.0 if relevant else 0.0,
                        "has_answer": relevant,
                        "evidence_ids": [],
                        "short_reason": "relevant" if relevant else "unrelated",
                    }
                )
            )
        if request.system == "MAP_REDUCE_DEEP_ANSWER":
            return ChatResponse(
                answer=json.dumps(
                    {
                        "answer": "claim-only answer",
                        "claims": [
                            {
                                "text": "The supported claim is on page 3.",
                                "evidence_ids": ["ev_rel_2"],
                            }
                        ],
                        "evidence_ids": [],
                        "confidence": 0.9,
                    }
                )
            )
        if request.system == "MAP_REDUCE_FINALIZER":
            return ChatResponse(answer="claim-only final answer")
        raise AssertionError(f"unexpected request: {request.system}")


def _pack_id_from_request(user: str) -> str:
    match = re.search(r"PACK_ID=([^\s]+)", user)
    assert match is not None
    return match.group(1)


def _file_name_from_request(user: str) -> str:
    match = re.search(r"\[([^\]]+\.pdf) \| pages", user)
    assert match is not None
    return match.group(1)


def test_document_map_reduce_triages_all_packs_deep_reads_relevant_pack_and_keeps_all_evidence() -> None:
    relevant, other, pages, chunks, evidences = _multi_doc_fixture()
    llm = MapReduceLLM()
    runner = DocumentMapReduceRunner(llm=llm, subagent_model="deepseek-v4-flash", max_deep_packs=1)

    result = runner.answer_question_many("Where is the RLM-specific answer?", [relevant, other], pages, chunks, evidences)

    assert result.answer == "Final map-reduce answer."
    assert [request.system for request in llm.requests].count("MAP_REDUCE_TRIAGE") == 2
    assert [request.system for request in llm.requests].count("MAP_REDUCE_DEEP_ANSWER") == 1
    assert all(request.model == "deepseek-v4-flash" for request in llm.requests[:3])
    assert len(result.evidences) == 5
    assert [evidence.evidence_id for evidence in result.evidences] == [f"ev_rel_{index}" for index in range(5)]
    assert result.runtime_status == "rlm_success"
    assert result.trajectory["strategy"] == "map_reduce"
    assert result.trajectory["triage_count"] == 2
    assert result.trajectory["triage_skipped"] is False
    assert result.trajectory["triage_skip_reason"] == ""
    assert result.trajectory["deep_count"] == 1
    assert result.trajectory["selected_pack_ids"] == ["pack_src_relevant_0001"]


def test_pipelined_mode_starts_deep_before_all_triage_calls_finish() -> None:
    relevant, other, pages, chunks, evidences = _multi_doc_fixture()
    llm = PipelinedBarrierLLM()
    runner = DocumentMapReduceRunner(
        llm=llm,
        max_workers=2,
        max_deep_packs=2,
        map_execution_mode="pipelined",
    )

    result = runner.answer_question_many(
        "Which document explains RLM?", [relevant, other], pages, chunks, evidences
    )

    assert result.answer == "pipelined final answer"
    assert result.trajectory["pipeline_used"] is True
    assert result.trajectory["pipeline_fallback_reason"] == ""
    assert result.trajectory["deep_count"] == 1
    assert llm.deep_started.is_set()


def test_pipelined_mode_keeps_ranking_barrier_when_pack_count_exceeds_cap() -> None:
    relevant, other, pages, chunks, evidences = _multi_doc_fixture()
    llm = MapReduceLLM()
    runner = DocumentMapReduceRunner(
        llm=llm,
        max_workers=2,
        max_deep_packs=1,
        map_execution_mode="pipelined",
    )

    result = runner.answer_question_many(
        "Which document explains RLM?", [relevant, other], pages, chunks, evidences
    )

    assert result.trajectory["pipeline_used"] is False
    assert result.trajectory["pipeline_fallback_reason"] == "pack_count_exceeds_max_deep_packs"
    assert result.trajectory["deep_count"] == 1


def test_one_pass_overlaps_planner_selects_relevant_pack_and_allowlists_evidence() -> None:
    relevant, other, pages, chunks, evidences = _multi_doc_fixture()
    llm = OnePassOverlapLLM()
    runner = DocumentMapReduceRunner(
        llm=llm,
        max_workers=3,
        max_deep_packs=1,
        map_execution_mode="one_pass",
    )

    result = runner.answer_question_many(
        "Which document explains RLM?", [relevant, other], pages, chunks, evidences
    )

    systems = [request.system for request in llm.requests]
    assert result.answer == "one-pass final answer"
    assert systems.count("MAP_REDUCE_ONE_PASS") == 2
    assert "MAP_REDUCE_TRIAGE" not in systems
    assert "MAP_REDUCE_DEEP_ANSWER" not in systems
    assert result.trajectory["planner_parallelized"] is True
    assert result.trajectory["one_pass_count"] == 2
    assert result.trajectory["selected_pack_ids"] == ["pack_src_relevant_0001"]
    assert [ref.evidence_id for ref in result.evidences] == ["ev_rel_0"]
    assert result.trajectory["one_pass"][0]["claims"] == [
        {"text": "RLM is explained here.", "evidence_ids": ["ev_rel_0"]}
    ]


def test_one_pass_all_selected_repairs_only_the_failed_pack() -> None:
    profiles, pages, chunks, evidences = _three_doc_fixture()
    llm = OnePassAllSelectedRepairLLM()
    runner = DocumentMapReduceRunner(
        llm=llm,
        max_workers=4,
        map_execution_mode="one_pass",
    )

    result = runner.answer_question_many(
        "Summarize every document", profiles, pages, chunks, evidences
    )

    systems = [request.system for request in llm.requests]
    assert systems.count("MAP_REDUCE_ONE_PASS") == 3
    assert systems.count("MAP_REDUCE_DEEP_ANSWER") == 1
    assert "MAP_REDUCE_TRIAGE" not in systems
    assert result.trajectory["one_pass_repair_count"] == 1
    assert result.trajectory["deep_missing_source_ids"] == []
    assert result.trajectory["selected_pack_ids"] == [
        "pack_src_first_0001",
        "pack_src_second_0001",
        "pack_src_third_0001",
    ]


def test_one_pass_relevant_subset_recovers_a_failed_map_with_staged_calls() -> None:
    relevant, other, pages, chunks, evidences = _multi_doc_fixture()
    llm = OnePassRelevantRepairLLM()
    runner = DocumentMapReduceRunner(
        llm=llm,
        max_workers=3,
        max_deep_packs=1,
        map_execution_mode="one_pass",
    )

    result = runner.answer_question_many(
        "Which document explains RLM?", [relevant, other], pages, chunks, evidences
    )

    systems = [request.system for request in llm.requests]
    assert result.answer == "recovered one-pass final answer"
    assert systems.count("MAP_REDUCE_ONE_PASS") == 2
    assert systems.count("MAP_REDUCE_TRIAGE") == 1
    assert systems.count("MAP_REDUCE_DEEP_ANSWER") == 1
    assert result.trajectory["one_pass_repair_count"] == 1
    assert result.trajectory["selected_pack_ids"] == ["pack_src_relevant_0001"]


def test_compact_reduce_keeps_claim_quotes_and_aliases_without_expanded_chunk_tail() -> None:
    relevant, other, pages, chunks, evidences = _multi_doc_fixture()
    chunks[0] = replace(
        chunks[0],
        text=chunks[0].text + " UNRELATED_EXPANDED_TAIL_MUST_NOT_REACH_FINALIZER",
    )
    llm = OnePassOverlapLLM()
    runner = DocumentMapReduceRunner(
        llm=llm,
        max_workers=3,
        max_deep_packs=1,
        map_execution_mode="one_pass",
        reduce_context_mode="compact",
    )

    result = runner.answer_question_many(
        "Which document explains RLM?", [relevant, other], pages, chunks, evidences
    )

    finalizer = next(request for request in llm.requests if request.system == "MAP_REDUCE_FINALIZER")
    assert "RLM is explained here." in finalizer.user
    assert "ev_rel_0" in finalizer.user
    assert "Relevant RLM content 0." in finalizer.user
    assert "[[ref:ref_001]]" in finalizer.user
    assert "UNRELATED_EXPANDED_TAIL_MUST_NOT_REACH_FINALIZER" not in finalizer.user
    assert result.trajectory["reduce_context_mode"] == "compact"


def test_compact_staged_reduce_promotes_claim_evidence_before_fallback() -> None:
    relevant, other, pages, chunks, evidences = _multi_doc_fixture()
    llm = ClaimsOnlyDeepLLM()
    runner = DocumentMapReduceRunner(
        llm=llm,
        max_deep_packs=1,
        fallback_evidence_per_pack=1,
        reduce_context_mode="compact",
    )

    result = runner.answer_question_many(
        "Where is the supported claim?", [relevant, other], pages, chunks, evidences
    )

    finalizer = next(request for request in llm.requests if request.system == "MAP_REDUCE_FINALIZER")
    assert [ref.evidence_id for ref in result.evidences] == ["ev_rel_2"]
    assert "The supported claim is on page 3." in finalizer.user
    assert '"reference_markers": [\n          "[[ref:ref_001]]"' in finalizer.user


def test_multi_document_bare_summary_uses_model_coverage_plan_without_keyword_routing() -> None:
    profiles, pages, chunks, evidences = _three_doc_fixture()
    llm = PlannedCoverageLLM()

    result = DocumentMapReduceRunner(llm=llm).answer_question_many("总结", profiles, pages, chunks, evidences)

    plan_requests = [request for request in llm.requests if request.system == "MAP_REDUCE_PLAN"]
    assert len(plan_requests) == 1
    assert "first.pdf" in plan_requests[0].user
    assert "PACK_ID=" not in plan_requests[0].user
    assert result.trajectory["coverage_contract"] == "all_selected"


def test_injected_execution_contract_skips_duplicate_coverage_planner() -> None:
    profiles, pages, chunks, evidences = _three_doc_fixture()
    llm = PlannedCoverageLLM()
    runner = DocumentMapReduceRunner(
        llm=llm,
        coverage_plan=CoveragePlan(
            coverage="all_selected",
            answer_shape="per_document",
            reason="root execution contract",
        ),
    )

    result = runner.answer_question_many("总结", profiles, pages, chunks, evidences)

    systems = [request.system for request in llm.requests]
    assert "MAP_REDUCE_PLAN" not in systems
    assert result.trajectory["coverage_contract"] == "all_selected"
    assert result.trajectory["coverage_plan_reason"] == "root execution contract"
    assert result.trajectory["planner_parallelized"] is False


def test_relevant_subset_does_not_force_an_unrelated_pack_into_deep_answer() -> None:
    relevant, other, pages, chunks, evidences = _multi_doc_fixture()
    llm = AllIrrelevantLLM()

    with pytest.raises(NoEvidenceError):
        DocumentMapReduceRunner(llm=llm).answer_question_many(
            "Which document contains an absent topic?",
            [relevant, other],
            pages,
            chunks,
            evidences,
        )

    assert [request.system for request in llm.requests].count("MAP_REDUCE_TRIAGE") == 2
    assert all(request.system != "MAP_REDUCE_DEEP_ANSWER" for request in llm.requests)


def test_all_selected_skips_triage_and_deep_reads_every_source() -> None:
    profiles, pages, chunks, evidences = _three_doc_fixture()
    llm = PlannedCoverageLLM()

    result = DocumentMapReduceRunner(llm=llm, max_deep_packs=1).answer_question_many(
        "总结", profiles, pages, chunks, evidences
    )

    assert result.trajectory["required_source_ids"] == [profile.source_id for profile in profiles]
    assert set(result.trajectory["deep_result_source_ids"]) == {profile.source_id for profile in profiles}
    assert [request.system for request in llm.requests].count("MAP_REDUCE_TRIAGE") == 0
    assert [request.system for request in llm.requests].count("MAP_REDUCE_DEEP_ANSWER") == 3
    assert result.trajectory["triage_count"] == 0
    assert result.trajectory["triage_skipped"] is True
    assert result.trajectory["triage_skip_reason"] == "coverage_all_selected"


def test_all_selected_non_stream_finalizer_fills_missing_source_locally() -> None:
    profiles, pages, chunks, evidences = _three_doc_fixture()
    llm = FinalizerDropsSecondThenRepairsLLM()

    result = DocumentMapReduceRunner(llm=llm).answer_question_many("总结", profiles, pages, chunks, evidences)

    assert "first.pdf" in result.answer
    assert "second.pdf" in result.answer
    assert "third.pdf" in result.answer
    assert "second.pdf summary" in result.answer
    assert result.trajectory["missing_source_ids"] == []
    assert result.trajectory["finalizer_local_fallback_used"] is True
    assert [request.system for request in llm.requests].count("MAP_REDUCE_FINAL_STREAM") == 1
    assert [request.system for request in llm.requests].count("MAP_REDUCE_FINALIZER") == 0
    assert [request.system for request in llm.requests].count("MAP_REDUCE_FINALIZER_REPAIR") == 0


def test_all_selected_retries_only_source_missing_after_deep_failure() -> None:
    profiles, pages, chunks, evidences = _three_doc_fixture()
    llm = DeepFailureThenRepairLLM()

    result = DocumentMapReduceRunner(llm=llm).answer_question_many("总结", profiles, pages, chunks, evidences)

    assert llm.second_attempt_count == 2
    assert result.trajectory["deep_repair_attempted"] is True
    assert result.trajectory["deep_missing_source_ids"] == []
    assert "## second.pdf" in result.answer


def test_page_count_question_stays_in_model_planned_map_reduce_contract() -> None:
    profiles, pages, chunks, evidences = _three_doc_fixture()
    llm = PlannedCoverageLLM()

    result = DocumentMapReduceRunner(llm=llm).answer_question_many(
        "How many pages do these documents have?", profiles, pages, chunks, evidences
    )

    assert result.trajectory["coverage_contract"] == "all_selected"
    assert "metadata_shortcut" not in result.trajectory
    assert [request.system for request in llm.requests].count("MAP_REDUCE_PLAN") == 1


def test_trajectory_never_exposes_raw_triage_exception_text() -> None:
    relevant, other, pages, chunks, evidences = _multi_doc_fixture()
    result = DocumentMapReduceRunner(llm=SecretFailingTriageLLM()).answer_question_many(
        "Which document explains RLM?", [relevant, other], pages, chunks, evidences
    )

    trajectory_text = json.dumps(result.trajectory)
    assert "sk-secret-must-not-reach-debug" not in trajectory_text
    assert {item["error"] for item in result.trajectory["triage"]} == {"LLM_CALL_FAILED"}


def test_multi_document_overview_deep_reads_every_selected_document_without_triage() -> None:
    profiles, pages, chunks, evidences = _three_doc_fixture()
    llm = MultiDocOverviewLLM()
    runner = DocumentMapReduceRunner(llm=llm, subagent_model="deepseek-v4-flash", max_deep_packs=1)

    result = runner.answer_question_many("这几篇文章说了什么", profiles, pages, chunks, evidences)

    assert "## first.pdf" in result.answer
    assert "## second.pdf" in result.answer
    assert "## third.pdf" in result.answer
    assert [request.system for request in llm.requests].count("MAP_REDUCE_TRIAGE") == 0
    assert [request.system for request in llm.requests].count("MAP_REDUCE_DEEP_ANSWER") == 3
    assert result.trajectory["selected_pack_ids"] == [
        "pack_src_first_0001",
        "pack_src_second_0001",
        "pack_src_third_0001",
    ]
    assert result.runtime_status == "rlm_success"
    assert {evidence.file_name for evidence in result.evidences} == {"first.pdf", "second.pdf", "third.pdf"}
    assert {evidence.evidence_id for evidence in result.evidences} == {"ev_first", "ev_second", "ev_third"}


def test_multi_document_overview_caps_evidence_per_pack_without_global_three_limit() -> None:
    profiles, pages, chunks, evidences = _two_doc_many_evidence_fixture()
    runner = DocumentMapReduceRunner(
        llm=EvidenceFloodOverviewLLM(),
        subagent_model="deepseek-v4-flash",
        fallback_evidence_per_pack=5,
    )

    result = runner.answer_question_many("这几篇文章说了什么", profiles, pages, chunks, evidences)

    assert len(result.evidences) == 10
    assert len(result.evidences) > 3
    assert sum(1 for evidence in result.evidences if evidence.file_name == "alpha.pdf") == 5
    assert sum(1 for evidence in result.evidences if evidence.file_name == "beta.pdf") == 5




def test_relevant_subset_streams_only_the_finalizer_output() -> None:
    relevant, other, pages, chunks, evidences = _multi_doc_fixture()
    llm = StreamingMapReduceLLM()
    deltas: list[str] = []
    runner = DocumentMapReduceRunner(
        llm=llm,
        on_final_delta=deltas.append,
        cancelled=lambda: False,
        stream_trace={},
    )

    result = runner.answer_question_many(
        "Which document explains RLM?", [relevant, other], pages, chunks, evidences
    )

    assert deltas == ["Final ", "streamed answer."]
    assert result.answer == "Final streamed answer."
    assert [request.system for request in llm.stream_requests] == ["MAP_REDUCE_FINALIZER"]
    assert "Citation guide:\n[[ref:ref_001]]:" in llm.stream_requests[0].user
    assert all(request.system != "MAP_REDUCE_FINALIZER" for request in llm.requests)


def test_all_selected_uses_one_final_stream_and_preserves_every_file() -> None:
    profiles, pages, chunks, evidences = _three_doc_fixture()
    llm = StreamingAllSelectedLLM()
    deltas: list[str] = []
    runner = DocumentMapReduceRunner(
        llm=llm,
        on_final_delta=deltas.append,
        cancelled=lambda: False,
        stream_trace={},
    )

    result = runner.answer_question_many("Summarize every document", profiles, pages, chunks, evidences)

    assert "## first.pdf" in result.answer
    assert "## second.pdf" in result.answer
    assert "## third.pdf" in result.answer
    assert "".join(deltas) == result.answer
    assert [request.system for request in llm.requests].count("MAP_REDUCE_FINALIZER") == 0
    assert [request.system for request in llm.stream_requests] == ["MAP_REDUCE_FINAL_STREAM"]
    assert "Citation guide:\n[[ref:ref_001]]:" in llm.stream_requests[0].user
    assert result.trajectory["finalizer_local_fallback_used"] is False


def test_all_selected_appends_missing_source_locally_without_repair_llm() -> None:
    profiles, pages, chunks, evidences = _three_doc_fixture()
    llm = OmittingStreamingAllSelectedLLM()
    deltas: list[str] = []
    runner = DocumentMapReduceRunner(
        llm=llm,
        on_final_delta=deltas.append,
        cancelled=lambda: False,
        stream_trace={},
    )

    result = runner.answer_question_many("Summarize every document", profiles, pages, chunks, evidences)

    assert "## second.pdf" in result.answer
    assert "second article summary" in result.answer
    assert "[[ref:ref_002]]" in result.answer
    assert "".join(deltas) == result.answer
    assert result.trajectory["finalizer_local_fallback_used"] is True
    assert result.trajectory["missing_source_ids"] == []
    assert all(request.system != "MAP_REDUCE_FINALIZER_REPAIR" for request in llm.requests)


def test_all_selected_normalizes_stream_evidence_before_final_generation() -> None:
    profiles, pages, chunks, evidences = _three_doc_fixture()
    llm = StreamingAllSelectedLLM()
    delivered: list[list] = []
    runner = DocumentMapReduceRunner(
        llm=llm,
        on_final_delta=lambda _delta: None,
        on_stream_evidence=delivered.append,
        cancelled=lambda: False,
        stream_trace={},
    )

    result = runner.answer_question_many("Summarize every document", profiles, pages, chunks, evidences)

    assert len(delivered) == 1
    assert delivered[0] == result.evidences
    assert [ref.quote_source for ref in delivered[0]] == ["exact_span", "exact_span", "exact_span"]
    assert [ref.quote for ref in delivered[0]] == [
        "First article text.",
        "Second article text.",
        "Third article text.",
    ]

def _multi_doc_fixture():
    relevant = DocProfile(
        source_id="src_relevant",
        file_name="relevant.pdf",
        page_count=5,
        token_estimate=1000,
        chunk_count=5,
        evidence_count=5,
        direct_ready=False,
        rlm_ready=True,
        status="rlm_ready",
    )
    other = DocProfile(
        source_id="src_other",
        file_name="unrelated.pdf",
        page_count=1,
        token_estimate=100,
        chunk_count=1,
        evidence_count=1,
        direct_ready=False,
        rlm_ready=True,
        status="rlm_ready",
    )
    pages = [
        *(Page(source_id="src_relevant", page_number=index + 1, text=f"Relevant page {index}") for index in range(5)),
        Page(source_id="src_other", page_number=1, text="Other page"),
    ]
    chunks = [
        *(
            Chunk(
                chunk_id=f"chk_rel_{index}",
                source_id="src_relevant",
                page_start=index + 1,
                page_end=index + 1,
                text=f"Relevant RLM content {index}.",
                evidence_ids=[f"ev_rel_{index}"],
            )
            for index in range(5)
        ),
        Chunk(
            chunk_id="chk_other",
            source_id="src_other",
            page_start=1,
            page_end=1,
            text="Unrelated coating content.",
            evidence_ids=["ev_other"],
        ),
    ]
    evidences = [
        *(
            Evidence(
                evidence_id=f"ev_rel_{index}",
                source_id="src_relevant",
                chunk_id=f"chk_rel_{index}",
                page_number=index + 1,
                quote=f"Relevant RLM content {index}.",
            )
            for index in range(5)
        ),
        Evidence(
            evidence_id="ev_other",
            source_id="src_other",
            chunk_id="chk_other",
            page_number=1,
            quote="Unrelated coating content.",
        ),
    ]
    return relevant, other, pages, chunks, evidences


def _three_doc_fixture():
    profiles = [
        DocProfile(
            source_id="src_first",
            file_name="first.pdf",
            page_count=1,
            token_estimate=100,
            chunk_count=1,
            evidence_count=1,
            direct_ready=False,
            rlm_ready=True,
            status="rlm_ready",
        ),
        DocProfile(
            source_id="src_second",
            file_name="second.pdf",
            page_count=1,
            token_estimate=100,
            chunk_count=1,
            evidence_count=1,
            direct_ready=False,
            rlm_ready=True,
            status="rlm_ready",
        ),
        DocProfile(
            source_id="src_third",
            file_name="third.pdf",
            page_count=1,
            token_estimate=100,
            chunk_count=1,
            evidence_count=1,
            direct_ready=False,
            rlm_ready=True,
            status="rlm_ready",
        ),
    ]
    pages = [
        Page(source_id="src_first", page_number=1, text="First article text."),
        Page(source_id="src_second", page_number=1, text="Second article text."),
        Page(source_id="src_third", page_number=1, text="Third article text."),
    ]
    chunks = [
        Chunk("chk_first", "src_first", 1, 1, "First article text.", ["ev_first"]),
        Chunk("chk_second", "src_second", 1, 1, "Second article text.", ["ev_second"]),
        Chunk("chk_third", "src_third", 1, 1, "Third article text.", ["ev_third"]),
    ]
    evidences = [
        Evidence("ev_first", "src_first", "chk_first", 1, "First article text."),
        Evidence("ev_second", "src_second", "chk_second", 1, "Second article text."),
        Evidence("ev_third", "src_third", "chk_third", 1, "Third article text."),
    ]
    return profiles, pages, chunks, evidences


def _two_doc_many_evidence_fixture():
    profiles = [
        DocProfile("src_alpha", "alpha.pdf", 7, 700, 7, 7, False, True, "rlm_ready"),
        DocProfile("src_beta", "beta.pdf", 7, 700, 7, 7, False, True, "rlm_ready"),
    ]
    pages = [
        *(Page("src_alpha", index + 1, f"Alpha text {index}") for index in range(7)),
        *(Page("src_beta", index + 1, f"Beta text {index}") for index in range(7)),
    ]
    chunks = [
        *(
            Chunk(f"chk_alpha_{index}", "src_alpha", index + 1, index + 1, f"Alpha text {index}", [f"ev_alpha_{index}"])
            for index in range(7)
        ),
        *(
            Chunk(f"chk_beta_{index}", "src_beta", index + 1, index + 1, f"Beta text {index}", [f"ev_beta_{index}"])
            for index in range(7)
        ),
    ]
    evidences = [
        *(
            Evidence(f"ev_alpha_{index}", "src_alpha", f"chk_alpha_{index}", index + 1, f"Alpha text {index}")
            for index in range(7)
        ),
        *(
            Evidence(f"ev_beta_{index}", "src_beta", f"chk_beta_{index}", index + 1, f"Beta text {index}")
            for index in range(7)
        ),
    ]
    return profiles, pages, chunks, evidences
