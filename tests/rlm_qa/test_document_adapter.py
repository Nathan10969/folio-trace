from __future__ import annotations

from rlm_qa.contracts import Chunk, DocProfile, Evidence, Page
from rlm_qa.document_adapter import DocumentRLMRunner
from rlm_qa.document_tools import build_document_packs, make_document_context, make_document_tools
from rlm_qa.llm import ChatRequest, ChatResponse, LLMClient


class ScriptedChatLLM(LLMClient):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("model called more times than expected")
        return ChatResponse(answer=self.responses.pop(0))


def sample_document() -> tuple[DocProfile, list[Page], list[Chunk], list[Evidence]]:
    profile = DocProfile(
        source_id="src_test",
        file_name="paper.pdf",
        page_count=2,
        token_estimate=100,
        chunk_count=1,
        evidence_count=1,
        direct_ready=False,
        rlm_ready=True,
        status="rlm_ready",
    )
    pages = [
        Page(source_id="src_test", page_number=1, text="Single-walled carbon nanotubes are one-dimensional carbon materials."),
        Page(source_id="src_test", page_number=2, text="They can be used in conductive composites."),
    ]
    chunks = [
        Chunk(
            chunk_id="chk_1",
            source_id="src_test",
            page_start=1,
            page_end=1,
            text="Single-walled carbon nanotubes are one-dimensional carbon materials.",
            evidence_ids=["ev_1"],
        )
    ]
    evidences = [
        Evidence(
            evidence_id="ev_1",
            source_id="src_test",
            chunk_id="chk_1",
            page_number=1,
            quote="Single-walled carbon nanotubes are one-dimensional carbon materials.",
        )
    ]
    return profile, pages, chunks, evidences


def test_document_context_contains_question_profile_pages_chunks_and_evidence() -> None:
    profile, pages, chunks, evidences = sample_document()

    context = make_document_context(profile, pages, chunks, evidences, question="What is this?")

    assert context["question"] == "What is this?"
    assert context["doc_profile"]["source_id"] == "src_test"
    assert context["pages"][0]["page_number"] == 1
    assert context["pages_by_number"]["1"]["page_number"] == 1
    assert context["chunks"][0]["chunk_id"] == "chk_1"
    assert context["chunks_by_id"]["chk_1"]["chunk_id"] == "chk_1"
    assert context["evidence"][0]["evidence_id"] == "ev_1"
    assert context["evidence_by_id"]["ev_1"]["evidence_id"] == "ev_1"
    assert context["packs"][0]["pack_id"].startswith("pack_")
    assert context["packs"][0]["doc_source"] == "src_test"
    assert context["packs"][0]["page_count"] == 2
    assert context["packs"][0]["page_range"] == "1-1"
    assert context["packs"][0]["chunk_count"] == 1
    assert context["packs"][0]["evidence_count"] == 1
    assert context["packs"][0]["chunk_ids"] == ["chk_1"]
    assert context["packs_by_id"][context["packs"][0]["pack_id"]]["evidence_ids"] == ["ev_1"]


def test_document_tools_return_structured_data() -> None:
    profile, pages, chunks, evidences = sample_document()
    tools = make_document_tools(profile, pages, chunks, evidences)

    assert tools["read_pages"]["tool"]([1])["1"]["text"].startswith("Single-walled")
    assert tools["read_chunks"]["tool"](["chk_1"])["chk_1"]["page_start"] == 1
    assert tools["lookup_evidence"]["tool"](["ev_1"])["ev_1"]["page_number"] == 1
    assert tools["search_context"]["tool"]("carbon nanotubes", top_k=1)[0]["chunk_id"] == "chk_1"
    pack_id = tools["list_packs"]["tool"]()[0]["pack_id"]
    assert tools["list_packs"]["tool"]()[0]["doc_source"] == "src_test"
    assert tools["list_packs"]["tool"]()[0]["page_count"] == 2
    assert tools["list_packs"]["tool"]()[0]["page_range"] == "1-1"
    pack_text = tools["read_packs"]["tool"]([pack_id])[pack_id]["text"]
    assert tools["read_packs"]["tool"]([pack_id])[pack_id]["doc_source"] == "src_test"
    assert tools["read_packs"]["tool"]([pack_id])[pack_id]["page_count"] == 2
    assert pack_text.startswith("PACK_ID=pack_src_test_0001\n[paper.pdf | pages 1-1]")
    assert tools["expand_evidence"]["tool"](["ev_1"])["ev_1"]["text"].startswith("Single-walled")


def test_build_document_packs_groups_many_chunks_into_large_budget_units() -> None:
    profile = DocProfile(
        source_id="src_big",
        file_name="big.pdf",
        page_count=29,
        token_estimate=30_000,
        chunk_count=56,
        evidence_count=56,
        direct_ready=False,
        rlm_ready=True,
        status="rlm_ready",
    )
    pages = [Page(source_id="src_big", page_number=index + 1, text=f"page {index + 1}") for index in range(29)]
    chunks = [
        Chunk(
            chunk_id=f"chk_{index}",
            source_id="src_big",
            page_start=(index // 2) + 1,
            page_end=(index // 2) + 1,
            text=(f"chunk {index} " * 40),
            evidence_ids=[f"ev_{index}"],
        )
        for index in range(56)
    ]
    evidences = [
        Evidence(
            evidence_id=f"ev_{index}",
            source_id="src_big",
            chunk_id=f"chk_{index}",
            page_number=(index // 2) + 1,
            quote=f"chunk {index}",
        )
        for index in range(56)
    ]

    packs = build_document_packs(profile, pages, chunks, evidences, target_tokens=80_000, hard_max_tokens=100_000)

    assert len(packs) == 1
    assert len(packs[0].chunk_ids) == 56
    assert len(packs[0].evidence_ids) == 56
    assert packs[0].page_start == 1
    assert packs[0].page_end == 28


def test_document_runner_converts_runtime_evidence_ids_to_answer_refs() -> None:
    profile, pages, chunks, evidences = sample_document()
    llm = ScriptedChatLLM(
        [
            """```repl
hits = search_context("carbon nanotubes", top_k=1)
refs = lookup_evidence(hits[0]["evidence_ids"])
ev = refs[hits[0]["evidence_ids"][0]]
answer["content"] = '{"answer": "The document defines SWCNTs as one-dimensional carbon materials.", "evidence_ids": ["' + ev["evidence_id"] + '"]}'
answer["ready"] = True
```""",
            "Final answer grounded in evidence.",
        ]
    )
    runner = DocumentRLMRunner(llm=llm, max_iterations=2)

    result = runner.answer_question("What are single-walled carbon nanotubes?", profile, pages, chunks, evidences)

    assert result.answer == "Final answer grounded in evidence."
    assert result.evidences[0].evidence_id == "ev_1"
    assert result.evidences[0].page_number == 1
    assert result.trajectory["iterations"][0]["code_blocks"]
    assert "Single-walled carbon nanotubes" in llm.requests[-1].user


def test_document_runner_accepts_dict_answer_content() -> None:
    profile, pages, chunks, evidences = sample_document()
    llm = ScriptedChatLLM(
        [
            """```repl
ev = context["evidence"][0]
answer["content"] = {"answer": "Dict answers are accepted.", "evidence_ids": [ev["evidence_id"]]}
answer["ready"] = True
```""",
            "Final dict answer.",
        ]
    )
    runner = DocumentRLMRunner(llm=llm, max_iterations=2)

    result = runner.answer_question("What is this document about?", profile, pages, chunks, evidences)

    assert result.answer == "Final dict answer."
    assert result.evidences[0].evidence_id == "ev_1"


def test_document_runner_parses_nested_fenced_json_answer_content() -> None:
    profile, pages, chunks, evidences = sample_document()
    llm = ScriptedChatLLM(
        [
            """```repl
fence = "`" * 3
answer["content"] = {
    "answer": fence + "json\\n{\\"answer\\": \\"Nested JSON answer.\\", \\"evidence_ids\\": [\\"ev_1\\"]}\\n" + fence,
    "evidence_ids": [],
    "metadata_refs": [],
}
answer["ready"] = True
```""",
            "Final nested JSON answer.",
        ]
    )
    runner = DocumentRLMRunner(llm=llm, max_iterations=2)

    result = runner.answer_question("What is this document about?", profile, pages, chunks, evidences)

    assert result.answer == "Final nested JSON answer."
    assert result.evidences[0].evidence_id == "ev_1"
    assert "Nested JSON answer." in llm.requests[-1].user


def test_document_runner_accepts_verified_metadata_refs_without_pdf_evidence() -> None:
    profile, pages, chunks, evidences = sample_document()
    llm = ScriptedChatLLM(
        [
            """```repl
profile = context["doc_profile"]
answer["content"] = {
    "answer": f"This document has {profile['page_count']} pages.",
    "evidence_ids": [],
    "metadata_refs": [f"doc_profile:{profile['source_id']}:page_count"],
}
answer["ready"] = True
```""",
            "Final metadata answer.",
        ]
    )
    runner = DocumentRLMRunner(llm=llm, max_iterations=2)

    result = runner.answer_question("How many pages does this document have?", profile, pages, chunks, evidences)

    assert result.answer == "Final metadata answer."
    assert result.evidences == []
    assert result.metadata_refs == ["doc_profile:src_test:page_count"]
    assert result.runtime_status == "rlm_success"
    assert "doc_profile:src_test:page_count" in llm.requests[-1].user
    assert "page_count: 2" in llm.requests[-1].user


def test_document_runner_finalizes_draft_without_verified_evidence_or_metadata() -> None:
    profile = DocProfile(
        source_id="src_no_refs",
        file_name="paper.pdf",
        page_count=26,
        token_estimate=100,
        chunk_count=0,
        evidence_count=0,
        direct_ready=False,
        rlm_ready=True,
        status="rlm_ready",
    )
    llm = ScriptedChatLLM(
        [
            """```repl
answer["content"] = {
    "answer": "这篇文章有 26 页。",
    "evidence_ids": [],
    "metadata_refs": [],
}
answer["ready"] = True
```""",
            "Final answer from relaxed finalizer.",
        ]
    )
    runner = DocumentRLMRunner(llm=llm, max_iterations=2)

    result = runner.answer_question("这篇文章有多少页？", profile, [], [], [])

    assert result.answer == "Final answer from relaxed finalizer."
    assert result.evidences == []
    assert result.metadata_refs == []
    assert "这篇文章有 26 页。" in llm.requests[-1].user
    assert "Do not reject the draft merely because supporting context is absent." in llm.requests[-1].system
    assert "Do not mention RLM draft" in llm.requests[-1].system


def test_document_runner_does_not_auto_recover_evidence_ids_from_runtime_trajectory() -> None:
    profile, pages, chunks, evidences = sample_document()
    llm = ScriptedChatLLM(
        [
            """```repl
print("saw evidence", context["evidence"][0]["evidence_id"])
answer["content"] = {"answer": "Draft forgot evidence ids.", "evidence_ids": []}
answer["ready"] = True
```""",
            "Final repaired answer.",
        ]
    )
    runner = DocumentRLMRunner(llm=llm, max_iterations=2)

    result = runner.answer_question("What is this document about?", profile, pages, chunks, evidences)

    assert result.answer == "Final repaired answer."
    assert result.evidences == []
    assert result.runtime_status == "rlm_degraded_recovered"
    assert "Verified evidence excerpts:\n(none)" in llm.requests[-1].user


def test_document_runner_recovers_representative_evidence_from_packs_read_by_subagents() -> None:
    profile, pages, chunks, evidences = sample_document()
    llm = ScriptedChatLLM(
        [
            """```repl
packs = list_packs()
pack_id = packs[0]["pack_id"]
full = read_packs([pack_id])
_ = llm_query(full[pack_id]["text"], model="glm-5.1")
answer["content"] = {"answer": "Draft read the full pack but forgot evidence ids.", "evidence_ids": []}
answer["ready"] = True
```""",
            "Subagent read full pack.",
            "Final recovered evidence answer.",
        ]
    )
    runner = DocumentRLMRunner(llm=llm, max_iterations=2, subagent_model="glm-5.1")

    result = runner.answer_question("What is this document about?", profile, pages, chunks, evidences)

    assert result.answer == "Final recovered evidence answer."
    assert [evidence.evidence_id for evidence in result.evidences] == ["ev_1"]
    assert result.runtime_status == "rlm_success"


def test_document_runner_marks_success_when_subagent_reads_full_pack() -> None:
    profile, pages, chunks, evidences = sample_document()
    llm = ScriptedChatLLM(
        [
            """```repl
packs = list_packs()
pack_id = packs[0]["pack_id"]
full = read_packs([pack_id])
_ = llm_query(full[pack_id]["text"], model="glm-5.1")
ev = packs[0]["evidence_ids"][0]
answer["content"] = {"answer": "Draft from full pack.", "evidence_ids": [ev]}
answer["ready"] = True
```""",
            "Subagent read full pack.",
            "Final full-pack answer.",
        ]
    )
    runner = DocumentRLMRunner(llm=llm, max_iterations=2, subagent_model="glm-5.1")

    result = runner.answer_question("What is this document about?", profile, pages, chunks, evidences)

    assert result.answer == "Final full-pack answer."
    assert result.runtime_status == "rlm_success"


def test_document_runner_keeps_success_when_verified_evidence_is_returned_even_if_subagent_prompt_is_short() -> None:
    profile, pages, chunks, evidences = sample_document()
    llm = ScriptedChatLLM(
        [
            """```repl
packs = list_packs()
pack_id = packs[0]["pack_id"]
full = read_packs([pack_id])
_ = llm_query(full[pack_id]["text"][:20], model="glm-5.1")
ev = packs[0]["evidence_ids"][0]
answer["content"] = {"answer": "Draft from partial pack.", "evidence_ids": [ev]}
answer["ready"] = True
```""",
            "Subagent read partial pack.",
            "Final partial-pack answer.",
        ]
    )
    runner = DocumentRLMRunner(llm=llm, max_iterations=2, subagent_model="glm-5.1")

    result = runner.answer_question("What is this document about?", profile, pages, chunks, evidences)

    assert result.answer == "Final partial-pack answer."
    assert result.runtime_status == "rlm_success"


def test_document_runner_passes_subagent_model_to_llm_query() -> None:
    profile, pages, chunks, evidences = sample_document()
    llm = ScriptedChatLLM(
        [
            """```repl
sub = llm_query("read the pack", model="glm-5.1")
ev = context["evidence"][0]
answer["content"] = {"answer": sub, "evidence_ids": [ev["evidence_id"]]}
answer["ready"] = True
```""",
            "Subagent answer.",
            "Final model-routed answer.",
        ]
    )
    runner = DocumentRLMRunner(llm=llm, max_iterations=2, subagent_model="glm-5.1")

    result = runner.answer_question("What is this document about?", profile, pages, chunks, evidences)

    assert result.answer == "Final model-routed answer."
    assert llm.requests[1].model == "glm-5.1"


def test_document_runner_finalizer_uses_expanded_chunk_text_not_only_short_quote() -> None:
    profile = DocProfile(
        source_id="src_long",
        file_name="long.pdf",
        page_count=1,
        token_estimate=900,
        chunk_count=1,
        evidence_count=1,
        direct_ready=False,
        rlm_ready=True,
        status="rlm_ready",
    )
    pages = [Page(source_id="src_long", page_number=1, text="page text")]
    long_chunk_text = "Short abstract opening. " + ("middle detail. " * 40) + "Critical conclusion only appears late."
    chunks = [
        Chunk(
            chunk_id="chk_long",
            source_id="src_long",
            page_start=1,
            page_end=1,
            text=long_chunk_text,
            evidence_ids=["ev_long"],
        )
    ]
    evidences = [
        Evidence(
            evidence_id="ev_long",
            source_id="src_long",
            chunk_id="chk_long",
            page_number=1,
            quote="Short abstract opening.",
        )
    ]
    llm = ScriptedChatLLM(
        [
            """```repl
answer["content"] = {"answer": "Draft", "evidence_ids": ["ev_long"]}
answer["ready"] = True
```""",
            "Final expanded answer.",
        ]
    )
    runner = DocumentRLMRunner(llm=llm, max_iterations=2)

    result = runner.answer_question("What does the paper conclude?", profile, pages, chunks, evidences)

    assert result.answer == "Final expanded answer."
    assert "Critical conclusion only appears late." in llm.requests[-1].user
