"""Pipeline behaviour, exercised against fake providers.

No network. The point of these tests is the state machine — that evidence is
always gathered, that the retrieval loop is genuinely bounded, that a failing
dependency degrades to escalation rather than to an answer, and that the audit
record comes out complete enough to replay.
"""

from __future__ import annotations

import pytest

from sharia_agent.agent.pipeline import CompliancePipeline
from sharia_agent.config import Settings
from sharia_agent.llm import LLMResponse, ToolCall, Usage
from sharia_agent.models import (
    Citation,
    Clause,
    DraftAssessment,
    ModelFinding,
    RetrievedClause,
    Verdict,
)
from sharia_agent.obs.trace import set_trace_id
from sharia_agent.resilience import UpstreamUnavailable

CLAUSE = Clause(
    chunk_id="SS8-3.1.1",
    standard_no="8",
    standard_name="Murabahah",
    clause_path="3/1/1",
    heading_path=["Murabahah", "3/1 Acquisition of the item"],
    text=(
        "The Institution shall not sell any item in a Murabahah transaction "
        "before it acquires such item."
    ),
    corpus_version="test@abc123",
)


class FakeRetriever:
    def __init__(self, results=None, fail=False):
        self._results = results if results is not None else [
            RetrievedClause(clause=CLAUSE, rerank_score=0.91)
        ]
        self._fail = fail
        self.calls: list[dict] = []

    async def search(self, query, *, standard_no=None, round_no=1, top_k=None):
        self.calls.append({"query": query, "standard_no": standard_no, "round": round_no})
        if self._fail:
            raise UpstreamUnavailable("openrouter.embeddings", "circuit open")
        return list(self._results)


class FakeLLM:
    """Scripted responses. `tool_script` drives the refinement loop."""

    def __init__(self, draft=None, tool_script=None):
        self._draft = draft
        self._tool_script = list(tool_script or [])
        self.complete_calls = 0
        self.structured_calls = 0

    async def complete(self, *, messages, system=None, tools=None, max_tokens=2048, model=None):
        self.complete_calls += 1
        calls = self._tool_script.pop(0) if self._tool_script else []
        return LLMResponse(
            text="" if calls else "DONE",
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
            usage=Usage(100, 20),
            raw_message={"role": "assistant", "content": None},
        )

    async def complete_structured(self, *, output_model, messages, system=None,
                                  max_tokens=4096, model=None):
        self.structured_calls += 1
        return self._draft, Usage(2000, 300)


def good_draft() -> DraftAssessment:
    return DraftAssessment(
        finding=ModelFinding.SUPPORTED_NON_COMPLIANT,
        confidence=0.93,
        reasoning="Selling before acquiring the item is excluded by the standard.",
        citations=[
            Citation(
                chunk_id="SS8-3.1.1",
                quote="shall not sell any item in a Murabahah transaction before it acquires such item",
                supports="ownership must precede sale",
            )
        ],
    )


def build(retriever, llm, **overrides) -> CompliancePipeline:
    overrides.setdefault("max_retrieval_rounds", 1)
    settings = Settings(_env_file=None, **overrides)
    return CompliancePipeline(retriever, llm, settings)


@pytest.fixture(autouse=True)
def _trace():
    set_trace_id("test-trace")


async def test_happy_path_produces_a_cited_verdict():
    retriever, llm = FakeRetriever(), FakeLLM(draft=good_draft())
    result = await build(retriever, llm).assess(
        query="Can Mal sell a car to a customer before buying it from the dealer?",
        principal_id="analyst@mal.ae",
    )
    assert result.assessment.verdict is Verdict.NON_COMPLIANT
    assert result.assessment.citations[0].chunk_id == "SS8-3.1.1"
    assert result.assessment.escalations == []


async def test_evidence_is_gathered_even_if_the_model_never_searches():
    """The seed search is unconditional — retrieval must not depend on the model
    choosing to call a tool."""
    retriever, llm = FakeRetriever(), FakeLLM(draft=good_draft())
    await build(retriever, llm).assess(query="a murabaha question", principal_id="p")
    assert retriever.calls and retriever.calls[0]["round"] == 1


async def test_retrieval_loop_is_bounded():
    """A model that keeps asking for more evidence cannot spend without limit."""
    script = [[ToolCall(id=f"t{i}", name="search_standards",
                        arguments={"query": f"variant {i}", "standard_no": None,
                                   "reason": "more"})] for i in range(10)]
    retriever, llm = FakeRetriever(), FakeLLM(draft=good_draft(), tool_script=script)
    pipeline = build(retriever, llm, max_retrieval_rounds=3)
    await pipeline.assess(query="a hard question about ijarah", principal_id="p")
    # 1 seed + at most 2 refinement rounds
    assert len(retriever.calls) <= 3
    assert llm.complete_calls <= 2


async def test_repeated_identical_search_is_suppressed():
    repeat = ToolCall(id="t1", name="search_standards",
                      arguments={"query": "same query", "standard_no": None, "reason": "r"})
    script = [[repeat], [repeat]]
    retriever, llm = FakeRetriever(), FakeLLM(draft=good_draft(), tool_script=script)
    pipeline = build(retriever, llm, max_retrieval_rounds=3)
    await pipeline.assess(query="original question here", principal_id="p")
    searched = [c["query"] for c in retriever.calls]
    assert searched.count("same query") == 1


async def test_no_evidence_skips_the_model_entirely():
    """Reasoning over nothing produces confident nonsense, so it is not attempted
    — which also means the request costs nothing."""
    retriever, llm = FakeRetriever(results=[]), FakeLLM(draft=good_draft())
    result = await build(retriever, llm).assess(query="something unrelated", principal_id="p")
    assert result.assessment.verdict is Verdict.NEEDS_REVIEW
    assert llm.structured_calls == 0


async def test_upstream_failure_degrades_to_review_not_to_an_answer():
    retriever, llm = FakeRetriever(fail=True), FakeLLM(draft=good_draft())
    result = await build(retriever, llm).assess(query="a murabaha question", principal_id="p")
    assert result.assessment.verdict is Verdict.NEEDS_REVIEW
    assert any(e.code.value == "upstream_error" for e in result.assessment.escalations)


async def test_pii_is_redacted_before_anything_leaves_the_process():
    """Redaction protects egress; it does not refuse the question.

    An analyst who pastes an account number into an otherwise sound query should
    still get an answer about the product structure — with the identifier
    stripped before it reaches an external API.
    """
    retriever, llm = FakeRetriever(), FakeLLM(draft=good_draft())
    result = await build(retriever, llm).assess(
        query="Customer 784-1990-1234567-1 asks whether this murabaha is valid",
        principal_id="p",
    )
    assert "784-1990" not in result.assessment.query
    assert "[EMIRATES_ID_REDACTED]" in result.assessment.query
    # The question is still answered on its merits.
    assert result.assessment.verdict is Verdict.NON_COMPLIANT
    assert result.assessment.escalations == []


async def test_redacted_query_is_what_reaches_retrieval():
    """The redaction has to happen upstream of the retriever, not just in the
    response — otherwise the identifier still reaches the embedding provider."""
    retriever, llm = FakeRetriever(), FakeLLM(draft=good_draft())
    await build(retriever, llm).assess(
        query="Does IBAN AE070331234567890123456 qualify for murabaha treatment?",
        principal_id="p",
    )
    searched = " ".join(c["query"] for c in retriever.calls)
    assert "AE07033" not in searched
    assert "IBAN_REDACTED" in searched


async def test_audit_record_is_replayable():
    retriever, llm = FakeRetriever(), FakeLLM(draft=good_draft())
    result = await build(retriever, llm).assess(query="a murabaha question", principal_id="p")
    audit = result.audit
    # The fields that make a verdict reproducible months later.
    assert audit.corpus_version == "test@abc123"
    assert audit.model and audit.prompt_version
    assert audit.retrieved and audit.retrieved[0].rerank_score == 0.91
    assert audit.retrieval_rounds[0]["source"] == "seed"
    assert audit.usage.input_tokens > 0
    assert {s.stage for s in audit.stages} >= {"intake", "reason", "guardrails"}


async def test_invalid_model_output_escalates():
    retriever, llm = FakeRetriever(), FakeLLM(draft=None)
    result = await build(retriever, llm).assess(query="a murabaha question", principal_id="p")
    assert result.assessment.verdict is Verdict.NEEDS_REVIEW
    assert any(e.code.value == "schema_validation_failed" for e in result.assessment.escalations)
