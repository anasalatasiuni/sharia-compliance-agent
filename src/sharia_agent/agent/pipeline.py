"""The assessment pipeline.

Eight stages, executed in a fixed order, with a validated boundary between each
one. Written as an explicit state machine rather than handed to an agent
framework for two reasons: a regulator reviewing a verdict has to be able to
read the control flow, and an unbounded ReAct loop produces a different
trajectory on every run, which makes a disputed verdict unreproducible.

    1 INTAKE     redact, classify, mint ids
    2 RETRIEVE   deterministic seed search
    3 REFINE     bounded agentic loop — the model may search again, N times max
    4 GATE A     enough evidence to reason over at all?
    5 REASON     structured output — the model's draft finding
    6 VALIDATE   citations resolve, quotes are real
    7 GATE B     guardrails; may escalate, may never relax
    8 EMIT       assessment + audit record

The model's agency is confined to stage 3. Stages 4 and 7 are the only places a
verdict is decided, and neither calls a model.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass

from ..config import Settings
from ..llm import LLM, Usage
from ..models import (
    Assessment,
    AuditRecord,
    DraftAssessment,
    Escalation,
    EscalationCode,
    RetrievedClause,
    StageTiming,
    TokenUsage,
)
from ..obs.trace import get_trace_id, log, span
from ..pii import redact
from ..resilience import UpstreamUnavailable
from ..retrieval.hybrid import Retriever
from . import guardrails, prompts
from .tools import TOOLS

# Clauses carried into the final reasoning call. Held deliberately low: retrieved
# context is the only part of the prompt that cannot be cached, so it dominates
# cost at volume — and extra weakly-relevant clauses degrade the answer anyway.
MAX_CLAUSES_TO_MODEL = 8


@dataclass
class PipelineResult:
    assessment: Assessment
    audit: AuditRecord


class CompliancePipeline:
    def __init__(self, retriever: Retriever, llm: LLM, settings: Settings) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    # -- public ------------------------------------------------------------

    async def assess(
        self, *, query: str, principal_id: str, index_snapshot: str = "unknown"
    ) -> PipelineResult:
        started = time.perf_counter()
        assessment_id = uuid.uuid4().hex
        trace_id = get_trace_id()

        audit = AuditRecord(
            trace_id=trace_id,
            assessment_id=assessment_id,
            principal_id=principal_id,
            query_hash=hashlib.sha256(query.encode()).hexdigest()[:16],
            corpus_version="unknown",
            index_snapshot=index_snapshot,
            model=self.settings.model,
            model_effort=self.settings.model_effort,
            prompt_version=prompts.PROMPT_VERSION,
        )

        # ---- 1 INTAKE ----------------------------------------------------
        with span("intake") as s:
            safe_query, report = redact(query)
            if report.total:
                log("pii.redacted", categories=dict(report), count=report.total)
            s.annotate(redactions=report.total)
            audit.query = safe_query if self.settings.log_full_prompts else None
        # Read after the block: span.ms is only set once the context exits.
        audit.stages.append(StageTiming(stage="intake", ms=s.ms))

        prior: list[Escalation] = []
        if report.total:
            prior.append(
                Escalation(
                    code=EscalationCode.ALWAYS_REVIEW_CATEGORY,
                    detail=(
                        "request contained personal identifiers; a compliance question "
                        "should concern a product structure, not an individual"
                    ),
                )
            )

        retrieved: list[RetrievedClause] = []
        draft: DraftAssessment | None = None

        try:
            # ---- 2 RETRIEVE + 3 REFINE -----------------------------------
            with span("evidence") as s:
                retrieved = await self._gather_evidence(safe_query, audit)
                s.annotate(clauses=len(retrieved))
            audit.stages.append(StageTiming(stage="evidence", ms=s.ms))

            if retrieved:
                audit.corpus_version = retrieved[0].clause.corpus_version
            audit.retrieved = retrieved

            # ---- 4 GATE A -------------------------------------------------
            gate_a = guardrails.check_retrieval(retrieved, self.settings)
            blocking = [
                e for e in gate_a if e.code is EscalationCode.WEAK_RETRIEVAL
            ]
            if blocking:
                # Reasoning over evidence this thin produces confident nonsense.
                # Skip the model call entirely — it saves the spend and the answer
                # would have been escalated regardless.
                log("gate_a.blocked", reasons=[e.code.value for e in blocking])
            else:
                # ---- 5 REASON ---------------------------------------------
                draft = await self._reason(safe_query, retrieved, audit)

        except UpstreamUnavailable as exc:
            log("pipeline.upstream_unavailable", service=exc.service, reason=exc.reason)
            audit.error = str(exc)
            prior.append(
                Escalation(
                    code=EscalationCode.UPSTREAM_ERROR,
                    detail=f"{exc.service} unavailable: {exc.reason}",
                )
            )

        # ---- 6 VALIDATE + 7 GATE B ---------------------------------------
        with span("guardrails") as s:
            verdict, escalations = guardrails.decide(
                query=safe_query,
                draft=draft,
                retrieved=retrieved,
                settings=self.settings,
                prior=prior,
            )
            s.annotate(
                verdict=verdict.value,
                escalations=[e.code.value for e in escalations],
            )
        audit.stages.append(StageTiming(stage="guardrails", ms=s.ms))

        audit.draft = draft
        audit.final_verdict = verdict
        audit.escalations = escalations

        # ---- 8 EMIT -------------------------------------------------------
        latency_ms = int((time.perf_counter() - started) * 1000)
        assessment = Assessment(
            assessment_id=assessment_id,
            trace_id=trace_id,
            query=safe_query,
            verdict=verdict,
            confidence=draft.confidence if draft else 0.0,
            reasoning=(
                draft.reasoning
                if draft
                else "No assessment was produced; this request requires human review."
            ),
            citations=draft.citations if draft else [],
            concerns=draft.concerns if draft else [],
            escalations=escalations,
            missing_information=draft.missing_information if draft else [],
            clauses_considered=[r.clause.chunk_id for r in retrieved],
            corpus_version=audit.corpus_version,
            model=self.settings.model,
            prompt_version=prompts.PROMPT_VERSION,
            latency_ms=latency_ms,
        )

        log(
            "assessment.complete",
            assessment_id=assessment_id,
            verdict=verdict.value,
            confidence=assessment.confidence,
            clauses=len(retrieved),
            escalations=[e.code.value for e in escalations],
            latency_ms=latency_ms,
            input_tokens=audit.usage.input_tokens,
            output_tokens=audit.usage.output_tokens,
        )
        return PipelineResult(assessment=assessment, audit=audit)

    # -- stages 2 & 3 ------------------------------------------------------

    async def _gather_evidence(
        self, query: str, audit: AuditRecord
    ) -> list[RetrievedClause]:
        """Seed search, then let the model refine within a hard budget.

        The seed search is unconditional: retrieval must not depend on the model
        choosing to call a tool. The refinement rounds exist because the
        requester's wording and the standards' wording routinely disagree, and
        reformulation is the one part of retrieval a model is genuinely good at.
        """
        collected: dict[str, RetrievedClause] = {}

        seed = await self.retriever.search(query, round_no=1)
        for item in seed:
            collected[item.clause.chunk_id] = item
        audit.retrieval_rounds.append(
            {"round": 1, "query": query, "source": "seed", "hits": len(seed)}
        )

        budget = self.settings.max_retrieval_rounds - 1
        if budget <= 0:
            return self._top_clauses(collected)

        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Question under assessment:\n{query}\n\n"
                    f"An initial search already returned these clauses:\n\n"
                    f"{prompts.render_clauses([c.clause for c in seed])}\n\n"
                    + prompts.SEARCH_TOOL_GUIDANCE.format(max_rounds=budget)
                    + "\n\nIf these clauses are sufficient, reply with the single "
                    "word DONE and make no tool call."
                ),
            }
        ]

        seen_queries: set[str] = {query.strip().lower()}

        for round_no in range(2, self.settings.max_retrieval_rounds + 1):
            with span("refine", round=round_no) as s:
                response = await self.llm.complete(
                    messages=messages,
                    system=prompts.SYSTEM,
                    tools=TOOLS,
                    max_tokens=1024,
                )
                self._add_usage(audit, response.usage)
                calls = response.tool_calls
                s.annotate(tool_calls=len(calls), finish_reason=response.finish_reason)

            if not calls:
                break

            messages.append(response.raw_message)
            results = []
            for block in calls:
                args = block.arguments or {}
                sub_query = (args.get("query") or "").strip()
                standard_no = args.get("standard_no") or None

                # Loop detection: an identical repeat search is the failure mode
                # a step cap alone does not catch, and it burns a full round.
                key = f"{sub_query.lower()}|{standard_no or ''}"
                if not sub_query or key in seen_queries:
                    results.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.id,
                            "content": (
                                "This search was already performed. Do not repeat it. "
                                "Assess with the clauses you have, or search differently."
                            ),
                        }
                    )
                    log("refine.repeat_suppressed", query=sub_query[:80])
                    continue
                seen_queries.add(key)

                hits = await self.retriever.search(
                    sub_query, standard_no=standard_no, round_no=round_no
                )
                for item in hits:
                    existing = collected.get(item.clause.chunk_id)
                    if existing is None or (item.rerank_score or 0) > (
                        existing.rerank_score or 0
                    ):
                        collected[item.clause.chunk_id] = item

                audit.retrieval_rounds.append(
                    {
                        "round": round_no,
                        "query": sub_query,
                        "standard_no": standard_no,
                        "reason": args.get("reason"),
                        "hits": len(hits),
                    }
                )
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.id,
                        "content": prompts.render_clauses([h.clause for h in hits])
                        or "No clauses matched.",
                    }
                )

            # OpenAI-shaped tool results are individual messages, not one bundle.
            messages.extend(results)

        return self._top_clauses(collected)

    def _top_clauses(self, collected: dict[str, RetrievedClause]) -> list[RetrievedClause]:
        return sorted(
            collected.values(),
            key=lambda r: (r.rerank_score or r.fused_score or 0.0),
            reverse=True,
        )[:MAX_CLAUSES_TO_MODEL]

    # -- stage 5 -----------------------------------------------------------

    async def _reason(
        self, query: str, retrieved: list[RetrievedClause], audit: AuditRecord
    ) -> DraftAssessment | None:
        """Produce the draft under a schema the model cannot deviate from.

        Structured output rather than 'respond in JSON': the format is enforced
        by the API, so a parsing failure here means a genuine outage rather than
        a formatting accident to be retried.
        """
        user = prompts.build_user_message(query, [r.clause for r in retrieved])

        with span("reason", clauses=len(retrieved)) as s:
            draft, usage = await self.llm.complete_structured(
                output_model=DraftAssessment,
                messages=[{"role": "user", "content": user}],
                system=prompts.SYSTEM,
                max_tokens=4096,
            )
            self._add_usage(audit, usage)
            s.annotate(
                finding=draft.finding.value if draft else None,
                confidence=draft.confidence if draft else None,
                citations=len(draft.citations) if draft else 0,
            )
        audit.stages.append(StageTiming(stage="reason", ms=s.ms))
        return draft

    # -- shared ------------------------------------------------------------

    @staticmethod
    def _add_usage(audit: AuditRecord, usage: Usage) -> None:
        """Token spend accumulates across every call a single assessment makes.

        Recorded per assessment rather than per call because the unit that
        matters for cost is a completed question, not an HTTP request — a cheap
        request that needed three retrieval rounds is not cheap.
        """
        audit.usage = TokenUsage(
            input_tokens=audit.usage.input_tokens + usage.input_tokens,
            output_tokens=audit.usage.output_tokens + usage.output_tokens,
            cache_read_input_tokens=audit.usage.cache_read_input_tokens,
            cache_creation_input_tokens=audit.usage.cache_creation_input_tokens,
        )


def audit_to_json(audit: AuditRecord) -> str:
    return json.dumps(audit.model_dump(mode="json"), ensure_ascii=False)
