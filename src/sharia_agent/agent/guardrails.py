"""Deterministic gates between the model's draft and the returned verdict.

The invariant this module exists to hold:

    **Guardrails can only move a result toward NEEDS_REVIEW. Nothing here can
    produce COMPLIANT, and nothing here can make a result less cautious.**

That is what lets the service claim NEEDS_REVIEW is an escalation decision made
in code rather than a label the model chose. It also means a regression in the
model can make the system noisier, but not more permissive.
"""

from __future__ import annotations

import re

from ..config import Settings
from ..models import (
    DraftAssessment,
    Escalation,
    EscalationCode,
    ModelFinding,
    RetrievedClause,
    Verdict,
)

# Categories that go to a human regardless of how confident the model is.
# These are not model-failure cases — they are matters where a machine
# assessment is not the appropriate artefact, however good it is.
ALWAYS_REVIEW: list[tuple[str, re.Pattern[str]]] = [
    (
        "novel structure with no settled treatment",
        re.compile(
            r"\b(novel|new product|first of its kind|unprecedented|bespoke structure)\b",
            re.I,
        ),
    ),
    (
        "capital or profit guarantee on a profit-sharing contract",
        re.compile(
            r"\b(guarantee\w*|capital[- ]protect\w*|principal[- ]protect\w*)\b.{0,60}"
            r"\b(mudarab\w*|musharak\w*|profit|return)\b|"
            r"\b(mudarab\w*|musharak\w*)\b.{0,60}\b(guarantee\w*|capital[- ]protect\w*)\b",
            re.I,
        ),
    ),
    (
        "cross-border or multi-jurisdiction structuring",
        re.compile(
            r"\b(cross[- ]border|offshore|multi[- ]jurisdiction|foreign branch)\b", re.I
        ),
    ),
    (
        "sukuk issuance or restructuring",
        re.compile(
            r"\bsukuk\b.{0,40}"
            r"\b(issu\w*|structur\w*|restructur\w*|securitis\w*|securitiz\w*)\b",
            re.I,
        ),
    ),
    (
        "request framed as seeking approval rather than assessment",
        re.compile(
            r"\b(approve|approval|sign[- ]off|authorise|authorize)\b.{0,30}"
            r"\b(this|the|our|product|deal)\b",
            re.I,
        ),
    ),
]

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def verify_citations(
    draft: DraftAssessment, retrieved: list[RetrievedClause]
) -> list[Escalation]:
    """Check every citation resolves, and that its quote is really in that clause.

    Quote verification is the cheap half of faithfulness checking: a fabricated
    or drifted quote is caught here without a second model call. It does not
    prove the reasoning is sound, only that the evidence it names is real.
    """
    by_id = {r.clause.chunk_id: r.clause for r in retrieved}
    problems: list[Escalation] = []

    for citation in draft.citations:
        clause = by_id.get(citation.chunk_id)
        if clause is None:
            problems.append(
                Escalation(
                    code=EscalationCode.UNRESOLVED_CITATION,
                    detail=(
                        f"citation {citation.chunk_id!r} was not among the clauses "
                        "retrieved for this request"
                    ),
                )
            )
            continue

        quote = _norm(citation.quote)
        if quote and quote not in _norm(clause.text):
            problems.append(
                Escalation(
                    code=EscalationCode.UNRESOLVED_CITATION,
                    detail=(
                        f"quoted text does not appear verbatim in {citation.chunk_id} "
                        f"({citation.quote[:70]!r})"
                    ),
                )
            )
    return problems


def check_query_category(query: str) -> list[Escalation]:
    return [
        Escalation(code=EscalationCode.ALWAYS_REVIEW_CATEGORY, detail=reason)
        for reason, pattern in ALWAYS_REVIEW
        if pattern.search(query)
    ]


def check_retrieval(
    retrieved: list[RetrievedClause], settings: Settings
) -> list[Escalation]:
    out: list[Escalation] = []
    if not retrieved:
        out.append(
            Escalation(code=EscalationCode.WEAK_RETRIEVAL, detail="no clauses retrieved")
        )
        return out

    scores = [r.rerank_score for r in retrieved if r.rerank_score is not None]
    if not scores:
        # The reranker degraded to fusion order. Retrieval quality is then
        # unverified rather than merely low, so the request goes to a human.
        out.append(
            Escalation(
                code=EscalationCode.WEAK_RETRIEVAL,
                detail="reranker unavailable; retrieval quality could not be verified",
            )
        )
        return out

    best = max(scores)
    if best < settings.min_rerank_score:
        out.append(
            Escalation(
                code=EscalationCode.WEAK_RETRIEVAL,
                detail=(
                    f"best reranked clause scored {best:.3f}, below the "
                    f"{settings.min_rerank_score:.2f} floor for reasoning"
                ),
            )
        )

    superseded = [r.clause.chunk_id for r in retrieved if r.clause.superseded_by]
    if superseded:
        out.append(
            Escalation(
                code=EscalationCode.SUPERSEDED_STANDARD,
                detail=f"retrieved superseded clauses: {', '.join(superseded[:5])}",
            )
        )
    return out


def decide(
    query: str,
    draft: DraftAssessment | None,
    retrieved: list[RetrievedClause],
    settings: Settings,
    prior: list[Escalation] | None = None,
) -> tuple[Verdict, list[Escalation]]:
    """Map a draft onto the returned verdict. The only place that mapping happens."""
    escalations: list[Escalation] = list(prior or [])
    escalations += check_query_category(query)
    escalations += check_retrieval(retrieved, settings)

    if draft is None:
        escalations.append(
            Escalation(
                code=EscalationCode.SCHEMA_VALIDATION_FAILED,
                detail="model did not return a valid assessment",
            )
        )
        return Verdict.NEEDS_REVIEW, _dedupe(escalations)

    escalations += verify_citations(draft, retrieved)

    if len(draft.citations) < settings.min_citations:
        escalations.append(
            Escalation(
                code=EscalationCode.NO_CITATIONS,
                detail=(
                    f"{len(draft.citations)} citations provided, "
                    f"minimum is {settings.min_citations}"
                ),
            )
        )

    if draft.confidence < settings.min_model_confidence:
        escalations.append(
            Escalation(
                code=EscalationCode.LOW_CONFIDENCE,
                detail=(
                    f"model confidence {draft.confidence:.2f} below the "
                    f"{settings.min_model_confidence:.2f} threshold"
                ),
            )
        )

    if draft.finding is ModelFinding.INSUFFICIENT_BASIS:
        escalations.append(
            Escalation(
                code=EscalationCode.INSUFFICIENT_BASIS,
                detail="retrieved clauses do not settle the question",
            )
        )
    elif draft.finding is ModelFinding.CONFLICTING_SOURCES:
        escalations.append(
            Escalation(
                code=EscalationCode.CONFLICTING_SOURCES,
                detail="retrieved clauses point in different directions",
            )
        )

    escalations = _dedupe(escalations)
    if escalations:
        return Verdict.NEEDS_REVIEW, escalations

    # Only reachable with a clean draft and every gate passed.
    return (
        Verdict.COMPLIANT
        if draft.finding is ModelFinding.SUPPORTED_COMPLIANT
        else Verdict.NON_COMPLIANT
    ), []


def _dedupe(items: list[Escalation]) -> list[Escalation]:
    seen: set[tuple[str, str]] = set()
    out: list[Escalation] = []
    for e in items:
        key = (e.code.value, e.detail)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out
