"""Deterministic gates between the model's draft and the returned verdict.

The invariant this module exists to hold:

    **Guardrails can only move a result toward NEEDS_REVIEW. Nothing here can
    produce COMPLIANT, and nothing here can make a result less cautious.**

That is what lets the service claim NEEDS_REVIEW is an escalation decision made
in code rather than a label the model chose. It also means a regression in the
model can make the system noisier, but not more permissive.
"""

from __future__ import annotations

import difflib
import re

from ..config import Settings
from ..models import (
    DraftAssessment,
    Escalation,
    EscalationCode,
    ModelFinding,
    RetrievedClause,
    ShariahConcern,
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
# "... " between two spans is a quoting convention, not an edit. A model that
# elides with an ellipsis is signalling the omission honestly; treating the
# skipped text as an interior change flagged three correct citations as
# polarity violations.
_ELLIPSIS = re.compile(r"\s*(?:\.\s*\.\s*\.|…|\[\s*\.{2,}\s*\])\s*")
_WORD = re.compile(r"[a-z0-9]+")

# Quote checking answers two separate questions, and one metric cannot do both.
#
#   PROVENANCE  — is this text really from the clause it names?
#   POLARITY    — does it say the same thing the clause says?
#   SUBSTANCE   — is it enough text to be evidence at all?
#
# Character containment answers provenance well: a quote lifted from a different
# clause scores 0.47-0.76 against the one it claims. It answers the other two not
# at all. Measured on this corpus, a negation flip ("shall not sell" -> "may
# sell") scores 0.978 and the fragment "the Institution and the customer" scores
# a perfect 1.0 because it is genuinely a substring. No single threshold
# separates those from a true quote at 0.998, so all three run.
QUOTE_SIMILARITY_FLOOR = 0.92
MIN_QUOTE_WORDS = 8

# Tokens that carry the legal force of a clause. If one of these is introduced,
# dropped or swapped inside the span a quote claims to reproduce, the quote
# asserts something the clause does not — the single most dangerous way for a
# citation to be wrong, because it is otherwise near-identical.
_POLARITY = frozenset({
    "not", "no", "never", "nor", "without", "unless", "except",
    "shall", "must", "may", "should", "cannot",
    "permissible", "impermissible", "permitted", "prohibited", "forbidden",
    "obligatory", "valid", "invalid", "void", "required", "allowed",
})


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _words(text: str) -> list[str]:
    return _WORD.findall(_norm(text))


def quote_fragments(quote: str) -> list[str]:
    """Split an elided quote into the spans it actually claims to reproduce."""
    return [f for f in (part.strip() for part in _ELLIPSIS.split(quote)) if f]


def quote_containment(quote: str, clause_text: str) -> float:
    """Fraction of `quote` recoverable from `clause_text`, preserving order."""
    q, c = _norm(quote), _norm(clause_text)
    if not q:
        return 0.0
    if q in c:
        return 1.0
    # autojunk treats common characters as noise in long strings and would
    # deflate the score on exactly the long quotes that matter most.
    matcher = difflib.SequenceMatcher(None, q, c, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(q)


def _aligned_window(q_words: list[str], c_words: list[str], slack: int = 8) -> list[str]:
    """The span of the clause the quote most plausibly came from.

    Aligning a quote against the *whole* clause misfires when a stem phrase
    recurs. "The institution should guarantee ... whereas it should not assume"
    contains "institution should" twice, so a quote beginning with those words
    anchors to the first occurrence and everything up to the real one reads as an
    interior edit — reporting a polarity change where the quote is verbatim.

    Anchoring on the longest matching run first, then comparing only within a
    window around it, removes that whole class of false positive without
    loosening what the check actually detects.
    """
    if not q_words or len(q_words) >= len(c_words):
        return c_words
    match = difflib.SequenceMatcher(
        None, q_words, c_words, autojunk=False
    ).find_longest_match(0, len(q_words), 0, len(c_words))
    if match.size == 0:
        return c_words
    start = max(0, match.b - match.a - slack)
    return c_words[start : start + len(q_words) + 2 * slack]


def polarity_mismatch(quote: str, clause_text: str) -> str | None:
    """Name a legal-force token the quote changes, or None if polarity holds.

    Aligns the quote against the clause token-wise and inspects only the edits
    *inside* the span the quote covers. Leading and trailing deletions are the
    rest of the clause and are expected; an edit in the middle that touches a
    polarity token is the quote rewriting the rule.
    """
    q_words, c_words = _words(quote), _words(clause_text)
    if not q_words:
        return None
    c_words = _aligned_window(q_words, c_words)

    opcodes = difflib.SequenceMatcher(None, q_words, c_words, autojunk=False).get_opcodes()
    interior = [op for op in opcodes if op[0] != "equal"]
    # A quote is a fragment, so the clause text before and after the span it
    # covers appears as an `insert` at either end (tokens in the clause, absent
    # from the quote). Those are expected; only edits *within* the span mean the
    # quote has rewritten something.
    if interior and interior[0][0] == "insert" and interior[0][1] == 0:
        interior = interior[1:]
    if interior and interior[-1][0] == "insert" and interior[-1][1] == len(q_words):
        interior = interior[:-1]

    for tag, i1, i2, j1, j2 in interior:
        touched = set(q_words[i1:i2]) | set(c_words[j1:j2])
        if offending := touched & _POLARITY:
            got = " ".join(q_words[i1:i2]) or "(nothing)"
            expected = " ".join(c_words[j1:j2]) or "(nothing)"
            return f"{tag} of {sorted(offending)}: quote says {got!r}, clause says {expected!r}"
    return None


def verify_citations(
    draft: DraftAssessment, retrieved: list[RetrievedClause]
) -> list[Escalation]:
    """Check every citation resolves, and that its quote is really in that clause.

    Quote verification is the cheap half of faithfulness checking: a fabricated
    or drifted quote is caught here without a second model call. It does not
    prove the reasoning is sound, only that the evidence it names is real.

    Matching is fuzzy rather than exact — see QUOTE_SIMILARITY_FLOOR for why.
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

        if not citation.quote:
            continue

        # Each elided span is verified on its own terms. The check asks whether
        # the text quoted is really in the clause and says the same thing — not
        # whether the quote is a complete restatement of it.
        fragments = quote_fragments(citation.quote)
        scores = [quote_containment(f, clause.text) for f in fragments]
        score = min(scores) if scores else 0.0
        if score < QUOTE_SIMILARITY_FLOOR:
            problems.append(
                Escalation(
                    code=EscalationCode.UNRESOLVED_CITATION,
                    detail=(
                        f"quote is only {score:.0%} recoverable from {citation.chunk_id} "
                        f"(floor {QUOTE_SIMILARITY_FLOOR:.0%}): {citation.quote[:70]!r}"
                    ),
                )
            )
            continue

        mismatch = next(
            (m for f in fragments if (m := polarity_mismatch(f, clause.text))), None
        )
        if mismatch is not None:
            problems.append(
                Escalation(
                    code=EscalationCode.UNRESOLVED_CITATION,
                    detail=f"quote alters the force of {citation.chunk_id} — {mismatch}",
                )
            )
            continue

        if len(_words(citation.quote)) < MIN_QUOTE_WORDS:
            problems.append(
                Escalation(
                    code=EscalationCode.NO_CITATIONS,
                    detail=(
                        f"quote from {citation.chunk_id} is too short to be evidence "
                        f"({len(_words(citation.quote))} words, minimum {MIN_QUOTE_WORDS})"
                    ),
                )
            )
    return problems


# Concerns that route to a human however the request was phrased. A capital or
# profit guarantee on a profit-sharing contract is the ISSC's call, and whether
# it escalates must not depend on the requester happening to use the word
# "guarantee" — "cover any capital loss" is the same arrangement and matched
# none of the patterns below.
ALWAYS_REVIEW_CONCERNS = frozenset({ShariahConcern.PROFIT_GUARANTEE})


def check_query_category(
    query: str, concerns: list[ShariahConcern] | None = None
) -> list[Escalation]:
    out = [
        Escalation(code=EscalationCode.ALWAYS_REVIEW_CATEGORY, detail=reason)
        for reason, pattern in ALWAYS_REVIEW
        if pattern.search(query)
    ]
    for concern in ALWAYS_REVIEW_CONCERNS.intersection(concerns or ()):
        out.append(
            Escalation(
                code=EscalationCode.ALWAYS_REVIEW_CATEGORY,
                detail=(
                    f"the assessment itself identifies {concern.value!r}, which is "
                    "reserved to the ISSC regardless of how the request was worded"
                ),
            )
        )
    return out


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
    escalations += check_query_category(query, draft.concerns if draft else None)
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
