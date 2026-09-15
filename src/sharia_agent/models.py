"""Domain models.

Two vocabularies are deliberately kept apart:

* What the **model** may say — `ModelFinding`, which is about *evidence*
  (does the retrieved text support a conclusion?).
* What the **service** returns — `Verdict`, which is about *routing*
  (does this go back to the requester, or to the ISSC?).

Only deterministic code maps the first onto the second. That is what keeps
NEEDS_REVIEW an escalation outcome rather than a label the model can pick to
avoid committing, and it is why a guardrail can never manufacture a COMPLIANT.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


class Clause(BaseModel):
    """One citable unit of a Shari'ah standard.

    The clause — not an arbitrary character window — is the chunk. It is also
    the unit a compliance officer cites, so a retrieved chunk maps 1:1 onto a
    reference they can verify by hand.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    standard_no: str
    standard_name: str
    clause_path: str
    heading_path: list[str] = Field(default_factory=list)
    text: str
    lang: Literal["en", "ar"] = "en"
    corpus_version: str
    supersedes: str | None = None
    superseded_by: str | None = None

    @property
    def citation_label(self) -> str:
        return f"AAOIFI SS No. {self.standard_no} ({self.standard_name}), clause {self.clause_path}"

    def for_prompt(self) -> str:
        """Render with the heading path attached.

        Disclosure documents repeat near-identical language across products —
        profit distribution for Mudaraba and for Wakala read alike and mean
        different things. Without the heading path in the text itself, neither
        the retriever nor the model can tell them apart.
        """
        trail = " > ".join(self.heading_path) if self.heading_path else self.standard_name
        return f"[{self.chunk_id}] {self.citation_label}\n{trail}\n\n{self.text}"


class RetrievedClause(BaseModel):
    """A clause plus the scores that selected it. Persisted into the audit record."""

    clause: Clause
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    retrieval_round: int = 1


# ---------------------------------------------------------------------------
# What the model is allowed to return
# ---------------------------------------------------------------------------


class ModelFinding(StrEnum):
    """The model's read of the evidence. Not a verdict."""

    SUPPORTED_COMPLIANT = "SUPPORTED_COMPLIANT"
    SUPPORTED_NON_COMPLIANT = "SUPPORTED_NON_COMPLIANT"
    INSUFFICIENT_BASIS = "INSUFFICIENT_BASIS"
    CONFLICTING_SOURCES = "CONFLICTING_SOURCES"


class ShariahConcern(StrEnum):
    """Named prohibitions and structural issues the analysis may implicate."""

    RIBA = "riba"                      # interest / unlawful increase
    GHARAR = "gharar"                  # excessive uncertainty
    MAYSIR = "maysir"                  # speculation / gambling
    HARAM_UNDERLYING = "haram_underlying"
    OWNERSHIP_SEQUENCE = "ownership_sequence"   # e.g. selling before possession
    LATE_PAYMENT_PENALTY = "late_payment_penalty"
    PROFIT_GUARANTEE = "profit_guarantee"       # guaranteeing return on PLS contracts
    WAAD_BINDING = "waad_binding"               # binding-promise structure
    OTHER = "other"


class Citation(BaseModel):
    """A pointer into the retrieved set. Validated — a citation that does not
    resolve to a clause actually retrieved this run fails the pipeline."""

    chunk_id: str
    quote: str = Field(description="Verbatim span from the cited clause.")
    supports: str = Field(description="Which part of the reasoning this backs.")


class DraftAssessment(BaseModel):
    """The model's structured output. Deliberately has no `verdict` field."""

    finding: ModelFinding
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    citations: list[Citation] = Field(default_factory=list)
    concerns: list[ShariahConcern] = Field(default_factory=list)
    missing_information: list[str] = Field(
        default_factory=list,
        description="What a reviewer would need in order to reach a firm conclusion.",
    )


# ---------------------------------------------------------------------------
# What the service returns
# ---------------------------------------------------------------------------


class Verdict(StrEnum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class EscalationCode(StrEnum):
    """Why a draft was routed to human review. Machine-readable so the
    escalation mix can be tracked as a product metric."""

    WEAK_RETRIEVAL = "weak_retrieval"
    NO_CITATIONS = "no_citations"
    UNRESOLVED_CITATION = "unresolved_citation"
    LOW_CONFIDENCE = "low_confidence"
    INSUFFICIENT_BASIS = "insufficient_basis"
    CONFLICTING_SOURCES = "conflicting_sources"
    SUPERSEDED_STANDARD = "superseded_standard"
    ALWAYS_REVIEW_CATEGORY = "always_review_category"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    UPSTREAM_ERROR = "upstream_error"


class Escalation(BaseModel):
    code: EscalationCode
    detail: str


DISCLAIMER = (
    "Decision-support output. This is not a fatwa and does not constitute Shari'ah "
    "approval. Under CBUAE rules, Shari'ah determinations are reserved to the "
    "institution's Internal Shari'ah Supervision Committee."
)


class Assessment(BaseModel):
    """The response body of POST /assess."""

    assessment_id: str
    trace_id: str
    query: str
    verdict: Verdict
    confidence: float
    reasoning: str
    citations: list[Citation]
    concerns: list[ShariahConcern]
    escalations: list[Escalation]
    missing_information: list[str]
    clauses_considered: list[str] = Field(description="chunk_ids sent to the model")
    corpus_version: str
    model: str
    prompt_version: str
    latency_ms: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    disclaimer: str = DISCLAIMER


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


class StageTiming(BaseModel):
    stage: str
    ms: int
    ok: bool = True
    note: str | None = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class AuditRecord(BaseModel):
    """Everything needed to replay a verdict months later.

    Pinning corpus_version, index_snapshot, model and prompt_version is what
    makes a disputed verdict reproducible — and therefore what lets you split a
    retrieval miss from a reasoning miss instead of guessing.
    """

    trace_id: str
    assessment_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    principal_id: str
    query_hash: str
    query: str | None = None  # omitted when SCA_LOG_FULL_PROMPTS is false
    # The exact system + user messages sent to the reasoning model. prompt_version
    # and retrieved[] make the prompt reconstructible, but reconstruction assumes
    # the template on disk still matches the one that ran; recording it removes
    # that assumption. Omitted when SCA_LOG_FULL_PROMPTS is false.
    prompt_sent: dict[str, str] | None = None

    corpus_version: str
    index_snapshot: str
    model: str
    model_effort: str
    prompt_version: str

    retrieval_rounds: list[dict] = Field(default_factory=list)
    retrieved: list[RetrievedClause] = Field(default_factory=list)
    draft: DraftAssessment | None = None
    final_verdict: Verdict | None = None
    escalations: list[Escalation] = Field(default_factory=list)

    stages: list[StageTiming] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = None
