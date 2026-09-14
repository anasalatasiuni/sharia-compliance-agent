"""Guardrail behaviour.

The property under test throughout: guardrails may escalate, never relax. A
regression that makes the system noisier is tolerable; one that makes it
permissive is not, so the permissive direction is what these tests pin down.
"""

from __future__ import annotations

import pytest

from sharia_agent.agent import guardrails
from sharia_agent.config import Settings
from sharia_agent.models import (
    Citation,
    Clause,
    DraftAssessment,
    EscalationCode,
    ModelFinding,
    RetrievedClause,
    Verdict,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, min_rerank_score=0.35, min_model_confidence=0.70)


def clause(chunk_id="SS8-3.1.1", text="The Institution shall not sell any item in a "
           "Murabahah transaction before it acquires such item.") -> Clause:
    return Clause(
        chunk_id=chunk_id,
        standard_no="8",
        standard_name="Murabahah",
        clause_path="3/1/1",
        heading_path=["Murabahah", "3/1 Acquisition"],
        text=text,
        corpus_version="test@abc",
    )


def retrieved(score: float = 0.9, **kw) -> list[RetrievedClause]:
    return [RetrievedClause(clause=clause(**kw), rerank_score=score)]


def draft(**kw) -> DraftAssessment:
    base = dict(
        finding=ModelFinding.SUPPORTED_COMPLIANT,
        confidence=0.95,
        reasoning="The clause permits the arrangement.",
        citations=[
            Citation(
                chunk_id="SS8-3.1.1",
                quote="shall not sell any item in a Murabahah transaction",
                supports="ownership must precede sale",
            )
        ],
    )
    base.update(kw)
    return DraftAssessment(**base)


def test_clean_draft_returns_the_models_finding(settings):
    verdict, escalations = guardrails.decide(
        "Can we sell before taking ownership?", draft(), retrieved(), settings
    )
    assert verdict is Verdict.COMPLIANT
    assert escalations == []


def test_non_compliant_finding_passes_through(settings):
    verdict, _ = guardrails.decide(
        "q", draft(finding=ModelFinding.SUPPORTED_NON_COMPLIANT), retrieved(), settings
    )
    assert verdict is Verdict.NON_COMPLIANT


def test_fabricated_quote_is_caught(settings):
    """A quote that is not verbatim in the cited clause fails without a second
    model call — the cheap half of faithfulness checking."""
    bad = draft(
        citations=[
            Citation(
                chunk_id="SS8-3.1.1",
                quote="the Institution may sell before acquiring the item",
                supports="fabricated",
            )
        ]
    )
    verdict, escalations = guardrails.decide("q", bad, retrieved(), settings)
    assert verdict is Verdict.NEEDS_REVIEW
    assert any(e.code.value == "unresolved_citation" for e in escalations)


def test_citation_to_unretrieved_clause_is_caught(settings):
    bad = draft(
        citations=[Citation(chunk_id="SS99-1.1", quote="anything", supports="x")]
    )
    verdict, escalations = guardrails.decide("q", bad, retrieved(), settings)
    assert verdict is Verdict.NEEDS_REVIEW
    assert any(e.code.value == "unresolved_citation" for e in escalations)


def test_low_confidence_escalates(settings):
    verdict, escalations = guardrails.decide(
        "q", draft(confidence=0.4), retrieved(), settings
    )
    assert verdict is Verdict.NEEDS_REVIEW
    assert any(e.code.value == "low_confidence" for e in escalations)


def test_weak_retrieval_escalates(settings):
    verdict, escalations = guardrails.decide("q", draft(), retrieved(score=0.05), settings)
    assert verdict is Verdict.NEEDS_REVIEW
    assert any(e.code.value == "weak_retrieval" for e in escalations)


def test_missing_rerank_scores_escalate(settings):
    """A degraded reranker leaves ranking unverified, which is not the same as
    ranking being fine."""
    unscored = [RetrievedClause(clause=clause(), fused_score=0.8)]
    verdict, escalations = guardrails.decide("q", draft(), unscored, settings)
    assert verdict is Verdict.NEEDS_REVIEW
    assert any(e.code.value == "weak_retrieval" for e in escalations)


def test_no_citations_escalates(settings):
    verdict, escalations = guardrails.decide("q", draft(citations=[]), retrieved(), settings)
    assert verdict is Verdict.NEEDS_REVIEW
    assert any(e.code.value == "no_citations" for e in escalations)


def test_missing_draft_escalates(settings):
    verdict, escalations = guardrails.decide("q", None, retrieved(), settings)
    assert verdict is Verdict.NEEDS_REVIEW
    assert any(e.code.value == "schema_validation_failed" for e in escalations)


@pytest.mark.parametrize(
    "query",
    [
        "Can we guarantee the capital on a mudarabah deposit?",
        "Please approve this product for launch next week",
        "We are structuring a cross-border sukuk issuance",
        "This is a novel structure with no precedent",
    ],
)
def test_always_review_categories_escalate_despite_a_clean_draft(settings, query):
    """These are not model failures. They are matters where a machine assessment
    is not the appropriate artefact however confident it is."""
    verdict, escalations = guardrails.decide(query, draft(), retrieved(), settings)
    assert verdict is Verdict.NEEDS_REVIEW
    assert any(e.code.value == "always_review_category" for e in escalations)


@pytest.mark.parametrize(
    "finding", [ModelFinding.INSUFFICIENT_BASIS, ModelFinding.CONFLICTING_SOURCES]
)
def test_evidence_findings_never_become_a_verdict(settings, finding):
    verdict, escalations = guardrails.decide("q", draft(finding=finding), retrieved(), settings)
    assert verdict is Verdict.NEEDS_REVIEW
    assert escalations


def test_guardrails_can_never_produce_compliant_from_a_non_compliant_finding(settings):
    """The one-directional invariant, stated as a test."""
    for confidence in (0.0, 0.5, 0.99):
        verdict, _ = guardrails.decide(
            "q",
            draft(finding=ModelFinding.SUPPORTED_NON_COMPLIANT, confidence=confidence),
            retrieved(),
            settings,
        )
        assert verdict is not Verdict.COMPLIANT


# ---------------------------------------------------------------------------
# Citation verification
#
# Written after the first live assessment escalated a correct answer. The clause
# text is OCR'd and reads "concludes a urchase contract"; the model quoted it as
# "purchase", which is right, and exact substring matching punished it. Fixing
# that by loosening the match alone would have opened a worse hole — measured on
# the real corpus, a negation flip scores 0.978 and a generic fragment scores a
# perfect 1.0. Hence three independent checks.
# ---------------------------------------------------------------------------

OCR_CLAUSE = (
    "The Institution shall not sell any item in a Murabahah transaction before it "
    "acquires such item. Hence, it is not valid for the Institution to conclude a "
    "Murabahah sale with the customer before the Institution concludes a urchase "
    "contract with the supplier of the item the subject matter of the Murabahah and "
    "before it acquires actual or constructive possession of such items"
)


def cited(quote: str, chunk_id: str = "SS8-3.1.1") -> DraftAssessment:
    return draft(citations=[Citation(chunk_id=chunk_id, quote=quote, supports="x")])


def ocr_retrieved() -> list[RetrievedClause]:
    return [RetrievedClause(clause=clause(text=OCR_CLAUSE), rerank_score=0.95)]


def test_quote_survives_an_ocr_defect_in_the_source(settings):
    """The regression this whole section exists for: a model that silently
    corrects 'urchase' to 'purchase' is quoting correctly, not fabricating."""
    quote = OCR_CLAUSE.replace("a urchase", "a purchase")
    _, escalations = guardrails.decide("q", cited(quote), ocr_retrieved(), settings)
    assert escalations == [], escalations


@pytest.mark.parametrize(
    ("label", "quote"),
    [
        ("negation flip", OCR_CLAUSE[:95].replace("shall not sell", "may sell")),
        ("dropped negation", OCR_CLAUSE[:95].replace("shall not sell", "shall sell")),
        ("modal weakened", OCR_CLAUSE[:95].replace("shall not", "may not")),
    ],
)
def test_quote_that_alters_legal_force_is_caught(settings, label, quote):
    """Character similarity cannot catch these — the negation flip scores 0.978
    against the clause it misquotes. Polarity alignment is what catches them."""
    verdict_, escalations = guardrails.decide("q", cited(quote), ocr_retrieved(), settings)
    assert verdict_ is Verdict.NEEDS_REVIEW, label
    assert any(e.code is EscalationCode.UNRESOLVED_CITATION for e in escalations), label


def test_quote_too_short_to_be_evidence_is_caught(settings):
    """'the Institution and the customer' is genuinely verbatim and scores 1.0,
    but carries no evidential weight."""
    verdict_, escalations = guardrails.decide(
        "q", cited("the Institution and the customer"), ocr_retrieved(), settings
    )
    assert verdict_ is Verdict.NEEDS_REVIEW
    assert escalations


def test_quote_from_a_different_clause_is_caught(settings):
    other = "It is obligatory that the Institutions actual or constructive possession"
    verdict_, escalations = guardrails.decide("q", cited(other), ocr_retrieved(), settings)
    assert verdict_ is Verdict.NEEDS_REVIEW
    assert any(e.code is EscalationCode.UNRESOLVED_CITATION for e in escalations)


@pytest.mark.parametrize("span", ["leading", "middle", "trailing"])
def test_partial_quotes_on_word_boundaries_are_accepted(settings, span):
    """A quote is a fragment; clause text outside the quoted span must not read
    as the quote having altered something."""
    words = OCR_CLAUSE.split()
    quote = {
        "leading": " ".join(words[:22]),
        "middle": " ".join(words[22:52]),
        "trailing": " ".join(words[-28:]),
    }[span]
    _, escalations = guardrails.decide("q", cited(quote), ocr_retrieved(), settings)
    assert escalations == [], (span, escalations)
