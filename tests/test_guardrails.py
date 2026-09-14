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
