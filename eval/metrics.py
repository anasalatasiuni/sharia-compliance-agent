"""Scoring functions. Pure, so they can be reasoned about without a network.

The organising idea is that retrieval and generation are scored separately.
A wrong verdict can mean the governing clause never reached the model, or that
it reached the model and the model misread it. Those have different fixes and
are indistinguishable from the verdict alone.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

VERDICTS = ("COMPLIANT", "NON_COMPLIANT", "NEEDS_REVIEW")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float | None:
    """Fraction of gold clauses present in the first k retrieved.

    None when the case has no gold clauses — an out-of-corpus question is not a
    retrieval failure and must not be averaged in as a zero.
    """
    if not gold:
        return None
    top = set(retrieved[:k])
    return sum(g in top for g in gold) / len(gold)


def reciprocal_rank(retrieved: list[str], gold: list[str]) -> float | None:
    """1/rank of the first gold clause. Rewards putting it near the top, because
    models attend unevenly across a long context."""
    if not gold:
        return None
    for i, cid in enumerate(retrieved, 1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def context_precision(retrieved: list[str], gold: list[str]) -> float | None:
    """Fraction of what was sent that was actually relevant. Noise degrades the
    answer even when the right clause is present, and it is paid for per token."""
    if not gold or not retrieved:
        return None
    return sum(c in gold for c in retrieved) / len(retrieved)


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


@dataclass
class VerdictScores:
    total: int = 0
    correct: int = 0
    confusion: Counter = field(default_factory=Counter)   # (gold, actual)
    escalation_codes: Counter = field(default_factory=Counter)
    errors: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def false_compliant(self) -> int:
        """Answered COMPLIANT when it should not have. The headline failure.

        Counted rather than rated because at n=40 a rate reads more precise than
        the sample supports — and because the number people need is "how many got
        through", not a percentage.
        """
        return sum(
            n for (gold, actual), n in self.confusion.items()
            if actual == "COMPLIANT" and gold != "COMPLIANT"
        )

    @property
    def false_non_compliant(self) -> int:
        return sum(
            n for (gold, actual), n in self.confusion.items()
            if actual == "NON_COMPLIANT" and gold != "NON_COMPLIANT"
        )

    @property
    def over_escalation(self) -> int:
        """Escalated something that had a definite answer. Cheap per instance,
        but it is what makes the product unusable at volume (PART2 §2.2)."""
        return sum(
            n for (gold, actual), n in self.confusion.items()
            if actual == "NEEDS_REVIEW" and gold != "NEEDS_REVIEW"
        )

    @property
    def missed_escalation(self) -> int:
        """Answered definitively where a human was required."""
        return sum(
            n for (gold, actual), n in self.confusion.items()
            if gold == "NEEDS_REVIEW" and actual != "NEEDS_REVIEW"
        )


def calibration_buckets(
    rows: list[tuple[float, bool]], edges=(0.0, 0.7, 0.8, 0.9, 0.95, 1.01)
) -> list[tuple[str, int, float]]:
    """Stated confidence against observed correctness.

    `min_model_confidence` is only a meaningful gate if confidence tracks
    accuracy. A model that is confident and wrong escalates *less* while being
    wrong more, which silently inverts the cost argument for using it.
    """
    out = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        bucket = [ok for conf, ok in rows if lo <= conf < hi]
        if bucket:
            out.append((f"{lo:.2f}-{min(hi, 1.0):.2f}", len(bucket),
                        sum(bucket) / len(bucket)))
    return out


def mean(values: list[float | None]) -> float | None:
    real = [v for v in values if v is not None]
    return sum(real) / len(real) if real else None
