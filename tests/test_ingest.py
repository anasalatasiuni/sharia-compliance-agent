"""Parser invariants.

The corpus is the system's whole evidence base, so a standard silently missing
from it is invisible at every later stage: retrieval cannot surface a clause that
was never indexed, and no guardrail fires on evidence that does not exist. These
tests run against the real source text when it is present.
"""

from __future__ import annotations

import pathlib

import pytest

from sharia_agent.ingest.parse import (
    STANDARD_HEADER,
    citable_entries,
    is_well_parsed,
    parse_standards,
)

SOURCE = (
    pathlib.Path(__file__).resolve().parent.parent
    / "corpus/raw/aaoifi-shariah-standards-en-2017.pdftext.txt"
)

# The 2017 English edition. Numbering runs 1-54 with none reserved.
EXPECTED_STANDARDS = set(range(1, 55))


@pytest.mark.parametrize(
    "line,number",
    [
        # Title pages carry the period, running page headers do not. Both forms
        # have to resolve to the same standard or frequency voting sees only one
        # of them and the title never resolves.
        ("Shari'ah Standard No. (46)", "46"),
        ("Shari'ah Standard No (46): Al-Wakalah Bi Al-Istithmar", "46"),
        ("Shari’ah Standard No (8): Murabahah", "8"),
        ("Shari'ah Standard No. (8): Murabahah", "8"),
    ],
)
def test_header_matches_both_punctuation_forms(line: str, number: str) -> None:
    found = STANDARD_HEADER.match(line)
    assert found is not None, f"header not recognised: {line!r}"
    assert found.group(1) == number


@pytest.mark.skipif(not SOURCE.exists(), reason="source text not fetched")
def test_every_standard_in_the_source_survives_parsing() -> None:
    """No standard may be dropped silently.

    Standards 42, 43, 44, 46, 47 and 48 were absent from the index for exactly
    this reason, and nothing downstream could have detected it.
    """
    kept = [s for s in parse_standards(SOURCE.read_text()) if is_well_parsed(s)]
    parsed = {int(s.number) for s in kept}
    assert not EXPECTED_STANDARDS - parsed, (
        f"standards dropped during parsing: {sorted(EXPECTED_STANDARDS - parsed)}"
    )


@pytest.mark.skipif(not SOURCE.exists(), reason="source text not fetched")
def test_every_standard_resolves_a_real_title() -> None:
    """A fallback title means the headers were never matched for that standard."""
    for standard in parse_standards(SOURCE.read_text()):
        assert standard.name != f"Standard {standard.number}", (
            f"SS{standard.number} title never resolved"
        )


@pytest.mark.skipif(not SOURCE.exists(), reason="source text not fetched")
def test_standards_carry_a_plausible_number_of_clauses() -> None:
    kept = [s for s in parse_standards(SOURCE.read_text()) if is_well_parsed(s)]
    for standard in kept:
        assert citable_entries(standard), f"SS{standard.number} has no citable clauses"
