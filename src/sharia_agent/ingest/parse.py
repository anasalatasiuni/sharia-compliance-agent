"""Parse the AAOIFI Shari'ah Standards text into citable clauses.

The chunking decision is made here, and it is the most consequential one in the
retrieval stack: **the chunk boundary is the clause boundary**, because the
clause is the unit a compliance officer cites. A generic recursive splitter
would produce chunks that straddle "2/2/2" and "2/2/3" and a citation could
then only point at a page.

Working against OCR text means the parser has to survive soft hyphens at line
breaks, running page headers repeated on every page, bare page numbers, and a
table of contents whose entries look exactly like clause openings. Each is
handled explicitly below rather than hoped away.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

# "Shariah Standard No. (8): Murabahah" — also appears as a running page header.
# The period after "No" is optional because the publisher is inconsistent about
# it: title pages read "Standard No. (46)" while the running page headers read
# "Standard No (46)". Requiring the period matched only the title page, whose
# title sits on the *next* line, so frequency voting had nothing to vote on, the
# name fell back to "Standard N", and six standards (42, 43, 44, 46, 47, 48)
# were dropped as unparseable while their text was silently absorbed by whatever
# standard was current. Pinned by tests/test_ingest.py.
STANDARD_HEADER = re.compile(
    r"^Shari[''’]?ah\s+Standard\s+No\.?\s*\(\s*(\d+)\s*\)\s*[:.]?\s*(.*)$", re.IGNORECASE
)

# "2/2/2 It is essential to exclude ..." — a numbered entry opening a line.
CLAUSE_OPEN = re.compile(r"^(\d+(?:/\d+)*)\s+(\S.*)$")

BARE_PAGE_NUMBER = re.compile(r"^\d{1,4}$")
APPENDIX = re.compile(r"^Appendix\s*\(", re.IGNORECASE)

# Entries shorter than this are treated as headings: useful for building the
# heading path, but too thin to retrieve on their own.
MIN_CLAUSE_CHARS = 150


@dataclass
class ParsedEntry:
    standard_no: str
    standard_name: str
    path: str
    text: str
    order: int


@dataclass
class ParsedStandard:
    number: str
    name: str
    entries: dict[str, ParsedEntry] = field(default_factory=dict)
    name_votes: Counter[str] = field(default_factory=Counter)

    def resolve_name(self) -> str:
        """Pick the title the running headers agree on.

        A standard's title is repeated as a page header dozens of times, while a
        cross-reference inside some other clause ("...Standard No. (8) on
        Murabahah and item 2/2/4 of...") looks identical to the regex but occurs
        once. Voting separates them without hand-maintained special cases.
        """
        for candidate, _ in self.name_votes.most_common():
            if _plausible_title(candidate):
                return candidate
        return self.name or f"Standard {self.number}"


_TITLE_REJECT = re.compile(r"\b(item|items|para|clause)\b|[\[\]]|^\s*on\b", re.IGNORECASE)


def _plausible_title(name: str) -> bool:
    """Reject fragments of running prose that happen to follow the header regex."""
    if not (3 <= len(name) <= 80):
        return False
    if _TITLE_REJECT.search(name):
        return False
    # A genuine title does not trail off into a connective.
    return not re.search(r"\b(of|and|the|in|to|for|is|it|also)\s*$", name, re.IGNORECASE)


def normalize_text(raw: str) -> str:
    """Undo the OCR artefacts that would otherwise corrupt tokens and citations."""
    text = unicodedata.normalize("NFKC", raw)
    # Soft hyphen / not-sign used as an end-of-line hyphen: "inter¬\nnational".
    text = re.sub(r"[¬­]\s*\n\s*", "", text)
    # A real hyphen at end of line splitting a lowercase word.
    text = re.sub(r"(?<=[a-z])-\s*\n\s*(?=[a-z])", "", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return text


def _clean_line(line: str) -> str:
    return re.sub(r"\s{2,}", " ", line).strip()


def _is_noise(line: str) -> bool:
    return (
        not line
        or BARE_PAGE_NUMBER.match(line) is not None
        or STANDARD_HEADER.match(line) is not None
    )


def parse_standards(raw: str, wanted: set[str] | None = None) -> list[ParsedStandard]:
    """Segment the document into standards, then into numbered entries.

    Assignment is positional: every numbered entry belongs to the most recently
    announced standard. Running headers re-announce the same standard, so they
    are harmless — and they double as the signal that a new standard has begun.
    """
    lines = [_clean_line(ln) for ln in normalize_text(raw).splitlines()]

    standards: dict[str, ParsedStandard] = {}
    current: ParsedStandard | None = None
    pending_path: str | None = None
    buffer: list[str] = []
    order = 0
    in_appendix = False

    def flush() -> None:
        nonlocal pending_path, buffer, order
        if current is None or pending_path is None:
            pending_path, buffer = None, []
            return
        body = " ".join(buffer).strip()
        body = re.sub(r"\s{2,}", " ", body)
        if body:
            prev = current.entries.get(pending_path)
            # The table of contents yields a stub for the same path. Keeping the
            # longest occurrence discards TOC entries without special-casing them.
            if prev is None or len(body) > len(prev.text):
                current.entries[pending_path] = ParsedEntry(
                    standard_no=current.number,
                    standard_name=current.name,
                    path=pending_path,
                    text=body,
                    order=prev.order if prev else order,
                )
                order += 1
        pending_path, buffer = None, []

    for line in lines:
        header = STANDARD_HEADER.match(line)
        if header:
            number, name = header.group(1), header.group(2).strip(" .:")
            flush()
            in_appendix = False
            if current is None or current.number != number:
                existing = standards.get(number)
                if existing is None:
                    existing = ParsedStandard(number=number, name=name or f"Standard {number}")
                    standards[number] = existing
                current = existing
            if name:
                current.name_votes[name.strip(" .:;,")] += 1
            continue

        if current is None:
            continue

        if APPENDIX.match(line):
            flush()
            in_appendix = True
            continue
        if in_appendix:
            continue

        opener = CLAUSE_OPEN.match(line)
        if opener:
            flush()
            pending_path = opener.group(1)
            buffer = [opener.group(2)]
            continue

        if _is_noise(line):
            continue
        if pending_path is not None:
            buffer.append(line)

    flush()

    for std in standards.values():
        std.name = std.resolve_name()
        for entry in std.entries.values():
            entry.standard_name = std.name

    result = [s for s in standards.values() if s.entries]
    if wanted:
        result = [s for s in result if s.number in wanted]
    return sorted(result, key=lambda s: int(s.number))


def heading_path_for(standard: ParsedStandard, path: str) -> list[str]:
    """Ancestor labels for a clause: ['Murabahah', '2 ...', '2/2 ...'].

    Prepending this to the chunk text is what lets the retriever distinguish
    clauses whose wording is near-identical across standards.
    """
    parts = path.split("/")
    trail = [standard.name]
    for depth in range(1, len(parts)):
        ancestor = "/".join(parts[:depth])
        entry = standard.entries.get(ancestor)
        if entry:
            label = entry.text[:90].rstrip()
            if len(entry.text) > 90:
                label += "..."
            trail.append(f"{ancestor} {label}")
    return trail


# A standard yielding fewer than this did not really parse — its headers were
# damaged in the source and the numbered entries were never attributed to it.
MIN_CLAUSES_FOR_INDEXING = 5


def is_well_parsed(standard: ParsedStandard) -> bool:
    """Whether a standard survived parsing well enough to be worth indexing.

    Two failure signatures, both from damage in the source rather than from the
    standard being short: the title never resolved (so `resolve_name` fell back
    to "Standard N"), or almost no numbered entries were attributed to it.
    Indexing either injects noise into retrieval without adding real coverage.
    """
    if standard.name == f"Standard {standard.number}":
        return False
    return len(citable_entries(standard)) >= MIN_CLAUSES_FOR_INDEXING


def citable_entries(standard: ParsedStandard) -> list[ParsedEntry]:
    """Entries substantial enough to retrieve. Short ones stay as heading context."""
    return sorted(
        (e for e in standard.entries.values() if len(e.text) >= MIN_CLAUSE_CHARS),
        key=lambda e: [int(p) for p in e.path.split("/")],
    )
