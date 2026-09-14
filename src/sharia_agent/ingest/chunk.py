"""Turn parsed standards into indexable clauses.

One clause is normally one chunk. The exception is a clause long enough that a
single embedding would average away its specifics — those are split on sentence
boundaries with overlap, and the parts keep the parent clause path so a citation
still resolves to a real clause number.
"""

from __future__ import annotations

import hashlib
import re

from ..models import Clause
from .parse import ParsedStandard, citable_entries, heading_path_for

# Beyond this a single vector stops being specific to anything in the clause.
MAX_CHUNK_CHARS = 1800
OVERLAP_CHARS = 220

_SENTENCE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    sentences = _SENTENCE.split(text)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > MAX_CHUNK_CHARS:
            parts.append(current.strip())
            # Carry the tail forward so a rule split across the boundary is
            # still retrievable from the following part.
            current = current[-OVERLAP_CHARS:].strip() + " " + sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        parts.append(current.strip())
    return parts or [text]


def chunk_id_for(standard_no: str, clause_path: str, part: int | None = None) -> str:
    base = f"SS{standard_no}-{clause_path.replace('/', '.')}"
    return base if part is None else f"{base}-p{part}"


def clauses_from_standard(standard: ParsedStandard, corpus_version: str) -> list[Clause]:
    out: list[Clause] = []
    for entry in citable_entries(standard):
        trail = heading_path_for(standard, entry.path)
        parts = _split_long(entry.text)
        for i, body in enumerate(parts):
            out.append(
                Clause(
                    chunk_id=chunk_id_for(
                        standard.number, entry.path, None if len(parts) == 1 else i + 1
                    ),
                    standard_no=standard.number,
                    standard_name=standard.name,
                    clause_path=entry.path,
                    heading_path=trail,
                    text=body,
                    lang="en",
                    corpus_version=corpus_version,
                )
            )
    return out


def corpus_version_for(source_bytes: bytes, standards: list[str]) -> str:
    """A version string that changes whenever the source or its selection changes.

    Pinned into every audit record. Without it, a verdict cannot be replayed —
    you would have no way to know which text the model actually saw.
    """
    digest = hashlib.sha256(source_bytes)
    digest.update(",".join(sorted(standards, key=int)).encode())
    return f"aaoifi-en-2017@{digest.hexdigest()[:16]}"
