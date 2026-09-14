"""Lexical (BM25) sparse vectors.

Dense retrieval alone fumbles exactly the tokens that matter most in standards
text: clause numbers ("2/2/2"), transliterated Arabic contract names
(murabaha / muraabaha / murābaḥa), and negation-bearing legal phrases. The
lexical half of hybrid search is what catches those.

Term frequencies are computed here; **IDF is left to Qdrant**, which applies it
server-side across the whole collection via `Modifier.IDF`. That keeps the
client stateless — no corpus statistics to persist, recompute, or let drift out
of sync with the index.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Latin words, Arabic words, and clause paths like 2/2/2 or 4-1-3
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’\-]*|[؀-ۿ]+|\d+(?:[/\-]\d+)+|\d+")

_ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")

# Light Arabic orthographic normalisation: the same term is written several ways
# across documents, and none of the variants should be a different token.
_AR_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ؤ": "و",
    "ة": "ه",
})

_STOP = frozenset(
    (
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is",
        "are", "be", "been", "being", "by", "with", "as", "at", "from", "that",
        "this", "these", "those", "it", "its", "if", "not", "no", "nor", "but",
        "shall", "may", "must", "can", "will", "would", "should", "there",
        "here", "which", "who", "whom", "whose",
    )
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ARABIC_DIACRITICS.sub("", text)
    return text.translate(_AR_FOLD).lower()


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(normalize(text)) if t not in _STOP and len(t) > 1]


def token_id(token: str) -> int:
    """Stable 32-bit id for a token.

    Deliberately not Python's `hash()`, which is salted per process and would
    make indices non-reproducible across restarts — silently breaking every
    previously indexed sparse vector.
    """
    h = 2166136261
    for ch in token.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def encode(text: str) -> tuple[list[int], list[float]]:
    """Return (indices, term-frequency values) for one document or query."""
    counts = Counter(tokenize(text))
    if not counts:
        return [], []
    # Collapse the rare hash collision by summing, keeping indices unique.
    merged: dict[int, float] = {}
    for token, n in counts.items():
        merged[token_id(token)] = merged.get(token_id(token), 0.0) + float(n)
    items = sorted(merged.items())
    return [i for i, _ in items], [v for _, v in items]
