"""Redaction applied before anything leaves the process.

CBUAE's Consumer Protection Regulations require licensed financial institutions
to store and process consumer and transaction data inside the UAE. Cohere and
the Anthropic API are both outside it, so in this deployment the honest position
is: **identifiers must not reach either.** Redaction is the compensating control
that makes a demo defensible; it is not a substitute for the in-country
deployment described in the technical document.

Redaction runs on the request path, not the logging path. Scrubbing logs while
sending raw text upstream would protect the wrong artefact.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Emirates ID: 784-YYYY-NNNNNNN-N
    ("EMIRATES_ID", re.compile(r"\b784[-\s]?\d{4}[-\s]?\d{7}[-\s]?\d\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}[A-Z0-9]{1,4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    ("PHONE", re.compile(r"(?:\+971|00971|\b0)\s?5\d(?:[\s-]?\d){7}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("ACCOUNT", re.compile(r"\b\d{9,}\b")),
    ("PASSPORT", re.compile(r"\b[A-Z]{1,2}\d{6,8}\b")),
]


class RedactionReport(dict):
    """Counts by category. Emitted into the audit record so a request carrying
    identifiers is visible even though the identifiers themselves are not."""

    @property
    def total(self) -> int:
        return sum(self.values())


def redact(text: str) -> tuple[str, RedactionReport]:
    report = RedactionReport()
    out = text
    for label, pattern in _PATTERNS:
        out, count = pattern.subn(f"[{label}_REDACTED]", out)
        if count:
            report[label] = report.get(label, 0) + count
    return out, report
