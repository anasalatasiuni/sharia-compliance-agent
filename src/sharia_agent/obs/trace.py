"""Tracing and structured logging.

Span names are OpenTelemetry-shaped (`sharia.<stage>`) so that swapping the
sink for a real collector is a wiring change, not a rewrite: this module is the
only place that knows logs currently go to stdout.

Every log line carries `trace_id`, so one request is greppable end to end and
joinable against the audit record for the same id.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
_principal: ContextVar[str] = ContextVar("principal", default="-")

_REDACT_KEYS = {"authorization", "api_key", "openrouter_api_key", "qdrant_api_key", "token"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": _trace_id.get(),
            "principal": _principal.get(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(_redact(extra))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def _redact(d: dict[str, Any]) -> dict[str, Any]:
    return {
        k: ("***" if k.lower() in _REDACT_KEYS else v)
        for k, v in d.items()
    }


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # uvicorn's own handlers would otherwise double-emit in plain text
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers[:] = [handler]
        lg.propagate = False


def new_trace_id() -> str:
    return uuid.uuid4().hex


def set_trace_id(value: str) -> None:
    _trace_id.set(value)


def get_trace_id() -> str:
    return _trace_id.get()


def set_principal(value: str) -> None:
    _principal.set(value)


def log(event: str, level: int = logging.INFO, **fields: Any) -> None:
    logging.getLogger("sharia").log(level, event, extra={"extra_fields": fields})


class Span:
    """A timed stage. Records duration and outcome whether or not it raised."""

    def __init__(self, name: str, **fields: Any) -> None:
        self.name = name
        self.fields = fields
        self.ms = 0
        self.ok = True
        self.note: str | None = None

    def annotate(self, **fields: Any) -> None:
        self.fields.update(fields)


@contextmanager
def span(name: str, **fields: Any):
    s = Span(name, **fields)
    started = time.perf_counter()
    try:
        yield s
    except Exception as exc:  # noqa: BLE001 — re-raised below, recorded first
        s.ok = False
        s.note = f"{type(exc).__name__}: {exc}"
        s.ms = int((time.perf_counter() - started) * 1000)
        log(f"sharia.{name}", level=logging.ERROR, duration_ms=s.ms, ok=False,
            error=s.note, **s.fields)
        raise
    else:
        s.ms = int((time.perf_counter() - started) * 1000)
        log(f"sharia.{name}", duration_ms=s.ms, ok=True, **s.fields)
