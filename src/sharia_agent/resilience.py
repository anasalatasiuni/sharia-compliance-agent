"""Retry, backoff and circuit breaking for outbound calls.

Three external services sit in the request path (embeddings, reranking, the
LLM) and each fails differently. A single global timeout would let the slowest
one dictate the latency of every request, so each gets its own budget, its own
retry policy, and its own breaker.

The breaker is the part that matters under load: retrying into a service that
is already failing turns a partial outage into a queue of stuck requests. Once
the breaker opens, calls fail immediately and the pipeline degrades to
NEEDS_REVIEW instead of hanging.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .obs.trace import log


class UpstreamUnavailable(RuntimeError):
    """Raised when a dependency is refusing work — breaker open, or retries exhausted."""

    def __init__(self, service: str, reason: str) -> None:
        super().__init__(f"{service} unavailable: {reason}")
        self.service = service
        self.reason = reason


@dataclass
class CircuitBreaker:
    """Closed -> open on repeated failure, half-open after a cool-off.

    A single success in half-open closes it; a single failure re-opens it. That
    asymmetry is deliberate — it is cheap to probe and expensive to flap.
    """

    service: str
    failure_threshold: int = 5
    reset_after_seconds: float = 30.0

    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.reset_after_seconds:
            return "half_open"
        return "open"

    def before_call(self) -> None:
        if self.state == "open":
            raise UpstreamUnavailable(self.service, "circuit open")

    def record_success(self) -> None:
        if self._opened_at is not None:
            log("breaker.closed", service=self.service)
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self.state == "half_open" or self._failures >= self.failure_threshold:
            if self._opened_at is None:
                log("breaker.opened", service=self.service, failures=self._failures)
            self._opened_at = time.monotonic()


async def call_with_resilience[T](
    fn: Callable[[], Awaitable[T]],
    *,
    service: str,
    breaker: CircuitBreaker,
    timeout: float,
    attempts: int = 3,
    base_delay: float = 0.4,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    do_not_retry_on: tuple[type[BaseException], ...] = (),
) -> T:
    """Run `fn` with a per-call timeout, bounded retries and breaker accounting.

    Backoff is exponential with full jitter. Without jitter, every request that
    fails during the same upstream blip retries in lockstep and re-creates the
    spike that caused the failure.
    """
    breaker.before_call()
    last: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            result = await asyncio.wait_for(fn(), timeout=timeout)
        except do_not_retry_on as exc:
            breaker.record_failure()
            raise UpstreamUnavailable(service, f"non-retryable: {exc}") from exc
        except (TimeoutError, *retry_on) as exc:
            last = exc
            breaker.record_failure()
            if attempt == attempts:
                break
            delay = random.uniform(0, base_delay * (2 ** (attempt - 1)))
            log(
                "upstream.retry",
                service=service,
                attempt=attempt,
                of=attempts,
                delay_ms=int(delay * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
            await asyncio.sleep(delay)
        else:
            breaker.record_success()
            return result

    raise UpstreamUnavailable(service, f"{attempts} attempts failed: {last}")
