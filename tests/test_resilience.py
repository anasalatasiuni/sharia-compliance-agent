"""Retry and circuit-breaker behaviour.

Written after a 40-case eval run collapsed: five concurrent requests hit a rate
limit, the breaker counted those as faults and opened, and the remaining 35 cases
then failed instantly against an open breaker without ever reaching the network.
The whole run finished in 60 seconds and reported numbers that measured nothing.

The distinction these tests pin is that a 429 says "you are sending too fast" and
a 500 says "I am broken", and only the second is a reason to stop sending.
"""

from __future__ import annotations

import pytest

from sharia_agent.resilience import (
    CircuitBreaker,
    UpstreamUnavailable,
    call_with_resilience,
    is_rate_limited,
    retry_after_seconds,
)


class FakeResponse:
    def __init__(self, status_code=429, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class RateLimitError(Exception):
    def __init__(self, retry_after=None):
        super().__init__("429 too many requests")
        self.status_code = 429
        self.response = FakeResponse(
            headers={"retry-after": str(retry_after)} if retry_after else {}
        )


class ServerError(Exception):
    def __init__(self):
        super().__init__("500 internal error")
        self.status_code = 500


def breaker() -> CircuitBreaker:
    return CircuitBreaker(service="test", failure_threshold=3, reset_after_seconds=30)


# ---------------------------------------------------------------------------


def test_detects_rate_limit_by_status_code():
    assert is_rate_limited(RateLimitError())
    assert not is_rate_limited(ServerError())
    assert not is_rate_limited(ValueError("unrelated"))


def test_reads_the_servers_own_backoff_hint():
    assert retry_after_seconds(RateLimitError(retry_after=4)) == 4.0
    assert retry_after_seconds(RateLimitError()) is None
    # A hint longer than a minute is clamped rather than obeyed literally.
    assert retry_after_seconds(RateLimitError(retry_after=9999)) == 60.0


async def test_rate_limits_do_not_open_the_breaker(monkeypatch):
    """The failure that caused the bad eval run. A throttled dependency is
    healthy; opening the breaker on it stops traffic that would have succeeded
    after a short wait."""
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    b = breaker()
    calls = {"n": 0}

    async def always_throttled():
        calls["n"] += 1
        raise RateLimitError(retry_after=1)

    with pytest.raises(UpstreamUnavailable) as exc:
        await call_with_resilience(
            always_throttled, service="test", breaker=b, timeout=5,
            attempts=2, rate_limit_attempts=3,
        )
    assert "rate limited" in str(exc.value)
    assert b.state == "closed", "a 429 must not trip the breaker"
    assert calls["n"] > 2, "rate limits get their own, longer retry budget"


async def test_real_faults_do_open_the_breaker(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    b = breaker()

    async def always_broken():
        raise ServerError()

    for _ in range(3):
        with pytest.raises(UpstreamUnavailable):
            await call_with_resilience(
                always_broken, service="test", breaker=b, timeout=5, attempts=1
            )
    assert b.state == "open", "repeated genuine faults must stop the traffic"


async def test_a_throttled_call_still_succeeds_once_the_limit_clears(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    b = breaker()
    attempts = {"n": 0}

    async def throttled_then_fine():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimitError(retry_after=1)
        return "ok"

    result = await call_with_resilience(
        throttled_then_fine, service="test", breaker=b, timeout=5, attempts=2
    )
    assert result == "ok"
    assert b.state == "closed"


async def test_open_breaker_fails_fast_without_calling(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    b = breaker()
    b._failures = 99
    b.record_failure()
    called = {"n": 0}

    async def should_not_run():
        called["n"] += 1
        return "ok"

    with pytest.raises(UpstreamUnavailable, match="circuit open"):
        await call_with_resilience(should_not_run, service="test", breaker=b, timeout=5)
    assert called["n"] == 0


async def _no_sleep(_seconds):
    return None
