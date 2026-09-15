"""HTTP-layer behaviour.

Everything here was previously untested: the whole suite sat below the API
boundary, so auth, error mapping and — most importantly — the job-ownership
check the README presents as a security control were asserted in prose and
verified nowhere.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sharia_agent.api.main import app
from sharia_agent.models import (
    Assessment,
    Citation,
    ShariahConcern,
    Verdict,
)

ANALYST = {"Authorization": "Bearer demo-token-analyst"}
OFFICER = {"Authorization": "Bearer demo-token-officer"}
QUERY = {"query": "Can Mal sell a vehicle under murabaha before buying it?"}


class StubPipeline:
    """Answers immediately so the HTTP layer can be tested without providers."""

    def __init__(self, raises: Exception | None = None):
        self.raises = raises

    async def assess(self, *, query, principal_id, index_snapshot="x"):
        if self.raises:
            raise self.raises
        from sharia_agent.agent.pipeline import PipelineResult
        from sharia_agent.models import AuditRecord

        a = Assessment(
            assessment_id="a1", trace_id="t1", query=query,
            verdict=Verdict.NON_COMPLIANT, confidence=0.9,
            reasoning="r",
            citations=[Citation(chunk_id="SS8-3.1.1", quote="q", supports="s")],
            concerns=[ShariahConcern.OWNERSHIP_SEQUENCE], escalations=[],
            missing_information=[], clauses_considered=["SS8-3.1.1"],
            corpus_version="c", model="m", prompt_version="v1", latency_ms=1,
        )
        return PipelineResult(assessment=a, audit=AuditRecord(
            trace_id="t1", assessment_id="a1", principal_id=principal_id,
            query_hash="h", corpus_version="c", index_snapshot="x",
            model="m", model_effort="none", prompt_version="v1"))


@pytest.fixture
def client(monkeypatch):
    with TestClient(app, raise_server_exceptions=False) as c:
        c.app.state.pipeline = StubPipeline()
        yield c


# -- auth -------------------------------------------------------------------


def test_assess_requires_a_token(client):
    assert client.post("/assess", json=QUERY).status_code == 401


def test_assess_rejects_an_unknown_token(client):
    r = client.post("/assess", json=QUERY, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    assert r.headers.get("www-authenticate") == "Bearer"


def test_assess_accepts_a_known_token(client):
    r = client.post("/assess", json=QUERY, headers=ANALYST)
    assert r.status_code == 200
    assert r.json()["verdict"] == "NON_COMPLIANT"
    assert r.headers.get("x-trace-id")


def test_short_queries_are_rejected_before_any_spend(client):
    assert client.post("/assess", json={"query": "hi"}, headers=ANALYST).status_code == 422


# -- the security claim the README makes ------------------------------------


def test_a_job_is_not_readable_by_another_principal(client):
    """README: 'otherwise a job id is a capability anyone can guess'.

    Asserted in prose, verified nowhere until now.
    """
    created = client.post("/assess?mode=async", json=QUERY, headers=ANALYST)
    assert created.status_code == 202
    job_id = created.json()["job_id"]

    assert client.get(f"/assess/{job_id}", headers=ANALYST).status_code == 200
    # A different, equally valid principal must not see it.
    assert client.get(f"/assess/{job_id}", headers=OFFICER).status_code == 404


def test_unknown_job_is_404_not_500(client):
    assert client.get("/assess/deadbeef", headers=ANALYST).status_code == 404


# -- failure shape ----------------------------------------------------------


def test_an_unhandled_failure_still_returns_json_with_a_trace_id(client):
    """A dependency outage previously escaped as Starlette's plain-text 500,
    breaking both the JSON contract and any hope of correlating the failure."""
    client.app.state.pipeline = StubPipeline(raises=RuntimeError("qdrant is gone"))
    r = client.post("/assess", json=QUERY, headers=ANALYST)
    assert r.status_code == 500
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["error"] == "internal_error"
    assert body["trace_id"]
    assert "qdrant is gone" not in str(body), "internal detail must not leak to the caller"


def test_liveness_is_independent_of_dependencies(client):
    assert client.get("/health/live").json() == {"status": "alive"}


def test_root_sends_a_visitor_to_the_docs(client):
    """The bare URL is the first thing anyone opening the service tries, and a
    404 is a poor answer to 'what is this'."""
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/docs"


def test_openapi_advertises_bearer_auth(client):
    """Swagger only renders an Authorize button when a scheme is declared.
    Without it a reviewer can read the API but not exercise it from a browser.
    """
    spec = client.get("/openapi.json").json()
    schemes = spec["components"]["securitySchemes"]
    assert any(s.get("scheme") == "bearer" for s in schemes.values()), schemes
    assert spec["paths"]["/assess"]["post"].get("security")
