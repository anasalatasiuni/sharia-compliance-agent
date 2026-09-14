"""HTTP surface.

`POST /assess` is offered in two modes because the workload genuinely has two
shapes. A typical assessment finishes in a few seconds and a synchronous reply
is the better developer experience; a hard one with three retrieval rounds can
run past any sane client timeout. Rather than pick the wrong one, the sync path
is the default and `?mode=async` returns a job id to poll — which is also the
shape this service has to take at volume.

Backpressure is explicit: a bounded number of assessments run concurrently, and
excess load is refused with 429 and `Retry-After` rather than queued invisibly
until every client times out at once.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from ..llm import breaker_state as llm_breaker_state
from ..models import Assessment
from ..obs.trace import get_trace_id, log, new_trace_id, set_principal, set_trace_id
from ..retrieval.embeddings import breaker_state as embed_breaker_state
from ..retrieval.rerank import breaker_state as rerank_breaker_state
from .deps import Principal, require_assess

router = APIRouter()


class AssessRequest(BaseModel):
    query: str = Field(
        min_length=8,
        max_length=4000,
        description="The proposal or question to assess, in plain English.",
        examples=["Can Mal offer a savings account that pays a fixed 4% annual return?"],
    )


class JobAccepted(BaseModel):
    job_id: str
    status: str
    trace_id: str
    poll: str


class JobState(BaseModel):
    job_id: str
    status: str
    trace_id: str
    result: Assessment | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


@router.post(
    "/assess",
    response_model=Assessment,
    responses={202: {"model": JobAccepted}, 429: {"description": "Service at capacity"}},
    summary="Assess a proposal against the AAOIFI Shari'ah Standards",
)
async def assess(
    body: AssessRequest,
    request: Request,
    response: Response,
    mode: str = Query("sync", pattern="^(sync|async)$"),
    principal: Principal = Depends(require_assess),
):
    set_trace_id(request.headers.get("x-request-id") or new_trace_id())
    set_principal(principal.id)
    response.headers["x-trace-id"] = get_trace_id()

    app = request.app
    pipeline = app.state.pipeline
    snapshot = app.state.index_snapshot

    if mode == "async":
        job = await app.state.jobs.create(
            trace_id=get_trace_id(), principal_id=principal.id
        )
        asyncio.create_task(
            _run_job(app, job.id, body.query, principal.id, snapshot, get_trace_id())
        )
        response.status_code = status.HTTP_202_ACCEPTED
        return JobAccepted(
            job_id=job.id,
            status="queued",
            trace_id=get_trace_id(),
            poll=f"/assess/{job.id}",
        )

    inflight: asyncio.Semaphore = app.state.inflight
    if inflight.locked():
        # Shed load rather than let every caller sit in a queue and time out.
        log("assess.rejected_backpressure")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="service at capacity; retry shortly or use ?mode=async",
            headers={"Retry-After": "5"},
        )

    async with inflight:
        result = await pipeline.assess(
            query=body.query, principal_id=principal.id, index_snapshot=snapshot
        )
    _emit_audit(result)
    return result.assessment


async def _run_job(app, job_id: str, query: str, principal_id: str, snapshot: str, trace_id: str):
    set_trace_id(trace_id)
    set_principal(principal_id)
    await app.state.jobs.update(job_id, status="running")
    try:
        async with app.state.inflight:
            result = await app.state.pipeline.assess(
                query=query, principal_id=principal_id, index_snapshot=snapshot
            )
        _emit_audit(result)
        await app.state.jobs.update(job_id, status="succeeded", result=result.assessment)
    except Exception as exc:  # noqa: BLE001 — surfaced to the caller via the job
        log("job.failed", job_id=job_id, error=f"{type(exc).__name__}: {exc}")
        await app.state.jobs.update(job_id, status="failed", error=str(exc))


@router.get("/assess/{job_id}", response_model=JobState, summary="Poll an async assessment")
async def get_assessment(
    job_id: str, request: Request, principal: Principal = Depends(require_assess)
):
    job = await request.app.state.jobs.get(job_id, principal_id=principal.id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobState(
        job_id=job.id,
        status=job.status,
        trace_id=job.trace_id,
        result=job.result,
        error=job.error,
    )


def _emit_audit(result) -> None:
    """The audit record is a log event, not a side channel.

    Shipping it through the same pipeline as everything else means it inherits
    retention, access control and shipping for free — and in production it is
    the artefact a Shari'ah reviewer is actually shown.
    """
    log("audit.record", record=result.audit.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", summary="Readiness: can this instance actually serve traffic?")
async def health(request: Request, response: Response):
    """Readiness, not liveness.

    A process that is up but whose collection is empty will answer every
    question with NEEDS_REVIEW and look healthy while doing it. That is the
    failure this endpoint exists to catch, so it checks the index has content —
    not merely that the vector store answered.
    """
    app = request.app
    settings = app.state.settings
    started = time.perf_counter()
    components: dict[str, dict] = {}
    ok = True

    manifest = app.state.manifest
    expected = manifest.get("chunk_count")
    try:
        points = await asyncio.wait_for(app.state.store.count(), timeout=5)
        detail = {"points": points, "collection": settings.collection}
        if points == 0:
            detail["status"] = "empty"
        elif expected and points != expected:
            # The manifest is what audit records cite as corpus_version and
            # index_snapshot. If it describes a different index than the one
            # actually being queried, every verdict is attributed to a corpus
            # that did not produce it — worse than having no manifest at all.
            detail["status"] = "manifest_mismatch"
            detail["expected_points"] = expected
            detail["hint"] = "re-run ingest, or point SCA_QDRANT_URL at the indexed collection"
        else:
            detail["status"] = "ok"
        components["vector_store"] = detail
        ok &= detail["status"] == "ok"
    except Exception as exc:  # noqa: BLE001
        components["vector_store"] = {"status": "error", "error": f"{type(exc).__name__}"}
        ok = False

    breakers = {**llm_breaker_state(), **embed_breaker_state(), **rerank_breaker_state()}
    components["breakers"] = breakers
    if any(state == "open" for state in breakers.values()):
        ok = False

    components["credentials"] = {"openrouter": bool(settings.openrouter_api_key)}
    ok &= all(components["credentials"].values())

    body = {
        "status": "ok" if ok else "degraded",
        "env": settings.env,
        "model": settings.model,
        "embed_model": settings.embed_model,
        "rerank_backend": app.state.reranker.name,
        "corpus_version": manifest.get("corpus_version", "unknown"),
        "index_snapshot": app.state.index_snapshot,
        "jobs_tracked": app.state.jobs.size,
        "components": components,
        "checked_in_ms": int((time.perf_counter() - started) * 1000),
    }
    response.status_code = 200 if ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return body


@router.get("/health/live", summary="Liveness: is the process running?")
async def liveness():
    return {"status": "alive"}
