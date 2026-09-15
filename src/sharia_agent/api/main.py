"""FastAPI application.

Every route handler is async and every downstream call is awaited. A single
synchronous SDK call in a handler would block the event loop and serialise the
whole service behind the slowest LLM request — which is precisely the failure
this workload invites, since the reasoning call routinely takes 10-30 seconds.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..agent.pipeline import CompliancePipeline
from ..config import get_settings
from ..ingest.index import load_manifest
from ..jobs import JobStore
from ..llm import LLM
from ..obs.trace import configure_logging, get_trace_id, log
from ..retrieval.embeddings import Embedder
from ..retrieval.hybrid import Retriever
from ..retrieval.rerank import build_reranker
from ..retrieval.store import VectorStore
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    for problem in settings.assert_production_safe():
        log("config.unsafe_for_production", problem=problem)

    store = VectorStore(settings)
    embedder = Embedder(settings)
    llm = LLM(settings)
    reranker = build_reranker(settings, llm)
    retriever = Retriever(store, embedder, reranker, settings)

    manifest = load_manifest()
    app.state.settings = settings
    app.state.store = store
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.reranker = reranker
    app.state.retriever = retriever
    app.state.manifest = manifest
    app.state.index_snapshot = manifest.get("index_snapshot", "unknown")
    app.state.pipeline = CompliancePipeline(retriever, llm, settings)
    app.state.jobs = JobStore(ttl_seconds=settings.job_ttl_seconds)
    app.state.inflight = asyncio.Semaphore(settings.max_concurrent_assessments)
    app.state.running = set()  # strong refs to in-flight async jobs

    log(
        "service.start",
        env=settings.env,
        model=settings.model,
        embed_model=settings.embed_model,
        rerank_backend=reranker.name,
        collection=settings.collection,
        corpus_version=manifest.get("corpus_version", "unknown"),
        index_snapshot=app.state.index_snapshot,
    )
    try:
        yield
    finally:
        await embedder.close()
        await llm.close()
        await store.close()
        log("service.stop")


app = FastAPI(
    title="Mal Shari'ah Compliance Agent",
    version="0.1.0",
    description=(
        "Decision support for Mal's internal compliance team. Returns an "
        "evidence-backed preliminary assessment against the AAOIFI Shari'ah "
        "Standards. Not a Shari'ah ruling — see the disclaimer on every response."
    ),
    lifespan=lifespan,
)
@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Any unhandled failure still leaves as structured JSON carrying the trace id.

    Starlette's default handler returns plain text, which breaks the contract
    that every response is JSON and — worse — hands back a failure with nothing
    to correlate against the logs.
    """
    trace_id = get_trace_id()
    log("request.unhandled_error", error=f"{type(exc).__name__}: {exc}", path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "The request could not be completed. Quote the trace id when reporting this.",
            "trace_id": trace_id,
        },
        headers={"x-trace-id": trace_id},
    )


app.include_router(router)
