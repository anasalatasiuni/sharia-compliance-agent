#!/usr/bin/env python
"""Inspect what retrieval actually did for a query, arm by arm.

Hybrid search merges two rankings that disagree, and the merged score alone
cannot tell you *why* a clause did or did not reach the model. This runs each
arm separately against the same index the service uses, so a retrieval failure
names its own cause:

    neither arm found it     -> candidate generation: chunking, the embedding
                                model, or the clause is not in the corpus
    one arm found it         -> the other arm's blind spot; usually fine, but
                                tells you which half is carrying the query
    found but ranked out     -> ranking: the reranker or rerank_top_k

The service does not store per-arm scores on the request path — two extra
queries on every request to serve the small fraction you investigate. It does
not need to: the audit record pins `index_snapshot`, so any past retrieval is
reproducible here after the fact.

    python scripts/retrieval_debug.py "can we charge for late payment"
    python scripts/retrieval_debug.py "..." --expect SS8-3.1.1 SS8-4.8
    python scripts/retrieval_debug.py "..." --no-rerank      # free
    python scripts/retrieval_debug.py "..." --standard 8     # test the filter
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from qdrant_client import models

from sharia_agent.config import get_settings
from sharia_agent.llm import LLM
from sharia_agent.retrieval import sparse as sp
from sharia_agent.retrieval.embeddings import Embedder
from sharia_agent.retrieval.rerank import build_reranker
from sharia_agent.retrieval.store import DENSE, SPARSE, VectorStore

DIM, BOLD, GREEN, YELLOW, RED, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"
)


def rank_of(chunk_id: str, ordered: list[str]) -> int | None:
    return ordered.index(chunk_id) + 1 if chunk_id in ordered else None


def show(title: str, rows: list[tuple[str, float, str]], limit: int) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    if not rows:
        print(f"  {DIM}(nothing){RESET}")
        return
    for i, (cid, score, text) in enumerate(rows[:limit], 1):
        print(f"  {i:>2}. {cid:<16} {score:>9.4f}  {text[:60]}")
    if len(rows) > limit:
        print(f"      {DIM}... {len(rows) - limit} more in the pool{RESET}")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--expect", nargs="*", default=[],
                    help="Gold chunk_ids that should be retrieved.")
    ap.add_argument("--standard", default=None, help="Restrict to one standard number.")
    ap.add_argument("--show", type=int, default=8, help="Rows to print per arm.")
    ap.add_argument("--no-rerank", action="store_true",
                    help="Skip reranking — no LLM call, so the run is effectively free.")
    args = ap.parse_args()

    logging.disable(logging.INFO)
    settings = get_settings()
    if not settings.openrouter_api_key:
        print(f"{RED}OPENROUTER_API_KEY is not set.{RESET}", file=sys.stderr)
        return 2

    store, embedder = VectorStore(settings), Embedder(settings)
    llm = LLM(settings)
    pool = settings.retrieve_candidates

    try:
        print(f"{DIM}query      : {args.query}{RESET}")
        dense_vec = await embedder.embed_query(args.query)
        sparse_vec = sp.encode(args.query)
        tokens = sp.tokenize(args.query)
        print(f"{DIM}bm25 tokens: {tokens}{RESET}")
        print(f"{DIM}pool       : {pool} candidates per arm"
              f"{f', filtered to standard {args.standard}' if args.standard else ''}{RESET}")

        qfilter = (
            models.Filter(must=[models.FieldCondition(
                key="standard_no", match=models.MatchValue(value=args.standard))])
            if args.standard else None
        )

        async def arm(using, query):
            res = await store.client.query_points(
                collection_name=store.collection, query=query, using=using,
                limit=pool, with_payload=True, query_filter=qfilter)
            return [(p.payload["chunk_id"], p.score, p.payload["text"]) for p in res.points]

        dense_rows = await arm(DENSE, dense_vec)
        sparse_rows = (
            await arm(SPARSE, models.SparseVector(indices=sparse_vec[0], values=sparse_vec[1]))
            if sparse_vec[0] else []
        )
        fused = await store.hybrid_search(
            dense_vec, sparse_vec, limit=pool, standard_no=args.standard)

        dense_ids = [r[0] for r in dense_rows]
        sparse_ids = [r[0] for r in sparse_rows]
        fused_ids = [r.clause.chunk_id for r in fused]

        show("DENSE arm — meaning, cosine 0..1", dense_rows, args.show)
        show("SPARSE arm — BM25 words, unbounded", sparse_rows, args.show)

        print(f"\n{BOLD}FUSED — reciprocal rank fusion{RESET}")
        print(f"  {DIM}scales above are incomparable (0.7 vs 20), so fusion uses rank, "
              f"not score{RESET}")
        for i, r in enumerate(fused[: args.show], 1):
            found = "+".join(
                x for x, ids in (("dense", dense_ids), ("sparse", sparse_ids))
                if r.clause.chunk_id in ids
            ) or "?"
            print(f"  {i:>2}. {r.clause.chunk_id:<16} {r.fused_score:>9.4f}  "
                  f"{DIM}via {found:<13}{RESET} {r.clause.text[:44]}")

        reranked_ids: list[str] = []
        if not args.no_rerank:
            reranker = build_reranker(settings, llm)
            ranked = await reranker.rerank(
                args.query, [r.clause.for_prompt() for r in fused], settings.rerank_top_k)
            print(f"\n{BOLD}RERANKED — top {settings.rerank_top_k} reach the model"
                  f"{RESET} {DIM}(backend: {reranker.name}){RESET}")
            for i, (idx, score) in enumerate(ranked, 1):
                cid = fused[idx].clause.chunk_id
                reranked_ids.append(cid)
                print(f"  {i:>2}. {cid:<16} {score:>9.4f}  {fused[idx].clause.text[:52]}")
        else:
            print(f"\n{DIM}(reranking skipped){RESET}")

        # -- the part that earns the script ---------------------------------
        if args.expect:
            print(f"\n{BOLD}GOLD CLAUSES{RESET}")
            # A mistyped gold id is indistinguishable from a retrieval miss —
            # both simply fail to appear — and chasing the wrong one costs an
            # afternoon. Check the index knows the id before blaming retrieval.
            known = {c.chunk_id for c in await store.get_by_chunk_ids(args.expect)}
            for cid in args.expect:
                if cid not in known:
                    print(f"\n  {BOLD}{cid}{RESET}")
                    print(f"    {RED}NOT IN THE INDEX{RESET} — this is a test-set bug, "
                          f"not a retrieval failure.\n      Check the id against "
                          f"corpus/manifest.json, or re-run ingest.")
                    continue
                d, s = rank_of(cid, dense_ids), rank_of(cid, sparse_ids)  # noqa: E741
                f, r = rank_of(cid, fused_ids), rank_of(cid, reranked_ids)
                print(f"\n  {BOLD}{cid}{RESET}")
                print(f"    dense  : {f'rank {d}' if d else 'absent from pool'}")
                print(f"    sparse : {f'rank {s}' if s else 'absent from pool'}")
                print(f"    fused  : {f'rank {f}' if f else 'absent'}")
                if not args.no_rerank:
                    print(f"    ranked : {f'rank {r}' if r else 'cut before the model'}")

                if d is None and s is None:
                    print(f"    {RED}CANDIDATE GENERATION failure{RESET} — neither arm "
                          f"surfaced it.\n      Fix lives in chunking, the embedding "
                          f"model, or the corpus (the clause may simply not cover this).")
                elif not args.no_rerank and r is None:
                    print(f"    {YELLOW}RANKING failure{RESET} — retrieved but cut before "
                          f"the model.\n      Fix lives in the reranker or "
                          f"SCA_RERANK_TOP_K (currently {settings.rerank_top_k}).")
                elif d is None:
                    print(f"    {YELLOW}dense missed it{RESET} — the lexical arm carried "
                          f"this query. The wording\n      likely diverges from the "
                          f"clause; query rewriting would help.")
                elif s is None:
                    print(f"    {DIM}sparse missed it — paraphrase matched where exact "
                          f"terms did not. Normal.{RESET}")
                elif args.no_rerank:
                    print(f"    {GREEN}both arms found it{RESET} — whether it survives to "
                          f"the model depends on\n      reranking, which this run skipped.")
                else:
                    print(f"    {GREEN}both arms found it and it reached the model{RESET}")
    finally:
        await embedder.close()
        await llm.close()
        await store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
