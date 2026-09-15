#!/usr/bin/env python
"""Run the labelled test set through the pipeline and score it.

    python -m eval.run_eval                      # all 40 cases
    python -m eval.run_eval --limit 5            # smoke test
    python -m eval.run_eval --model anthropic/claude-opus-5
    python -m eval.run_eval --concurrency 6

Results land in `eval/results/<timestamp>.json` with the full per-case detail and
the configuration that produced them. Pinning the config is not bookkeeping: a
score is meaningless six weeks later if you cannot say which corpus, prompt,
model and thresholds produced it, and "did this change help?" is the only
question the eval exists to answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from eval.metrics import (
    VERDICTS,
    VerdictScores,
    calibration_buckets,
    context_precision,
    mean,
    recall_at_k,
    reciprocal_rank,
)
from sharia_agent.agent.pipeline import CompliancePipeline
from sharia_agent.agent.prompts import PROMPT_VERSION
from sharia_agent.config import get_settings
from sharia_agent.ingest.index import load_manifest
from sharia_agent.llm import LLM
from sharia_agent.obs.trace import new_trace_id, set_trace_id
from sharia_agent.retrieval.embeddings import Embedder
from sharia_agent.retrieval.hybrid import Retriever
from sharia_agent.retrieval.rerank import build_reranker
from sharia_agent.retrieval.store import VectorStore

TESTSET = Path("eval/testset.yaml")
RESULTS = Path("eval/results")

BOLD, DIM, GREEN, YELLOW, RED, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"
)


async def run_case(case: dict, pipeline: CompliancePipeline, snapshot: str,
                   sem: asyncio.Semaphore) -> dict:
    async with sem:
        set_trace_id(new_trace_id())
        started = time.perf_counter()
        try:
            result = await pipeline.assess(
                query=case["query"], principal_id="eval", index_snapshot=snapshot
            )
        except Exception as exc:  # noqa: BLE001 — one bad case must not end the run
            return {**case, "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": round(time.perf_counter() - started, 1)}

        a, audit = result.assessment, result.audit
        return {
            **case,
            "actual_verdict": a.verdict.value,
            "confidence": a.confidence,
            "retrieved": a.clauses_considered,
            # Keep the citation text, not just the id. A citation fails for one
            # of three reasons and only the quote distinguishes them.
            "cited": [
                {"chunk_id": c.chunk_id, "quote": c.quote, "supports": c.supports}
                for c in a.citations
            ],
            "escalations": [e.code.value for e in a.escalations],
            # The detail says *which* citation failed and why. Storing only the
            # code made the first run's three failures undiagnosable without
            # spending money to reproduce them.
            "escalation_details": [
                {"code": e.code.value, "detail": e.detail} for e in a.escalations
            ],
            "concerns": [c.value for c in a.concerns],
            "reasoning": a.reasoning,
            "tokens_in": audit.usage.input_tokens,
            "tokens_out": audit.usage.output_tokens,
            "elapsed_s": round(time.perf_counter() - started, 1),
        }


def score(rows: list[dict]) -> dict:
    scores = VerdictScores()
    recalls: dict[int, list] = {k: [] for k in (1, 3, 5, 8)}
    rr, cp, calib = [], [], []
    by_difficulty: dict[str, list[bool]] = {}
    fail_examples: list[dict] = []

    for r in rows:
        scores.total += 1
        if r.get("error"):
            scores.errors += 1
            continue

        gold_v, actual_v = r["gold_verdict"], r["actual_verdict"]
        ok = gold_v == actual_v
        scores.correct += ok
        scores.confusion[(gold_v, actual_v)] += 1
        scores.escalation_codes.update(r["escalations"])
        by_difficulty.setdefault(r["difficulty"], []).append(ok)
        calib.append((r["confidence"], ok))

        gold_c, retrieved = r["gold_clauses"], r["retrieved"]
        for k in recalls:
            recalls[k].append(recall_at_k(retrieved, gold_c, k))
        rr.append(reciprocal_rank(retrieved, gold_c))
        cp.append(context_precision(retrieved, gold_c))

        if not ok:
            fail_examples.append({
                "id": r["id"], "difficulty": r["difficulty"],
                "gold": gold_v, "actual": actual_v,
                "gold_clauses": gold_c,
                "gold_retrieved": [g for g in gold_c if g in retrieved],
                "escalations": r["escalations"],
                "escalation_details": r.get("escalation_details", []),
            })

    return {
        "scores": scores,
        "recall": {k: mean(v) for k, v in recalls.items()},
        "mrr": mean(rr),
        "context_precision": mean(cp),
        "by_difficulty": {d: sum(v) / len(v) for d, v in sorted(by_difficulty.items())},
        "calibration": calibration_buckets(calib),
        "failures": fail_examples,
    }


def report(rows: list[dict], s: dict, cfg: dict) -> None:
    sc: VerdictScores = s["scores"]
    ok_rows = [r for r in rows if not r.get("error")]
    tin = sum(r.get("tokens_in", 0) for r in ok_rows)
    tout = sum(r.get("tokens_out", 0) for r in ok_rows)

    print(f"\n{BOLD}{'=' * 66}{RESET}")
    print(f"{BOLD}  EVAL — {cfg['model']}{RESET}")
    print(f"{DIM}  corpus {cfg['corpus_version']} · index {cfg['index_snapshot']}")
    print(f"  prompt {cfg['prompt_version']} · top_k {cfg['rerank_top_k']} · "
          f"min_conf {cfg['min_model_confidence']} · min_rerank {cfg['min_rerank_score']}{RESET}")
    print(f"{BOLD}{'=' * 66}{RESET}")

    # A run where the provider was failing measures the provider, not the model.
    # Reporting those numbers as quality is worse than reporting nothing, because
    # they look like quality — the first run of this harness scored 0.525
    # accuracy purely because a rate limit had opened the circuit breaker.
    upstream = sc.escalation_codes.get("upstream_error", 0)
    if upstream:
        share = upstream / sc.total
        colour = RED if share > 0.1 else YELLOW
        print(f"\n{colour}{BOLD}  WARNING — {upstream}/{sc.total} cases hit an upstream "
              f"failure.{RESET}")
        print(f"{colour}  These scores measure the provider, not the system. "
              f"Lower --concurrency and re-run.{RESET}")
        if share > 0.1:
            print(f"{RED}  Above 10% the run is not usable as a baseline.{RESET}")

    print(f"\n{BOLD}RETRIEVAL{RESET}  {DIM}(over the clauses that reached the model){RESET}")
    for k, v in s["recall"].items():
        print(f"  recall@{k:<2}          {v:.3f}" if v is not None else f"  recall@{k:<2}  n/a")
    print(f"  MRR                {s['mrr']:.3f}")
    print(f"  context precision  {s['context_precision']:.3f}"
          f"   {DIM}how much of what we sent was relevant{RESET}")

    print(f"\n{BOLD}VERDICTS{RESET}")
    print(f"  accuracy           {sc.accuracy:.3f}  ({sc.correct}/{sc.total})")
    flag = RED if sc.false_compliant else GREEN
    print(f"  {flag}false COMPLIANT    {sc.false_compliant}{RESET}"
          f"   {DIM}<- the headline. Approved something it should not have{RESET}")
    print(f"  false NON_COMPLIANT {sc.false_non_compliant}")
    print(f"  over-escalated     {sc.over_escalation}   "
          f"{DIM}had a definite answer, sent to a human{RESET}")
    print(f"  missed escalation  {sc.missed_escalation}   "
          f"{DIM}answered where a human was required{RESET}")
    if sc.errors:
        print(f"  {RED}errors             {sc.errors}{RESET}")

    print(f"\n{BOLD}CONFUSION{RESET}  {DIM}rows = gold, cols = actual{RESET}")
    print(f"    {'':<16}" + "".join(f"{v[:13]:>15}" for v in VERDICTS))
    for g in VERDICTS:
        cells = "".join(f"{sc.confusion.get((g, a), 0):>15}" for a in VERDICTS)
        print(f"    {g:<16}{cells}")

    print(f"\n{BOLD}BY DIFFICULTY{RESET}")
    for d in ("easy", "medium", "hard"):
        if d in s["by_difficulty"]:
            print(f"  {d:<8} {s['by_difficulty'][d]:.3f}")

    if sc.escalation_codes:
        print(f"\n{BOLD}ESCALATION MIX{RESET}  {DIM}which gate fired, and how often{RESET}")
        for code, n in sc.escalation_codes.most_common():
            print(f"  {code:<28} {n}")

    print(f"\n{BOLD}CALIBRATION{RESET}  {DIM}stated confidence vs observed accuracy{RESET}")
    for label, n, acc in s["calibration"]:
        # Compare observed accuracy against the floor of the stated band. Below
        # it the model is overconfident, which is the direction that matters:
        # min_model_confidence stops being a meaningful gate, and a confidently
        # wrong model escalates less while being wrong more.
        gap = acc - float(label.split("-")[0])
        mark = ""
        if n >= 3 and gap < -0.15:
            mark = f"  {RED}<- OVERCONFIDENT{RESET}"
        elif n >= 3 and gap > 0.15:
            mark = f"  {DIM}<- underconfident{RESET}"
        print(f"  {label}   n={n:<3} correct {acc:.2f}{mark}")

    if s["failures"]:
        print(f"\n{BOLD}FAILURES{RESET}")
        for f in s["failures"]:
            got = (f"{GREEN}gold clause retrieved{RESET}" if f["gold_retrieved"]
                   else f"{RED}gold clause NOT retrieved{RESET}" if f["gold_clauses"]
                   else f"{DIM}no gold clauses{RESET}")
            print(f"  {f['id']} ({f['difficulty']:<6}) {f['gold']:<14} -> "
                  f"{f['actual']:<14} {got}")
            for d in f.get("escalation_details", []):
                print(f"       {DIM}{d['code']}: {d['detail'][:96]}{RESET}")

    cost = tin / 1e6 * cfg["price_in"] + tout / 1e6 * cfg["price_out"]
    print(f"\n{DIM}tokens {tin:,} in / {tout:,} out · approx ${cost:.2f} · "
          f"{sum(r['elapsed_s'] for r in ok_rows):.0f}s of model time{RESET}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model")
    ap.add_argument("--concurrency", type=int, default=2,
                help="Keep low. Concurrency 5 tripped provider rate limits.")
    ap.add_argument("--price-in", type=float, default=1.0, help="$/Mtok, for the cost line")
    ap.add_argument("--price-out", type=float, default=5.0)
    args = ap.parse_args()

    logging.disable(logging.INFO)
    settings = get_settings()
    if args.model:
        settings = settings.model_copy(update={"model": args.model})
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 2

    cases = yaml.safe_load(TESTSET.read_text())
    if args.limit:
        cases = cases[: args.limit]

    manifest = load_manifest()
    snapshot = manifest.get("index_snapshot", "unknown")

    store, embedder, llm = VectorStore(settings), Embedder(settings), LLM(settings)
    retriever = Retriever(store, embedder, build_reranker(settings, llm), settings)
    pipeline = CompliancePipeline(retriever, llm, settings)
    sem = asyncio.Semaphore(args.concurrency)

    print(f"{DIM}running {len(cases)} cases, concurrency {args.concurrency}, "
          f"model {settings.model}{RESET}")
    started = time.perf_counter()
    try:
        rows = await asyncio.gather(
            *(run_case(c, pipeline, snapshot, sem) for c in cases)
        )
    finally:
        await embedder.close()
        await llm.close()
        await store.close()

    cfg = {
        "model": settings.model,
        "rerank_backend": settings.rerank_backend,
        "rerank_model": settings.rerank_model,
        "embed_model": settings.embed_model,
        "corpus_version": manifest.get("corpus_version", "unknown"),
        "index_snapshot": snapshot,
        "prompt_version": PROMPT_VERSION,
        "rerank_top_k": settings.rerank_top_k,
        "min_model_confidence": settings.min_model_confidence,
        "min_rerank_score": settings.min_rerank_score,
        "max_retrieval_rounds": settings.max_retrieval_rounds,
        "price_in": args.price_in,
        "price_out": args.price_out,
    }
    s = score(rows)
    report(rows, s, cfg)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / f"{stamp}.json"
    sc: VerdictScores = s["scores"]
    out.write_text(json.dumps({
        "config": cfg,
        "wall_clock_s": round(time.perf_counter() - started, 1),
        "summary": {
            "accuracy": sc.accuracy,
            "false_compliant": sc.false_compliant,
            "false_non_compliant": sc.false_non_compliant,
            "over_escalation": sc.over_escalation,
            "missed_escalation": sc.missed_escalation,
            "errors": sc.errors,
            "recall": s["recall"],
            "mrr": s["mrr"],
            "context_precision": s["context_precision"],
            "by_difficulty": s["by_difficulty"],
            "escalation_codes": dict(sc.escalation_codes),
            "confusion": {f"{g}->{a}": n for (g, a), n in sc.confusion.items()},
        },
        "cases": rows,
    }, indent=2, default=str) + "\n")
    print(f"{DIM}written: {out}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
