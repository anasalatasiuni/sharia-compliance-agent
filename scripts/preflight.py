#!/usr/bin/env python
"""Verify the three provider wire formats before spending on a full ingest.

Each check is the smallest request that can prove its point, and each is
independent, so a failure names exactly one thing. The structured-output check
deliberately uses the **real** `DraftAssessment` schema rather than a toy one —
nested models and enums are precisely where strict-mode support tends to differ
between gateways, and a toy schema would pass while the real one fails.

    python scripts/preflight.py                       # Haiku, ~$0.005
    python scripts/preflight.py --model anthropic/claude-opus-5
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sharia_agent.config import get_settings
from sharia_agent.llm import LLM, to_strict_schema
from sharia_agent.models import DraftAssessment
from sharia_agent.retrieval.embeddings import Embedder

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

CLAUSE = (
    "[SS8-3.1.1] AAOIFI SS No. 8 (Murabahah), clause 3/1/1\n"
    "Murabahah > 3/1 Acquisition of the item\n\n"
    "The Institution shall not sell any item in a Murabahah transaction before "
    "it acquires such item."
)

_tokens = {"in": 0, "out": 0}


def record(usage) -> None:
    if usage:
        _tokens["in"] += getattr(usage, "prompt_tokens", 0) or 0
        _tokens["out"] += getattr(usage, "completion_tokens", 0) or 0


def ok(label: str, detail: str = "") -> bool:
    print(f"  {GREEN}PASS{RESET}  {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    return True


def fail(label: str, detail: str) -> bool:
    print(f"  {RED}FAIL{RESET}  {label}\n        {RED}{detail}{RESET}")
    return False


# ---------------------------------------------------------------------------


async def check_embeddings(settings) -> bool:
    """Does /v1/embeddings exist, serve bge-m3, and what width does it return?"""
    print(f"\n{YELLOW}1. Embeddings{RESET}  ({settings.embed_model})")
    embedder = Embedder(settings)
    try:
        vec = await embedder.embed_query("murabaha sale before taking ownership")
    except Exception as exc:
        return fail(
            "embeddings endpoint",
            f"{type(exc).__name__}: {str(exc)[:220]}\n        "
            "If the endpoint or model is unavailable, the fallback is a self-hosted "
            "BGE-M3 via fastembed, or another embedding vendor. Only Embedder changes.",
        )
    finally:
        await embedder.close()

    if not vec or not isinstance(vec[0], float):
        return fail("embeddings endpoint", f"unexpected payload: {str(vec)[:120]}")
    return ok("embeddings endpoint", f"{len(vec)}-dimensional vector returned")


async def check_tool_calling(llm: LLM, model: str) -> bool:
    """Will the model emit a tool_use block in the OpenAI function format?"""
    print(f"\n{YELLOW}2. Tool calling{RESET}  ({model})")
    tool = {
        "type": "function",
        "function": {
            "name": "search_standards",
            "description": "Search AAOIFI Shari'ah Standards for relevant clauses.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
    try:
        res = await llm.complete(
            messages=[{"role": "user", "content":
                       "Find the AAOIFI rule on selling an item before owning it. "
                       "Use the search tool."}],
            system="You research Shari'ah standards. Use the tool when asked.",
            tools=[tool],
            max_tokens=512,
            model=model,
        )
    except Exception as exc:
        return fail("tool calling", f"{type(exc).__name__}: {str(exc)[:220]}")

    record(None)
    _tokens["in"] += res.usage.input_tokens
    _tokens["out"] += res.usage.output_tokens

    if not res.tool_calls:
        return fail(
            "tool calling",
            f"no tool call emitted (finish_reason={res.finish_reason!r}). "
            "The refinement loop would silently no-op and fall back to seed-only "
            "retrieval — degraded, not broken.",
        )
    call = res.tool_calls[0]
    if not isinstance(call.arguments, dict) or "query" not in call.arguments:
        return fail("tool calling", f"arguments did not parse: {call.arguments!r}")
    return ok("tool calling", f'called {call.name}(query={call.arguments["query"][:44]!r})')


async def check_structured_output(llm: LLM, model: str) -> bool:
    """THE critical check: strict json_schema against the real nested schema."""
    print(f"\n{YELLOW}3. Structured output{RESET}  ({model}, real DraftAssessment schema)")
    schema = to_strict_schema(DraftAssessment)
    print(f"  {DIM}schema: {len(schema.get('properties', {}))} fields, "
          f"{len(schema.get('$defs', {}))} nested defs{RESET}")

    user = (
        "Assess this proposal against the retrieved clause.\n\n"
        "## Proposal\nCan Mal sell a vehicle to a customer under murabaha before "
        "buying it from the dealer?\n\n"
        f"## Retrieved clauses\n{CLAUSE}\n\n"
        "## Task\nDecide whether the clause supports, prohibits, or fails to settle "
        "this. Cite the clause id and quote it verbatim."
    )
    try:
        draft, usage = await llm.complete_structured(
            output_model=DraftAssessment,
            messages=[{"role": "user", "content": user}],
            system="You are a Shari'ah compliance research assistant. Cite every claim.",
            max_tokens=2048,
            model=model,
        )
    except Exception as exc:
        return fail("structured output", f"{type(exc).__name__}: {str(exc)[:220]}")

    _tokens["in"] += usage.input_tokens
    _tokens["out"] += usage.output_tokens

    if draft is None:
        return fail(
            "structured output",
            "returned no valid object. THIS IS THE BLOCKING FAILURE — every "
            "assessment would escalate to NEEDS_REVIEW and the service would look "
            "healthy while being useless. Fallback: drop strict mode, prompt for "
            "JSON, validate with Pydantic, accept a retry loop.",
        )

    ok("structured output", f"finding={draft.finding.value} confidence={draft.confidence}")
    if not draft.citations:
        print(f"  {YELLOW}WARN{RESET}  no citations — guardrails would escalate this")
        return True

    cite = draft.citations[0]
    verbatim = " ".join(cite.quote.split()).lower() in " ".join(CLAUSE.split()).lower()
    if cite.chunk_id != "SS8-3.1.1":
        print(f"  {YELLOW}WARN{RESET}  cited {cite.chunk_id!r}, expected 'SS8-3.1.1'")
    print(f"  {GREEN if verbatim else YELLOW}{'PASS' if verbatim else 'WARN'}{RESET}  "
          f"quote is {'verbatim' if verbatim else 'NOT verbatim (guardrails would escalate)'}"
          f"\n        {DIM}{cite.quote[:88]}{RESET}")
    return True


# ---------------------------------------------------------------------------


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-haiku-4.5")
    args = ap.parse_args()

    settings = get_settings()
    if not settings.openrouter_api_key:
        print(f"{RED}OPENROUTER_API_KEY is not set.{RESET} Put it in .env first.")
        return 2

    print(f"{DIM}key ...{settings.openrouter_api_key[-6:]} · "
          f"model {args.model} · effort {settings.model_effort}{RESET}")

    llm = LLM(settings)
    try:
        results = [
            await check_embeddings(settings),
            await check_tool_calling(llm, args.model),
            await check_structured_output(llm, args.model),
        ]
    finally:
        await llm.close()

    # Haiku-rate arithmetic; Opus is 5x. Either way this run costs cents.
    est = _tokens["in"] / 1e6 * 1.0 + _tokens["out"] / 1e6 * 5.0
    print(f"\n{DIM}tokens: {_tokens['in']} in / {_tokens['out']} out "
          f"· approx ${est:.4f} at Haiku rates{RESET}")

    if all(results):
        print(f"\n{GREEN}All checks passed.{RESET} Safe to run the ingest.")
        return 0
    print(f"\n{RED}{results.count(False)} check(s) failed.{RESET} "
          "Do not ingest yet — fix these first.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
