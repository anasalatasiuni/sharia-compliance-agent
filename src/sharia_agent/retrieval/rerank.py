"""Reranking — three interchangeable backends behind one interface.

Reranking is the second retrieval stage and the one that decides what the
reasoning model actually sees. Hybrid search casts a wide net; the reranker
scores each candidate *jointly* with the query, which is what separates clauses
that are topically identical but differ on the single condition that settles the
question — the normal case in a standards corpus.

OpenRouter exposes no rerank endpoint, so the backend is a deployment choice:

  LocalCrossEncoder  bge-reranker-v2-m3 (Apache-2.0), ~560MB, no data egress.
                     The production answer under CBUAE residency rules, since
                     clause text and the query never leave the deployment.
                     Note: jina-reranker-v2-multilingual scores well but is
                     CC-BY-NC-4.0 and therefore unusable commercially.

  LLMReranker        A small model scores relevance. No extra key, no RAM, and
                     acceptable latency at demo volume — but it sends clause
                     text to a third party and costs tokens per query, so it
                     does not survive contact with 50k requests/day.

  NullReranker       Falls back to fusion order. Retrieval quality is then
                     unverified, and the guardrails treat that as grounds for
                     escalation rather than pretending the order is meaningful.
"""

from __future__ import annotations

import json
from typing import Protocol

from ..config import Settings
from ..llm import record_usage, usage_from
from ..obs.trace import log
from ..resilience import CircuitBreaker, UpstreamUnavailable, call_with_resilience

_rerank_breaker = CircuitBreaker(service="rerank", failure_threshold=5)


class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """Return [(original_index, score)] best first, at most top_n items."""
        ...


# ---------------------------------------------------------------------------


class NullReranker:
    """No reranking. Preserves fusion order and reports no scores.

    Returning no scores is deliberate: it is an honest signal that ranking is
    unverified, and the guardrails act on it.
    """

    name = "null"

    async def rerank(self, query: str, documents: list[str], top_n: int):
        return []


# ---------------------------------------------------------------------------


_RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "number"},
                },
                "required": ["index", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rankings"],
    "additionalProperties": False,
}

_RERANK_SYSTEM = """You score how well each numbered passage answers a search query.

Score 0.0 to 1.0:
  0.9-1.0  directly settles the query
  0.6-0.8  clearly on point but partial
  0.3-0.5  same topic, does not address the query
  0.0-0.2  unrelated

Judge only whether the passage bears on the query. Do not judge whether the \
underlying proposal is permissible. Return every passage index exactly once."""


class LLMReranker:
    """Cross-encoding by prompting. One call scores the whole candidate set."""

    name = "llm"

    def __init__(self, llm, model: str) -> None:
        self._llm = llm
        self._model = model

    async def rerank(self, query: str, documents: list[str], top_n: int):
        # Truncated: the reranker needs enough text to judge relevance, not the
        # whole clause, and the candidate pool is large enough that full text
        # would dominate the token bill.
        listing = "\n\n".join(
            f"[{i}] {doc[:700]}" for i, doc in enumerate(documents)
        )
        user = f"Query: {query}\n\nPassages:\n\n{listing}"

        async def call():
            return await self._llm.client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _RERANK_SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_tokens=2048,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Rankings",
                        "strict": True,
                        "schema": _RERANK_SCHEMA,
                    },
                },
            )

        response = await call_with_resilience(
            call,
            service="rerank",
            breaker=_rerank_breaker,
            timeout=60.0,
            attempts=2,
        )
        # Reranking is a real line item — roughly 43% of an assessment's token
        # cost on Haiku — so it has to reach the audit record like any other call.
        record_usage(usage_from(response))

        content = response.choices[0].message.content or "{}"
        try:
            rankings = json.loads(content).get("rankings", [])
        except json.JSONDecodeError:
            raise UpstreamUnavailable("rerank", "unparseable ranking response") from None

        scored = [
            (int(r["index"]), float(r["score"]))
            for r in rankings
            if isinstance(r, dict)
            and isinstance(r.get("index"), int)
            and 0 <= r["index"] < len(documents)
        ]
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:top_n]


# ---------------------------------------------------------------------------


class LocalCrossEncoder:
    """bge-reranker-v2-m3 through FastEmbed's ONNX runtime, on CPU.

    Loaded lazily and scored in a worker thread: the model is synchronous and
    CPU-bound, and running it inline would block the event loop for every other
    in-flight request.
    """

    name = "local"

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self._model_name = model_name
        self._encoder = None

    def _load(self):
        if self._encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            log("rerank.loading_local_model", model=self._model_name)
            self._encoder = TextCrossEncoder(model_name=self._model_name)
        return self._encoder

    async def rerank(self, query: str, documents: list[str], top_n: int):
        import asyncio

        def score() -> list[float]:
            return list(self._load().rerank(query, documents))

        try:
            scores = await asyncio.to_thread(score)
        except Exception as exc:  # noqa: BLE001 — surfaced as an upstream failure
            raise UpstreamUnavailable("rerank", f"local model failed: {exc}") from exc

        ranked = sorted(enumerate(scores), key=lambda p: p[1], reverse=True)
        return [(i, float(s)) for i, s in ranked[:top_n]]


# ---------------------------------------------------------------------------


def build_reranker(settings: Settings, llm) -> Reranker:
    backend = (settings.rerank_backend or "llm").lower()
    if backend == "none":
        return NullReranker()
    if backend == "local":
        return LocalCrossEncoder(settings.rerank_local_model)
    return LLMReranker(llm, settings.rerank_model)


def breaker_state() -> dict[str, str]:
    return {_rerank_breaker.service: _rerank_breaker.state}
