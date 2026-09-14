"""Dense embeddings via OpenRouter.

Model choice: `baai/bge-m3`. It is trained for cross-lingual retrieval across
100+ languages with genuine Arabic coverage, which matters because Mal's clause
corpus and its users' questions will not always be in the same language — and an
English-optimised embedder silently returns plausible-but-wrong clauses rather
than failing.

The vector dimension is discovered from the provider on first use rather than
hardcoded. Hardcoding it is a classic way to produce a collection whose vectors
are the wrong width for the model that later queries it.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from ..config import Settings
from ..obs.trace import log
from ..resilience import CircuitBreaker, call_with_resilience

EMBED_BATCH = 64

_embed_breaker = CircuitBreaker(service="openrouter.embeddings", failure_threshold=5)


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._dim: int | None = settings.embed_dim or None
        self.client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=60.0,
            max_retries=0,
        )

    @property
    def dimension(self) -> int | None:
        return self._dim

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        async def call() -> list[list[float]]:
            res = await self.client.embeddings.create(
                model=self._s.embed_model, input=texts
            )
            return [item.embedding for item in res.data]

        vectors = await call_with_resilience(
            call,
            service="openrouter.embeddings",
            breaker=_embed_breaker,
            timeout=60.0,
            attempts=3,
        )
        if vectors and self._dim is None:
            self._dim = len(vectors[0])
            log("embeddings.dimension_detected", model=self._s.embed_model, dim=self._dim)
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            out.extend(await self._embed(texts[start : start + EMBED_BATCH]))
        return out

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text]))[0]

    async def close(self) -> None:
        await self.client.close()


def breaker_state() -> dict[str, str]:
    return {_embed_breaker.service: _embed_breaker.state}
