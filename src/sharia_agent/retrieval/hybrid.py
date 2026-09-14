"""Retrieval: hybrid candidate generation, then cross-encoder reranking.

Two stages rather than one because they fix different failures. Hybrid search
decides *what gets considered* — dense catches paraphrase, BM25 catches clause
numbers and transliterations. Reranking decides *what gets sent to the model* —
and sending five well-ordered clauses instead of forty is simultaneously the
accuracy win and, at volume, the dominant cost lever, since retrieved context
is the one part of the prompt that cannot be cached.
"""

from __future__ import annotations

from ..config import Settings
from ..models import RetrievedClause
from ..obs.trace import log, span
from ..resilience import UpstreamUnavailable
from . import sparse
from .embeddings import Embedder
from .rerank import Reranker
from .store import VectorStore


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings

    async def search(
        self,
        query: str,
        *,
        standard_no: str | None = None,
        round_no: int = 1,
        top_k: int | None = None,
    ) -> list[RetrievedClause]:
        top_k = top_k or self.settings.rerank_top_k

        with span("retrieve", round=round_no, standard_no=standard_no) as s:
            dense = await self.embedder.embed_query(query)
            sparse_vec = sparse.encode(query)

            candidates = await self.store.hybrid_search(
                dense_vector=dense,
                sparse_vector=sparse_vec,
                limit=self.settings.retrieve_candidates,
                standard_no=standard_no,
            )
            s.annotate(candidates=len(candidates), sparse_terms=len(sparse_vec[0]))

        if not candidates:
            return []

        with span("rerank", round=round_no, candidates=len(candidates)) as s:
            try:
                ranked = await self.reranker.rerank(
                    query=query,
                    documents=[c.clause.for_prompt() for c in candidates],
                    top_n=top_k,
                )
                if not ranked:
                    # NullReranker, or a backend that scored nothing. Fusion order
                    # stands, but without scores the guardrails escalate.
                    for c in candidates[:top_k]:
                        c.retrieval_round = round_no
                    s.annotate(backend=self.reranker.name, scored=False)
                    return candidates[:top_k]
            except UpstreamUnavailable as exc:
                # Degrade to fusion order rather than failing the request. The
                # absence of rerank scores is itself a signal: guardrails read
                # it as weak retrieval and escalate to human review.
                log("rerank.degraded", service=exc.service, reason=exc.reason)
                s.annotate(degraded=True)
                for c in candidates[:top_k]:
                    c.retrieval_round = round_no
                return candidates[:top_k]

            out: list[RetrievedClause] = []
            for index, score in ranked:
                item = candidates[index]
                item.rerank_score = score
                item.retrieval_round = round_no
                out.append(item)
            s.annotate(
                backend=self.reranker.name,
                returned=len(out),
                top_score=round(out[0].rerank_score, 4) if out else None,
            )
            return out
