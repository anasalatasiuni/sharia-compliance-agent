"""Qdrant vector store.

Why Qdrant rather than Chroma or pgvector: Qdrant applies payload filters
*during* ANN traversal rather than after it. That matters here more than it
does for a general knowledge base — a query about ijara must never be answered
from murabaha clauses, and the two are lexically near-identical. Post-filtering
would silently drop recall exactly when the filter is doing useful work.

Payload indexes below are not optional decoration: without them Qdrant cannot
use a filter during traversal and falls back to a slower path.

Every client call is async — the service is I/O bound end to end, and a sync
client in a FastAPI route handler would block the event loop under load.
"""

from __future__ import annotations

from contextlib import suppress

from qdrant_client import AsyncQdrantClient, models

from ..config import Settings
from ..models import Clause, RetrievedClause

DENSE = "dense"
SPARSE = "bm25"

_FILTERABLE: dict[str, models.PayloadSchemaType] = {
    "standard_no": models.PayloadSchemaType.KEYWORD,
    "standard_name": models.PayloadSchemaType.KEYWORD,
    "lang": models.PayloadSchemaType.KEYWORD,
    "corpus_version": models.PayloadSchemaType.KEYWORD,
    "is_superseded": models.PayloadSchemaType.BOOL,
}


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=30,
        )
        self.collection = settings.collection

    async def close(self) -> None:
        await self.client.close()

    # -- lifecycle ---------------------------------------------------------

    async def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        exists = await self.client.collection_exists(self.collection)
        if exists and recreate:
            await self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)
                },
                # IDF is computed by Qdrant across the collection, so the client
                # ships raw term frequencies and holds no corpus statistics.
                sparse_vectors_config={
                    SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        for field, schema in _FILTERABLE.items():
            # Idempotent: re-running ingest against an existing collection should
            # not fail because the index is already there.
            with suppress(Exception):
                await self.client.create_payload_index(
                    collection_name=self.collection, field_name=field, field_schema=schema
                )

    async def count(self) -> int:
        res = await self.client.count(self.collection, exact=True)
        return res.count

    # -- writes ------------------------------------------------------------

    async def upsert(
        self,
        clauses: list[Clause],
        dense_vectors: list[list[float]],
        sparse_vectors: list[tuple[list[int], list[float]]],
    ) -> None:
        points = []
        for clause, dense, (idx, val) in zip(clauses, dense_vectors, sparse_vectors, strict=True):
            points.append(
                models.PointStruct(
                    id=_point_id(clause.chunk_id),
                    vector={
                        DENSE: dense,
                        SPARSE: models.SparseVector(indices=idx, values=val),
                    },
                    payload={
                        "chunk_id": clause.chunk_id,
                        "standard_no": clause.standard_no,
                        "standard_name": clause.standard_name,
                        "clause_path": clause.clause_path,
                        "heading_path": clause.heading_path,
                        "text": clause.text,
                        "lang": clause.lang,
                        "corpus_version": clause.corpus_version,
                        "supersedes": clause.supersedes,
                        "superseded_by": clause.superseded_by,
                        "is_superseded": clause.superseded_by is not None,
                    },
                )
            )
        await self.client.upsert(self.collection, points=points, wait=True)

    # -- reads -------------------------------------------------------------

    async def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: tuple[list[int], list[float]],
        limit: int,
        standard_no: str | None = None,
        include_superseded: bool = False,
        corpus_version: str | None = None,
    ) -> list[RetrievedClause]:
        """Dense + lexical retrieval fused with Reciprocal Rank Fusion.

        RRF rather than a weighted score blend: cosine similarity and BM25 are
        on incomparable scales, and any fixed weighting between them is a
        constant someone has to re-tune whenever the corpus changes. RRF only
        consumes rank, so it has nothing to drift.
        """
        query_filter = _build_filter(standard_no, include_superseded, corpus_version)
        idx, val = sparse_vector

        prefetch = [
            models.Prefetch(
                query=dense_vector, using=DENSE, limit=limit * 2, filter=query_filter
            )
        ]
        if idx:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=idx, values=val),
                    using=SPARSE,
                    limit=limit * 2,
                    filter=query_filter,
                )
            )

        res = await self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            RetrievedClause(clause=_clause_from_payload(p.payload), fused_score=p.score)
            for p in res.points
        ]

    async def get_by_chunk_ids(self, chunk_ids: list[str]) -> list[Clause]:
        if not chunk_ids:
            return []
        res = await self.client.retrieve(
            self.collection, ids=[_point_id(c) for c in chunk_ids], with_payload=True
        )
        return [_clause_from_payload(p.payload) for p in res]


# ---------------------------------------------------------------------------


def _build_filter(
    standard_no: str | None, include_superseded: bool, corpus_version: str | None
) -> models.Filter | None:
    must: list[models.FieldCondition] = []
    if standard_no:
        must.append(
            models.FieldCondition(key="standard_no", match=models.MatchValue(value=standard_no))
        )
    if corpus_version:
        must.append(
            models.FieldCondition(
                key="corpus_version", match=models.MatchValue(value=corpus_version)
            )
        )
    if not include_superseded:
        must.append(
            models.FieldCondition(key="is_superseded", match=models.MatchValue(value=False))
        )
    return models.Filter(must=must) if must else None


def _point_id(chunk_id: str) -> str:
    """Qdrant needs a UUID or unsigned int id; derive one deterministically so
    re-ingesting the same clause updates in place instead of duplicating."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"sharia://{chunk_id}"))


def _clause_from_payload(payload: dict) -> Clause:
    return Clause(
        chunk_id=payload["chunk_id"],
        standard_no=payload["standard_no"],
        standard_name=payload["standard_name"],
        clause_path=payload["clause_path"],
        heading_path=payload.get("heading_path") or [],
        text=payload["text"],
        lang=payload.get("lang", "en"),
        corpus_version=payload["corpus_version"],
        supersedes=payload.get("supersedes"),
        superseded_by=payload.get("superseded_by"),
    )
