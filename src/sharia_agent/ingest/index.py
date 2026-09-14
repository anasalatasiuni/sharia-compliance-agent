"""Build the index and the manifest that describes it.

The manifest is not documentation — it is the thing that makes verdicts
replayable. `index_snapshot` folds in the corpus content, the clause selection
*and* the embedding model, so it changes whenever anything that could alter
retrieval changes. Any cache keyed on it is invalidated for free when the
corpus is re-indexed, which is the failure mode behind "the knowledge base was
updated but answers are still stale".
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ..config import Settings
from ..models import Clause
from ..retrieval import sparse
from ..retrieval.embeddings import Embedder
from ..retrieval.store import VectorStore

MANIFEST_PATH = Path("corpus/manifest.json")


def compute_index_snapshot(clauses: list[Clause], settings: Settings) -> str:
    h = hashlib.sha256()
    for chunk_id in sorted(c.chunk_id for c in clauses):
        h.update(chunk_id.encode())
    if clauses:
        h.update(clauses[0].corpus_version.encode())
    h.update(f"{settings.embed_model}:{settings.embed_dim}".encode())
    return h.hexdigest()[:16]


async def build_index(
    clauses: list[Clause],
    settings: Settings,
    *,
    recreate: bool = False,
    source: dict | None = None,
    batch: int = 96,
) -> dict:
    store = VectorStore(settings)
    embedder = Embedder(settings)
    dim = settings.embed_dim or 0
    try:
        # Embed one batch first so the collection is created at the width the
        # model actually produces, rather than at a width we guessed.
        first = clauses[:batch]
        first_texts = [c.for_prompt() for c in first]
        first_dense = await embedder.embed_documents(first_texts)
        detected = len(first_dense[0]) if first_dense else dim
        if dim and detected != dim:
            raise ValueError(
                f"SCA_EMBED_DIM={dim} but {settings.embed_model} returned {detected}-d vectors"
            )
        dim = detected

        await store.ensure_collection(dim=dim, recreate=recreate)
        await store.upsert(first, first_dense, [sparse.encode(t) for t in first_texts])
        print(f"  indexed {len(first)}/{len(clauses)} (dim={dim})", flush=True)

        for start in range(batch, len(clauses), batch):
            window = clauses[start : start + batch]
            texts = [c.for_prompt() for c in window]
            dense = await embedder.embed_documents(texts)
            sparse_vectors = [sparse.encode(t) for t in texts]
            await store.upsert(window, dense, sparse_vectors)
            print(f"  indexed {min(start + batch, len(clauses))}/{len(clauses)}", flush=True)

        indexed = await store.count()
    finally:
        await embedder.close()
        await store.close()

    by_standard: dict[str, dict] = {}
    for c in clauses:
        entry = by_standard.setdefault(
            c.standard_no, {"name": c.standard_name, "chunks": 0}
        )
        entry["chunks"] += 1

    manifest = {
        "corpus_version": clauses[0].corpus_version if clauses else "",
        "index_snapshot": compute_index_snapshot(clauses, settings),
        "built_at": datetime.now(UTC).isoformat(),
        "collection": settings.collection,
        "embed_model": settings.embed_model,
        "embed_dim": dim,
        "chunk_count": len(clauses),
        "points_in_collection": indexed,
        "standards": {
            k: by_standard[k] for k in sorted(by_standard, key=int)
        },
        "source": source or {},
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())
