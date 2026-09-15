#!/usr/bin/env python
"""Browse and search the indexed clauses. No API calls — reads Qdrant directly.

    python scripts/clauses.py --standards                 # what is indexed
    python scripts/clauses.py --standard 8                # list one standard
    python scripts/clauses.py --grep "late payment"       # substring search
    python scripts/clauses.py --id SS8-3.1.1 SS13-8.5     # full text by id
    python scripts/clauses.py --exists SS8-3.1.1 SS35-3.1 # verify ids resolve
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from qdrant_client import models

from sharia_agent.config import get_settings
from sharia_agent.retrieval.store import VectorStore


async def scroll_all(store: VectorStore, standard: str | None = None):
    flt = (
        models.Filter(must=[models.FieldCondition(
            key="standard_no", match=models.MatchValue(value=standard))])
        if standard else None
    )
    out, offset = [], None
    while True:
        points, offset = await store.client.scroll(
            collection_name=store.collection, scroll_filter=flt,
            limit=512, offset=offset, with_payload=True, with_vectors=False)
        out.extend(p.payload for p in points)
        if offset is None:
            break
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standards", action="store_true")
    ap.add_argument("--standard")
    ap.add_argument("--grep")
    ap.add_argument("--id", nargs="*")
    ap.add_argument("--exists", nargs="*")
    ap.add_argument("--full", action="store_true", help="Print whole clause text.")
    args = ap.parse_args()

    logging.disable(logging.INFO)
    store = VectorStore(get_settings())
    try:
        if args.exists:
            found = {c.chunk_id for c in await store.get_by_chunk_ids(args.exists)}
            bad = 0
            for cid in args.exists:
                ok = cid in found
                bad += not ok
                print(f"  {'OK     ' if ok else 'MISSING'} {cid}")
            return 1 if bad else 0

        if args.id:
            for c in await store.get_by_chunk_ids(args.id):
                print(f"\n{c.chunk_id}  |  {c.citation_label}")
                print(f"  {' > '.join(c.heading_path)[:110]}")
                print(f"  {c.text}\n")
            return 0

        rows = await scroll_all(store, args.standard)

        if args.standards:
            by: dict[str, tuple[str, int]] = {}
            for p in rows:
                n = p["standard_no"]
                name, cnt = by.get(n, (p["standard_name"], 0))
                by[n] = (name, cnt + 1)
            for n in sorted(by, key=int):
                print(f"  SS{n:<4} {by[n][1]:>4} clauses  {by[n][0][:58]}")
            print(f"\n  {len(by)} standards, {len(rows)} clauses")
            return 0

        rows.sort(key=lambda p: (int(p["standard_no"]),
                                 [int(x) for x in p["clause_path"].split("/")]))
        needle = (args.grep or "").lower()
        n = 0
        for p in rows:
            if needle and needle not in p["text"].lower():
                continue
            n += 1
            print(f"\n  {p['chunk_id']:<16} SS{p['standard_no']} {p['standard_name'][:34]}")
            print(f"    {p['text'] if args.full else p['text'][:220]}")
        print(f"\n  {n} clause(s)")
        return 0
    finally:
        await store.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
