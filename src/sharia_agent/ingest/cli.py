"""`sharia-ingest` — parse the AAOIFI source, chunk it, and build the index."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

from ..config import get_settings
from ..obs.trace import configure_logging
from .chunk import clauses_from_standard, corpus_version_for
from .index import build_index
from .parse import citable_entries, parse_standards

DEFAULT_SOURCE = Path("corpus/raw/aaoifi-standards-en-2017.txt")
SOURCE_URL = (
    "https://archive.org/details/AAOIFIShariaaStandardsENG1"
    " (AAOIFI Shari'ah Standards, English, 2017)"
)

# A deliberately narrow slice: the contract families Mal actually issues, plus
# the two cross-cutting standards (gharar, promise) that most product questions
# end up turning on.
DEFAULT_STANDARDS = ["8", "9", "12", "13", "17", "23", "31", "49"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest AAOIFI standards into Qdrant.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--standards",
        nargs="*",
        default=DEFAULT_STANDARDS,
        help="AAOIFI standard numbers to index.",
    )
    parser.add_argument("--recreate", action="store_true", help="Drop the collection first.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk, do not index.")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    if not args.source.exists():
        print(f"source not found: {args.source}", file=sys.stderr)
        print("Download it first — see README, 'Corpus'.", file=sys.stderr)
        return 2

    raw_bytes = args.source.read_bytes()
    raw = raw_bytes.decode("utf-8", errors="replace")
    wanted = set(args.standards)

    standards = parse_standards(raw, wanted=wanted)
    found = {s.number for s in standards}
    if missing := wanted - found:
        print(
            f"warning: requested standards not found: {sorted(missing, key=int)}",
            file=sys.stderr,
        )

    corpus_version = corpus_version_for(raw_bytes, sorted(found))
    clauses = []
    for std in standards:
        produced = clauses_from_standard(std, corpus_version)
        clauses.extend(produced)
        print(
            f"  SS{std.number:>2}  {std.name[:44]:46s} "
            f"{len(citable_entries(std)):>3} clauses -> {len(produced):>3} chunks"
        )

    print(f"\ncorpus_version: {corpus_version}")
    print(f"total chunks:   {len(clauses)}")

    if args.dry_run:
        print("\n--dry-run: nothing indexed")
        return 0

    if not settings.openrouter_api_key:
        print("\nOPENROUTER_API_KEY is not set — cannot embed. See .env.example.", file=sys.stderr)
        return 2

    source_meta = {
        "url": SOURCE_URL,
        "file": str(args.source),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "standards_requested": sorted(wanted, key=int),
        "standards_indexed": sorted(found, key=int),
    }

    print("\nindexing...")
    manifest = asyncio.run(
        build_index(clauses, settings, recreate=args.recreate, source=source_meta)
    )
    print(f"\nindex_snapshot: {manifest['index_snapshot']}")
    print(f"points:         {manifest['points_in_collection']}")
    print("manifest:       corpus/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
