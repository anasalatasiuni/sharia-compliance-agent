"""`sharia-ingest` — parse the AAOIFI source, chunk it, and build the index."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from ..config import get_settings
from ..obs.trace import configure_logging
from .chunk import clauses_from_standard, corpus_version_for
from .index import build_index
from .parse import citable_entries, is_well_parsed, parse_standards

# The publisher's own PDF, not the Internet Archive's OCR of it. Both are
# available; the PDF carries a real text layer and parses measurably cleaner
# (48/48 standards resolve vs 46/52, and ~8% more clauses survive).
DEFAULT_SOURCE = Path("corpus/raw/aaoifi-shariah-standards-en-2017.pdf")
SOURCE_URL = (
    "https://archive.org/details/AAOIFIShariaaStandardsENG1"
    " (AAOIFI Shari'ah Standards, English, 2017)"
)


def extract_text(source: Path) -> tuple[bytes, str]:
    """Return (bytes used for versioning, text to parse).

    A PDF is converted with `pdftotext`. The PDF's own bytes are what version
    the corpus, so the recorded `corpus_version` identifies the document rather
    than a derived artefact that a different poppler build might render
    slightly differently.
    """
    raw_bytes = source.read_bytes()
    if source.suffix.lower() != ".pdf":
        return raw_bytes, raw_bytes.decode("utf-8", errors="replace")

    if shutil.which("pdftotext") is None:
        raise SystemExit(
            "pdftotext not found — install poppler-utils, or pass a pre-extracted "
            ".txt with --source"
        )
    cached = source.with_suffix(".pdftext.txt")
    if not cached.exists() or cached.stat().st_mtime < source.stat().st_mtime:
        print(f"  extracting text from {source.name} ...", flush=True)
        subprocess.run(
            ["pdftotext", str(source), str(cached)], check=True, capture_output=True
        )
    return raw_bytes, cached.read_text(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest AAOIFI standards into Qdrant.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--standards",
        nargs="*",
        default=None,
        help="Standard numbers to index. Omit to index every one that parses cleanly.",
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

    raw_bytes, raw = extract_text(args.source)
    wanted = set(args.standards) if args.standards else None

    parsed = parse_standards(raw, wanted=wanted)
    keep = [s for s in parsed if is_well_parsed(s)]
    dropped = [s for s in parsed if not is_well_parsed(s)]

    if wanted and (missing := wanted - {s.number for s in parsed}):
        print(f"warning: requested standards not found: {sorted(missing, key=int)}",
              file=sys.stderr)

    corpus_version = corpus_version_for(raw_bytes, sorted(s.number for s in keep))
    clauses = []
    for std in keep:
        produced = clauses_from_standard(std, corpus_version)
        clauses.extend(produced)
        print(f"  SS{std.number:>2}  {std.name[:44]:46s} "
              f"{len(citable_entries(std)):>3} clauses -> {len(produced):>3} chunks")

    if dropped:
        print(f"\n  excluded {len(dropped)} standard(s) that did not parse cleanly "
              f"(damaged headers in the source):")
        for std in dropped:
            print(f"    SS{std.number:<4} {len(citable_entries(std)):>3} clauses  {std.name[:40]}")

    print(f"\nsource:         {args.source}")
    print(f"corpus_version: {corpus_version}")
    print(f"standards:      {len(keep)} indexed, {len(dropped)} excluded")
    print(f"total chunks:   {len(clauses)}")

    if args.dry_run:
        print("\n--dry-run: nothing indexed")
        return 0

    if not settings.openrouter_api_key:
        print("\nOPENROUTER_API_KEY is not set — cannot embed. See .env.example.",
              file=sys.stderr)
        return 2

    source_meta = {
        "url": SOURCE_URL,
        "file": str(args.source),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "standards_indexed": sorted((s.number for s in keep), key=int),
        "standards_excluded": sorted((s.number for s in dropped), key=int),
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
