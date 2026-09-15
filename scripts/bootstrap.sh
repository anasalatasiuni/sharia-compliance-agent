#!/usr/bin/env bash
# Bring up a working system from a clean clone.
#
# The index ships with the repo as a Qdrant snapshot, so nobody has to spend
# money re-embedding 1,518 clauses or depend on the source PDF still being
# downloadable. Restoring it takes seconds; rebuilding it costs about $0.04 and
# three minutes, and is only necessary if you change the corpus.
set -euo pipefail
cd "$(dirname "$0")/.."

QDRANT_URL="${SCA_QDRANT_URL:-http://localhost:6333}"
COLLECTION="${SCA_COLLECTION:-aaoifi_standards}"
SNAPSHOT="corpus/aaoifi_standards.snapshot.gz"

say() { printf '  %s\n' "$*"; }
die() { say "$*"; exit 1; }

command -v docker >/dev/null || die "docker is not installed."
docker compose version >/dev/null 2>&1 || die "docker compose v2 is required."
[ -f "$SNAPSHOT" ] || die "$SNAPSHOT is missing — clone the repo rather than downloading a source archive."

if [ ! -f .env ]; then
  cp .env.example .env
  say ".env created from .env.example — add your OPENROUTER_API_KEY before assessing."
fi

say "starting qdrant..."
docker compose up -d qdrant >/dev/null
for _ in $(seq 1 60); do
  curl -sf --max-time 2 "$QDRANT_URL/healthz" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf --max-time 2 "$QDRANT_URL/healthz" >/dev/null 2>&1 || { say "qdrant did not start"; exit 1; }

points=$(curl -sf "$QDRANT_URL/collections/$COLLECTION" 2>/dev/null \
         | grep -o '"points_count":[0-9]*' | cut -d: -f2 || true)
if [ "${points:-0}" -gt 0 ]; then
  say "collection already holds ${points} points — nothing to restore."
else
  say "restoring the prebuilt index..."
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  gunzip -c "$SNAPSHOT" > "$tmp/index.snapshot"
  curl -sf -X POST "$QDRANT_URL/collections/$COLLECTION/snapshots/upload?priority=snapshot" \
       -H 'Content-Type:multipart/form-data' -F "snapshot=@$tmp/index.snapshot" >/dev/null
  points=$(curl -sf "$QDRANT_URL/collections/$COLLECTION" | grep -o '"points_count":[0-9]*' | cut -d: -f2)
  expected=$(grep -o '"chunk_count": *[0-9]*' corpus/manifest.json | grep -o '[0-9]*' | head -1)
  say "restored ${points} clauses (manifest expects ${expected})."
  [ "${points:-0}" = "$expected" ] || die "count mismatch — /health will report manifest_mismatch."
fi

say ""
if grep -q '^OPENROUTER_API_KEY=sk-or-v1-\.\.\.$' .env 2>/dev/null; then
  say "the index is ready. .env still holds the placeholder key, so /health will"
  say "report degraded until you replace it:"
  say ""
  say "  1. set OPENROUTER_API_KEY in .env"
  say "  2. docker compose up api        (or: uvicorn sharia_agent.api.main:app)"
  say "  3. curl localhost:8000/health"
  say ""
  say "retrieval needs no key and works now:"
  say "  python scripts/retrieval_debug.py 'can we sell before we own it' \\"
  say "    --expect SS8-3.1.1 --no-rerank"
else
  say "ready:"
  say "  docker compose up api           (or: uvicorn sharia_agent.api.main:app)"
  say "  curl localhost:8000/health"
fi
