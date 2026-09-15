# Shari'ah Compliance Agent

Decision support for Mal's internal compliance team. Give it a proposed product
or transaction in plain English; it returns an evidence-backed preliminary
assessment against the AAOIFI Shari'ah Standards, with every claim cited to a
specific clause.

> **It does not issue Shari'ah rulings.** Under CBUAE rules, Shari'ah
> determinations are reserved to the institution's Internal Shari'ah Supervision
> Committee (ISSC). This system triages and prepares a defensible memo for that
> committee. `NEEDS_REVIEW` is its primary output, not its failure mode.

## The central design decision

The brief asks for an AI agent, and this is one: it plans its own retrieval,
judges whether the evidence it has is enough, and searches again when it is not.
The design decision that shaped everything else is where that agency stops.

> **The model has genuine agency over what evidence to gather, and none over how
> the verdict is decided.**

It can reformulate a question into the vocabulary the standards actually use, and
it can search again. It cannot choose the control flow, and it has no `verdict`
field to fill in. The two places a verdict is decided, Gate A and Gate B, are
ordinary Python that calls no model.

That buys three things a ReAct loop cannot. **Reproducibility**, because an
unbounded loop takes a different trajectory each run and a regulator asking "why
did it say this?" needs the same answer twice. **Auditability**, because the
control flow and the gates are ~800 lines of ordinary Python you can read end to
end rather than a framework's internals. And
**asymmetric safety**, because guardrails only ever escalate, so a model
regression can make the system noisier but cannot make it more permissive.

## Architecture

```
  POST /assess                                        ┌──────────────────┐
  Authorization: Bearer <token>                       │  Qdrant          │
         │                                            │  1,518 clauses   │
         ▼                                            │  dense + sparse  │
  ┌─────────────────────────────────────────┐         └────────▲─────────┘
  │ 1  INTAKE                               │                  │
  │    authorize · redact PII · trace_id    │                  │
  └────────────────────┬────────────────────┘                  │
                       ▼                                       │
  ┌─────────────────────────────────────────┐                  │
  │ 2  RETRIEVE   seed search — always runs │──────────────────┤
  │      bge-m3 dense ─┐                    │                  │
  │      BM25 sparse  ─┴─▶ RRF fusion       │                  │
  │      cross-encoder rerank ─▶ top 5      │                  │
  └────────────────────┬────────────────────┘                  │
                       ▼                                       │
  ┌─────────────────────────────────────────┐                  │
  │ 3  REFINE     bounded agentic loop      │──────────────────┘
  │    model may search again · max 3 total │
  │    repeat-query detection               │   ◀── the only stage
  └────────────────────┬────────────────────┘       with model agency
                       ▼
  ┌═════════════════════════════════════════┐
  ║ 4  GATE A     enough evidence at all?   ║──── no ──┐
  ╚════════════════════╤════════════════════╝          │
                       ▼ yes                           │
  ┌─────────────────────────────────────────┐          │
  │ 5  REASON     claude-haiku-4.5          │          │
  │    structured output · schema-enforced  │          │
  │    returns a *finding*, not a verdict   │          │
  └────────────────────┬────────────────────┘          │
                       ▼                               │
  ┌─────────────────────────────────────────┐          │
  │ 6  VALIDATE   citations resolve?        │          │
  │               quotes verbatim?          │          │
  └────────────────────┬────────────────────┘          │
                       ▼                               │
  ┌═════════════════════════════════════════┐          │
  ║ 7  GATE B     guardrails                ║          │
  ║    may escalate · may never relax       ║◀─────────┘
  ╚════════════════════╤════════════════════╝
                       ▼
  ┌─────────────────────────────────────────┐
  │ 8  EMIT     assessment + audit record   │
  │  COMPLIANT │ NON_COMPLIANT │ NEEDS_REVIEW│
  └─────────────────────────────────────────┘

  ══ double border = decides a verdict, calls no model
```

The model returns a **finding** about evidence; deterministic code maps that onto
the **verdict**. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for that mapping,
the escalation codes, and why a chunk is a clause.

## Quick start

```bash
git clone https://github.com/anasalatasiuni/sharia-compliance-agent
cd sharia-compliance-agent
./scripts/bootstrap.sh          # starts Qdrant, restores the prebuilt index
$EDITOR .env                    # add OPENROUTER_API_KEY
docker compose up api
```

```bash
curl -s localhost:8000/health | jq          # expect points: 1518

curl -sS -X POST localhost:8000/assess \
  -H "Authorization: Bearer demo-token-analyst" \
  -H "Content-Type: application/json" \
  -d '{"query": "Can Mal offer a savings account paying a fixed 4% annual return?"}'
```

The index ships with the repo as a Qdrant snapshot, so nothing has to be
re-embedded. An OpenRouter key is needed only for the reasoning and reranking
calls, about $0.02 per assessment; until one is set, `/health` reports `degraded`
with `credentials.openrouter: false` rather than failing later at the first
request. Retrieval itself is free and works immediately:

```bash
python scripts/retrieval_debug.py "can we sell before we own it" \
  --expect SS8-3.1.1 --no-rerank
```

<details>
<summary>Running without Docker, or rebuilding the index</summary>

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
docker compose up -d qdrant && ./scripts/bootstrap.sh
uvicorn sharia_agent.api.main:app --reload
```

Rebuilding is only needed if the corpus changes. It costs ~$0.04 and needs
poppler plus the source PDF:

```bash
# Debian/Ubuntu: sudo apt install poppler-utils      macOS: brew install poppler
curl -L -o corpus/raw/aaoifi-shariah-standards-en-2017.pdf \
  "https://archive.org/download/AAOIFIShariaaStandardsENG1/AAOIFI_Shariaa-Standards-ENG%201.pdf"

python -m sharia_agent.ingest.cli --dry-run   # parse + chunk, no API calls, free
python -m sharia_agent.ingest.cli             # embed + index (~1,518 chunks)
```

The first run caches the extracted text and writes `corpus/manifest.json` with the
`corpus_version` and an `index_snapshot` folding in corpus content, clause
selection and embedding model. Every audit record pins those, which is what makes
a verdict replayable.

</details>

## Example response

```jsonc
{
  "verdict": "NON_COMPLIANT",
  "confidence": 0.95,
  "reasoning": "The proposal requests that Mal sell a vehicle to a customer via murabaha before Mal has purchased it from the dealer. […] The proposal reverses steps 1 and 3, which the standards forbid.",
  "citations": [
    {
      "chunk_id": "SS8-3.1.1",
      "quote": "The Institution shall not sell any item in a Murabahah transaction before it acquires such item…",
      "supports": "The core prohibition against selling before acquisition"
    }
  ],
  "concerns": ["ownership_sequence"],
  "escalations": [],
  "trace_id": "6b9ec77b8b9a45bb82751b59873650b9",
  "corpus_version": "aaoifi-en-2017@145a0995711cf7ed",
  "model": "anthropic/claude-haiku-4.5",
  "latency_ms": 18032,
  "disclaimer": "Decision-support output. This is not a fatwa…"
}
```

Full response, the async path and every setting are in
[docs/API.md](docs/API.md).

## Evaluated

Two runs of 40 labelled cases, committed under `eval/results/`:

| | run 1 | run 2 |
|---|---|---|
| accuracy | 0.925 | 0.925 |
| **false `COMPLIANT`** | **0** | **0** |
| recall@5 · MRR | 0.838 · 0.840 | 0.811 · 0.870 |

Zero false `COMPLIANT` held across both runs, and across a third in which a rate
limit had broken every upstream call: a failing dependency degrades to escalation
by construction. `scripts/verify_docs.py` re-reads the saved runs and fails if any
figure quoted in the docs drifts from them.

## Known limitations

**No live URL.** Free container hosting without a payment method has closed:
Render, Koyeb and Fly require a card, Hugging Face Spaces made the Docker runtime
paid in July 2026, and Back4App's free URLs expire an hour after issue. The
deployment path is built and was exercised against a Qdrant Cloud cluster, and
`./scripts/bootstrap.sh && docker compose up` brings the whole system up locally
with the index already populated.

**No supersession data.** `superseded_by`, the guardrail reading it and the
retrieval filter excluding it are all built, and nothing populates them, because
the 2017 edition was ingested as a flat snapshot. Until then a repealed clause
would be cited as though live.

**Six standards are missing from the shipped index: 42, 43, 44, 46, 47, 48.**
The 2017 edition has 54 and the index holds 48. They are in the source; the header
regex required `Standard No.` with a period and the publisher's running headers
omit it, so those standards never resolved a title and the quality gate dropped
them. `parse.py` is fixed and `tests/test_ingest.py` fails on the old behaviour,
but **the index has not been rebuilt**, so the gap is live. It matters beyond the
count: SS 46 governs wakala investment accounts, the compliant alternative to the
fixed-return savings account in the example above, and a wakala question retrieves
plausible neighbours from SS 40 and SS 23 instead, so no guardrail fires.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the full diagnosis.

**One corpus edition, English.** CBUAE circulars, HSA resolutions and Mal's own
product policy are the other half of what governs a real product decision, and
none is public. Arabic retrieval works by construction and has not been measured.

**Demo scaffolding with the seams cut.** Reranking defaults to an LLM (`local`
removes both the cost and the data egress), the job store is in-process (`jobs.py`
is narrow enough that Redis is a one-file change), and auth is static tokens (the
`scopes` shape downstream is already what an IdP would populate). Redaction on the
egress path is a compensating control rather than a sufficient one, which
[docs/PART2-technical-decisions.md](docs/PART2-technical-decisions.md) §5 covers
against CBUAE residency rules.

## Documentation

| | |
|---|---|
| [docs/PART2-technical-decisions.md](docs/PART2-technical-decisions.md) | Architecture decisions and what was ruled out, scaling to 50k queries/day, the evaluation framework, production observability, CBUAE and PDPL risk, and what was deprioritised. Also as a [4-page PDF](docs/PART2-technical-decisions.pdf). |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The two vocabularies, escalation codes, the corpus, and why a chunk is a clause |
| [docs/API.md](docs/API.md) | Endpoint reference and every configuration variable |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Observability, the audit record, tests, diagnostic scripts, deployment |
