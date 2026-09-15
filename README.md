---
title: Shari'ah Compliance Agent
emoji: ⚖️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: Auditable RAG over the AAOIFI Shari'ah Standards
---

# Shari'ah Compliance Agent

Decision support for Mal's internal compliance team. Give it a proposed product
or transaction in plain English; it returns an evidence-backed preliminary
assessment against the AAOIFI Shari'ah Standards, with every claim cited to a
specific clause.

> **It does not issue Shari'ah rulings.** Under CBUAE rules, Shari'ah
> determinations are reserved to the institution's Internal Shari'ah Supervision
> Committee (ISSC). This system triages and prepares a defensible memo for that
> committee. `NEEDS_REVIEW` is its primary output, not its failure mode.

---

## The central design decision

The brief asks for an "AI agent". For a compliance verdict, an *autonomous*
agent is the wrong shape, and the difference is the whole architecture:

> **The model has genuine agency over what evidence to gather, and none over how
> the verdict is decided.**

It can reformulate a question into the vocabulary the standards actually use, and
it can search again. It cannot choose the control flow, and it has no `verdict`
field to fill in. The two places a verdict is decided — Gate A and Gate B — are
ordinary Python that calls no model.

This buys three things a ReAct loop cannot:

| | |
|---|---|
| **Reproducibility** | An unbounded loop takes a different trajectory each run. A regulator asking "why did it say this?" needs the same answer twice. |
| **Auditability** | The control flow is 200 lines you can read, not a framework's internals. |
| **Asymmetric safety** | Guardrails only ever escalate. A model regression can make the system noisier; it cannot make it more permissive. |

---

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
  │ 5  REASON     claude-opus-5             │          │
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

### Two vocabularies, deliberately separated

The model returns a **finding** about *evidence*. Deterministic code maps that
onto the **verdict** the service returns. There is no code path where a guardrail
produces `COMPLIANT`.

```
  model may return              code decides
  ──────────────────            ─────────────
  SUPPORTED_COMPLIANT      ──▶  COMPLIANT      (only if every gate passes)
  SUPPORTED_NON_COMPLIANT  ──▶  NON_COMPLIANT  (only if every gate passes)
  INSUFFICIENT_BASIS       ──▶  NEEDS_REVIEW
  CONFLICTING_SOURCES      ──▶  NEEDS_REVIEW
                           ──▶  NEEDS_REVIEW   ← any guardrail firing
```

### Escalation triggers

`NEEDS_REVIEW` is produced by code, for reasons that are recorded as machine-readable
codes so the escalation mix can be tracked as a product metric.

| Code | Meaning |
|---|---|
| `weak_retrieval` | Top reranked clause below the score floor, or ranking unverified |
| `no_citations` / `unresolved_citation` | A claim with no evidence, a citation to a clause that was never retrieved, or a quote that is not verbatim in the clause it names |
| `low_confidence` | Model's own confidence below threshold |
| `insufficient_basis` / `conflicting_sources` | The model's read of the evidence |
| `superseded_standard` | A retrieved clause has been superseded |
| `always_review_category` | Not a model failure — see below |
| `schema_validation_failed` / `upstream_error` | The model or a dependency failed |

**`always_review_category`** covers matters where a machine assessment is not the
appropriate artefact however confident it is: novel structures, capital or profit
guarantees on profit-sharing contracts, cross-border structuring, sukuk issuance,
requests phrased as seeking approval.

---

## Corpus

Real AAOIFI Shari'ah Standards (English, 2017). The source is **the publisher's
own PDF**, not the Internet Archive's OCR of it — both are available, and the PDF
carries a genuine text layer that parses measurably cleaner (48 of 48 standards
resolve, against 46 of 52 from the OCR, with ~8% more clauses surviving).

**1518 clauses across 48 standards**, every standard that parses cleanly.
`corpus_version: aaoifi-en-2017@145a0995711cf7ed`

<details>
<summary>Standards indexed</summary>

| No. | Standard | Chunks |
|----:|----------|-------:|
| 1 | Trading in Currencies | 24 |
| 2 | Debit Card, Charge Card and Credit Card | 24 |
| 3 | Procrastinating Debtor | 14 |
| 4 | Settlement of Debts by Set-Off | 9 |
| 5 | Guarantees | 43 |
| 6 | Conversion of a Conventional Bank to an Islamic Bank | 32 |
| 7 | Hawalah | 27 |
| 8 | Murabahah | 64 |
| 9 | Ijarah and Ijarah Muntahia Bittamleek | 57 |
| 10 | Salam and Parallel Salam | 30 |
| 11 | Istisna'a and Parallel Istisna'a | 54 |
| 12 | Sharikah (Musharakah), and Modern Corporations | 91 |
| 13 | Mudarabah | 32 |
| 14 | Documentary Credit | 37 |
| 15 | Ju'alah | 24 |
| 16 | Commercial Papers | 16 |
| 17 | Investment Sukuk | 58 |
| 18 | Possession (Qabd) | 18 |
| 19 | Loan (Qard) | 14 |
| 20 | Sale of Commodities in Organized Markets | 28 |
| 21 | Financial Paper (Shares and Bonds) | 37 |
| 22 | Concession Contracts | 38 |
| 23 | Agency and the Act of an Uncommissioned Agent (Fodoo | 33 |
| 24 | Syndicated Financing | 17 |
| 25 | Combination of Contracts | 25 |
| 26 | Islamic Insurance | 41 |
| 27 | Indices | 18 |
| 28 | Banking Services in Islamic Banks | 10 |
| 29 | Stipulations and Ethics of Fatwa in the Institutiona | 38 |
| 30 | Monetization (Tawarruq) | 13 |
| 31 | Controls on Gharar in Financial Transactions | 29 |
| 32 | Arbitration | 43 |
| 33 | Waqf | 42 |
| 34 | Hiring of Persons | 38 |
| 35 | Zakah | 115 |
| 36 | Impact of Contingent Incidents on Commitments | 9 |
| 37 | Credit Agreement | 32 |
| 38 | Online Financial Dealings | 26 |
| 39 | Mortgage and Its Contemporary Applications | 26 |
| 40 | Distribution of Profit in Mudarabah-Based Investment | 47 |
| 41 | Islamic Reinsurance | 17 |
| 45 | Protection of Capital and Investments | 16 |
| 49 | Unilateral and Bilateral Promise | 17 |
| 50 | Irrigation Partnership (Musaqat) | 23 |
| 51 | Options to Revoke Contracts Due to Incomplete Perfor | 22 |
| 52 | Options to Reconsider (Cooling-Off Options, Either-O | 27 |
| 53 | 'Arboun (Earnest Money) | 11 |
| 54 | Revocation of Contracts by Exercise of a Cooling-Off | 12 |

</details>

Ingestion applies a quality gate (`ingest/parse.py`, `is_well_parsed`): a standard
whose title never resolved, or which yielded almost no numbered entries, did not
really parse — its headers were damaged in the source — and indexing it would
inject noise without adding coverage. On the PDF text nothing is excluded; on the
OCR text six standards are.

### Why the chunk is the clause

AAOIFI text is hierarchically numbered — `2/2/2` sits under `2/2` under `2`. So:

> **The chunk boundary is the citation unit.**

A retrieved chunk *is* a reference a compliance officer can verify by hand
(`AAOIFI SS No. 8 (Murabahah), clause 3/1/1`). A generic recursive splitter would
straddle `2/2/2` and `2/2/3` and a citation could then only point at a page.

Each chunk carries its heading path, because disclosure text repeats
near-identical language across products — profit distribution under Mudarabah and
under Wakala read alike and mean different things. Clauses longer than 1,800
characters are split on sentence boundaries with overlap, keeping the parent
clause path.

### Source defects are permanent, and the system assumes it

Clause SS8-3.1.1 reads `"concludes a urchase contract"` — in the publisher's own
PDF, under every extraction mode, and it is the only dropped-letter instance in
1,264 pages. It is a typo in the published standard, not an OCR artefact, and no
cleaner source fixes it.

That is why citation checking tolerates imperfect source text rather than
demanding character-perfect equality (`agent/guardrails.py`). Mal will eventually
index its own term sheets and
counsel memos, which will be far messier than AAOIFI's typesetting. A system that
requires a clean corpus is one that breaks on contact with production.

## Quick start

**Prerequisites:** Python 3.12+, Docker, and an
[OpenRouter](https://openrouter.ai) API key. That key is the only credential
needed — it serves both reasoning and embeddings. Qdrant runs locally.

```bash
git clone <repo> && cd sharia-compliance-agent

cp .env.example .env
$EDITOR .env                      # set OPENROUTER_API_KEY

uv venv --python 3.12
uv pip install -e ".[dev]"

docker compose up -d qdrant       # vector store on :6333
```

### Fetch the corpus

Not committed — 12 MB of third-party standards. Ingestion reads the PDF directly
and shells out to `pdftotext`, so you also need poppler.

```bash
# Debian/Ubuntu: sudo apt install poppler-utils
#         macOS: brew install poppler

curl -L -o corpus/raw/aaoifi-shariah-standards-en-2017.pdf \
  "https://archive.org/download/AAOIFIShariaaStandardsENG1/AAOIFI_Shariaa-Standards-ENG%201.pdf"
```

### Build the index

```bash
python -m sharia_agent.ingest.cli --dry-run   # parse + chunk, no API calls, free
python -m sharia_agent.ingest.cli             # embed + index (~1,518 chunks)
```

The first run extracts text from the PDF and caches it alongside. It writes
`corpus/manifest.json` with the `corpus_version`, the detected embedding
dimension, and an `index_snapshot` folding in corpus content, clause selection
and embedding model. Every audit record pins those, which is what makes a verdict
replayable — and what invalidates any cache keyed on the snapshot the moment the
corpus is re-indexed.

`--standards 8 9 13` indexes a subset; omit it for everything that parses cleanly.

### Run

```bash
uvicorn sharia_agent.api.main:app --reload     # http://localhost:8000/docs
```

---

## Using the API

### Assess (synchronous)

```bash
curl -sS -X POST http://localhost:8000/assess \
  -H "Authorization: Bearer demo-token-analyst" \
  -H "Content-Type: application/json" \
  -d '{"query": "Can Mal sell a vehicle to a customer under murabaha before we have purchased it from the dealer?"}'
```

```jsonc
{
  "assessment_id": "3f2a…",
  "trace_id": "9c81…",
  "verdict": "NON_COMPLIANT",
  "confidence": 0.93,
  "reasoning": "The arrangement requires the Institution to contract a sale …",
  "citations": [
    {
      "chunk_id": "SS8-3.1.1",
      "quote": "shall not sell any item in a Murabahah transaction before it acquires such item",
      "supports": "ownership must precede the sale contract"
    }
  ],
  "concerns": ["ownership_sequence"],
  "escalations": [],
  "missing_information": [],
  "clauses_considered": ["SS8-3.1.1", "SS8-3.2.1", "…"],
  "corpus_version": "aaoifi-en-2017@145a0995711cf7ed",
  "model": "anthropic/claude-opus-5",
  "prompt_version": "v1",
  "latency_ms": 7412,
  "disclaimer": "Decision-support output. This is not a fatwa …"
}
```

### Assess (asynchronous)

A hard question with three retrieval rounds can outrun a client timeout. The
async path is also the shape this service takes at volume.

```bash
curl -sS -X POST "http://localhost:8000/assess?mode=async" \
  -H "Authorization: Bearer demo-token-analyst" \
  -H "Content-Type: application/json" \
  -d '{"query": "Is a diminishing musharaka home finance structure with a binding purchase undertaking acceptable?"}'
# → 202 {"job_id": "a1b2…", "status": "queued", "poll": "/assess/a1b2…"}

curl -sS http://localhost:8000/assess/a1b2… \
  -H "Authorization: Bearer demo-token-analyst"
```

Jobs are readable only by the principal that created them — otherwise a job id is
a capability anyone can guess.

### Health

`/health` is **readiness**, not liveness. A process that is up but whose
collection is empty answers every question with `NEEDS_REVIEW` and looks healthy
doing it, so the check requires the index to have content.

```bash
curl -sS http://localhost:8000/health | jq
curl -sS http://localhost:8000/health/live     # pure liveness
```

Returns `503` when the index is empty, a credential is missing, or a circuit
breaker is open.

---

## Deployment

The API runs on Render's free tier; Qdrant Cloud's free tier holds the index.
They are separate because Render's free web service has no persistent disk, so
the index has to live somewhere it survives a restart.

The service idles at ~148 MB against Render's 512 MB, and the index is ~5 MB of
vectors against Qdrant's 1 GB — neither is close to a limit.

### 1. Index a Qdrant Cloud cluster

Create a free cluster (no card required), then ingest against it rather than
against localhost:

```bash
SCA_QDRANT_URL=https://<cluster>.<region>.cloud.qdrant.io:6333 \
SCA_QDRANT_API_KEY=<key> \
python -m sharia_agent.ingest.cli --recreate
```

### 2. Deploy

Two paths, both free.

**Hugging Face Spaces** — no card, 2 vCPU / 16 GB, sleeps only after 48 hours of
inactivity. Create a Space with SDK **Docker**, add the repo as a git remote and
push; the config block at the top of this file supplies the rest. Secrets go in
the Space's *Settings → Variables and secrets*.

**Render** — create a service from [`deploy/render.yaml`](deploy/render.yaml) via
*New → Blueprint*, which applies the region, plan, health-check path and tuning
values automatically. Note the free instance is 512 MB / 0.1 CPU and spins down
after 15 minutes, so the first request after idle takes about a minute.

Either way, four values must be set by hand and never committed:

| variable | |
|---|---|
| `OPENROUTER_API_KEY` | Prefer a key funded with a small balance. It is the only control that fails closed if the bearer token is ever forwarded. |
| `SCA_QDRANT_URL` | the cluster URL |
| `SCA_QDRANT_API_KEY` | the cluster key |
| `SCA_PRINCIPALS` | **Not the demo token.** See below. |

### 3. Use a token that is not in this repo

`.env.example` is committed, so `demo-token-analyst` is public and worthless as
an access control. Generate a real one:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
# SCA_PRINCIPALS=<token>:reviewer@mal.ae:assess
```

### 4. Keep the cluster alive

[`.github/workflows/keepalive.yml`](.github/workflows/keepalive.yml) pings
`/health` once a day. Set the repository variable `DEPLOY_URL` to enable it.

This is not about the cold start. **Qdrant Cloud suspends a free cluster after a
week idle and deletes it after four** — and the failure is silent: the URL still
resolves, `/health` just starts reporting an empty index. A daily request
prevents that at no cost. One ping covers both services, because `/health`
queries the collection.

Preventing Render's 15-minute spin-down would be a different matter: staying
awake is ~720 hours against a 750-hour monthly allowance. Not worth the entire
budget to avoid one minute of waiting, so **the first request after idle takes
about a minute.** Subsequent ones are normal.

### Verifying a deployment

```bash
curl -s https://<your-service>.onrender.com/health | jq   # expect points: 1518
curl -sS -X POST https://<your-service>.onrender.com/assess \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"query": "Can Mal offer a savings account paying a fixed 4% annual return?"}'
```

`/health` returning `503` with `manifest_mismatch` means the service is pointed at
a different index than the one it was built against — re-run the ingest, or fix
`SCA_QDRANT_URL`.

### What this deployment is not

A demo. `SCA_ENV=prod` only tightens logging; the posture described in
[`docs/PART2-technical-decisions.md`](docs/PART2-technical-decisions.md) §5 —
in-country inference under CBUAE residency rules, OIDC instead of static tokens,
audit records on WORM storage — is not what is running here.

## Configuration

Everything tunable is an environment variable, so a deployment is reproducible
from its environment alone. Full list in [`.env.example`](.env.example).

| Variable | Default | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | — | The only credential required |
| `SCA_MODEL` | `anthropic/claude-opus-5` | Reasoning model |
| `SCA_EMBED_MODEL` | `baai/bge-m3` | Multilingual, real Arabic coverage |
| `SCA_EMBED_DIM` | `0` | `0` = detect from the provider at ingest |
| `SCA_RERANK_BACKEND` | `llm` | `llm` \| `local` \| `none` — see below |
| `SCA_RETRIEVE_CANDIDATES` | `40` | Hybrid pool before reranking |
| `SCA_RERANK_TOP_K` | `5` | Clauses sent to the model |
| `SCA_MAX_RETRIEVAL_ROUNDS` | `3` | Hard cap on the search loop |
| `SCA_MIN_RERANK_SCORE` | `0.35` | Below this → `weak_retrieval` |
| `SCA_MIN_MODEL_CONFIDENCE` | `0.70` | Below this → `low_confidence` |
| `SCA_LOG_FULL_PROMPTS` | `true` | **Must be `false` in production** |

### Reranking backends

OpenRouter exposes no rerank endpoint, so this is a deployment choice.

| Backend | Cost | Data egress | Use |
|---|---|---|---|
| `llm` | ~$0.01/query | Clause text leaves | Demo default — no extra setup |
| `local` | free | **None** | Production. `bge-reranker-v2-m3`, Apache-2.0, ~560 MB on CPU. `pip install -e ".[local-rerank]"` |
| `none` | free | None | Fusion order only; guardrails escalate for unverified ranking |

`jina-reranker-v2-multilingual` scores better but is CC-BY-NC-4.0 and therefore
unusable commercially at a bank.

---

## Observability

Structured JSON to stdout, one line per stage, every line carrying `trace_id`.
Span names are OpenTelemetry-shaped (`sharia.retrieve`, `sharia.reason`), so
swapping stdout for a collector is a wiring change.

```jsonc
{"ts":"…","level":"INFO","msg":"sharia.retrieve","trace_id":"9c81…",
 "principal":"analyst@mal.ae","duration_ms":412,"candidates":40,"sparse_terms":14}
{"ts":"…","level":"INFO","msg":"sharia.rerank","trace_id":"9c81…",
 "backend":"llm","returned":5,"top_score":0.91}
{"ts":"…","level":"INFO","msg":"sharia.reason","trace_id":"9c81…",
 "clauses":5,"finding":"SUPPORTED_NON_COMPLIANT","confidence":0.93}
```

### The audit record

Emitted as one `audit.record` event per assessment, so it inherits retention,
access control and shipping from the normal log pipeline. It is what a Shari'ah
reviewer is actually shown, and it contains everything needed to replay a verdict
months later:

```
trace_id · assessment_id · principal_id · query_hash
corpus_version · index_snapshot · model · model_effort · prompt_version
retrieval_rounds[]   each query, why it was issued, how many hits
retrieved[]          chunk_id + fused (RRF) and rerank scores
draft                the model's finding, confidence, citations
final_verdict · escalations[]
stages[]             per-stage latency
usage                tokens across every call the assessment made
```

Per-arm scores are deliberately **not** stored. Qdrant's fusion returns a single
merged score, and recovering the dense and sparse ranks separately would cost two
extra queries on every request to serve the small fraction ever investigated.
They are reconstructible on demand instead — `index_snapshot` pins the index, so
any past retrieval can be replayed:

```bash
python scripts/retrieval_debug.py "<the query>" --expect SS8-3.1.1
```

which reports each arm's rank separately and names the failure:

```
  SS8-3.1.1
    dense  : rank 25          <- the lexical arm carried this query
    sparse : rank 4
    fused  : rank 6
    ranked : rank 1           <- the reranker promoted it five places
```

Pinning `corpus_version` and `prompt_version` is the point: without them you
cannot tell whether a disputed verdict came from a **retrieval miss** (wrong
clauses reached the model) or a **reasoning miss** (right clauses, wrong
conclusion). Those have different fixes and the distinction is unrecoverable
after the fact if it was not recorded.

---

## Testing

```bash
pytest -q          # 51 tests, no network
ruff check .
```

`tests/test_guardrails.py` pins the safety properties — fabricated quotes caught,
citations to unretrieved clauses caught, always-review categories escalating
despite a clean draft, and the one-directional invariant stated directly as a
test. `tests/test_pipeline.py` drives the state machine against fake providers:
the seed search always runs, the retrieval loop is genuinely bounded, repeated
identical searches are suppressed, and an upstream failure degrades to escalation
rather than to an answer.

---

## Tools

Two scripts, both outside the request path.

### `scripts/preflight.py` — verify the provider before spending

Checks embeddings, tool calling, and strict structured output as three
independent requests, so a failure names exactly one thing. The structured-output
check sends the **real** `DraftAssessment` schema rather than a toy one — nested
models and enums are where gateway strict-mode support tends to differ, and a toy
schema would pass while the real one fails.

```bash
python scripts/preflight.py                            # ~$0.005
python scripts/preflight.py --model anthropic/claude-opus-5
```

Run it before an ingest. A failure here means every assessment would escalate
while the service looks healthy.

### `scripts/retrieval_debug.py` — why retrieval found, or missed, a clause

Runs each search arm separately against the live index and separates three
failures that are indistinguishable from the outside.

```bash
python scripts/retrieval_debug.py "can we sell before we own it" \
  --expect SS8-3.1.1 --no-rerank      # --no-rerank makes the run free
```

| reported | meaning | fix lives in |
|---|---|---|
| no arm surfaced it | candidate generation | chunking, embedding model, or the corpus |
| retrieved but cut | ranking | the reranker, or `SCA_RERANK_TOP_K` |
| id not in the index | the test set is wrong, not retrieval | the label |

The last row matters most when building an eval: a mistyped gold clause id fails
exactly like a retrieval miss, and chasing the wrong one costs an afternoon.

## Known limitations

**Corpus is one edition, English only.** All 48 standards that parse cleanly are
indexed, but that is still a single 2017 edition with no CBUAE circulars, no HSA
resolutions and no Mal-internal product policy — any of which would bind in
practice, and which together are the larger half of what actually governs a UAE
product decision.

**No supersession data.** The schema and the guardrail exist, but nothing
populates `superseded_by`, because the 2017 edition was ingested as a flat
snapshot. A superseded clause would currently be cited as though live. This is
the most dangerous gap in the system.

**Reranking defaults to an LLM.** Costs ~$0.01/query and sends clause text to a
third party. Fine for a demo, wrong for production; `local` fixes both.

**Job store is in-process.** Async jobs die with the process and do not survive a
second replica.

**Auth is static tokens.** Demo scaffolding. Real deployment resolves against the
bank's OIDC provider; the `scopes` shape downstream is already the right one.

**Redaction is compensating, not sufficient.** CBUAE requires consumer and
transaction data to be stored and processed inside the UAE. This deployment sends
redacted text to a US-hosted gateway. That is defensible for a demo and not for
production — see [`docs/PART2-technical-decisions.md`](docs/PART2-technical-decisions.md) §5.

**English-only retrieval, tested.** The embedding model and the sparse tokenizer
both handle Arabic (the tokenizer folds alef and ta-marbuta variants), but the
indexed corpus is the English edition and no Arabic query has been evaluated.

**Evaluated on one model, 40 cases.** Baseline is 0.925 accuracy with zero false
`COMPLIANT` (`docs/PART2-technical-decisions.md` §3.0). Not yet run against a
second model, so the cost/quality tradeoff in §2.2.1 remains open, and at n=40 the
calibration finding is suggestive rather than settled.

---

## Layout

```
src/sharia_agent/
  config.py          settings; the only source of tunables
  models.py          domain types — the two vocabularies live here
  llm.py             OpenRouter client; strict-schema structured output
  pii.py             redaction, applied on the request path
  resilience.py      per-service timeouts, backoff, circuit breakers
  jobs.py            async job store
  ingest/            parse → chunk → embed → index, + manifest
  retrieval/         sparse BM25 · Qdrant store · embeddings · rerank · hybrid
  agent/             prompts · tools · guardrails · pipeline
  api/               routes · auth · app wiring
  obs/               trace ids, spans, JSON logging
```

## Further reading

[`docs/PART2-technical-decisions.md`](docs/PART2-technical-decisions.md) — the
architecture decisions and what was ruled out, scaling to 50k queries/day, the
evaluation framework, production observability, security and regulatory risk
under CBUAE and PDPL, and an honest account of what was cut.
