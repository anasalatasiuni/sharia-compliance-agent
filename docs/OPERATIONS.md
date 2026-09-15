# Operations

Observability, deployment, tests, and the diagnostic scripts.

## Observability

Structured JSON to stdout, one line per stage, every line carrying `trace_id`.
Span names are OpenTelemetry-shaped (`sharia.retrieve`, `sharia.reason`), so
swapping stdout for a collector is a wiring change.

A full trace, filtered to one `trace_id`:

```jsonc
{"msg":"sharia.intake",     "duration_ms":0,     "ok":true, "redactions":0}
{"msg":"sharia.retrieve",   "duration_ms":1725,  "ok":true, "round":1, "candidates":40, "sparse_terms":11}
{"msg":"sharia.rerank",     "duration_ms":3438,  "ok":true, "backend":"llm", "returned":5, "top_score":0.95}
{"msg":"sharia.refine",     "duration_ms":6292,  "ok":true, "round":2, "tool_calls":0, "finish_reason":"stop"}
{"msg":"sharia.evidence",   "duration_ms":11457, "ok":true, "clauses":5}
{"msg":"sharia.reason",     "duration_ms":6571,  "ok":true, "clauses":5,
                            "finding":"SUPPORTED_NON_COMPLIANT", "confidence":0.95, "citations":3}
{"msg":"sharia.guardrails", "duration_ms":3,     "ok":true, "verdict":"NON_COMPLIANT", "escalations":[]}
{"msg":"assessment.complete","assessment_id":"8dc253f0…", "verdict":"NON_COMPLIANT", "latency_ms":18032}
```

Two things are legible here that matter. The model reported a **finding**
(`SUPPORTED_NON_COMPLIANT`) and the guardrail stage produced the **verdict**
(`NON_COMPLIANT`), so the two vocabularies stay separate all the way into the
logs and it is always visible whether a verdict came from the evidence or from a
gate. And reranking cost 3.4s against 1.7s of retrieval, which is why it is a
configurable backend rather than a fixed part of the pipeline.

### The audit record

One `audit.record` event per assessment, so it inherits retention, access control
and shipping from the normal log pipeline. It is what a Shari'ah reviewer is
actually shown, and it carries everything needed to replay a verdict months
later:

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
They are reconstructible on demand instead, since `index_snapshot` pins the index
and any past retrieval can be replayed:

```bash
python scripts/retrieval_debug.py "<the query>" --expect SS8-3.1.1
```

Pinning `corpus_version` and `prompt_version` is the point. Without them you
cannot tell whether a disputed verdict came from a **retrieval miss** (wrong
clauses reached the model) or a **reasoning miss** (right clauses, wrong
conclusion). Those have different fixes, and the distinction is unrecoverable
after the fact if it was not recorded.

## Testing

```bash
pytest -q          # 70 tests, no network
ruff check .
```

`tests/test_guardrails.py` pins the safety properties: fabricated quotes caught,
citations to unretrieved clauses caught, always-review categories escalating
despite a clean draft, and the one-directional invariant stated directly as a
test. `tests/test_pipeline.py` drives the state machine against fake providers.
`tests/test_api.py` covers the HTTP boundary, including the job-ownership check.
`tests/test_resilience.py` pins the distinction between a fault and a rate limit,
which is the bug that contaminated the first eval run. `tests/test_ingest.py`
asserts the parser recovers every standard the source contains, which is the
check whose absence let six standards go missing.

The eval runs the documentation quotes are committed under `eval/results/`, with
per-case verdicts, retrieved ids, citations and escalation codes, so the numbers
in [PART2-technical-decisions.md](PART2-technical-decisions.md) §3 can be checked
against the runs that produced them.

## Scripts

### `scripts/bootstrap.sh` — a working index without re-embedding

Starts Qdrant and restores `corpus/aaoifi_standards.snapshot.gz`, the prebuilt
index committed with the repo. Idempotent: if the collection already holds points
it leaves them alone.

```bash
./scripts/bootstrap.sh          # ~2s, no API key, no cost
```

The snapshot carries the payload indexes with it (`standard_no`, `standard_name`,
`corpus_version`, `lang`, `is_superseded`), which is what makes metadata filters
run inside the ANN traversal rather than after it.

### `scripts/preflight.py` — verify the provider before spending

Checks embeddings, tool calling, and strict structured output as three
independent requests, so a failure names exactly one thing. The structured-output
check sends the **real** `DraftAssessment` schema rather than a toy one, because
nested models and enums are where gateway strict-mode support tends to differ and
a toy schema would pass while the real one fails.

```bash
python scripts/preflight.py                            # ~$0.005
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

### `scripts/verify_docs.py` — check the docs against reality

Counts, file references, environment variables and eval figures, each checked
against the manifest, the filesystem, the `Settings` model and the saved results.

```bash
python scripts/verify_docs.py      # exits non-zero on any mismatch
```

Documentation drifts silently. A number is right when written and wrong three
commits later, and nobody re-reads a README looking for arithmetic, so the parts
a machine can check now get checked on every change.

### `scripts/clauses.py` — browse the corpus

```bash
python scripts/clauses.py --standard 8 | head          # clauses of SS 8
python scripts/clauses.py --exists SS8-3.1.1           # verify a gold id
```

### `scripts/build_pdf.py` — render the architecture document

```bash
uv pip install -e ".[docs]"
python scripts/build_pdf.py        # exits non-zero above 4 pages
```

## Deployment

The image builds and runs anywhere that takes a Dockerfile. It needs ~150 MB of
RAM and no persistent disk, because the index lives in a managed Qdrant cluster.
The service idles at ~148 MB, and the index is ~5 MB of vectors.

Two constraints worth knowing before choosing a host. The service holds a
long-lived process, since circuit breakers accumulate failures across requests and
the async job store lives in memory, so a serverless target silently degrades both
rather than failing loudly. And an assessment takes 15–30 seconds, so any platform
with a request timeout under ~60s will cut off the synchronous path.

### 1. Index a Qdrant Cloud cluster

Create a free cluster (no card required), then ingest against it rather than
against localhost:

```bash
SCA_QDRANT_URL=https://<cluster>.<region>.cloud.qdrant.io:6333 \
SCA_QDRANT_API_KEY=<key> \
python -m sharia_agent.ingest.cli --recreate
```

### 2. Deploy

[`deploy/render.yaml`](../deploy/render.yaml) is a working Render blueprint
(*New → Blueprint*) that applies the region, plan, health-check path and the
tuning values the eval baseline was measured at. Any other container host works
the same way: build from the Dockerfile and set the environment.

### 3. Use a token that is not in this repo

`.env.example` is committed, so `demo-token-analyst` is public and worthless as an
access control. Generate a real one:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
# SCA_PRINCIPALS=<token>:reviewer@mal.ae:assess
```

### 4. Keep the cluster alive

[`.github/workflows/keepalive.yml`](../.github/workflows/keepalive.yml) pings
`/health` once a day. Set the repository variable `DEPLOY_URL` to enable it.

This is not about cold starts. **Qdrant Cloud suspends a free cluster after a week
idle and deletes it after four**, and the failure is silent: the URL still
resolves, `/health` just starts reporting an empty index. A daily request prevents
that at no cost, and one ping covers both services because `/health` queries the
collection.

### What this deployment is not

Production. `SCA_ENV=prod` only tightens logging. The posture described in
[PART2-technical-decisions.md](PART2-technical-decisions.md) §5, meaning
in-country inference under CBUAE residency rules, OIDC instead of static tokens,
and audit records on WORM storage, is not what runs here.
