# API and configuration

Interactive docs are at `/docs` once the service is running. `/` redirects there.

## POST /assess

```bash
curl -sS -X POST http://localhost:8000/assess \
  -H "Authorization: Bearer demo-token-analyst" \
  -H "Content-Type: application/json" \
  -d '{"query": "Can Mal sell a vehicle to a customer under murabaha before we have purchased it from the dealer?"}'
```

A real response, trimmed only where marked:

```jsonc
{
  "assessment_id": "8dc253f0f0cd43599da0e9aaf7058427",
  "trace_id": "6b9ec77b8b9a45bb82751b59873650b9",
  "verdict": "NON_COMPLIANT",
  "confidence": 0.95,
  "reasoning": "The proposal requests that Mal sell a vehicle to a customer via murabaha before Mal has purchased it from the dealer. The retrieved AAOIFI standards explicitly prohibit this arrangement. […] The sequence mandated by the standards is: (1) Institution contracts with supplier, (2) Institution acquires actual or constructive possession, (3) Institution sells to customer. The proposal reverses steps 1 and 3, which the standards forbid.",
  "citations": [
    {
      "chunk_id": "SS8-3.1.1",
      "quote": "The Institution shall not sell any item in a Murabahah transaction before it acquires such item. Hence, it is not valid for the Institution to conclude a Murabahah sale with the customer before the Institution concludes a purchase contract with the supplier…",
      "supports": "The core prohibition against selling before acquisition and before concluding a supplier contract"
    },
    {
      "chunk_id": "SS8-3.2.1",
      "quote": "It is obligatory that the Institution's actual or constructive possession of the item be ascertained before its sale to the customer on the basis of Murabahah.",
      "supports": "The mandatory sequencing: possession must be established before the customer sale, not after"
    },
    {
      "chunk_id": "SS8-2.3.5",
      "quote": "If the customer then does not purchase the item, the Institution is able to return it to the supplier within the specified period on the basis of the conditional option that is established in Shari'ah…",
      "supports": "Even under sale-or-return structures, the Institution must first acquire before offering to the customer"
    }
  ],
  "concerns": ["ownership_sequence"],
  "escalations": [],
  "missing_information": [],
  "clauses_considered": ["SS8-3.1.1", "SS8-3.2.1", "SS1-2.6.5", "SS8-2.2.3", "SS8-2.3.5"],
  "corpus_version": "aaoifi-en-2017@145a0995711cf7ed",
  "model": "anthropic/claude-haiku-4.5",
  "prompt_version": "v1",
  "latency_ms": 18032,
  "created_at": "2026-09-15T11:58:07.629464Z",
  "disclaimer": "Decision-support output. This is not a fatwa and does not constitute Shari'ah approval. Under CBUAE rules, Shari'ah determinations are reserved to the institution's Internal Shari'ah Supervision Committee."
}
```

That request cost 11,761 input and 1,746 output tokens, $0.021 at Haiku list
price. The third citation is worth noting: nothing asked the model to address
sale-or-return, and it went looking for the nearest thing to a counter-argument
rather than stopping at the two clauses that made its case.

## POST /assess?mode=async

A hard question with three retrieval rounds can outrun a client timeout, and the
async path is the shape this service takes at volume.

```bash
curl -sS -X POST "http://localhost:8000/assess?mode=async" \
  -H "Authorization: Bearer demo-token-analyst" \
  -H "Content-Type: application/json" \
  -d '{"query": "Is a diminishing musharaka home finance structure with a binding purchase undertaking acceptable?"}'
# → 202 {"job_id": "a1b2…", "status": "queued", "poll": "/assess/a1b2…"}

curl -sS http://localhost:8000/assess/a1b2… \
  -H "Authorization: Bearer demo-token-analyst"
```

Jobs are readable only by the principal that created them, so a job id is not a
capability anyone can guess.

## GET /health

`/health` is **readiness**, not liveness. A process that is up but whose
collection is empty answers every question with `NEEDS_REVIEW` and looks healthy
doing it, so the check requires the index to have content.

```bash
curl -sS http://localhost:8000/health | jq
curl -sS http://localhost:8000/health/live     # pure liveness
```

It returns `503` when the index is empty, a credential is missing, or a circuit
breaker is open. A `503` carrying `manifest_mismatch` means the service is pointed
at a different index than the one it was built against: re-run the ingest, or fix
`SCA_QDRANT_URL`.

## Configuration

Everything tunable is an environment variable, so a deployment is reproducible
from its environment alone. Full list in [`.env.example`](../.env.example).

| Variable | Default | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | — | The only credential required |
| `SCA_MODEL` | `anthropic/claude-haiku-4.5` | Reasoning model, and the one the eval was measured on |
| `SCA_EMBED_MODEL` | `baai/bge-m3` | Multilingual, real Arabic coverage |
| `SCA_EMBED_DIM` | `0` | `0` = detect from the provider at ingest |
| `SCA_RERANK_BACKEND` | `llm` | `llm` \| `local` \| `none`, see below |
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
| `llm` | ~$0.01/query | Clause text leaves | Demo default, no extra setup |
| `local` | free | **None** | Production. `bge-reranker-v2-m3`, Apache-2.0, ~560 MB on CPU. `pip install -e ".[local-rerank]"` |
| `none` | free | None | Fusion order only; guardrails escalate for unverified ranking |

`jina-reranker-v2-multilingual` scores better and is CC-BY-NC-4.0, so a bank
cannot use it commercially.
