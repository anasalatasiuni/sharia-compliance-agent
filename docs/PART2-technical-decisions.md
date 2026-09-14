# Technical decisions and the path to production

Shari'ah Compliance Agent — Mal internal compliance tooling.

This covers why the system is built the way it is, what was explicitly rejected,
and what would have to change to run it at 50,000 assessments a day inside a
CBUAE-licensed institution.

**Contents**
1. [Architecture decisions](#1-architecture-decisions)
2. [Scaling to production](#2-scaling-to-production)
3. [AI evaluation and quality](#3-ai-evaluation-and-quality)
4. [Observability and debugging](#4-observability-and-debugging)
5. [Security and regulatory compliance](#5-security-and-regulatory-compliance)
6. [What was cut](#6-what-was-cut)

---

## 1. Architecture decisions

### 1.0 The decision everything else follows from

The brief asks for an agent. The strongest thing I can say about this design is
where I refused to put the agency:

> **The model chooses what evidence to gather. It has no say in how the verdict
> is decided.**

It can reformulate a question into the vocabulary the standards use, and it can
search again. It cannot alter the control flow, and `DraftAssessment` has no
`verdict` field for it to fill in. The two stages that decide a verdict — Gate A
and Gate B — call no model.

This was not a hedge against model quality. It follows from what the artefact is
for. A compliance verdict has to survive being questioned months later by someone
who was not in the room, and an unbounded ReAct loop takes a different trajectory
on every run. Reproducibility is a requirement here in a way it is not for a
support chatbot.

Two design consequences worth stating separately:

**The vocabularies are split.** The model returns a `finding` about *evidence*
(`SUPPORTED_COMPLIANT`, `INSUFFICIENT_BASIS`, `CONFLICTING_SOURCES`). Code maps
that onto the returned `verdict`. This makes the safety property structural
rather than conventional — there is no code path in which a guardrail produces
`COMPLIANT`.

**Guardrails are one-directional.** They may escalate toward `NEEDS_REVIEW` and
may never relax. A model regression therefore makes the system noisier, never
more permissive. That asymmetry is the whole point, because the error costs are
asymmetric: a wrong `NEEDS_REVIEW` costs a reviewer five minutes, a wrong
`COMPLIANT` costs a mispriced product and a regulatory finding.

**`NEEDS_REVIEW` is the product, not the failure mode.** Under CBUAE rules every
UAE Islamic financial institution must maintain an Internal Shari'ah Supervision
Committee, and Shari'ah determinations are reserved to it. A machine verdict is
not a fatwa and cannot be. So the system's actual job is to triage and to prepare
a defensible, cited memo for that committee — which means escalation is a
successful outcome, and the metric that matters is not "how often did it avoid
escalating" but "how often was a non-escalated verdict wrong".

### 1.1 Vector store — Qdrant

**Chosen because it applies payload filters during ANN traversal rather than
after it.**

This matters more here than for a general knowledge base. A query about *ijara*
must never be answered from *murabaha* clauses, and across contract families the
language is close to identical — profit distribution under Mudarabah and under
Wakala read alike and mean different things. The category has to be a hard filter,
not a hope about cosine distance.

| Rejected | Why |
|---|---|
| **Chroma** | Filters *after* the ANN search and over-fetches to compensate. Recall degrades silently, precisely when the filter is doing useful work. Fine to prototype on, wrong to build a compliance control on. |
| **pgvector** | Reasonable under a few million vectors and attractive if you are already on Postgres. But hybrid dense+lexical has to be assembled by hand, and the fusion logic then becomes ours to maintain. |
| **Pinecone** | Managed and capable, but proprietary, and data residency is a live constraint here (§5). Self-hostability is not optional. |

Qdrant also runs the same engine locally in Docker and in Qdrant Cloud, so
retrieval behaves identically in dev and prod. Payload indexes are created
explicitly at ingest — without them Qdrant cannot use a filter during traversal
and quietly falls back to a slower path.

### 1.2 Embedding model — `baai/bge-m3`

**Chosen for genuine cross-lingual retrieval, because Mal's users and Mal's
corpus will not always be in the same language.**

An English-optimised embedder does not fail loudly on an Arabic query; it returns
plausible, wrong clauses. BGE-M3 is trained for retrieval across 100+ languages
with real Arabic coverage.

| Rejected | Why |
|---|---|
| **Cohere `embed-v4`** | Strong Arabic and was the original choice. Dropped when the provider consolidated onto one gateway — a second vendor, a second key, a second DPA, and a second thing to move in-country later. |
| **`text-embedding-3-large`** | Strongest English scores, materially weaker cross-lingual. Wrong optimisation target for a UAE bank. |
| **Self-hosted BGE-M3** | What I wanted originally, and the right production answer. Ruled out for the demo by ~2.5 GB resident memory; free-tier compute is 512 MB. Reachable via API instead, and the seam to self-host later is intact. |

The vector dimension is **detected from the provider at ingest** and written to
the manifest, not hardcoded. Hardcoding it is a well-worn way to build a
collection whose vectors are the wrong width for the model that later queries it.

### 1.3 Retrieval — hybrid, then cross-encoder rerank

Two stages, because they fix different failures.

**Hybrid (dense + BM25, fused with RRF)** decides what gets considered. Dense
retrieval catches paraphrase; lexical retrieval catches the tokens that decide
questions in a standards corpus — clause numbers (`2/2/2`), transliteration
variants (murabaha / muraabaha / murābaḥa), and negation-bearing legal phrases.
Dense-only retrieval fumbles all three.

Fusion is **Reciprocal Rank Fusion rather than a weighted score blend**: cosine
similarity and BM25 sit on incomparable scales, and any fixed weighting between
them is a constant someone has to re-tune whenever the corpus shifts. RRF consumes
only rank, so it has nothing to drift.

The sparse side ships **term frequencies and lets Qdrant apply IDF** across the
collection. The client therefore holds no corpus statistics that could fall out of
sync with the index. Token ids use FNV-1a rather than Python's `hash()`, which is
salted per process and would make every previously indexed sparse vector
unreadable after a restart.

**Reranking** decides what reaches the model. A cross-encoder scores query and
clause *jointly*, which is what separates clauses that are topically identical but
differ on the single condition that settles the question — the normal case here.
It is simultaneously the accuracy win and the dominant cost lever (§2.2).

OpenRouter has no rerank endpoint, so the backend is pluggable: `local`
(`bge-reranker-v2-m3`, Apache-2.0, no data egress — the production answer), `llm`
(a small model scores candidates; the demo default because it needs no extra
setup), or `none` (fusion order, which the guardrails treat as unverified ranking
and escalate on). `jina-reranker-v2-multilingual` benchmarks better than BGE but
is **CC-BY-NC-4.0** and therefore unusable commercially at a bank — a licensing
constraint that is easy to miss when picking models on benchmark scores.

### 1.4 Chunking — the clause is the chunk

**The chunk boundary is the citation unit.**

AAOIFI text is hierarchically numbered (`2/2/2` under `2/2` under `2`), so this
alignment is available for free, and it is worth taking: a retrieved chunk *is* a
reference a compliance officer can verify by hand. That makes the system's output
auditable against the source rather than merely plausible.

| Rejected | Why |
|---|---|
| **Fixed-size windows** | Straddle `2/2/2` and `2/2/3`. A citation could then only point at a page, which destroys the property above. |
| **Sentence-level** | Too granular. "This ratio applies to all Wakala deposits" means nothing alone. |
| **Whole document** | One embedding for 60 pages averages everything and is specific to nothing. |
| **Generic recursive splitter** | The usual right answer, and what I would use on unstructured text. Here the document already carries a better boundary than any splitter would infer. |

Each chunk carries its **heading path**, which is what lets the retriever tell
near-identical clauses apart across standards. Clauses over 1,800 characters split
on sentence boundaries with overlap, keeping the parent clause path so citations
still resolve.

The parser handles OCR reality explicitly — soft hyphens at line breaks, running
page headers, bare page numbers, and a table of contents whose entries match the
clause-opening pattern exactly. Standard titles resolve by **frequency voting**: a
title appears as a running header dozens of times, while a cross-reference inside
another clause (`...Standard No. (8) on Murabahah and item 2/2/4 of...`) matches
the same regex once.

### 1.5 Agent framework — none

**Plain Python, ~200 lines of explicit state machine.**

The brief rules out black-box agent abstractions, but I would have made this call
anyway. A framework's value is abstraction over many backends; here that value is
negative, because the control flow is the thing a reviewer needs to read. LangChain's
`AgentExecutor` or LlamaIndex's agents would hide exactly what has to be legible,
and add a dependency surface to audit for no compensating benefit.

Not using a framework also made the bounded loop natural to express: a hard round
cap, repeat-query detection, and an unconditional seed search so retrieval never
depends on the model choosing to call a tool.

### 1.6 Model access — one OpenAI-compatible gateway

Reasoning is `anthropic/claude-opus-5` via OpenRouter. Correctness is the product
here, so the reasoning step uses the strongest available model and the cost
question is answered by retrieving less (§2.2), not by reasoning worse.

Structured output is enforced by the API (`json_schema`, strict), then re-validated
through Pydantic — the wire schema has to drop the value constraints strict mode
rejects (`ge`, `max_length`), so validating the parsed object restores them. A
failure at that point is a real signal and escalates, rather than being a
formatting accident to retry.

Going through one **OpenAI-compatible seam** is a deliberate hedge against §5:
when residency rules force an in-country host, that is a base URL and a model
string, not a rewrite.

---

## 2. Scaling to production

Target: **50,000 assessments/day** within 12 months.

### 2.1 What does not break

Worth saying first, because the intuitive answer is wrong. 50,000/day is ~0.6
requests/second average. Even at a 10× peak that is 6 rps. Qdrant holds 345
vectors today and would hold the full 61-standard corpus plus Mal's internal
policy in perhaps 50,000 — still trivially small; Qdrant serves millions on a
single node. **Retrieval infrastructure is not the constraint and will not become
one.**

### 2.2 What breaks first: escalation capacity

Escalation is the product (§1.0), which means every escalated assessment
consumes a human. That arithmetic is unforgiving at this volume:

| Escalation rate | Reviews/day | Full-time reviewers* |
|---:|---:|---:|
| 10% | 5,000 | ~179 |
| 25% | 12,500 | ~446 |
| 35% | 17,500 | ~625 |

\* 15 minutes per review, 7 productive hours per day.

**No plausible staffing absorbs this.** Mal's ISSC is a committee, not a call
centre. So the binding constraint on the product at 50,000/day is not
infrastructure and not model spend — it is **how often the system declines to
answer**, and that has to land in the low single digits for the target volume to
mean anything.

Three things follow, and they reorder the roadmap:

1. **Deduplication is not an optimisation, it is load-bearing.** If 50,000 daily
   queries contain 3,000 genuinely novel questions, a semantic cache (§2.3) turns
   an impossible review queue into a feasible one. The 50k figure should be
   interrogated before it is engineered against: it is almost certainly mostly
   repeats.
2. **Escalations need tiering.** Not every `NEEDS_REVIEW` needs a Shari'ah
   scholar. `weak_retrieval` is a corpus-coverage problem for a curator;
   `always_review_category` is a routing decision; only genuine
   `insufficient_basis` and `conflicting_sources` need scholarly judgement. The
   escalation codes already carry this distinction — nothing currently uses it.
3. **Escalation rate becomes a first-class SLO**, measured and alerted on, not an
   emergent property nobody owns.

### 2.2.1 What this means for choosing a model

The instinct to run a smaller model is a good one — the whole architecture was
built to reduce dependence on model strength, and reaching for a frontier model
can be a way of papering over retrieval that should have been fixed instead. But
the economics do not turn on the per-token price:

| | Per day at 50,000 assessments |
|---|---|
| Saving from Haiku 4.5 over Opus 5 | **$1,900** |
| Cost of escalating 1pp more, at $20/review | **$10,000** |
| Cost of escalating 5pp more | **$50,000** |

**Break-even is a 0.19 percentage-point increase in escalation rate.** A cheaper
model that is even slightly less able to reach a defensible conclusion on thin
evidence is dramatically more expensive, and the model line item is small enough
to be noise against the human one.

So the model decision is not "which is cheaper per call" but:

> **Which model escalates least, at a false-`COMPLIANT` rate of zero?**

That is measurable, and measuring it costs about $2 (§3). Two specific things to
watch in that comparison, because they are where a smaller model would fail
without it being obvious:

- **Calibration.** The `min_model_confidence` guardrail assumes self-reported
  confidence tracks accuracy. A model that is less capable *and* overconfident
  does not escalate more — it escalates *less*, and the extra answers are wrong.
  That inverts the whole analysis above and is a silent failure.
- **Conditional structure.** Shari'ah clauses are dense with provisos — "it is
  permissible... provided that... except where...". Missing a proviso yields a
  confident, well-cited, wrong `COMPLIANT`. Citation checks do not catch it,
  because the citation is real.

Until that comparison is run, the default stays on the stronger model — not
because it is known to be better here, but because the failure it guards against
is the expensive, silent one. This is a decision awaiting evidence, not a
conclusion.

### 2.3 What breaks second: model spend

At current settings, per assessment:

| Call | Model | Tokens (in/out) | Cost |
|---|---|---|---|
| Reasoning | `claude-opus-5` | 4,250 / 600 | $0.0363 |
| Refinement (~60% of requests) | `claude-opus-5` | 3,500 / 80 | $0.0117 |
| Rerank (~1.5 rounds) | `claude-haiku-4.5` | 7,000 / 400 | $0.0135 |
| Embeddings | `bge-m3` | — | ~$0.0005 |
| **Total** | | | **≈ $0.062** |

**50,000/day ≈ $3,100/day ≈ $93,000/month ≈ $1.1M/year.**

That is a real number and worth attacking, but note the ordering: it is roughly
a fifth of what a 5-percentage-point swing in escalation rate costs (§2.2). Model
spend is the second constraint, not the first.

The useful detail is *where* the tokens are. Of the 4,250 input tokens on the
reasoning call, **~3,200 are retrieved clauses** — which vary per query and
therefore **cannot be cached**. Only the ~850-token system prompt is cacheable.
So the obvious lever is nearly worthless here, and the real lever is retrieving
fewer, better clauses.

Ranked by saving, quality-neutral first:

| Lever | Saving/day | Notes |
|---|---|---|
| **Semantic cache** on normalised query, keyed to `index_snapshot` | $930–1,550 | Compliance questions repeat heavily; the same product question arrives from five people. A 30–50% hit rate is realistic. Keying on the snapshot means a re-index invalidates it for free. |
| **Local reranker** (`SCA_RERANK_BACKEND=local`) | $675 | Removes the entire rerank line item *and* stops clause text leaving the jurisdiction. Free win twice over. |
| **Tighter context** — 8 clauses → 5 | $300 | Also usually *improves* answers: noise degrades reasoning even when the right clause is present. |
| **Prompt caching** on the system prompt | $190 | Modest, because only 20% of input is cacheable. Already implemented. |
| **Batch API** (`:batch`, −50%) for eval and backfill | — | Not the live path, but makes running evals on every PR cheap. |

Free wins alone take **$93k → ~$40k/month**.

Beyond that the levers cost quality and need evidence: `claude-sonnet-5` for
reasoning would save a further ~$1,090/day, but that is a decision the eval
framework in §3 has to license, not one to make on price. A cheaper model that
escalates more is not cheaper — escalation consumes a scholar's time, which costs
more per hour than the model does.

### 2.4 What breaks third: the request path

Each assessment holds a slot for 5–15 seconds. The current concurrency limit is a
semaphore of 8 in a single process, so sustained throughput saturates near 1 rps —
under the 10× peak.

- **Horizontal scale** behind a load balancer. The service is stateless apart from
  the job store, which is the blocker (§2.4).
- **Async becomes the default**, not an option. The sync path stays for fast
  queries with a short timeout.
- Backpressure already sheds with `429` + `Retry-After` rather than queueing
  invisibly until every client times out at once. That behaviour is reasoned, not
  load-tested (§6).
- **Gateway rate limits** become a real constraint at 6 rps of Opus traffic and
  need to be negotiated ahead of the ramp, not discovered during it.

### 2.5 What breaks fourth: state

The job store is an in-process dict. It dies with the process, is invisible to a
second replica, and loses in-flight work on every deploy.

Replace with Redis or a proper queue (SQS/Celery/Arq). The interface in `jobs.py`
is deliberately narrow so this is a single-file change.

### 2.6 What breaks fifth: the data pipeline

Today ingestion is a manual CLI run against a file someone downloaded. AAOIFI
issues revisions; CBUAE issues circulars; Mal's own product policy changes weekly.
Manual re-ingestion guarantees staleness, and stale Shari'ah guidance is worse
than no guidance because it carries the same confidence.

Production needs:

- **Scheduled ingestion** with change detection on source checksums.
- **Versioned, additive indexing.** Never mutate in place; write a new
  `index_snapshot` and cut over atomically, so rollback is a pointer change.
- **Supersession as first-class data.** The schema and guardrail exist;
  nothing populates them (§6). Until they are populated, a repealed clause is
  cited as though live.
- **A curator loop.** Escalations tagged "the corpus does not cover this" are a
  free, precise backlog for whoever owns corpus coverage. That signal should be
  routed, not discarded.

---

## 3. AI evaluation and quality

Everything in this section is specified and **not yet run** (§6). The tuning
constants currently in the code — `top_k=5`, `min_rerank_score=0.35`,
`min_model_confidence=0.70` — are reasoned guesses, and the purpose of this
framework is to replace them with measurements.

### 3.1 Measure retrieval separately from reasoning

This is the single most important structural choice in evaluating a RAG system,
because the two failures have different fixes and are indistinguishable from the
output alone.

**Retrieval eval** needs no model, costs nothing, and runs on every corpus
re-index:

| Metric | Reads as |
|---|---|
| **recall@k** (k = 1, 3, 5, 10) | Can generation succeed at all? It cannot recover from a clause that was never retrieved. |
| **MRR / nDCG@5** | Is the governing clause near the top? Models attend unevenly across long context. |
| **context precision** | How much noise rides along. Noise degrades answers even when the right clause is present. |

Diagnostics: a large gap between recall@10 and recall@3 is a **ranking** problem
(fix the reranker). Low recall@10 is a **candidate generation** problem (fix
hybrid weighting, chunking, or accept a corpus gap). Those get different work.

### 3.2 Verdict quality, with an asymmetric headline metric

The headline metric is **false-`COMPLIANT` rate**, not accuracy.

Accuracy treats all three errors alike, and they are not alike. A wrong
`NEEDS_REVIEW` costs a reviewer minutes. A wrong `NON_COMPLIANT` costs a product
delay and an argument. A wrong `COMPLIANT` puts a non-compliant product in market
and is the failure the ISSC exists to prevent. Optimising the single number that
averages them is how a system gets quietly more permissive.

Alongside it:

- **Full confusion matrix** over the three verdicts.
- **Escalation rate and escalation mix by code.** The mix is the diagnostic: if
  `weak_retrieval` dominates, retrieval is the problem; if `low_confidence`
  dominates, the threshold is miscalibrated; if `always_review_category`
  dominates, the product is being pointed at questions it was not built for.
- **Citation faithfulness**, in two tiers — the quote-is-verbatim check is
  deterministic and already enforced in-line at zero cost; whether the cited
  clause actually *supports* the claim needs an LLM judge, calibrated against
  human labels before its scores are trusted.
- **Calibration.** A reliability curve of stated confidence against observed
  accuracy. `min_model_confidence` is only meaningful if the confidence is
  calibrated, and it currently is not known to be.
- **Escalation rate at a fixed false-`COMPLIANT` rate.** The metric that decides
  model selection (§2.2.1). Comparing models on accuracy alone hides the thing
  that actually costs money: how often each one declines to answer. Run the suite
  against both `claude-opus-5` and `claude-haiku-4.5` and report both columns —
  a ~$2 experiment that settles a $1,900/day question in the right direction.

### 3.3 Three test sets, kept separate

| Set | Size | Built from | Weakness |
|---|---|---|---|
| **Clause-derived** | ~150 | For each indexed clause, a compliant and a violating scenario | Lexically too close to the source. **Will flatter the retriever.** Hand-validate a sample; never report it alone. |
| **Adversarial** | ~50 | Hand-written near-miss pairs differing on one decisive condition — murabaha where the bank does own the asset vs. does not; a unilateral vs. bilateral wa'd; a mudarabah with and without a capital guarantee | Expensive to write. This is where the system actually fails, so it is worth the time. |
| **Production-sampled** | grows | Real queries with ISSC determinations attached | Only set reflecting the real distribution. Starts empty; becomes the most valuable one. |

Split train / validation / test, and report only the test split. Cost of a full
200-case run is ~$12, or ~$6 on the batch endpoint — cheap enough to gate every PR.

### 3.4 Human-in-the-loop: the escalation queue is the labelling pipeline

The highest-leverage observation available: **every `NEEDS_REVIEW` already goes to
a human who makes a determination.** Capturing that determination as a label costs
one extra field in the review UI and produces perfectly-distributed training and
eval data for free. A system that escalates 40% of queries and throws away 40%
worth of expert labels is wasting its best asset.

Checkpoints:

| Checkpoint | Coverage | Purpose |
|---|---|---|
| Every `NEEDS_REVIEW` | 100% (they go to a human anyway) | Free labels |
| Blind audit of `COMPLIANT` | 5% sample | **The only way to measure false-`COMPLIANT` in production.** Non-negotiable. |
| Review of `NON_COMPLIANT` | 100% initially, then sample | Low volume, high stakes, catches over-restriction |
| Weekly triage | Failures → adversarial set | A wrong verdict that does not become a test case will recur |

### 3.5 Gating

- No prompt, model, corpus or threshold change ships without the eval passing on
  the held-out split.
- **Hard gate: false-`COMPLIANT` rate must not increase.** Other metrics trade;
  this one does not.
- Retrieval eval runs on every re-index — it is free and catches corpus
  regressions before they reach a model.
- Change one variable at a time. Stacked changes make attribution impossible and
  waste the eval.

---

## 4. Observability and debugging

### 4.1 What exists

Structured JSON to stdout, one event per stage, every line carrying `trace_id`
and principal. Span names are already OpenTelemetry-shaped (`sharia.retrieve`,
`sharia.reason`), and `obs/trace.py` is the only module that knows where logs go.

Per assessment, one `audit.record` event carrying corpus version, index snapshot,
model, effort, prompt version, every retrieval round with its query and rationale,
every retrieved clause with dense/sparse/fused/rerank scores, the model's draft,
the final verdict, which guardrails fired, per-stage latency, and token usage.

### 4.2 Production stack

- **OpenTelemetry SDK → collector**, fanning out to traces (Tempo/Jaeger), metrics
  (Prometheus), logs (Loki). The span shape already fits.
- **An LLM-specific layer** — Langfuse or Braintrust — for prompt/response
  inspection and eval runs. Generic APM shows a slow span; it does not show which
  clause the model ignored.
- **Audit records to append-only, WORM-retained storage**, separate from
  operational logs. These have a regulatory retention obligation (typically 5
  years in UAE banking) and a different access-control profile from debug logs.

### 4.3 Metrics that actually signal

| Metric | Alert on |
|---|---|
| Verdict distribution | **`NEEDS_REVIEW` rate dropping >20% week-over-week.** Reads like an improvement; is usually a weakened guardrail or a changed threshold. This is the most important alarm in the system. |
| Escalation mix by code | Any single code's share shifting sharply |
| Top-1 rerank score, p50 and p10 | Downward drift = corpus or embedding drift |
| `unresolved_citation` rate | Rising = model regression, or the gateway silently rerouting to a different model |
| Cost per assessment | >30% rise |
| Per-stage latency p95/p99 | Budget breach |
| Breaker state per dependency | Any open breaker |

### 4.4 Debugging a wrong verdict — the runbook

The user reports assessment `3f2a…` came back `COMPLIANT` and the ISSC disagrees.

**1. Pull the audit record** by `assessment_id`. It pins `corpus_version`,
`index_snapshot`, `model`, `model_effort` and `prompt_version`. If the corpus has
been re-indexed since, you know immediately, because the snapshot differs from
current.

**2. Split the failure.** This is the whole reason retrieval is logged separately,
and it is the first question to ask, not the last:

> Was the governing clause in `retrieved[]`?

**3a. No — retrieval miss.** Replay retrieval against the same `index_snapshot`.
Per-arm scores are recorded, so the sub-diagnosis is direct:

- Retrieved by neither dense nor sparse → **candidate generation**. Fix hybrid
  weighting or chunking. Or the clause genuinely is not in the corpus, which is a
  content gap and belongs to whoever curates the corpus, not to engineering.
- Retrieved but ranked out → **ranking**. Fix the reranker or raise `top_k`.
- Check `retrieval_rounds[]` — the model records *why* it issued each search. A
  reformulation into the wrong vocabulary is visible there and is a prompt fix.

**3b. Yes — reasoning miss.** The right clauses reached the model and it drew the
wrong conclusion. Citations were already verified verbatim, so this is not
fabrication; the model misread real text. Reconstruct the exact prompt from
`retrieved[]` plus `prompt_version` and re-run. Fixes live in the prompt, the
model, or `effort` — and any of those is a change that must clear §3's gate.

**4. Ask why no guardrail caught it.** A wrong `COMPLIANT` that passed every gate
is a gap in the gates, not only in the model. Should this query have matched an
`always_review_category`? Was confidence high but miscalibrated?

**5. Add the case to the adversarial eval set.** Non-optional. A wrong verdict
that does not become a regression test will recur.

None of this works without `corpus_version` and `prompt_version` in the record.
They are two strings, and they are the difference between diagnosing a failure and
guessing at it.

---

## 5. Security and regulatory compliance

### Risk 1 — Cross-border transfer of consumer data (highest severity)

**The exposure.** CBUAE's Consumer Protection Regulations require licensed
financial institutions to store and process consumer and transaction data inside
the UAE. PDPL (Federal Decree-Law 45/2021) separately restricts cross-border
transfer to adequacy-listed jurisdictions or transfers carrying specific
safeguards. This deployment sends query text to a US-hosted gateway which routes
to a US-hosted model.

**Current mitigation.** PII redaction runs on the **request path, before egress** —
Emirates ID, IBAN, card and account numbers, phone, email, passport. Deliberately
not on the logging path: scrubbing logs while sending raw text upstream protects
the wrong artefact. Any request containing identifiers is additionally forced to
`NEEDS_REVIEW`, on the reasoning that a compliance question should concern a
product structure rather than an individual.

**Why that is not sufficient.** Redaction is regex-based and therefore
pattern-bound. "The largest depositor at our Sheikh Zayed Road branch" identifies
a person and matches nothing. And a product proposal can be commercially
confidential with no personal data in it at all — residency obligations do not
only attach to PII.

**Production.** Move inference in-country. CBUAE launched a sovereign financial
cloud with Core42 in February 2026; Azure UAE North and AWS `me-central-1` are
alternatives. Embeddings and reranking can go fully local today —
`SCA_RERANK_BACKEND=local` already exists and BGE-M3 self-hosts — which removes
two of the three egress points without touching the architecture. For the LLM,
an in-country endpoint or an on-prem model, reached through the same
OpenAI-compatible seam. Plus the paperwork that makes it lawful rather than merely
technical: a DPA with each processor, a documented lawful basis, and a record of
processing.

### Risk 2 — Prompt injection through the corpus

**The exposure.** The corpus is AAOIFI text and is trusted today. That stops being
true the moment Mal indexes its own term sheets, customer-submitted documents, or
external counsel memos — which is the obvious next step for this system. Retrieved
text then becomes attacker-influenced, and an instruction embedded in a term sheet
could aim to flip a verdict on the product that term sheet describes.

**Current mitigation.** The system prompt states that retrieved text is reference
material and never instruction, and to report anything that tries to redirect it.
Output is schema-constrained, so the model cannot emit free-form actions. Citations
must resolve to clauses retrieved this run, and quotes must appear verbatim — so a
wholly fabricated justification fails the gate without a second model call.

**Honest assessment.** That is instruction plus output constraint, which is the
weakest tier of defence. **Structural separation is not implemented.**

**Production.** Retrieved content in a structurally distinct channel, never
concatenated into the instruction block. An ingestion-time scanner flagging
imperative or meta-referential language in newly ingested documents. Per-chunk
provenance with a trust tier, so AAOIFI text and a customer PDF are not treated
alike. And the control that actually holds: **a verdict causes no action.** Nothing
downstream executes on `COMPLIANT`; it routes to a human queue. An injection that
flips a verdict wins a wrong line in a memo a scholar then reads, not a product
launch.

### Risk 3 — Authorisation, audit integrity, and log leakage

**The exposure.** Three distinct problems:

- Auth is static bearer tokens from an environment variable.
- Any principal holding `assess` can query anything. Fine now; wrong once
  assessments reference named internal products and business units.
- `SCA_LOG_FULL_PROMPTS` defaults to `true`, writing query text into operational
  logs — which have a wider access profile than audit records should.

**Current mitigation.** Authorisation is checked **before the pipeline runs**, not
inside it — by the time a tool call exists, the decision to permit retrieval and
spend must already be made, because the model is untrusted input. Jobs are readable
only by the principal that created them, so a job id is not a capability. Log
fields matching credential names are redacted. `assert_production_safe()` inspects
these settings at startup.

**Production.** OIDC against the bank's IdP, with group claims mapped onto the
existing `scopes` shape — nothing downstream of `deps.py` changes. Per-assessment
ACLs tied to the requesting business unit. Audit records to append-only WORM
storage on the regulatory retention clock. And `assert_production_safe()` should
**refuse to boot** rather than log a warning, which is what it does today.

### Risk 4 — Automation bias: the verdict treated as sign-off

**The exposure.** The realistic failure is not a jailbreak. It is a product manager
screenshotting a `COMPLIANT` verdict and treating it as Shari'ah approval. That is
a human-factors risk and it is more likely than any technical attack here.

**Mitigation.** A disclaimer on every response stating that determinations are
reserved to the ISSC. `always_review_category` catches requests phrased as seeking
approval ("please approve this product"). And a product rule that belongs in the UI
as much as the API: **`COMPLIANT` is never the end of a workflow** — it feeds an
ISSC queue, not a ship button, and it should never render without its citations
beside it.

### Risk 5 — Secrets and supply chain

One key with broad account access, no rotation, and `SCA_QDRANT_API_KEY` unset by
default so the vector store is unauthenticated in dev. Production: a secrets
manager with short-lived credentials, per-environment keys, spend caps per
principal (nothing currently stops a script from running up the bill), and network
policy so Qdrant is not reachable outside the application.

---

## 6. What was cut

Ordered by what it would actually cost in production.

**1. No measured eval numbers.** The framework in §3 is specified and has not been
run. *Cost:* every tuning constant in the system — `top_k=5`,
`min_rerank_score=0.35`, `min_model_confidence=0.70` — is currently an informed
guess with no evidence behind it. There is no baseline, so there is no way to know
whether a change helps, and no licence for the cost reductions in §2.2 that trade
quality. This is the largest gap and the first thing I would close.

**2. No supersession data.** `superseded_by`, the guardrail that reads it, and the
retrieval filter that excludes it all exist. Nothing populates them, because the
2017 edition was ingested as a flat snapshot. *Cost:* a repealed clause is cited as
though it were live, with full confidence and a correct-looking citation. This is
the highest-severity correctness defect in the system, and it is invisible — the
output looks exactly like a right answer.

**3. Reranking defaults to an LLM.** *Cost:* ~$675/day at target volume, and clause
text crossing the jurisdiction on every query. The `local` backend is implemented
and one environment variable away; it is not the default only because the demo
should run on a single key.

**4. In-process job store.** *Cost:* no horizontal scaling, and in-flight
assessments lost on every deploy. Blocks §2.3 entirely.

**5. Static token auth.** *Cost:* not deployable inside a bank as-is. Cheap to fix —
the `scopes` shape is already right — but it is a hard blocker, not a nice-to-have.

**6. No semantic cache.** *Cost:* $930–1,550/day, the single largest saving left on
the table. Not built because without §1's eval there is no safe way to decide what
counts as "the same question".

**7. Structural prompt-injection separation.** *Cost:* acceptable while the corpus
is AAOIFI-only. Unacceptable on the day Mal indexes its own or customer documents,
which is the obvious next feature.

**8. Single corpus edition, manual ingestion.** Eight of 61 standards, English only,
no CBUAE circulars, no HSA resolutions, no Mal product policy. *Cost:* the useful
question surface is narrow, and staleness is guaranteed rather than merely possible.

**9. No Arabic evaluation.** The embedder and the sparse tokenizer both handle
Arabic — the tokenizer folds alef and ta-marbuta variants — but the indexed corpus
is the English edition and no Arabic query has been measured. *Cost:* unknown
quality on a significant share of real queries at a UAE bank, which is worse than
known-bad quality.

**10. No load test.** The concurrency limit of 8 and the `429` shedding behaviour
are reasoned from the latency profile, not measured. *Cost:* the backpressure
threshold is a guess, and backpressure that is wrong under real load fails exactly
when it matters.

**11. No per-principal spend controls.** *Cost:* one script can run up an unbounded
bill against a $0.062/request endpoint.

---

## Appendix — where the numbers come from

Token estimates in §2.2 assume ~850-token system prompt, 8 clauses at ~400 tokens,
~600-token reasoning output, refinement on ~60% of requests, and reranking over 40
candidates truncated to 700 characters each. Prices are OpenRouter list at the time
of writing: `claude-opus-5` $5/$25 per M, `claude-haiku-4.5` $1/$5 per M. These are
estimates for sizing decisions, not a forecast — the point is the *ratio* between
line items, which is what determines which lever to pull first.
