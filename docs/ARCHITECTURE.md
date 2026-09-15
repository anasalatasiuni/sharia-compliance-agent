# Architecture

How the pipeline is put together, and why the corpus is chunked the way it is.
The decisions behind these choices, and what was ruled out, are in
[PART2-technical-decisions.md](PART2-technical-decisions.md).

## Two vocabularies, deliberately separated

The model returns a **finding** about evidence. Deterministic code maps that onto
the **verdict** the service returns. No code path lets a guardrail produce
`COMPLIANT`.

```
  model may return              code decides
  ──────────────────            ─────────────
  SUPPORTED_COMPLIANT      ──▶  COMPLIANT      (only if every gate passes)
  SUPPORTED_NON_COMPLIANT  ──▶  NON_COMPLIANT  (only if every gate passes)
  INSUFFICIENT_BASIS       ──▶  NEEDS_REVIEW
  CONFLICTING_SOURCES      ──▶  NEEDS_REVIEW
                           ──▶  NEEDS_REVIEW   ← any guardrail firing
```

Both types live in `models.py`. `DraftAssessment`, the model's output, has no
`verdict` field to fill in.

## Escalation triggers

Code produces `NEEDS_REVIEW`, and records why as a machine-readable code, so the
escalation mix can be tracked as a product metric rather than inferred.

| Code | Meaning |
|---|---|
| `weak_retrieval` | Top reranked clause below the score floor, or ranking unverified |
| `no_citations` / `unresolved_citation` | A claim with no evidence, a citation to a clause that was never retrieved, or a quote that is not verbatim in the clause it names |
| `low_confidence` | Model's own confidence below threshold |
| `insufficient_basis` / `conflicting_sources` | The model's read of the evidence |
| `superseded_standard` | A retrieved clause has been superseded |
| `always_review_category` | Reserved to the ISSC regardless of confidence |
| `schema_validation_failed` / `upstream_error` | The model or a dependency failed |

`always_review_category` covers matters where a machine assessment is not the
appropriate artefact however confident it is: novel structures, capital or profit
guarantees on profit-sharing contracts, cross-border structuring, sukuk issuance,
and requests phrased as seeking approval.

## Corpus

Real AAOIFI Shari'ah Standards, English, 2017 edition, parsed from **the
publisher's own PDF** rather than the Internet Archive's OCR of it. Both are
available; the PDF carries a genuine text layer that parses measurably cleaner.
With the current parser the PDF text yields all 54 standards against the OCR
text's 52.

**The shipped index holds 1,518 clauses across 48 of the edition's 54
standards.** `corpus_version: aaoifi-en-2017@145a0995711cf7ed`

> **Six standards are absent from the shipped index: 42, 43, 44, 46, 47 and 48.**
> They are in the source. The header regex required `Standard No.` with a period,
> and the publisher's running page headers omit it, so only each standard's title
> page matched — and the title sits on the *next* line, leaving frequency voting
> nothing to resolve. The name fell back to `"Standard N"` and the quality gate
> dropped it.
>
> `parse.py` now accepts both punctuations, which recovers all six (54 standards,
> 1,629 clauses) and also stops 8 clauses being mis-attributed to SS 40.
> `tests/test_ingest.py` fails on the old behaviour naming the exact six.
> **The index has not been rebuilt**, so the gap is live and the numbers above
> describe what actually ships. `scripts/verify_docs.py` prints the coverage on
> every run and fails if anything goes missing beyond these six.
>
> This matters more than a count. SS 46 is *Al-Wakalah Bi Al-Istithmar*, which
> governs wakala investment accounts — the compliant alternative to the
> fixed-return savings account used as the example query in the README — and
> SS 47 governs profit calculation. A wakala question retrieves adjacent clauses
> from SS 40 and SS 23 that score well, so `weak_retrieval` does not fire and
> nothing in the audit record reveals that the governing standard was never
> indexed. The eval cannot see it either: its 40 cases were written from indexed
> clauses, so no case can test a standard that is not there.

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
| 23 | Agency and the Act of an Uncommissioned Agent | 33 |
| 24 | Syndicated Financing | 17 |
| 25 | Combination of Contracts | 25 |
| 26 | Islamic Insurance | 41 |
| 27 | Indices | 18 |
| 28 | Banking Services in Islamic Banks | 10 |
| 29 | Stipulations and Ethics of Fatwa | 38 |
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
| 40 | Distribution of Profit in Mudarabah-Based Investments | 47 |
| 41 | Islamic Reinsurance | 17 |
| 45 | Protection of Capital and Investments | 16 |
| 49 | Unilateral and Bilateral Promise | 17 |
| 50 | Irrigation Partnership (Musaqat) | 23 |
| 51 | Options to Revoke Contracts Due to Incomplete Performance | 22 |
| 52 | Options to Reconsider | 27 |
| 53 | 'Arboun (Earnest Money) | 11 |
| 54 | Revocation of Contracts by Exercise of a Cooling-Off Option | 12 |

</details>

Ingestion applies a quality gate (`ingest/parse.py`, `is_well_parsed`). A standard
whose title never resolved, or which yielded almost no numbered entries, did not
really parse, and indexing it would inject noise without adding coverage.

The gate is the right idea and it was reading a bad signal: an unresolved title
meant "the headers were damaged in the source" when it could equally mean "the
header pattern is too strict". That is what happened to the six standards above,
and it is why the gate now has a test asserting the source's full numbering
rather than trusting its own output.

## Why the chunk is the clause

AAOIFI text is hierarchically numbered, with `2/2/2` sitting under `2/2` under
`2`. So:

> **The chunk boundary is the citation unit.**

A retrieved chunk *is* a reference a compliance officer can verify by hand
(`AAOIFI SS No. 8 (Murabahah), clause 3/1/1`). A generic recursive splitter would
straddle `2/2/2` and `2/2/3`, and a citation could then only point at a page.

Each chunk carries its heading path, because the standards repeat near-identical
language across products: profit distribution under Mudarabah and under Wakala
read alike and mean different things. Clauses longer than 1,800 characters split
on sentence boundaries with overlap, keeping the parent clause path so citations
still resolve.

## Source defects are permanent, and the system assumes it

Clause SS8-3.1.1 reads `"concludes a urchase contract"`. That appears in the
publisher's own PDF under every extraction mode, and it is the only
dropped-letter instance in 1,264 pages, so it is a typo in the published standard
rather than an OCR artefact. No cleaner source fixes it.

This is why citation checking tolerates imperfect source text instead of demanding
character-perfect equality (`agent/guardrails.py`). It verifies provenance
fuzzily, polarity exactly, and substance by length. Mal will eventually index its
own term sheets and counsel memos, which are messier than AAOIFI's typesetting,
and a system that requires a clean corpus breaks on contact with production.

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
