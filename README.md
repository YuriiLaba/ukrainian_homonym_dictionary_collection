# Ukrainian homonym pipeline

This project implements two input workflows that converge on cached ГРАК example retrieval,
OpenAI embedding reranking, Luna validation/sense assignment, and final dictionary aggregation.

The repository currently contains the source dictionary JSON. The implementation uses its
`lemma_normalized` and `definition` fields for Workflow A and preserves source identifiers and
paragraph metadata in structured provenance.

## Setup

```bash
uv sync --extra dev
export OPENAI_API_KEY=...
```

Run the baseline workflow:

```bash
uv run python scripts/run_baseline.py \
  --input ukrainian_homonym_dictionary.json \
  --output outputs/
```

Run the Wikipedia workflow:

```bash
uv run python scripts/run_wikipedia_augmented.py \
  --input data/input/homonym_lemmas.txt \
  --output outputs/
```

## Retrieve sentences from ГРАК without an OpenAI key

```bash
uv run python scripts/retrieve_grac.py \
  --lemma автомат --max-examples 20 \
  --output outputs/grac/automat.json
```

The default corpus is `grac19` (the backend identifier for Grac v.19).
Use `--corpus` to select another installed corpus. Output is a list of
`CandidateExample` records; use a `.jsonl` extension for JSONL instead.
In the installed local environment, `.venv/bin/python` can replace `uv run python`.

Both dictionary workflows now use the same live adapter by default. It queries the
website's Bonito backend at `https://sketch.uacorpus.org/bonito/run.cgi/concordance`
with `<s/> containing [lemma="автомат"]`: one query per lemma, independent of glosses,
including inflected forms. The whole sentence is returned as the match; `structs=g,s`
preserves corpus punctuation glue and sentence boundaries. Text is reconstructed from
the returned chunks with HTML entity decoding and spaces between chunks, matching the
website's rendering. Spelling, accents and historical orthography are preserved; this
is corpus-rendered text, not a claim of original document whitespace fidelity.

The adapter discovers available document metadata using `corp_info`, then requests
those fields in each concordance page. Examples retain corpus/version information,
document and sentence token identifiers, available author/title/date/genre/source URL,
the exact query and request parameters, timestamps, backend versions and raw rows.
Unavailable values (`===NONE===`) are omitted from the interpreted metadata.

Queries can compute asynchronously: retrieval polls until completion, paginates, and
retries transient HTTP failures with bounded backoff. Non-JSON responses, permission
failures, backend errors and incompatible response shapes raise explicit errors.
The standalone command exits unsuccessfully on such errors and does not write an empty
success output. GRAC pages are retrieved with bounded concurrency
(`grac.max_page_concurrency`, default 4) when seeded sampling requires multiple pages.
Set it to 1 to use sequential retrieval; in that mode, `request_interval_seconds` spaces
requests by default.

By default examples are taken in corpus order. This is reproducible but is not a
representative sample and may favor older sources. Pass `--seed 42` (or configure
`grac.seed`) for reproducible sampling of sentence ranks across the full concordance.
Sampling is implemented locally without assuming a server seed API; it can require
more page requests. Page requests are prefetched in deterministic windows, but sampled
ranks are processed in their original order. Record counts refer to sentences, not
individual lemma occurrences.
Sentences above 100 tokens are skipped because Bonito can truncate long matches; their
raw rows and skip reasons remain in the cache. Fewer examples may be returned if there
are insufficient eligible sentences.

The shared stage retrieves up to 1,000 GRAC sentences per lemma by default. It then
embeds the complete candidate pool and every gloss with the configured OpenAI embedding
model (`text-embedding-3-small` by default), calculates cosine similarity locally, and
passes the top 50 candidates independently for each gloss to Luna. The same sentence can
therefore be shortlisted for different glosses, but final assignment still keeps one
sense per sentence. The exact shortlist and scores are stored in
`grac/embedding_rankings.jsonl`; embedding request metadata is stored in
`grac/embedding_calls.jsonl`. Set `embeddings.enabled: false` or use `--no-embeddings`
to use deterministic GRAC order instead of semantic reranking.

Embedding batches run with bounded concurrency. `embeddings.batch_size` controls how many
texts are sent in one API request, while `embeddings.max_concurrency` controls how many
of those requests can be active at once (default: 4). Use
`--embedding-max-concurrency` to override it for a run. All response vectors are combined
in the original input order before cosine scoring; request provenance includes batch timing,
attempt count, and input hashes.

Embedding ranking is cached per lemma and includes the model, gloss inventory, GRAC
candidate pool, and top-k setting in its cache key. A later validation-only rerun does
not repeat GRAC or embedding requests. Ranked candidate provenance is copied into each
validated example's `source_metadata`, including the embedding model, cosine score,
rank, and candidate-pool size.

Luna validation uses bounded parallelism: independent gloss/batch requests run in a
worker pool while GRAC retrieval remains sequential. The default is eight concurrent Luna
requests; configure
`validation.max_concurrency` or pass `--max-concurrency`. Results and JSONL writes are
ordered deterministically after requests complete, and each request retains its own
cache key and OpenAI request identifier.

Raw response snapshots, completed page caches and completed lemma results are stored
under `outputs/grac/cache/`. Rerunning the same request reuses the cache without network
calls; an interrupted run can reuse completed pages. `--force` refreshes results while
retaining previous raw snapshots. Cache identity includes the endpoint, corpus,
adapter version, limits, page size and sampling seed. Fixture data use a separate identity.
Keep the cache with research artifacts to preserve the exact source snapshot.

The baseline `--dry-run` retrieves and caches candidates without LLM validation. For
offline work, both workflows accept `--grac-fixture path/to/examples.json`; this supports
the retrieval command's JSON list or a mapping such as
`{"автомат": [{"sentence": "...", "metadata": {}}]}`.

Every full workflow also writes a Hugging Face-compatible export under
`outputs/final/huggingface.json` and `outputs/final/huggingface.jsonl`. Its schema is
intentionally restricted to the published dataset shape: each row contains only
`lemma` (string), `gloss` (list of strings), and `examples` (list of strings). One row
is emitted per final sense, so a single normalized gloss is represented as a one-item
`gloss` list. Only accepted examples are included, and Ukrainian sentence text is
copied without rewriting. Provenance remains available in
`outputs/final/homonym_dictionary.json`; it is not added to the Hugging Face export
because that would change the dataset schema.

In Workflow B, confirmed Terra `merge` decisions are consolidated into one gloss;
all contributing Wikipedia references are retained. `merge_uncertain` decisions are
kept as separate glosses and point to the possible target sense for review. An
`llm_added` gloss is accepted only when at least one returned evidence ID matches a
retrieved Wikipedia candidate.

Final output is limited to five validated examples per sense by default
(`validation.max_final_examples_per_sense`). If more examples pass validation, the
highest-confidence examples are selected, with original validation order breaking
ties. All validation results remain in `validation/llm_assignments.jsonl`. The limit
can be changed from the CLI with `--max-final-examples-per-sense`.

GRAC candidates are retrieved once per lemma, but Luna validation is performed in
separate batches for each gloss. This lets each sense receive its own examples. If the
same sentence is accepted for multiple glosses, the pipeline keeps the highest-confidence
assignment and records the other assignment as `duplicate_assignment_conflict`.

The full workflow commands use compact, machine-readable terminal labels. The shared
stage emits records like:

```text
[filter] Removed 2 single-gloss lemmas. Processing 1749 of 1751 lemmas.
[input] 1749 lemmas, 4260 glosses; 1749 lemmas have multiple glosses.
[stage 2/3] GRAC retrieval, embedding reranking, and Luna validation started. GRAC page concurrency: 4; embedding concurrency: 4; Luna concurrency: 8.

[2/1749] аверс
  GRAC candidates: 1000
  Embedding shortlist: up to 50/gloss (100 selected; 2000 scored pairs)
  Supported senses: 1/2
  Lemmas with 2+ supported senses: 1/1749
  Processed glosses: 4/4260
  GRAC retrieval time: 18.42 seconds
  Embedding time: 2.31 seconds
  Luna validation time: 31.76 seconds
  Finalization time: 0.02 seconds
  Total processing time: 52.51 seconds

[stage 2/3] Complete.
  Processed lemmas: 1749/1749
  Supported glosses: 6/4260
  Lemmas with 2+ supported senses: 1/1749
  GRAC candidates: 1749000
  Embedding pairs scored: 4260000
  Embedding candidates sent to Luna: 213000
```

Single-gloss lemmas are filtered before GRAC retrieval and validation. The filter is
reported as, for example, `[filter] Removed 2 single-gloss lemmas. Processing 1749 of 1751 lemmas.`
Workflow B performs Wikipedia/Terra augmentation first so that a lemma can acquire a
second evidence-supported gloss before this filter is applied.

The progress record includes the current lemma's GRAC candidate count, gloss count,
glosses with at least one final validated example, and the cumulative number of lemmas
with at least two such glosses out of the total input lemmas. Detailed validation
records remain in the output JSONL files; the manifest and audit files are the
authoritative source for later statistical analysis.

Each CLI run also writes `run_manifest.json` and `config.snapshot.yaml` at the output
root. The manifest records the workflow, input SHA-256, configuration, Python and
dependency versions, Git revision when available, completion status, aggregate
statistics, and output artifact hashes. `audit/lemma_audit.jsonl` contains one
cache-addressed record per lemma with Wikipedia/Terra, GRAC, validation, rejection,
final-cap, stage-specific timing, total timing, and failure metrics. The timing fields
are `grac_elapsed_seconds`, `embedding_elapsed_seconds`,
`validation_elapsed_seconds`, `finalization_elapsed_seconds`, and `elapsed_seconds`.
GRAC audit metrics additionally include total corpus hits, pages requested, raw rows
examined, skipped rows by reason, duplicate rows, and cache hits. The run-level
`run_input_statistics.json` records original lemmas, filtering, `--max-lemmas`, and
the Wikipedia augmentation count. `pipeline_statistics.json` aggregates these values
and separates LLM calls and token usage by stage. Optional cost estimates can be
enabled with per-model prices in `config.yaml`, for example:

```yaml
llm:
  pricing:
    gpt-5.6-luna:
      input_per_million_tokens: 1.0
      output_per_million_tokens: 5.0
embeddings:
  pricing:
    text-embedding-3-small: 0.02
```

If a price is missing, the corresponding estimated cost is reported as `null` rather
than being guessed.

Each completed manifest also records run-level elapsed time and hashes the main log,
audit, intermediate, and final artifacts. LLM call records are tagged with run,
workflow, stage, lemma, sense, request attempts, and elapsed time so token and cost
statistics remain scoped to the current run.
Aggregate statistics use the latest audit and cache records for the current run, so
historical retries do not inflate counts.

The same concise progress messages shown in the terminal are also appended to
`logs/pipeline.log`, including per-lemma GRAC, embedding, Luna, finalization, and total
processing times. Low-level HTTP and OpenAI transport messages are excluded from this
log.

The integration was verified with the public Grac v.19 backend
(`open-5.71.15`). It is a website interface and can change; the adapter intentionally
fails on incompatible responses. References:
[ГРАК lemma search](https://uacorpus.org/poshuk-u-graku/poshuk-u-graku),
[site backend configuration](https://sketch.uacorpus.org/config.js),
[CQL containing and the 100-token display limit](https://www.sketchengine.eu/documentation/cql-within-containing/).
