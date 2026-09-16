# Ukrainian homonym pipeline

This project implements two input workflows that converge on cached ГРАК example retrieval,
OpenAI validation/sense assignment, and final dictionary aggregation.

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
success output. Requests are spaced 0.5 seconds apart by default.

By default examples are taken in corpus order. This is reproducible but is not a
representative sample and may favor older sources. Pass `--seed 42` (or configure
`grac.seed`) for reproducible sampling of sentence ranks across the full concordance.
Sampling is implemented locally without assuming a server seed API; it can require
more page requests. Record counts refer to sentences, not individual lemma occurrences.
Sentences above 100 tokens are skipped because Bonito can truncate long matches; their
raw rows and skip reasons remain in the cache. Fewer examples may be returned if there
are insufficient eligible sentences.

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

The full workflow commands report progress in the terminal after each lemma, including
the number of glosses processed and the number with at least one final validated
example, as well as the current count of lemmas with at least two glosses. They also
print the number of GRAC candidates for each lemma and a cumulative summary; detailed
validation records remain in the output JSONL files.

Each CLI run also writes `run_manifest.json` and `config.snapshot.yaml` at the output
root. The manifest records the workflow, input SHA-256, configuration, Python and
dependency versions, Git revision when available, completion status, aggregate
statistics, and output artifact hashes. `audit/lemma_audit.jsonl` contains one
cache-addressed record per lemma with Wikipedia/Terra, GRAC, validation, rejection,
final-cap, timing, and failure metrics. Aggregate statistics use the latest audit and
cache records for the current run, so historical retries do not inflate counts.

The integration was verified with the public Grac v.19 backend
(`open-5.71.15`). It is a website interface and can change; the adapter intentionally
fails on incompatible responses. References:
[ГРАК lemma search](https://uacorpus.org/poshuk-u-graku/poshuk-u-graku),
[site backend configuration](https://sketch.uacorpus.org/config.js),
[CQL containing and the 100-token display limit](https://www.sketchengine.eu/documentation/cql-within-containing/).
