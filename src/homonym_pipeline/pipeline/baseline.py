from __future__ import annotations

import logging
from pathlib import Path

from homonym_pipeline.config import AppConfig
from homonym_pipeline.embeddings.client import EmbeddingClient
from homonym_pipeline.hashing import content_hash
from homonym_pipeline.inputs.dictionary import parse_dictionary
from homonym_pipeline.llm.client import LLMClient
from homonym_pipeline.pipeline.shared import run_shared
from homonym_pipeline.retrieval.grac import GracClient
from homonym_pipeline.storage import append_jsonl, atomic_write_json


logger = logging.getLogger(__name__)


def run_baseline(input_path: str | Path, output_dir: str | Path, config: AppConfig,
                 llm: LLMClient | None = None, grac: GracClient | None = None, max_lemmas: int | None = None,
                 resume: bool = True, run_id: str | None = None,
                 embedder: EmbeddingClient | None = None) -> list:
    entries = parse_dictionary(input_path)
    input_lemma_count = len(entries)
    input_gloss_count = sum(len(entry.glosses) for entry in entries)
    single_gloss_count = sum(len(entry.glosses) < 2 for entry in entries)
    entries = [entry for entry in entries if len(entry.glosses) >= 2]
    if len(entries) < input_lemma_count:
        logger.info(
            "[filter] Removed %d single-gloss lemmas. Processing %d of %d lemmas.",
            input_lemma_count - len(entries),
            len(entries),
            input_lemma_count,
        )
    if max_lemmas:
        entries = entries[:max_lemmas]
    atomic_write_json(Path(output_dir) / "run_input_statistics.json", {
        "run_id": run_id,
        "original_lemmas": input_lemma_count,
        "original_glosses": input_gloss_count,
        "single_gloss_lemmas_removed": single_gloss_count,
        "lemmas_after_max_lemmas": len(entries),
        "requested_max_lemmas": max_lemmas,
    })
    for entry in entries:
        append_jsonl(Path(output_dir) / "gloss_inventory" / "dictionary_glosses.jsonl", {
            "cache_key": content_hash({"stage": "dictionary", "lemma": entry.lemma,
                                        "glosses": [item.model_dump(mode="json") for item in entry.glosses]}),
            "lemma": entry.lemma,
            "glosses": [item.model_dump(mode="json") for item in entry.glosses],
        })
    llm = llm or LLMClient(config.llm)
    embedder = embedder or EmbeddingClient(config.embeddings)
    owns_grac = grac is None
    grac = grac or GracClient.from_config(
        config.grac, cache_dir=Path(output_dir) / "grac" / "cache", resume=resume)
    try:
        return run_shared(entries, output_dir, config, grac, llm, resume=resume,
                          embedder=embedder, workflow="baseline", run_id=run_id)
    finally:
        if owns_grac:
            grac.close()
