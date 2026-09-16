from __future__ import annotations

from pathlib import Path

from homonym_pipeline.config import AppConfig
from homonym_pipeline.hashing import content_hash
from homonym_pipeline.inputs.dictionary import parse_dictionary
from homonym_pipeline.llm.client import LLMClient
from homonym_pipeline.pipeline.shared import run_shared
from homonym_pipeline.retrieval.grac import GracClient
from homonym_pipeline.storage import append_jsonl


def run_baseline(input_path: str | Path, output_dir: str | Path, config: AppConfig,
                 llm: LLMClient | None = None, grac: GracClient | None = None, max_lemmas: int | None = None,
                 resume: bool = True) -> list:
    entries = parse_dictionary(input_path)
    if max_lemmas:
        entries = entries[:max_lemmas]
    for entry in entries:
        append_jsonl(Path(output_dir) / "gloss_inventory" / "dictionary_glosses.jsonl", {
            "cache_key": content_hash({"stage": "dictionary", "lemma": entry.lemma,
                                        "glosses": [item.model_dump(mode="json") for item in entry.glosses]}),
            "lemma": entry.lemma,
            "glosses": [item.model_dump(mode="json") for item in entry.glosses],
        })
    llm = llm or LLMClient(config.llm)
    owns_grac = grac is None
    grac = grac or GracClient.from_config(
        config.grac, cache_dir=Path(output_dir) / "grac" / "cache", resume=resume)
    try:
        return run_shared(entries, output_dir, config, grac, llm, resume=resume)
    finally:
        if owns_grac:
            grac.close()
