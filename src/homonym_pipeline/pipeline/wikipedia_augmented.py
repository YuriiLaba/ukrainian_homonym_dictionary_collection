from __future__ import annotations

from pathlib import Path

from homonym_pipeline.config import AppConfig
from homonym_pipeline.glosses.llm_augmentation import augment_glosses
from homonym_pipeline.glosses.llm_augmentation import PROMPT_VERSION as GLOSS_PROMPT_VERSION
from homonym_pipeline.glosses.wikipedia import WikipediaClient
from homonym_pipeline.hashing import content_hash
from homonym_pipeline.inputs.lemma_list import parse_lemma_list
from homonym_pipeline.llm.client import LLMClient
from homonym_pipeline.models import LemmaEntry
from homonym_pipeline.models import FailureRecord, Gloss
from homonym_pipeline.pipeline.shared import run_shared
from homonym_pipeline.retrieval.grac import GracClient
from homonym_pipeline.storage import append_jsonl, latest_by_cache_key


def run_wikipedia_augmented(input_path: str | Path, output_dir: str | Path, config: AppConfig,
                            llm: LLMClient | None = None, grac: GracClient | None = None,
                            wikipedia: WikipediaClient | None = None, max_lemmas: int | None = None,
                            resume: bool = True) -> list:
    root = Path(output_dir)
    lemmas = parse_lemma_list(input_path)
    if max_lemmas:
        lemmas = lemmas[:max_lemmas]
    llm = llm or LLMClient(config.llm)
    wikipedia = wikipedia or WikipediaClient()
    raw_path = root / "gloss_inventory" / "wikipedia_raw.jsonl"
    normalized_path = root / "gloss_inventory" / "wikipedia_normalized.jsonl"
    calls_path = root / "validation" / "llm_calls.jsonl"
    failures_path = root / "validation" / "failures.jsonl"
    raw_cache = latest_by_cache_key(raw_path)
    normalized_cache = latest_by_cache_key(normalized_path)
    entries: list[LemmaEntry] = []
    for lemma in lemmas:
        try:
            raw_key = content_hash({"stage": "wikipedia_raw", "lemma": lemma, "fallback": config.pipeline.wikipedia_search_fallback})
            raw_record = raw_cache.get(raw_key) if resume else None
            if raw_record:
                from homonym_pipeline.models import WikipediaCandidate
                candidates = [WikipediaCandidate.model_validate(item) for item in raw_record["candidates"]]
            else:
                candidates = wikipedia.retrieve_candidates(lemma, config.pipeline.wikipedia_search_fallback)
                raw_record = {"cache_key": raw_key, "lemma": lemma,
                              "candidates": [item.model_dump(mode="json") for item in candidates]}
                append_jsonl(raw_path, raw_record)
                raw_cache[raw_key] = raw_record
            norm_key = content_hash({"stage": "wikipedia_normalized", "lemma": lemma,
                                     "candidates": [item.model_dump(mode="json") for item in candidates],
                                     "model": config.llm.model_gloss,
                                     "prompt_version": GLOSS_PROMPT_VERSION})
            cached = normalized_cache.get(norm_key) if resume else None
            if cached:
                glosses = [Gloss.model_validate(item) for item in cached["glosses"]]
            else:
                glosses, call_record = augment_glosses(lemma, candidates, llm)
                if not config.pipeline.allow_llm_added_senses:
                    glosses = [item for item in glosses if item.source != "llm_added"]
                normalized_record = {"cache_key": norm_key, "lemma": lemma,
                                     "glosses": [item.model_dump(mode="json") for item in glosses],
                                     "llm_call": call_record.model_dump(mode="json")}
                append_jsonl(normalized_path, normalized_record)
                normalized_cache[norm_key] = normalized_record
                append_jsonl(calls_path, {"cache_key": norm_key, **call_record.model_dump(mode="json")})
            entries.append(LemmaEntry(lemma=lemma, glosses=glosses))
        except Exception as error:
            append_jsonl(failures_path, FailureRecord(stage="wikipedia_augmentation", lemma=lemma,
                                                      error_type=type(error).__name__, message=str(error)))
    owns_grac = grac is None
    grac = grac or GracClient.from_config(
        config.grac, cache_dir=root / "grac" / "cache", resume=resume)
    try:
        return run_shared(entries, output_dir, config, grac, llm, resume=resume)
    finally:
        if owns_grac:
            grac.close()
