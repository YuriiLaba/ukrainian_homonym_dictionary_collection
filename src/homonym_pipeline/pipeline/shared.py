from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from homonym_pipeline.config import AppConfig
from homonym_pipeline.hashing import content_hash
from homonym_pipeline.llm.client import LLMClient
from homonym_pipeline.models import FailureRecord, FinalLemmaEntry, FinalSenseEntry, LemmaEntry
from homonym_pipeline.retrieval.grac import GracClient
from homonym_pipeline.storage import append_jsonl, latest_by_cache_key
from homonym_pipeline.validation.example_validator import validate_batch


logger = logging.getLogger(__name__)


def _select_final_examples(examples: list, limit: int) -> list:
    """Select a deterministic, quality-first final subset for one sense."""
    ranked = sorted(
        enumerate(examples),
        key=lambda item: (-item[1].model_confidence, item[0]),
    )
    return [example for _, example in ranked[:limit]]


def run_shared(entries: list[LemmaEntry], output_dir: str | Path, config: AppConfig,
               grac: GracClient, llm: LLMClient, resume: bool = True) -> list[FinalLemmaEntry]:
    root = Path(output_dir)
    candidate_path = root / "grac" / "candidate_examples.jsonl"
    assignment_path = root / "validation" / "llm_assignments.jsonl"
    calls_path = root / "validation" / "llm_calls.jsonl"
    rejected_path = root / "validation" / "rejected_examples.jsonl"
    failures_path = root / "validation" / "failures.jsonl"
    cache = latest_by_cache_key(assignment_path)
    candidate_cache = latest_by_cache_key(candidate_path)
    finals: list[FinalLemmaEntry] = []
    total_lemmas = len(entries)
    processed_lemmas = 0
    processed_glosses = 0
    glosses_with_examples = 0
    logger.info("Starting shared pipeline for %d lemmas.", total_lemmas)

    for index, entry in enumerate(entries, start=1):
        try:
            retrieval_key = content_hash({"stage": "grac", "lemma": entry.lemma,
                                          "max": config.grac.max_examples_per_lemma,
                                          "adapter": grac.cache_identity})
            candidate_record = candidate_cache.get(retrieval_key) if resume else None
            if candidate_record:
                from homonym_pipeline.models import CandidateExample
                candidate_models = [CandidateExample.model_validate(item) for item in candidate_record["examples"]]
            else:
                candidate_models = grac.retrieve_examples(entry.lemma, config.grac.max_examples_per_lemma)
                record = {"cache_key": retrieval_key, "lemma": entry.lemma,
                          "examples": [item.model_dump(mode="json") for item in candidate_models]}
                append_jsonl(candidate_path, record)
                candidate_cache[retrieval_key] = record

            accepted_examples = []
            for start in range(0, len(candidate_models), config.validation.batch_size):
                batch = candidate_models[start:start + config.validation.batch_size]
                if getattr(llm, "dry_run", False):
                    continue
                validation_key = content_hash({"stage": "validation", "lemma": entry.lemma,
                                               "glosses": [item.model_dump(mode="json") for item in entry.glosses],
                                               "examples": [item.model_dump(mode="json") for item in batch],
                                               "model": config.llm.model_validation,
                                               "prompt_version": "grac_example_assignment_v1"})
                cached = cache.get(validation_key) if resume else None
                from_cache = cached is not None
                if cached:
                    from homonym_pipeline.models import ValidatedExample
                    validated = [ValidatedExample.model_validate(item) for item in cached["validated"]]
                else:
                    validated, call_record = validate_batch(entry.lemma, entry.glosses, batch, llm)
                    record = {"cache_key": validation_key, "lemma": entry.lemma,
                              "validated": [item.model_dump(mode="json") for item in validated]}
                    append_jsonl(assignment_path, record)
                    cache[validation_key] = record
                    append_jsonl(calls_path, {"cache_key": validation_key, **call_record.model_dump(mode="json")})
                for item in validated:
                    if item.accepted and item.model_confidence >= config.validation.min_confidence:
                        accepted_examples.append(item)
                    elif not from_cache:
                        append_jsonl(rejected_path, item)

            final_senses = []
            for gloss in entry.glosses:
                sense_examples = [item for item in accepted_examples if item.sense_id == gloss.sense_id]
                final_senses.append(FinalSenseEntry(sense_id=gloss.sense_id, lemma=entry.lemma, gloss=gloss.gloss,
                                                    gloss_source=gloss.source, original_gloss=gloss.original_gloss,
                                                    source_references=gloss.source_references,
                                                    merge_status=gloss.merge_status,
                                                    merged_into_sense_id=gloss.merged_into_sense_id,
                                                    llm_reason=gloss.llm_reason,
                                                    examples=_select_final_examples(
                                                        sense_examples,
                                                        config.validation.max_final_examples_per_sense,
                                                    )))
            final_entry = FinalLemmaEntry(lemma=entry.lemma, glosses=sorted(final_senses, key=lambda item: item.sense_id))
            finals.append(final_entry)
            processed_lemmas += 1
            processed_glosses += len(final_entry.glosses)
            glosses_with_examples += sum(1 for sense in final_entry.glosses if sense.examples)
            logger.info(
                "[%d/%d] %s — glosses processed: %d; glosses with >=1 final example: %d "
                "(cumulative: %d/%d glosses)",
                index,
                total_lemmas,
                entry.lemma,
                len(final_entry.glosses),
                sum(1 for sense in final_entry.glosses if sense.examples),
                glosses_with_examples,
                processed_glosses,
            )
        except Exception as error:
            append_jsonl(failures_path, FailureRecord(stage="shared", lemma=entry.lemma,
                                                      error_type=type(error).__name__, message=str(error)))
            logger.error(
                "[%d/%d] %s — failed: %s (details saved to %s)",
                index,
                total_lemmas,
                entry.lemma,
                error,
                failures_path,
            )
    logger.info(
        "Completed shared pipeline: %d/%d lemmas; %d glosses processed; "
        "%d glosses with >=1 final example.",
        processed_lemmas,
        total_lemmas,
        processed_glosses,
        glosses_with_examples,
    )
    return finals
