from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

from homonym_pipeline.config import AppConfig
from homonym_pipeline.hashing import content_hash
from homonym_pipeline.llm.client import LLMClient
from homonym_pipeline.models import FailureRecord, FinalLemmaEntry, FinalSenseEntry, LemmaAuditRecord, LemmaEntry
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
               grac: GracClient, llm: LLMClient, resume: bool = True, *,
               run_id: str | None = None, workflow: str = "shared",
               audit_path: str | Path | None = None,
               stage_metadata: dict[str, dict[str, Any]] | None = None) -> list[FinalLemmaEntry]:
    root = Path(output_dir)
    candidate_path = root / "grac" / "candidate_examples.jsonl"
    assignment_path = root / "validation" / "llm_assignments.jsonl"
    calls_path = root / "validation" / "llm_calls.jsonl"
    rejected_path = root / "validation" / "rejected_examples.jsonl"
    failures_path = root / "validation" / "failures.jsonl"
    audit_path = Path(audit_path) if audit_path else root / "audit" / "lemma_audit.jsonl"
    run_id = run_id or "untracked"
    stage_metadata = stage_metadata or {}
    cache = latest_by_cache_key(assignment_path)
    candidate_cache = latest_by_cache_key(candidate_path)
    finals: list[FinalLemmaEntry] = []
    total_lemmas = len(entries)
    processed_lemmas = 0
    processed_glosses = 0
    glosses_with_examples = 0
    lemmas_with_multiple_glosses = 0
    grac_candidates_retrieved = 0
    logger.info(
        "[input] lemmas=%d glosses=%d multi_gloss_lemmas=%d",
        total_lemmas,
        sum(len(entry.glosses) for entry in entries),
        sum(len(entry.glosses) >= 2 for entry in entries),
    )
    logger.info("[stage] grac_retrieval_and_luna_validation total_lemmas=%d", total_lemmas)

    for index, entry in enumerate(entries, start=1):
        lemma_started = time.perf_counter()
        candidate_count = 0
        validation_batches = 0
        validation_cache_hits = 0
        validated_items = []
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
            candidate_count = len(candidate_models)

            accepted_examples = []
            for start in range(0, len(candidate_models), config.validation.batch_size):
                batch = candidate_models[start:start + config.validation.batch_size]
                validation_batches += 1
                if getattr(llm, "dry_run", False):
                    continue
                validation_key = content_hash({"stage": "validation", "lemma": entry.lemma,
                                               "glosses": [item.model_dump(mode="json") for item in entry.glosses],
                                               "examples": [item.model_dump(mode="json") for item in batch],
                                               "model": config.llm.model_validation,
                                               "prompt_version": "grac_example_assignment_v1"})
                cached = cache.get(validation_key) if resume else None
                from_cache = cached is not None
                if from_cache:
                    validation_cache_hits += 1
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
                    validated_items.append(item)
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
            rejected_by_reason = Counter()
            for item in validated_items:
                if item.accepted and item.model_confidence < config.validation.min_confidence:
                    rejected_by_reason["below_confidence_threshold"] += 1
                elif not item.accepted:
                    rejected_by_reason[item.validation_reason] += 1
            metadata = stage_metadata.get(entry.lemma, {})
            audit = LemmaAuditRecord(
                run_id=run_id,
                workflow=workflow,
                lemma=entry.lemma,
                status="success",
                elapsed_seconds=round(time.perf_counter() - lemma_started, 6),
                input_glosses=len(entry.glosses),
                wikipedia_candidates=int(metadata.get("wikipedia_candidates", 0)),
                terra_actions=dict(metadata.get("terra_actions", {})),
                grac_candidates_retrieved=candidate_count,
                validation_batches=validation_batches,
                validation_cache_hits=validation_cache_hits,
                llm_assignments=len(validated_items),
                accepted_before_final_cap=len(accepted_examples),
                rejected_by_reason=dict(rejected_by_reason),
                final_glosses=len(final_entry.glosses),
                final_glosses_with_examples=sum(1 for sense in final_entry.glosses if sense.examples),
                final_examples=sum(len(sense.examples) for sense in final_entry.glosses),
            )
            append_jsonl(audit_path, {
                "cache_key": content_hash({"stage": "lemma_audit", "run_id": run_id, "lemma": entry.lemma}),
                **audit.model_dump(mode="json"),
            })
            processed_lemmas += 1
            processed_glosses += len(final_entry.glosses)
            glosses_with_examples += sum(1 for sense in final_entry.glosses if sense.examples)
            grac_candidates_retrieved += len(candidate_models)
            if len(final_entry.glosses) >= 2:
                lemmas_with_multiple_glosses += 1
            logger.info(
                "[progress] %d/%d lemma=%s lemma_grac_candidates=%d lemma_glosses=%d "
                "lemma_glosses_with_examples=%d cumulative_multi_gloss_lemmas=%d "
                "cumulative_glosses=%d/%d",
                index,
                total_lemmas,
                entry.lemma,
                len(candidate_models),
                len(final_entry.glosses),
                sum(1 for sense in final_entry.glosses if sense.examples),
                lemmas_with_multiple_glosses,
                processed_glosses,
                sum(len(item.glosses) for item in entries),
            )
        except Exception as error:
            append_jsonl(failures_path, FailureRecord(stage="shared", lemma=entry.lemma,
                                                      error_type=type(error).__name__, message=str(error)))
            audit = LemmaAuditRecord(
                run_id=run_id,
                workflow=workflow,
                lemma=entry.lemma,
                status="failed",
                elapsed_seconds=round(time.perf_counter() - lemma_started, 6),
                input_glosses=len(entry.glosses),
                grac_candidates_retrieved=candidate_count,
                validation_batches=validation_batches,
                validation_cache_hits=validation_cache_hits,
                llm_assignments=len(validated_items),
                error_type=type(error).__name__,
                error_message=str(error),
            )
            append_jsonl(audit_path, {
                "cache_key": content_hash({"stage": "lemma_audit", "run_id": run_id, "lemma": entry.lemma}),
                **audit.model_dump(mode="json"),
            })
            logger.error(
                "[%d/%d] %s — failed: %s (details saved to %s)",
                index,
                total_lemmas,
                entry.lemma,
                error,
                failures_path,
            )
    logger.info(
        "[summary] processed_lemmas=%d glosses=%d glosses_with_examples=%d "
        "multi_gloss_lemmas=%d grac_candidates=%d",
        processed_lemmas,
        processed_glosses,
        glosses_with_examples,
        lemmas_with_multiple_glosses,
        grac_candidates_retrieved,
    )
    return finals
