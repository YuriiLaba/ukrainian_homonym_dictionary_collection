from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from homonym_pipeline.config import AppConfig
from homonym_pipeline.hashing import content_hash
from homonym_pipeline.llm.client import LLMClient
from homonym_pipeline.models import (
    CandidateExample,
    FailureRecord,
    FinalLemmaEntry,
    FinalSenseEntry,
    LemmaAuditRecord,
    LemmaEntry,
    ValidatedExample,
)
from homonym_pipeline.retrieval.grac import GracClient
from homonym_pipeline.retrieval.reranking import rank_candidates_by_gloss
from homonym_pipeline.storage import append_jsonl, latest_by_cache_key
from homonym_pipeline.validation.example_validator import validate_gloss_batch


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
               embedder: Any | None = None,
               run_id: str | None = None, workflow: str = "shared",
               audit_path: str | Path | None = None,
               stage_metadata: dict[str, dict[str, Any]] | None = None) -> list[FinalLemmaEntry]:
    root = Path(output_dir)
    candidate_path = root / "grac" / "candidate_examples.jsonl"
    assignment_path = root / "validation" / "llm_assignments.jsonl"
    calls_path = root / "validation" / "llm_calls.jsonl"
    embedding_calls_path = root / "grac" / "embedding_calls.jsonl"
    embedding_ranking_path = root / "grac" / "embedding_rankings.jsonl"
    rejected_path = root / "validation" / "rejected_examples.jsonl"
    failures_path = root / "validation" / "failures.jsonl"
    audit_path = Path(audit_path) if audit_path else root / "audit" / "lemma_audit.jsonl"
    run_id = run_id or "untracked"
    stage_metadata = stage_metadata or {}
    cache = latest_by_cache_key(assignment_path)
    candidate_cache = latest_by_cache_key(candidate_path)
    embedding_cache = latest_by_cache_key(embedding_ranking_path)
    dropped_single_gloss_lemmas = sum(len(entry.glosses) < 2 for entry in entries)
    if dropped_single_gloss_lemmas:
        logger.info(
            "[filter] Removed %d single-gloss lemmas. Processing %d of %d lemmas.",
            dropped_single_gloss_lemmas,
            len(entries) - dropped_single_gloss_lemmas,
            len(entries),
        )
        entries = [entry for entry in entries if len(entry.glosses) >= 2]
    finals: list[FinalLemmaEntry] = []
    total_lemmas = len(entries)
    processed_lemmas = 0
    processed_glosses = 0
    glosses_with_examples = 0
    lemmas_with_multiple_glosses_with_examples = 0
    grac_candidates_retrieved = 0
    embedding_pairs_scored = 0
    embedding_candidates_selected = 0
    total_glosses = sum(len(entry.glosses) for entry in entries)
    logger.info(
        "[input] %d lemmas, %d glosses; %d lemmas have multiple glosses.",
        total_lemmas,
        total_glosses,
        sum(len(entry.glosses) >= 2 for entry in entries),
    )
    logger.info(
        "[stage 2/3] GRAC retrieval, embedding reranking, and Luna validation started."
        " Luna concurrency: %d.",
        config.validation.max_concurrency,
    )

    for index, entry in enumerate(entries, start=1):
        lemma_started = time.perf_counter()
        candidate_count = 0
        validation_batches = 0
        validation_cache_hits = 0
        embedding_calls = 0
        embedding_cache_hits = 0
        lemma_embedding_pairs_scored = 0
        lemma_embedding_candidates_selected = 0
        embedding_model = config.embeddings.model if config.embeddings.enabled else None
        validated_items = []
        try:
            retrieval_key = content_hash({"stage": "grac", "lemma": entry.lemma,
                                          "max": config.grac.max_examples_per_lemma,
                                          "adapter": grac.cache_identity})
            candidate_record = candidate_cache.get(retrieval_key) if resume else None
            if candidate_record:
                candidate_models = [CandidateExample.model_validate(item) for item in candidate_record["examples"]]
            else:
                candidate_models = grac.retrieve_examples(entry.lemma, config.grac.max_examples_per_lemma)
                record = {"cache_key": retrieval_key, "lemma": entry.lemma,
                          "examples": [item.model_dump(mode="json") for item in candidate_models]}
                append_jsonl(candidate_path, record)
                candidate_cache[retrieval_key] = record
            candidate_count = len(candidate_models)

            ranking_key = content_hash({
                "stage": "embedding_reranking",
                "lemma": entry.lemma,
                "glosses": [{"sense_id": gloss.sense_id, "gloss": gloss.gloss} for gloss in entry.glosses],
                "examples": [item.model_dump(mode="json") for item in candidate_models],
                "enabled": config.embeddings.enabled,
                "model": config.embeddings.model,
                "top_k": config.validation.max_candidates_per_gloss,
            })
            ranking_record = embedding_cache.get(ranking_key) if resume else None
            if ranking_record:
                ranked_by_sense = {
                    sense_id: [CandidateExample.model_validate(item) for item in items]
                    for sense_id, items in ranking_record.get("ranked_by_sense", {}).items()
                }
                embedding_cache_hits = 1
                embedding_model = ranking_record.get("embedding_model") or embedding_model
                lemma_embedding_pairs_scored = int(ranking_record.get("pairs_scored", 0))
                lemma_embedding_candidates_selected = int(ranking_record.get("candidates_selected", 0))
            else:
                ranking = rank_candidates_by_gloss(
                    entry.glosses,
                    candidate_models,
                    embedder if config.embeddings.enabled else None,
                    top_k=config.validation.max_candidates_per_gloss,
                    model=embedding_model,
                    status=("disabled" if not config.embeddings.enabled else
                            "dry_run" if getattr(embedder, "dry_run", False) else
                            "no_embedder" if embedder is None else "embedded"),
                )
                ranked_by_sense = ranking.by_sense
                embedding_model = ranking.model or embedding_model
                embedding_calls = len(ranking.embedding_calls)
                lemma_embedding_pairs_scored = ranking.pairs_scored
                lemma_embedding_candidates_selected = ranking.candidates_selected
                ranking_record = {
                    "cache_key": ranking_key,
                    "lemma": entry.lemma,
                    "embedding_model": embedding_model,
                    "top_k": config.validation.max_candidates_per_gloss,
                    "pool_size": candidate_count,
                    "pairs_scored": ranking.pairs_scored,
                    "candidates_selected": ranking.candidates_selected,
                    "ranked_by_sense": {
                        sense_id: [item.model_dump(mode="json") for item in items]
                        for sense_id, items in ranked_by_sense.items()
                    },
                }
                append_jsonl(embedding_ranking_path, ranking_record)
                embedding_cache[ranking_key] = ranking_record
                for call in ranking.embedding_calls:
                    append_jsonl(embedding_calls_path, {"cache_key": ranking_key, **call})

            # Build all cache keys first, then execute only cache misses in a
            # bounded pool. Results are written below in job order, keeping the
            # JSONL artifacts deterministic even though requests finish out of order.
            validation_results = []
            pending_jobs = []
            job_order = 0
            for gloss in entry.glosses:
                gloss_candidates = ranked_by_sense.get(gloss.sense_id, [])
                for start in range(0, len(gloss_candidates), config.validation.batch_size):
                    batch = gloss_candidates[start:start + config.validation.batch_size]
                    validation_batches += 1
                    if getattr(llm, "dry_run", False):
                        job_order += 1
                        continue
                    validation_key = content_hash({"stage": "gloss_validation", "lemma": entry.lemma,
                                                   "sense_id": gloss.sense_id, "gloss": gloss.gloss,
                                                   "examples": [item.model_dump(mode="json") for item in batch],
                                                   "model": config.llm.model_validation,
                                                   "prompt_version": "grac_gloss_example_validation_v1"})
                    cached = cache.get(validation_key) if resume else None
                    if cached is not None:
                        validation_cache_hits += 1
                        from homonym_pipeline.models import ValidatedExample
                        validated = [ValidatedExample.model_validate(item) for item in cached["validated"]]
                        validation_results.append((job_order, validation_key, gloss, batch, validated, None, True))
                    else:
                        pending_jobs.append((job_order, validation_key, gloss, batch))
                    job_order += 1

            if pending_jobs:
                with ThreadPoolExecutor(max_workers=config.validation.max_concurrency) as executor:
                    futures = [
                        (job, executor.submit(validate_gloss_batch, entry.lemma, gloss, batch, llm))
                        for job_order, validation_key, gloss, batch in pending_jobs
                        for job in [(job_order, validation_key, gloss, batch)]
                    ]
                    for (job_order, validation_key, gloss, batch), future in futures:
                        validated, call_record = future.result()
                        validation_results.append((job_order, validation_key, gloss, batch, validated, call_record, False))

            for _, validation_key, gloss, batch, validated, call_record, from_cache in sorted(
                validation_results, key=lambda item: item[0]
            ):
                if not from_cache:
                    record = {"cache_key": validation_key, "lemma": entry.lemma,
                              "sense_id": gloss.sense_id, "gloss": gloss.gloss,
                              "validation_mode": "one_gloss_at_a_time",
                              "validated": [item.model_dump(mode="json") for item in validated]}
                    append_jsonl(assignment_path, record)
                    cache[validation_key] = record
                    append_jsonl(calls_path, {"cache_key": validation_key, **call_record.model_dump(mode="json")})
                for item in validated:
                    validated_items.append(item)
                    if not item.accepted and not from_cache:
                        append_jsonl(rejected_path, item)

            accepted_by_example: dict[str, list[tuple[int, ValidatedExample]]] = defaultdict(list)
            for order, item in enumerate(validated_items):
                if item.accepted and item.model_confidence >= config.validation.min_confidence:
                    accepted_by_example[item.example_id].append((order, item))
            accepted_examples = []
            conflict_count = 0
            for options in accepted_by_example.values():
                _, winner = max(options, key=lambda pair: (pair[1].model_confidence, -pair[0]))
                accepted_examples.append(winner)
                conflict_count += len(options) - 1

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
            if conflict_count:
                rejected_by_reason["duplicate_assignment_conflict"] += conflict_count
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
                embedding_model=embedding_model,
                embedding_calls=embedding_calls,
                embedding_cache_hits=embedding_cache_hits,
                embedding_pairs_scored=lemma_embedding_pairs_scored,
                embedding_candidates_selected=lemma_embedding_candidates_selected,
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
            embedding_pairs_scored += lemma_embedding_pairs_scored
            embedding_candidates_selected += lemma_embedding_candidates_selected
            lemma_glosses_with_examples = sum(1 for sense in final_entry.glosses if sense.examples)
            if lemma_glosses_with_examples >= 2:
                lemmas_with_multiple_glosses_with_examples += 1
            logger.info(
                "[%d/%d] %s\n"
                "  GRAC candidates: %d\n"
                "  Embedding shortlist: up to %d/gloss (%d selected; %d scored pairs)\n"
                "  Supported senses: %d/%d\n"
                "  Lemmas with 2+ supported senses: %d/%d\n"
                "  Processed glosses: %d/%d",
                index,
                total_lemmas,
                entry.lemma,
                len(candidate_models),
                config.validation.max_candidates_per_gloss,
                lemma_embedding_candidates_selected,
                lemma_embedding_pairs_scored,
                lemma_glosses_with_examples,
                len(final_entry.glosses),
                lemmas_with_multiple_glosses_with_examples,
                total_lemmas,
                processed_glosses,
                total_glosses,
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
                embedding_model=embedding_model,
                embedding_calls=embedding_calls,
                embedding_cache_hits=embedding_cache_hits,
                embedding_pairs_scored=lemma_embedding_pairs_scored,
                embedding_candidates_selected=lemma_embedding_candidates_selected,
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
                "[error %d/%d] %s — %s. Details saved to %s.",
                index,
                total_lemmas,
                entry.lemma,
                error,
                failures_path,
            )
    logger.info(
        "[stage 2/3] Complete.\n"
        "  Processed lemmas: %d/%d\n"
        "  Supported glosses: %d/%d\n"
        "  Lemmas with 2+ supported senses: %d/%d\n"
        "  GRAC candidates: %d\n"
        "  Embedding pairs scored: %d\n"
        "  Embedding candidates sent to Luna: %d",
        processed_lemmas,
        total_lemmas,
        glosses_with_examples,
        total_glosses,
        lemmas_with_multiple_glosses_with_examples,
        total_lemmas,
        grac_candidates_retrieved,
        embedding_pairs_scored,
        embedding_candidates_selected,
    )
    return finals
