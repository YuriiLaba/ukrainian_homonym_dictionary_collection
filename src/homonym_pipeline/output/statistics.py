from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from homonym_pipeline.config import AppConfig
from homonym_pipeline.models import FinalLemmaEntry, LemmaAuditRecord
from homonym_pipeline.storage import atomic_write_json, latest_by_cache_key, read_jsonl


def calculate_statistics(entries: list[FinalLemmaEntry], output_dir: str | Path,
                         *, run_id: str | None = None,
                         config: AppConfig | None = None) -> dict[str, Any]:
    counts = [len(sense.examples) for entry in entries for sense in entry.glosses]
    accepted = sum(counts)
    audit_records = latest_by_cache_key(Path(output_dir) / "audit" / "lemma_audit.jsonl")
    audits = [LemmaAuditRecord.model_validate(item) for item in audit_records.values()
              if run_id is None or item.get("run_id") == run_id]
    rejected_by_reason: Counter[str] = Counter()
    terra_actions: Counter[str] = Counter()
    grac_skipped_by_reason: Counter[str] = Counter()
    for audit in audits:
        rejected_by_reason.update(audit.rejected_by_reason)
        terra_actions.update(audit.terra_actions)
        grac_skipped_by_reason.update(audit.grac_skipped_by_reason)

    if audits:
        grac_candidates = sum(item.grac_candidates_retrieved for item in audits)
        grac_total_sentence_hits = sum(item.grac_total_sentence_hits for item in audits)
        grac_pages_requested = sum(item.grac_pages_requested for item in audits)
        grac_raw_rows_examined = sum(item.grac_raw_rows_examined for item in audits)
        grac_duplicate_rows_skipped = sum(item.grac_duplicate_rows_skipped for item in audits)
        grac_request_attempts = sum(item.grac_request_attempts for item in audits)
        grac_retry_count = sum(item.grac_retry_count for item in audits)
        grac_cache_hits = sum(item.grac_cache_hit for item in audits)
        embedding_calls = sum(item.embedding_calls for item in audits)
        embedding_cache_hits = sum(item.embedding_cache_hits for item in audits)
        embedding_pairs_scored = sum(item.embedding_pairs_scored for item in audits)
        embedding_candidates_selected = sum(item.embedding_candidates_selected for item in audits)
        embedding_models = sorted({item.embedding_model for item in audits if item.embedding_model})
        rejected_count = sum(rejected_by_reason.values())
        accepted_before_cap = sum(item.accepted_before_final_cap for item in audits)
        validation_batches = sum(item.validation_batches for item in audits)
        validation_cache_hits = sum(item.validation_cache_hits for item in audits)
        llm_assignments = sum(item.llm_assignments for item in audits)
        validation_model_accepted = sum(item.validation_model_accepted for item in audits)
        validation_model_rejected = sum(item.validation_model_rejected for item in audits)
        validation_confidence_sum = sum(item.validation_confidence_sum for item in audits)
        validation_confidence_count = sum(item.validation_confidence_count for item in audits)
        failed_lemmas = sum(item.status == "failed" for item in audits)
        total_elapsed = sum(item.elapsed_seconds for item in audits)
        grac_elapsed = sum(item.grac_elapsed_seconds for item in audits)
        embedding_elapsed = sum(item.embedding_elapsed_seconds for item in audits)
        validation_elapsed = sum(item.validation_elapsed_seconds for item in audits)
        finalization_elapsed = sum(item.finalization_elapsed_seconds for item in audits)
        wikipedia_elapsed = sum(item.wikipedia_elapsed_seconds for item in audits)
        gloss_augmentation_elapsed = sum(item.gloss_augmentation_elapsed_seconds for item in audits)
        wikipedia_cache_hits = sum(item.wikipedia_cache_hit for item in audits)
        terra_cache_hits = sum(item.terra_cache_hit for item in audits)
        lemma_elapsed_values = [item.elapsed_seconds for item in audits]
        input_glosses = sum(item.input_glosses for item in audits)
        input_multi_gloss_lemmas = sum(item.input_glosses >= 2 for item in audits)
        wikipedia_candidates = sum(item.wikipedia_candidates for item in audits)
    else:
        # Backward-compatible fallback for outputs created before per-lemma audits.
        candidates = read_jsonl(Path(output_dir) / "grac" / "candidate_examples.jsonl")
        rejected = read_jsonl(Path(output_dir) / "validation" / "rejected_examples.jsonl")
        grac_candidates = sum(len(item.get("examples", [])) for item in candidates)
        grac_total_sentence_hits = 0
        grac_pages_requested = 0
        grac_raw_rows_examined = 0
        grac_duplicate_rows_skipped = 0
        grac_request_attempts = 0
        grac_retry_count = 0
        grac_cache_hits = 0
        rejected_count = len(rejected)
        accepted_before_cap = accepted
        validation_batches = 0
        validation_cache_hits = 0
        llm_assignments = 0
        validation_model_accepted = 0
        validation_model_rejected = 0
        validation_confidence_sum = 0.0
        validation_confidence_count = 0
        failed_lemmas = 0
        total_elapsed = 0.0
        grac_elapsed = 0.0
        embedding_elapsed = 0.0
        validation_elapsed = 0.0
        finalization_elapsed = 0.0
        wikipedia_elapsed = 0.0
        gloss_augmentation_elapsed = 0.0
        wikipedia_cache_hits = 0
        terra_cache_hits = 0
        lemma_elapsed_values = []
        input_glosses = sum(len(entry.glosses) for entry in entries)
        input_multi_gloss_lemmas = sum(len(entry.glosses) >= 2 for entry in entries)
        wikipedia_candidates = 0
        embedding_calls = 0
        embedding_cache_hits = 0
        embedding_pairs_scored = 0
        embedding_candidates_selected = 0
        embedding_models = []

    input_metadata_path = Path(output_dir) / "run_input_statistics.json"
    if input_metadata_path.exists():
        input_metadata = json.loads(input_metadata_path.read_text(encoding="utf-8"))
        if run_id is not None and input_metadata.get("run_id") not in {None, run_id}:
            input_metadata = {}
    else:
        input_metadata = {}
    stats: dict[str, Any] = {
        "run_id": run_id,
        "original_lemmas": input_metadata.get("original_lemmas", len(entries)),
        "original_glosses": input_metadata.get("original_glosses", input_glosses),
        "single_gloss_lemmas_removed": input_metadata.get("single_gloss_lemmas_removed", 0),
        "lemmas_after_max_lemmas": input_metadata.get("lemmas_after_max_lemmas", len(entries)),
        "lemmas_after_gloss_filter": input_metadata.get("lemmas_after_gloss_filter", len(entries)),
        "glosses_after_wikipedia_augmentation": input_metadata.get(
            "glosses_after_wikipedia_augmentation"
        ),
        "requested_max_lemmas": input_metadata.get("requested_max_lemmas"),
        "total_lemmas": len(entries),
        "successful_lemmas": sum(item.status == "success" for item in audits) if audits else len(entries),
        "total_glosses": sum(len(entry.glosses) for entry in entries),
        "input_glosses": input_glosses,
        "input_multi_gloss_lemmas": input_multi_gloss_lemmas,
        "final_multi_gloss_lemmas": sum(len(entry.glosses) >= 2 for entry in entries),
        "glosses_by_source": {},
        "glosses_from_dictionary": sum(1 for entry in entries for sense in entry.glosses if sense.gloss_source == "dictionary"),
        "glosses_from_wikipedia": sum(1 for entry in entries for sense in entry.glosses if sense.gloss_source == "wikipedia"),
        "glosses_refined_by_llm": sum(1 for entry in entries for sense in entry.glosses if sense.gloss_source == "llm_refined"),
        "glosses_added_by_llm": sum(1 for entry in entries for sense in entry.glosses if sense.gloss_source == "llm_added"),
        "wikipedia_candidates_retrieved": wikipedia_candidates,
        "terra_actions": dict(terra_actions),
        "grac_candidates_retrieved": grac_candidates,
        "grac_total_sentence_hits": grac_total_sentence_hits,
        "grac_pages_requested": grac_pages_requested,
        "grac_raw_rows_examined": grac_raw_rows_examined,
        "grac_skipped_by_reason": dict(grac_skipped_by_reason),
        "grac_duplicate_rows_skipped": grac_duplicate_rows_skipped,
        "grac_request_attempts": grac_request_attempts,
        "grac_retry_count": grac_retry_count,
        "grac_cache_hits": grac_cache_hits,
        "wikipedia_cache_hits": wikipedia_cache_hits,
        "terra_cache_hits": terra_cache_hits,
        "embedding_models": embedding_models,
        "embedding_calls": embedding_calls,
        "embedding_cache_hits": embedding_cache_hits,
        "embedding_pairs_scored": embedding_pairs_scored,
        "embedding_candidates_selected": embedding_candidates_selected,
        "accepted_examples": accepted,
        "accepted_examples_before_final_cap": accepted_before_cap,
        "examples_excluded_by_final_cap": max(accepted_before_cap - accepted, 0),
        "rejected_examples": rejected_count,
        "rejected_examples_by_reason": dict(rejected_by_reason),
        "validation_batches": validation_batches,
        "validation_cache_hits": validation_cache_hits,
        "llm_assignments": llm_assignments,
        "validation_model_accepted": validation_model_accepted,
        "validation_model_rejected": validation_model_rejected,
        "validation_confidence_mean": (
            validation_confidence_sum / validation_confidence_count
            if validation_confidence_count else 0.0
        ),
        "failed_lemmas": failed_lemmas,
        "lemma_audit_records": len(audits),
        "total_lemma_processing_seconds": total_elapsed,
        "total_grac_seconds": grac_elapsed,
        "total_embedding_seconds": embedding_elapsed,
        "total_validation_seconds": validation_elapsed,
        "total_finalization_seconds": finalization_elapsed,
        "total_wikipedia_seconds": wikipedia_elapsed,
        "total_gloss_augmentation_seconds": gloss_augmentation_elapsed,
        "mean_time_per_lemma_seconds": statistics.mean(lemma_elapsed_values) if lemma_elapsed_values else 0.0,
        "median_time_per_lemma_seconds": statistics.median(lemma_elapsed_values) if lemma_elapsed_values else 0.0,
        "accepted_examples_per_lemma": {entry.lemma: sum(len(sense.examples) for sense in entry.glosses) for entry in entries},
        "accepted_examples_per_gloss": {sense.sense_id: len(sense.examples) for entry in entries for sense in entry.glosses},
        "senses_with_zero_validated_examples": [sense.sense_id for entry in entries for sense in entry.glosses if not sense.examples],
        "mean_examples_per_sense": statistics.mean(counts) if counts else 0.0,
        "median_examples_per_sense": statistics.median(counts) if counts else 0.0,
    }
    for entry in entries:
        for sense in entry.glosses:
            stats["glosses_by_source"][sense.gloss_source] = stats["glosses_by_source"].get(sense.gloss_source, 0) + 1
    all_calls = read_jsonl(Path(output_dir) / "validation" / "llm_calls.jsonl")
    calls = [item for item in all_calls if run_id is None or item.get("run_id") == run_id]
    usage_by_stage: dict[str, dict[str, int]] = {}
    calls_by_stage: Counter[str] = Counter()
    for item in calls:
        stage = item.get("stage") or "unknown"
        calls_by_stage[stage] += 1
        stage_usage = usage_by_stage.setdefault(stage, {"input_tokens": 0, "output_tokens": 0})
        usage = item.get("usage") or {}
        stage_usage["input_tokens"] += usage.get("input_tokens", 0)
        stage_usage["output_tokens"] += usage.get("output_tokens", 0)
    stats["llm_token_usage"] = {
        "input_tokens": sum((item.get("usage") or {}).get("input_tokens", 0) for item in calls),
        "output_tokens": sum((item.get("usage") or {}).get("output_tokens", 0) for item in calls),
    }
    stats["llm_calls"] = len(calls)
    stats["llm_calls_by_stage"] = dict(calls_by_stage)
    stats["llm_token_usage_by_stage"] = usage_by_stage
    llm_cost_by_model: dict[str, float | None] = {}
    for item in calls:
        model = item.get("model") or "unknown"
        usage = item.get("usage") or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        prices = (config.llm.pricing.get(model, {}) if config else {})
        input_price = prices.get("input_per_million_tokens")
        output_price = prices.get("output_per_million_tokens")
        if input_price is None or output_price is None:
            llm_cost_by_model.setdefault(model, None)
        elif llm_cost_by_model.get(model) is not None or model not in llm_cost_by_model:
            llm_cost_by_model[model] = (llm_cost_by_model.get(model) or 0.0) + (
                input_tokens * input_price + output_tokens * output_price
            ) / 1_000_000
    stats["estimated_llm_cost_usd_by_model"] = llm_cost_by_model
    configured_llm_costs = [value for value in llm_cost_by_model.values() if value is not None]
    stats["estimated_llm_cost_usd"] = (
        sum(configured_llm_costs)
        if len(configured_llm_costs) == len(llm_cost_by_model)
        else None
    )

    embedding_path = Path(output_dir) / "grac" / "embedding_calls.jsonl"
    all_embedding_calls = read_jsonl(embedding_path)
    embedding_calls_for_run = [
        item for item in all_embedding_calls
        if run_id is None or item.get("run_id") == run_id
    ]
    embedding_input_tokens = sum(
        (item.get("usage") or {}).get("prompt_tokens", (item.get("usage") or {}).get("total_tokens", 0))
        for item in embedding_calls_for_run
    )
    stats["embedding_token_usage"] = {"input_tokens": embedding_input_tokens}
    embedding_models_usage = Counter(item.get("model", "unknown") for item in embedding_calls_for_run)
    stats["embedding_calls_by_model"] = dict(embedding_models_usage)
    embedding_cost_by_model: dict[str, float | None] = {}
    for model in embedding_models_usage:
        price = config.embeddings.pricing.get(model) if config else None
        embedding_cost_by_model[model] = (
            embedding_input_tokens * price / 1_000_000 if price is not None else None
        )
    stats["estimated_embedding_cost_usd_by_model"] = embedding_cost_by_model
    configured_embedding_costs = [value for value in embedding_cost_by_model.values() if value is not None]
    stats["estimated_embedding_cost_usd"] = (
        sum(configured_embedding_costs)
        if len(configured_embedding_costs) == len(embedding_cost_by_model)
        else None
    )
    atomic_write_json(Path(output_dir) / "statistics" / "pipeline_statistics.json", stats)
    return stats
