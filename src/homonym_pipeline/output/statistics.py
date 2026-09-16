from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from homonym_pipeline.models import FinalLemmaEntry, LemmaAuditRecord
from homonym_pipeline.storage import atomic_write_json, latest_by_cache_key, read_jsonl


def calculate_statistics(entries: list[FinalLemmaEntry], output_dir: str | Path,
                         *, run_id: str | None = None) -> dict[str, Any]:
    counts = [len(sense.examples) for entry in entries for sense in entry.glosses]
    accepted = sum(counts)
    audit_records = latest_by_cache_key(Path(output_dir) / "audit" / "lemma_audit.jsonl")
    audits = [LemmaAuditRecord.model_validate(item) for item in audit_records.values()
              if run_id is None or item.get("run_id") == run_id]
    rejected_by_reason: Counter[str] = Counter()
    terra_actions: Counter[str] = Counter()
    for audit in audits:
        rejected_by_reason.update(audit.rejected_by_reason)
        terra_actions.update(audit.terra_actions)

    if audits:
        grac_candidates = sum(item.grac_candidates_retrieved for item in audits)
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
        failed_lemmas = sum(item.status == "failed" for item in audits)
        total_elapsed = sum(item.elapsed_seconds for item in audits)
        input_glosses = sum(item.input_glosses for item in audits)
        input_multi_gloss_lemmas = sum(item.input_glosses >= 2 for item in audits)
        wikipedia_candidates = sum(item.wikipedia_candidates for item in audits)
    else:
        # Backward-compatible fallback for outputs created before per-lemma audits.
        candidates = read_jsonl(Path(output_dir) / "grac" / "candidate_examples.jsonl")
        rejected = read_jsonl(Path(output_dir) / "validation" / "rejected_examples.jsonl")
        grac_candidates = sum(len(item.get("examples", [])) for item in candidates)
        rejected_count = len(rejected)
        accepted_before_cap = accepted
        validation_batches = 0
        validation_cache_hits = 0
        llm_assignments = 0
        failed_lemmas = 0
        total_elapsed = 0.0
        input_glosses = sum(len(entry.glosses) for entry in entries)
        input_multi_gloss_lemmas = sum(len(entry.glosses) >= 2 for entry in entries)
        wikipedia_candidates = 0
        embedding_calls = 0
        embedding_cache_hits = 0
        embedding_pairs_scored = 0
        embedding_candidates_selected = 0
        embedding_models = []
    stats: dict[str, Any] = {
        "run_id": run_id,
        "total_lemmas": len(entries),
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
        "failed_lemmas": failed_lemmas,
        "lemma_audit_records": len(audits),
        "total_lemma_processing_seconds": total_elapsed,
        "accepted_examples_per_lemma": {entry.lemma: sum(len(sense.examples) for sense in entry.glosses) for entry in entries},
        "accepted_examples_per_gloss": {sense.sense_id: len(sense.examples) for entry in entries for sense in entry.glosses},
        "senses_with_zero_validated_examples": [sense.sense_id for entry in entries for sense in entry.glosses if not sense.examples],
        "mean_examples_per_sense": statistics.mean(counts) if counts else 0.0,
        "median_examples_per_sense": statistics.median(counts) if counts else 0.0,
    }
    for entry in entries:
        for sense in entry.glosses:
            stats["glosses_by_source"][sense.gloss_source] = stats["glosses_by_source"].get(sense.gloss_source, 0) + 1
    calls = read_jsonl(Path(output_dir) / "validation" / "llm_calls.jsonl")
    stats["llm_token_usage"] = {
        "input_tokens": sum((item.get("usage") or {}).get("input_tokens", 0) for item in calls),
        "output_tokens": sum((item.get("usage") or {}).get("output_tokens", 0) for item in calls),
    }
    atomic_write_json(Path(output_dir) / "statistics" / "pipeline_statistics.json", stats)
    return stats
