from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from homonym_pipeline.models import FinalLemmaEntry
from homonym_pipeline.storage import atomic_write_json, read_jsonl


def calculate_statistics(entries: list[FinalLemmaEntry], output_dir: str | Path) -> dict[str, Any]:
    counts = [len(sense.examples) for entry in entries for sense in entry.glosses]
    accepted = sum(counts)
    candidates = read_jsonl(Path(output_dir) / "grac" / "candidate_examples.jsonl")
    rejected = read_jsonl(Path(output_dir) / "validation" / "rejected_examples.jsonl")
    stats: dict[str, Any] = {
        "total_lemmas": len(entries),
        "total_glosses": sum(len(entry.glosses) for entry in entries),
        "glosses_by_source": {},
        "glosses_from_dictionary": sum(1 for entry in entries for sense in entry.glosses if sense.gloss_source == "dictionary"),
        "glosses_from_wikipedia": sum(1 for entry in entries for sense in entry.glosses if sense.gloss_source == "wikipedia"),
        "glosses_refined_by_llm": sum(1 for entry in entries for sense in entry.glosses if sense.gloss_source == "llm_refined"),
        "glosses_added_by_llm": sum(1 for entry in entries for sense in entry.glosses if sense.gloss_source == "llm_added"),
        "grac_candidates_retrieved": sum(len(item.get("examples", [])) for item in candidates),
        "accepted_examples": accepted,
        "rejected_examples": len(rejected),
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
