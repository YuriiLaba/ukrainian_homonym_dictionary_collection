from __future__ import annotations

from pathlib import Path

from homonym_pipeline.models import FinalLemmaEntry
from homonym_pipeline.storage import atomic_write_json, atomic_write_jsonl


def write_final(entries: list[FinalLemmaEntry], output_dir: str | Path) -> None:
    root = Path(output_dir) / "final"
    jsonl = root / "homonym_dictionary.jsonl"
    ordered = sorted(entries, key=lambda item: item.lemma)
    atomic_write_jsonl(jsonl, ordered)
    atomic_write_json(root / "homonym_dictionary.json", {"entries": [item.model_dump(mode="json") for item in ordered]})
