"""Export the final dictionary in the published Hugging Face row schema.

The research output keeps provenance and validation metadata.  This adapter is
intentionally lossy: the Hugging Face dataset schema contains only ``lemma``,
``gloss`` and ``examples``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from homonym_pipeline.models import FinalLemmaEntry
from homonym_pipeline.storage import atomic_write_json, atomic_write_jsonl


class HuggingFaceRow(TypedDict):
    lemma: str
    gloss: list[str]
    examples: list[str]


def to_huggingface_rows(entries: list[FinalLemmaEntry]) -> list[HuggingFaceRow]:
    """Convert final senses to exactly the dataset's three-field row shape."""
    rows: list[HuggingFaceRow] = []
    for entry in sorted(entries, key=lambda item: item.lemma):
        for sense in sorted(entry.glosses, key=lambda item: item.sense_id):
            rows.append(
                {
                    "lemma": entry.lemma,
                    "gloss": [sense.gloss],
                    "examples": [
                        example.sentence
                        for example in sense.examples
                        if example.accepted and example.sense_id == sense.sense_id
                    ],
                }
            )
    return rows


def write_huggingface(entries: list[FinalLemmaEntry], output_dir: str | Path) -> None:
    """Write JSON and JSONL files loadable as the HF dataset's train rows."""
    root = Path(output_dir) / "final"
    rows = to_huggingface_rows(entries)
    atomic_write_json(root / "huggingface.json", rows)
    atomic_write_jsonl(root / "huggingface.jsonl", rows)
