from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from homonym_pipeline.hashing import normalize_text, sense_id
from homonym_pipeline.models import Gloss, LemmaEntry, SourceReference


def _load_records(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def parse_dictionary(path: str | Path) -> list[LemmaEntry]:
    """Parse the repository's dictionary JSON without modifying the raw source."""
    source_path = Path(path)
    payload = _load_records(source_path)
    entries = payload.get("entries", []) if isinstance(payload, dict) else payload
    grouped: dict[str, LemmaEntry] = {}
    seen: dict[tuple[str, str], Gloss] = {}

    for entry in entries:
        lemma = normalize_text(str(entry.get("lemma_normalized") or entry.get("lemma") or ""))
        if not lemma:
            continue
        target = grouped.setdefault(lemma, LemmaEntry(lemma=lemma))
        for source_sense in entry.get("senses", []):
            definition = normalize_text(str(source_sense.get("definition") or ""))
            if not definition:
                continue
            key = (lemma, definition)
            reference = SourceReference(
                source="dictionary",
                locator=f"entry:{entry.get('entry_id')};sense:{source_sense.get('sense_id')}",
                document_id=str(entry.get("entry_id")) if entry.get("entry_id") is not None else None,
                excerpt=source_sense.get("source_text"),
                **{"source_file": str(source_path), "source_paragraph": source_sense.get("source_paragraph")},
            )
            if key in seen:
                seen[key].source_references.append(reference)
                continue
            gloss = Gloss(
                sense_id=sense_id(lemma, definition),
                lemma=lemma,
                gloss=definition,
                source="dictionary",
                source_references=[reference],
            )
            seen[key] = gloss
            target.glosses.append(gloss)
    return sorted(grouped.values(), key=lambda item: item.lemma)
