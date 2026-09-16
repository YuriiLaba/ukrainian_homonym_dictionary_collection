from __future__ import annotations

from pathlib import Path

from homonym_pipeline.hashing import normalize_text


def parse_lemma_list(path: str | Path) -> list[str]:
    values = {
        normalize_text(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if normalize_text(line) and not normalize_text(line).startswith("#")
    }
    return sorted(values)
