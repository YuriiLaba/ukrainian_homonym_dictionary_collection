from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: BaseModel | dict[str, Any]) -> None:
    ensure_parent(path)
    payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def latest_by_cache_key(path: Path) -> dict[str, dict[str, Any]]:
    return {
        record["cache_key"]: record
        for record in read_jsonl(path)
        if record.get("cache_key")
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, records: Iterable[BaseModel | dict[str, Any]]) -> None:
    ensure_parent(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_many(path: Path, records: Iterable[BaseModel | dict[str, Any]]) -> None:
    for record in records:
        append_jsonl(path, record)
