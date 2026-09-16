from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).strip().split())


def content_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sense_id(lemma: str, gloss: str) -> str:
    return "s_" + content_hash({"lemma": normalize_text(lemma), "gloss": normalize_text(gloss)})[:16]


def example_id(lemma: str, sentence: str, source_id: str | int | None = None) -> str:
    return "e_" + content_hash({"lemma": lemma, "sentence": sentence, "source_id": source_id})[:20]
