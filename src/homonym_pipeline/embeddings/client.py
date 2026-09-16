from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from homonym_pipeline.config import EmbeddingConfig


class EmbeddingError(RuntimeError):
    """An embedding request failed or returned an incompatible response."""


@dataclass
class EmbeddingBatch:
    vectors: list[list[float]]
    calls: list[dict[str, Any]]


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


class EmbeddingClient:
    """Small OpenAI embeddings adapter with bounded retries.

    The pipeline caches complete per-lemma rankings. Keeping this client focused
    on API calls makes it straightforward to replace OpenAI with a local model.
    """

    def __init__(self, config: EmbeddingConfig, client: Any | None = None,
                 *, dry_run: bool = False):
        self.config = config
        self._client = client
        self.dry_run = dry_run

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise RuntimeError(f"Missing OpenAI API key in {self.config.api_key_env}")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        if self.dry_run:
            raise EmbeddingError("Embedding request attempted in dry-run mode")
        if not texts:
            return EmbeddingBatch(vectors=[], calls=[])
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("embedding texts must be non-empty strings")

        vectors: list[list[float]] = []
        calls: list[dict[str, Any]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start:start + self.config.batch_size]
            last_error: Exception | None = None
            for attempt in range(self.config.max_retries + 1):
                try:
                    request: dict[str, Any] = {
                        "model": self.config.model,
                        "input": batch,
                    }
                    if self.config.dimensions is not None:
                        request["dimensions"] = self.config.dimensions
                    response = self.client.embeddings.create(**request)
                    raw = (
                        response.model_dump(mode="json")
                        if hasattr(response, "model_dump")
                        else dict(response)
                    )
                    data = _value(response, "data", raw.get("data"))
                    if not isinstance(data, list) or len(data) != len(batch):
                        raise ValueError("OpenAI embedding response has unexpected data length")
                    ordered = sorted(data, key=lambda item: _value(item, "index", 0))
                    batch_vectors = []
                    for item in ordered:
                        vector = _value(item, "embedding")
                        if (
                            not isinstance(vector, list)
                            or not vector
                            or not all(isinstance(x, (int, float)) for x in vector)
                        ):
                            raise ValueError("OpenAI embedding response contains an invalid vector")
                        batch_vectors.append([float(x) for x in vector])
                    if len({len(vector) for vector in batch_vectors}) != 1:
                        raise ValueError("OpenAI embedding response contains inconsistent dimensions")
                    vectors.extend(batch_vectors)
                    calls.append({
                        "request_id": _value(response, "id", raw.get("id")),
                        "model": raw.get("model", self.config.model),
                        "requested_at": datetime.now(timezone.utc).isoformat(),
                        "input_count": len(batch),
                        "input_text_sha256": [
                            hashlib.sha256(text.encode("utf-8")).hexdigest()
                            for text in batch
                        ],
                        "dimensions": len(batch_vectors[0]),
                        "usage": raw.get("usage") or {},
                    })
                    last_error = None
                    break
                except Exception as error:
                    last_error = error
                    if attempt >= self.config.max_retries:
                        break
                    time.sleep(2 ** attempt)
            if last_error is not None:
                raise EmbeddingError(
                    f"OpenAI embedding request failed after retries: {last_error}"
                ) from last_error
        return EmbeddingBatch(vectors=vectors, calls=calls)
