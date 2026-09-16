from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from homonym_pipeline.embeddings.client import EmbeddingClient
from homonym_pipeline.models import CandidateExample, Gloss


@dataclass
class RerankingResult:
    by_sense: dict[str, list[CandidateExample]]
    model: str | None
    embedding_calls: list[dict[str, Any]]
    pairs_scored: int
    candidates_selected: int


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left or not right:
        raise ValueError("cosine vectors must have the same non-zero dimension")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def rank_candidates_by_gloss(
    glosses: list[Gloss],
    candidates: list[CandidateExample],
    embedder: EmbeddingClient | None,
    *,
    top_k: int,
    model: str | None = None,
    status: str = "embedded",
) -> RerankingResult:
    """Rank one shared GRAC pool independently for every gloss.

    Ties are resolved by the original GRAC result order and then example_id,
    making the selected top-k set deterministic.
    """
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not candidates or not glosses or top_k == 0:
        return RerankingResult(
            by_sense={gloss.sense_id: [] for gloss in glosses}, model=model,
            embedding_calls=[], pairs_scored=0, candidates_selected=0,
        )

    if embedder is None or getattr(embedder, "dry_run", False):
        # Dry-run and direct library callers retain a deterministic corpus-order
        # preview without making an embedding request.
        ranked_by_sense: dict[str, list[CandidateExample]] = {}
        for gloss in glosses:
            ranked_by_sense[gloss.sense_id] = [
                _ranked_copy(candidate, gloss, rank + 1, None, len(candidates), top_k, status, model)
                for rank, candidate in enumerate(candidates[:top_k])
            ]
        return RerankingResult(
            by_sense=ranked_by_sense, model=model, embedding_calls=[],
            pairs_scored=0, candidates_selected=sum(len(items) for items in ranked_by_sense.values()),
        )

    sentence_batch = embedder.embed_texts([candidate.sentence for candidate in candidates])
    gloss_batch = embedder.embed_texts([gloss.gloss for gloss in glosses])
    if len(sentence_batch.vectors) != len(candidates) or len(gloss_batch.vectors) != len(glosses):
        raise ValueError("embedding response count does not match input count")

    ranked_by_sense = {}
    for gloss, gloss_vector in zip(glosses, gloss_batch.vectors):
        scored = [
            (cosine_similarity(gloss_vector, sentence_vector), index, candidate)
            for index, (candidate, sentence_vector) in enumerate(zip(candidates, sentence_batch.vectors))
        ]
        scored.sort(key=lambda item: (-item[0], item[1], item[2].example_id))
        ranked_by_sense[gloss.sense_id] = [
            _ranked_copy(candidate, gloss, rank + 1, score, len(candidates), top_k, status,
                         embedder.config.model)
            for rank, (score, _, candidate) in enumerate(scored[:top_k])
        ]
    return RerankingResult(
        by_sense=ranked_by_sense,
        model=embedder.config.model,
        embedding_calls=sentence_batch.calls + gloss_batch.calls,
        pairs_scored=len(candidates) * len(glosses),
        candidates_selected=sum(len(items) for items in ranked_by_sense.values()),
    )


def _ranked_copy(candidate: CandidateExample, gloss: Gloss, rank: int,
                 score: float | None, pool_size: int, top_k: int, status: str,
                 model: str | None) -> CandidateExample:
    copy = candidate.model_copy(deep=True)
    copy.source_metadata.update({
        "embedding_status": status,
        "embedding_gloss_sense_id": gloss.sense_id,
        "embedding_gloss": gloss.gloss,
        "embedding_rank": rank,
        "embedding_candidate_pool_size": pool_size,
        "embedding_top_k": top_k,
    })
    if model is not None:
        copy.source_metadata["embedding_model"] = model
    if score is not None:
        copy.source_metadata["embedding_cosine_similarity"] = score
    return copy
