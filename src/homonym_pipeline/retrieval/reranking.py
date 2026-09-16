from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from homonym_pipeline.embeddings.client import EmbeddingClient
from homonym_pipeline.hashing import normalize_text
from homonym_pipeline.models import CandidateExample, Gloss

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in minimal installations
    np = None


@dataclass
class RerankingResult:
    by_sense: dict[str, list[CandidateExample]]
    model: str | None
    embedding_calls: list[dict[str, Any]]
    pairs_scored: int
    candidates_selected: int
    exact_duplicates_removed: int = 0
    mmr_enabled: bool = False
    mmr_lambda: float | None = None


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
    mmr_enabled: bool = False,
    mmr_lambda: float = 0.7,
    exact_deduplication: bool = True,
) -> RerankingResult:
    """Rank one shared GRAC pool independently for every gloss.

    Ties are resolved by the original GRAC result order and then example_id,
    making the selected top-k set deterministic.
    """
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError("mmr_lambda must be between 0 and 1")
    if not candidates or not glosses or top_k == 0:
        return RerankingResult(
            by_sense={gloss.sense_id: [] for gloss in glosses}, model=model,
            embedding_calls=[], pairs_scored=0, candidates_selected=0,
            exact_duplicates_removed=0, mmr_enabled=False, mmr_lambda=None,
        )

    ranked_candidates, exact_duplicates_removed = _deduplicate_candidates(
        candidates, enabled=exact_deduplication
    )

    if embedder is None or getattr(embedder, "dry_run", False):
        # Dry-run and direct library callers retain a deterministic corpus-order
        # preview without making an embedding request.
        ranked_by_sense: dict[str, list[CandidateExample]] = {}
        for gloss in glosses:
            ranked_by_sense[gloss.sense_id] = [
                _ranked_copy(
                    candidate, gloss, rank + 1, None, len(ranked_candidates), top_k, status, model,
                    mmr_enabled=False, mmr_lambda=None, mmr_score=None, mmr_redundancy=None,
                    mmr_rank=None,
                )
                for rank, candidate in enumerate(ranked_candidates[:top_k])
            ]
        return RerankingResult(
            by_sense=ranked_by_sense, model=model, embedding_calls=[],
            pairs_scored=0, candidates_selected=sum(len(items) for items in ranked_by_sense.values()),
            exact_duplicates_removed=exact_duplicates_removed, mmr_enabled=False, mmr_lambda=None,
        )

    sentence_batch = embedder.embed_texts([candidate.sentence for candidate in ranked_candidates])
    gloss_batch = embedder.embed_texts([gloss.gloss for gloss in glosses])
    if len(sentence_batch.vectors) != len(ranked_candidates) or len(gloss_batch.vectors) != len(glosses):
        raise ValueError("embedding response count does not match input count")

    normalized_sentence_vectors = _normalized_matrix(sentence_batch.vectors)
    ranked_by_sense = {}
    for gloss, gloss_vector in zip(glosses, gloss_batch.vectors):
        relevance_scores = _cosine_scores(gloss_vector, sentence_batch.vectors, normalized_sentence_vectors)
        scored = [
            (relevance_scores[index], index, candidate)
            for index, candidate in enumerate(ranked_candidates)
        ]
        scored.sort(key=lambda item: (-item[0], item[1], item[2].example_id))
        relevance_rank = {index: rank + 1 for rank, (_, index, _) in enumerate(scored)}
        if mmr_enabled:
            selected = _select_mmr(
                scored, sentence_batch.vectors, top_k, mmr_lambda, normalized_sentence_vectors
            )
        else:
            selected = [
                (score, index, candidate, score, 0.0, rank + 1)
                for rank, (score, index, candidate) in enumerate(scored[:top_k])
            ]
        ranked_by_sense[gloss.sense_id] = [
            _ranked_copy(
                candidate, gloss, relevance_rank[index], score, len(ranked_candidates), top_k, status,
                embedder.config.model, mmr_enabled=mmr_enabled, mmr_lambda=mmr_lambda if mmr_enabled else None,
                mmr_score=mmr_score, mmr_redundancy=redundancy, mmr_rank=mmr_rank,
            )
            for score, index, candidate, mmr_score, redundancy, mmr_rank in selected
        ]
    return RerankingResult(
        by_sense=ranked_by_sense,
        model=embedder.config.model,
        embedding_calls=sentence_batch.calls + gloss_batch.calls,
        pairs_scored=len(ranked_candidates) * len(glosses),
        candidates_selected=sum(len(items) for items in ranked_by_sense.values()),
        exact_duplicates_removed=exact_duplicates_removed,
        mmr_enabled=mmr_enabled,
        mmr_lambda=mmr_lambda if mmr_enabled else None,
    )


def _deduplicate_candidates(
    candidates: list[CandidateExample], *, enabled: bool
) -> tuple[list[CandidateExample], int]:
    if not enabled:
        return list(candidates), 0
    unique: list[CandidateExample] = []
    by_text: dict[str, CandidateExample] = {}
    duplicate_ids: dict[str, list[str]] = {}
    for candidate in candidates:
        key = normalize_text(candidate.sentence)
        existing = by_text.get(key)
        if existing is None:
            copy = candidate.model_copy(deep=True)
            by_text[key] = copy
            unique.append(copy)
            duplicate_ids[key] = []
        else:
            duplicate_ids[key].append(candidate.example_id)
    for candidate in unique:
        duplicates = duplicate_ids[normalize_text(candidate.sentence)]
        if duplicates:
            candidate.source_metadata["exact_duplicate_example_ids"] = duplicates
            candidate.source_metadata["exact_duplicate_count"] = len(duplicates)
    return unique, len(candidates) - len(unique)


def _normalized_matrix(vectors: list[list[float]]) -> Any:
    if np is None:
        return None
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(vectors):
        raise ValueError("embedding vectors must form a rectangular matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _cosine_scores(
    query: list[float], vectors: list[list[float]], normalized_vectors: Any = None
) -> list[float]:
    if normalized_vectors is not None and np is not None:
        query_array = np.asarray(query, dtype=float)
        if query_array.ndim != 1 or query_array.shape[0] != normalized_vectors.shape[1]:
            raise ValueError("cosine vectors must have the same dimension")
        norm = np.linalg.norm(query_array)
        if norm == 0.0:
            return [0.0] * len(vectors)
        return (normalized_vectors @ (query_array / norm)).tolist()
    return [cosine_similarity(query, vector) for vector in vectors]


def _select_mmr(
    scored: list[tuple[float, int, CandidateExample]],
    sentence_vectors: list[list[float]],
    top_k: int,
    mmr_lambda: float,
    normalized_sentence_vectors: Any = None,
) -> list[tuple[float, int, CandidateExample, float, float, int]]:
    """Select relevant, non-redundant candidates with deterministic tie-breaking."""
    if not scored:
        return []
    selected: list[tuple[float, int, CandidateExample, float, float, int]] = []
    first_score, first_index, first_candidate = scored[0]
    selected.append((first_score, first_index, first_candidate, mmr_lambda * first_score, 0.0, 1))
    selected_indices = {first_index}
    relevance_by_index = {index: relevance for relevance, index, _ in scored}
    normalized = (
        normalized_sentence_vectors
        if normalized_sentence_vectors is not None
        else _normalized_matrix(sentence_vectors)
    )
    while len(selected) < min(top_k, len(scored)):
        if normalized is not None:
            selected_array = normalized[list(selected_indices)]
            redundancies = normalized @ selected_array.T
            redundancies = redundancies.max(axis=1)
            best_index = min(
                (index for _, index, _ in scored if index not in selected_indices),
                key=lambda index: (
                    -float(
                        mmr_lambda * relevance_by_index[index]
                        - (1.0 - mmr_lambda) * redundancies[index]
                    ),
                    index,
                ),
            )
            redundancy = float(redundancies[best_index])
            mmr_score = mmr_lambda * relevance_by_index[best_index] - (1.0 - mmr_lambda) * redundancy
            candidate = next(candidate for _, index, candidate in scored if index == best_index)
            selected_indices.add(best_index)
            selected.append((relevance_by_index[best_index], best_index, candidate,
                             mmr_score, redundancy, len(selected) + 1))
            continue
        best: tuple[float, float, int, int, CandidateExample] | None = None
        for relevance, index, candidate in scored:
            if index in selected_indices:
                continue
            redundancy = max(
                cosine_similarity(sentence_vectors[index], sentence_vectors[selected_index])
                for selected_index in selected_indices
            )
            mmr_score = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
            proposal = (mmr_score, relevance, -index, index, candidate)
            if best is None or proposal > best:
                best = proposal
        if best is None:
            break
        mmr_score, relevance, _, index, candidate = best
        redundancy = max(
            cosine_similarity(sentence_vectors[index], sentence_vectors[selected_index])
            for selected_index in selected_indices
        )
        selected_indices.add(index)
        selected.append((relevance, index, candidate, mmr_score, redundancy, len(selected) + 1))
    return selected


def _ranked_copy(candidate: CandidateExample, gloss: Gloss, rank: int,
                 score: float | None, pool_size: int, top_k: int, status: str,
                 model: str | None, *, mmr_enabled: bool, mmr_lambda: float | None,
                 mmr_score: float | None, mmr_redundancy: float | None,
                 mmr_rank: int | None) -> CandidateExample:
    copy = candidate.model_copy(deep=True)
    copy.source_metadata.update({
        "embedding_status": status,
        "embedding_gloss_sense_id": gloss.sense_id,
        "embedding_gloss": gloss.gloss,
        "embedding_rank": rank,
        "embedding_candidate_pool_size": pool_size,
        "embedding_top_k": top_k,
        "embedding_mmr_enabled": mmr_enabled,
    })
    if model is not None:
        copy.source_metadata["embedding_model"] = model
    if score is not None:
        copy.source_metadata["embedding_cosine_similarity"] = score
    if mmr_lambda is not None:
        copy.source_metadata["embedding_mmr_lambda"] = mmr_lambda
    if mmr_score is not None:
        copy.source_metadata["embedding_mmr_score"] = mmr_score
    if mmr_redundancy is not None:
        copy.source_metadata["embedding_mmr_redundancy"] = mmr_redundancy
    if mmr_rank is not None:
        copy.source_metadata["embedding_mmr_rank"] = mmr_rank
    return copy
