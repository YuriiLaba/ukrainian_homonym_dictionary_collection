from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable

from homonym_pipeline.hashing import sense_id
from homonym_pipeline.llm.client import LLMClient
from homonym_pipeline.models import Gloss, GlossAugmentationResponse, GlossDecision, SourceReference, WikipediaCandidate


PROMPT_VERSION = "wikipedia_gloss_augmentation_v2"


SYSTEM_PROMPT = """Ти укладач словника українських омонімів. Працюй лише з наданими даними.

Для кожного кандидата прийми одне рішення:
- keep: зберегти кандидата як окреме значення;
- refine: зберегти значення, але нормалізувати формулювання;
- merge: об'єднати кандидата з семантично еквівалентним кандидатом;
- merge_uncertain: позначити можливу еквівалентність, але не об'єднувати автоматично;
- remove: вилучити явно непридатного кандидата;
- add: додати нове значення, яке прямо підтверджене текстом одного або кількох
  наданих Wikipedia-кандидатів.

Об'єднуй лише семантично еквівалентні значення. Для merge вкажи
merge_into_candidate_id — ідентифікатор канонічного кандидата. Для merge_uncertain
також вкажи можливий merge_into_candidate_id, але не вважай значення еквівалентними.
Для add candidate_id має бути null, а evidence_candidate_ids має містити лише
ідентифікатори наданих кандидатів, які прямо підтверджують нове значення. Не додавай
значення на основі загальних знань або припущень. Не вигадуй джерел. Поверни рішення
лише за заданою JSON-схемою."""


def _decision_references(
    decision: GlossDecision,
    candidate_map: dict[str, WikipediaCandidate],
) -> list[SourceReference]:
    """Collect all source references supporting a decision, without duplicates."""
    candidate_ids = list(decision.evidence_candidate_ids)
    if decision.candidate_id and decision.candidate_id in candidate_map:
        candidate_ids.append(decision.candidate_id)

    references: list[SourceReference] = []
    seen: set[str] = set()
    for candidate_id in candidate_ids:
        candidate = candidate_map.get(candidate_id)
        if candidate is None:
            continue
        reference = candidate.source_reference
        key = json.dumps(reference.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            references.append(reference)
    return references


def _resolve_merge_target(
    candidate_id: str,
    confirmed_merges: dict[str, str],
) -> str:
    """Resolve chained confirmed merges deterministically and safely."""
    current = candidate_id
    visited: set[str] = set()
    while current in confirmed_merges and current not in visited:
        visited.add(current)
        current = confirmed_merges[current]
    return current


def _decisions_to_glosses(
    lemma: str,
    decisions: Iterable[GlossDecision],
    candidates: list[WikipediaCandidate],
) -> list[Gloss]:
    """Convert Terra decisions into glosses, consolidating confirmed merges."""
    decisions = list(decisions)
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    decision_by_candidate = {
        decision.candidate_id: decision
        for decision in decisions
        if decision.candidate_id is not None
    }

    # Only a confident ``merge`` creates a union.  An uncertain merge remains a
    # separate gloss and is marked so that a researcher can review it later.
    confirmed_merges = {
        decision.candidate_id: decision.merge_into_candidate_id
        for decision in decisions
        if decision.action == "merge"
        and decision.candidate_id is not None
        and decision.merge_into_candidate_id is not None
        and decision.merge_into_candidate_id in decision_by_candidate
        and decision_by_candidate[decision.merge_into_candidate_id].action != "remove"
    }

    groups: dict[str, list[tuple[int, GlossDecision]]] = defaultdict(list)
    for index, decision in enumerate(decisions):
        if decision.action == "remove":
            continue
        if decision.action == "add" and not any(
            candidate_id in candidate_map for candidate_id in decision.evidence_candidate_ids
        ):
            # An added sense without a valid Wikipedia evidence link is not
            # allowed into the inventory, even if the LLM returned valid JSON.
            continue
        if decision.action == "merge" and decision.candidate_id in confirmed_merges:
            root = _resolve_merge_target(decision.candidate_id, confirmed_merges)
        else:
            root = decision.candidate_id or f"decision_{index}"
        groups[root].append((index, decision))

    plans: list[tuple[str, list[tuple[int, GlossDecision]], GlossDecision, bool]] = []
    for root, group in groups.items():
        non_merge = [item for item in group if item[1].action != "merge"]
        if non_merge:
            canonical = next(
                (item[1] for item in non_merge if item[1].candidate_id == root),
                non_merge[0][1],
            )
        else:
            canonical = group[0][1]
        was_merged = len(group) > 1 and any(item[1].action == "merge" for item in group)
        plans.append((root, group, canonical, was_merged))

    # Build this before the output objects so uncertain merges can point to the
    # sense that they were compared against.
    candidate_to_sense_id: dict[str, str] = {}
    for _, group, canonical, _ in plans:
        target_sense_id = sense_id(lemma, canonical.gloss)
        for _, decision in group:
            if decision.candidate_id:
                candidate_to_sense_id[decision.candidate_id] = target_sense_id

    glosses: list[Gloss] = []
    for _, group, canonical, was_merged in plans:
        source = (
            "llm_added"
            if canonical.action == "add"
            else "wikipedia"
            if canonical.action == "keep"
            else "llm_refined"
        )
        original = canonical.original_gloss
        if source == "llm_refined" and not original and canonical.candidate_id in candidate_map:
            original = candidate_map[canonical.candidate_id].extract

        references: list[SourceReference] = []
        seen_references: set[str] = set()
        for _, decision in group:
            for reference in _decision_references(decision, candidate_map):
                key = json.dumps(reference.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
                if key not in seen_references:
                    seen_references.add(key)
                    references.append(reference)

        merge_status = "merged" if was_merged else (
            "merge_uncertain"
            if canonical.action in {"merge", "merge_uncertain"}
            else "canonical"
        )
        merged_into = None
        if canonical.action == "merge_uncertain" and canonical.merge_into_candidate_id:
            merged_into = candidate_to_sense_id.get(canonical.merge_into_candidate_id)

        glosses.append(
            Gloss(
                sense_id=sense_id(lemma, canonical.gloss),
                lemma=lemma,
                gloss=canonical.gloss,
                original_gloss=original,
                source=source,
                source_references=references,
                merge_status=merge_status,
                merged_into_sense_id=merged_into,
                llm_reason=canonical.reason,
            )
        )
    return glosses


def augment_glosses(lemma: str, candidates: list[WikipediaCandidate], client: LLMClient) -> tuple[list[Gloss], object]:
    payload = {"lemma": lemma, "candidates": [candidate.model_dump(mode="json") for candidate in candidates]}
    result = client.structured(model=client.config.model_gloss, prompt_version=PROMPT_VERSION,
                               system=SYSTEM_PROMPT, user=json.dumps(payload, ensure_ascii=False),
                               schema=GlossAugmentationResponse)
    return _decisions_to_glosses(lemma, result.parsed.decisions, candidates), result.call_record
