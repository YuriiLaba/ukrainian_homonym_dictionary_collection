from __future__ import annotations

import json

from homonym_pipeline.llm.client import LLMClient
from homonym_pipeline.models import AssignmentResponse, CandidateExample, Gloss, ValidatedExample
from homonym_pipeline.validation.prompts import (
    GLOSS_PROMPT_VERSION,
    GLOSS_SYSTEM_PROMPT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    assignment_prompt,
    gloss_assignment_prompt,
)


def _validated_assignments(lemma: str, glosses: list[Gloss], examples: list[CandidateExample], parsed: AssignmentResponse) -> list[ValidatedExample]:
    by_id = {example.example_id: example for example in examples}
    valid_sense_ids = {gloss.sense_id for gloss in glosses}
    output: list[ValidatedExample] = []
    for assignment in parsed.assignments:
        example = by_id.get(assignment.example_id)
        if example is None:
            continue
        sense_id = assignment.sense_id if assignment.sense_id in valid_sense_ids else None
        accepted = assignment.accepted and sense_id is not None
        reason = assignment.reason
        if assignment.accepted and sense_id is None:
            accepted = False
            reason = "invalid_sense_id_returned_by_llm"
        output.append(ValidatedExample(example_id=example.example_id, lemma=lemma, sentence=example.sentence,
                                       sense_id=sense_id if accepted else None, accepted=accepted,
                                       model_confidence=assignment.model_confidence,
                                       validation_reason=reason, source_metadata=example.source_metadata,
                                       query=example.query, retrieved_at=example.retrieved_at))
    assigned = {item.example_id for item in output}
    for example in examples:
        if example.example_id not in assigned:
            output.append(ValidatedExample(example_id=example.example_id, lemma=lemma, sentence=example.sentence,
                                           accepted=False, model_confidence=0.0,
                                           validation_reason="missing_assignment", source_metadata=example.source_metadata,
                                           query=example.query, retrieved_at=example.retrieved_at))
    return output


def validate_batch(lemma: str, glosses: list[Gloss], examples: list[CandidateExample], client: LLMClient) -> tuple[list[ValidatedExample], object]:
    result = client.structured(model=client.config.model_validation, prompt_version=PROMPT_VERSION,
                               system=SYSTEM_PROMPT, user=assignment_prompt(lemma, glosses, examples),
                               schema=AssignmentResponse)
    return _validated_assignments(lemma, glosses, examples, result.parsed), result.call_record


def validate_gloss_batch(lemma: str, gloss: Gloss, examples: list[CandidateExample], client: LLMClient) -> tuple[list[ValidatedExample], object]:
    """Validate examples against exactly one gloss."""
    result = client.structured(model=client.config.model_validation,
                               prompt_version=GLOSS_PROMPT_VERSION,
                               system=GLOSS_SYSTEM_PROMPT,
                               user=gloss_assignment_prompt(lemma, gloss, examples),
                               schema=AssignmentResponse)
    return _validated_assignments(lemma, [gloss], examples, result.parsed), result.call_record
