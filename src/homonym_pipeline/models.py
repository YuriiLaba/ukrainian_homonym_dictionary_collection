from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


GlossSource = Literal["dictionary", "wikipedia", "llm_added", "llm_refined"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str
    url: str | None = None
    title: str | None = None
    document_id: str | int | None = None
    revision_id: str | int | None = None
    locator: str | None = None
    retrieved_at: datetime | None = None
    excerpt: str | None = None
    excerpt_sha256: str | None = None


class Gloss(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sense_id: str
    lemma: str
    gloss: str
    original_gloss: str | None = None
    source: GlossSource
    source_references: list[SourceReference] = Field(default_factory=list)
    merge_status: Literal["canonical", "merged", "merge_uncertain"] = "canonical"
    merged_into_sense_id: str | None = None
    llm_reason: str | None = None

    @field_validator("gloss", "lemma")
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class LemmaEntry(BaseModel):
    lemma: str
    glosses: list[Gloss] = Field(default_factory=list)


class CandidateExample(BaseModel):
    example_id: str
    lemma: str
    sentence: str
    source: Literal["grac"] = "grac"
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    query: str | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)


class LLMAssignment(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    example_id: str
    accepted: bool
    sense_id: str | None
    model_confidence: float = Field(ge=0.0, le=1.0, alias="confidence")
    reason: str


class AssignmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignments: list[LLMAssignment]


class ValidatedExample(BaseModel):
    example_id: str
    lemma: str
    sentence: str
    sense_id: str | None = None
    accepted: bool
    model_confidence: float
    validation_reason: str
    source: Literal["grac"] = "grac"
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    query: str | None = None
    retrieved_at: datetime | None = None


class WikipediaCandidate(BaseModel):
    candidate_id: str
    lemma: str
    title: str
    page_id: str | int | None = None
    revision_id: str | int | None = None
    extract: str
    url: str | None = None
    source_reference: SourceReference
    raw_response: dict[str, Any] = Field(default_factory=dict)


class GlossDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["keep", "refine", "add", "remove", "merge", "merge_uncertain"]
    candidate_id: str | None
    gloss: str
    original_gloss: str | None
    evidence_candidate_ids: list[str]
    merge_into_candidate_id: str | None
    reason: str


class GlossAugmentationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[GlossDecision]


class FinalSenseEntry(BaseModel):
    sense_id: str
    lemma: str
    gloss: str
    gloss_source: GlossSource
    original_gloss: str | None = None
    source_references: list[SourceReference] = Field(default_factory=list)
    merge_status: Literal["canonical", "merged", "merge_uncertain"] = "canonical"
    merged_into_sense_id: str | None = None
    llm_reason: str | None = None
    examples: list[ValidatedExample] = Field(default_factory=list)


class FinalLemmaEntry(BaseModel):
    lemma: str
    glosses: list[FinalSenseEntry] = Field(default_factory=list)


class LLMCallRecord(BaseModel):
    run_id: str | None = None
    workflow: str | None = None
    stage: str | None = None
    lemma: str | None = None
    sense_id: str | None = None
    request_id: str | None = None
    model: str
    prompt_version: str
    temperature: float | None = None
    reasoning_effort: str | None = None
    requested_at: datetime = Field(default_factory=utc_now)
    elapsed_seconds: float | None = None
    attempts: int | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class FailureRecord(BaseModel):
    stage: str
    lemma: str | None = None
    cache_key: str | None = None
    error_type: str
    message: str
    retry_count: int = 0
    request_id: str | None = None
    occurred_at: datetime = Field(default_factory=utc_now)


class LemmaAuditRecord(BaseModel):
    """Per-lemma audit metrics for reproducible pipeline analysis."""

    run_id: str
    workflow: str
    lemma: str
    status: Literal["success", "failed"]
    elapsed_seconds: float = 0.0
    grac_elapsed_seconds: float = 0.0
    embedding_elapsed_seconds: float = 0.0
    validation_elapsed_seconds: float = 0.0
    finalization_elapsed_seconds: float = 0.0
    wikipedia_elapsed_seconds: float = 0.0
    gloss_augmentation_elapsed_seconds: float = 0.0
    wikipedia_cache_hit: bool = False
    terra_cache_hit: bool = False
    grac_total_sentence_hits: int = 0
    grac_pages_requested: int = 0
    grac_raw_rows_examined: int = 0
    grac_skipped_by_reason: dict[str, int] = Field(default_factory=dict)
    grac_duplicate_rows_skipped: int = 0
    grac_request_attempts: int = 0
    grac_retry_count: int = 0
    grac_cache_hit: bool = False
    input_glosses: int = 0
    wikipedia_candidates: int = 0
    terra_actions: dict[str, int] = Field(default_factory=dict)
    grac_candidates_retrieved: int = 0
    embedding_model: str | None = None
    embedding_calls: int = 0
    embedding_cache_hits: int = 0
    embedding_pairs_scored: int = 0
    embedding_candidates_selected: int = 0
    embedding_exact_duplicates_removed: int = 0
    embedding_mmr_enabled: bool = False
    embedding_mmr_lambda: float | None = None
    validation_batches: int = 0
    validation_cache_hits: int = 0
    llm_assignments: int = 0
    validation_model_accepted: int = 0
    validation_model_rejected: int = 0
    validation_confidence_sum: float = 0.0
    validation_confidence_count: int = 0
    accepted_before_final_cap: int = 0
    rejected_by_reason: dict[str, int] = Field(default_factory=dict)
    final_glosses: int = 0
    final_glosses_with_examples: int = 0
    final_examples: int = 0
    error_type: str | None = None
    error_message: str | None = None
