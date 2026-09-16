from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model_gloss: str = "gpt-5.6-terra"
    model_validation: str = "gpt-5.6-luna"
    temperature: float = 0.0
    reasoning_effort: str = "medium"
    max_retries: int = 3
    api_key_env: str = "OPENAI_API_KEY"


class GracConfig(BaseModel):
    endpoint: str = "https://sketch.uacorpus.org/bonito/run.cgi/"
    corpus: str = "grac19"
    max_examples_per_lemma: int = Field(default=100, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    user_agent: str = "ukrainian-homonym-pipeline/0.1"
    page_size: int = Field(default=100, ge=1, le=1000)
    max_retries: int = Field(default=3, ge=0)
    poll_attempts: int = Field(default=15, ge=1)
    request_interval_seconds: float = Field(default=0.5, ge=0)
    max_sentence_tokens: int = Field(default=100, ge=1, le=100)
    seed: int | None = None


class ValidationConfig(BaseModel):
    min_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    batch_size: int = Field(default=20, gt=0)
    max_final_examples_per_sense: int = Field(default=5, ge=0)


class PipelineConfig(BaseModel):
    resume: bool = True
    wikipedia_search_fallback: bool = True
    allow_llm_added_senses: bool = True


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    grac: GracConfig = Field(default_factory=GracConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(raw)
