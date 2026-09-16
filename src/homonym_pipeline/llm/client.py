from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from homonym_pipeline.config import LLMConfig
from homonym_pipeline.models import LLMCallRecord


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


@dataclass
class StructuredResult:
    parsed: BaseModel
    call_record: LLMCallRecord


class LLMClient:
    """Small OpenAI Responses API adapter with strict JSON-schema output."""

    def __init__(self, config: LLMConfig, client: Any | None = None, dry_run: bool = False):
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

    def structured(
        self,
        *,
        model: str,
        prompt_version: str,
        system: str,
        user: str,
        schema: type[ResponseModel],
    ) -> StructuredResult:
        if self.dry_run:
            raise RuntimeError("LLM call requested in dry-run mode")
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                request = {
                    "model": model,
                    "instructions": system,
                    "input": user,
                    "reasoning": {"effort": self.config.reasoning_effort},
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": schema.__name__.lower(),
                            "strict": True,
                            "schema": schema.model_json_schema(),
                        }
                    },
                }
                if self.config.temperature is not None:
                    request["temperature"] = self.config.temperature
                response = self.client.responses.create(**request)
                raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
                output_text = getattr(response, "output_text", None)
                if not output_text:
                    raise ValueError("OpenAI response contained no output_text")
                parsed = schema.model_validate(json.loads(output_text))
                usage = raw.get("usage") or {}
                record = LLMCallRecord(
                    request_id=raw.get("id"),
                    model=model,
                    prompt_version=prompt_version,
                    temperature=self.config.temperature,
                    reasoning_effort=self.config.reasoning_effort,
                    usage=usage,
                    raw_response=raw,
                )
                return StructuredResult(parsed=parsed, call_record=record)
            except Exception as error:  # retry policy intentionally covers SDK/JSON/Pydantic failures
                last_error = error
                if attempt >= self.config.max_retries:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenAI structured request failed after retries: {last_error}") from last_error
