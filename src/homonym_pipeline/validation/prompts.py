from __future__ import annotations

import json

from homonym_pipeline.models import CandidateExample, Gloss


PROMPT_VERSION = "grac_example_assignment_v1"

SYSTEM_PROMPT = """Ти перевіряєш приклади для словника українських омонімів.
Для кожного прикладу визнач, чи придатний він як словникова ілюстрація, і якщо так,
признач рівно один наданий sense_id. Не вигадуй значень, не переписуй речення і не
створюй прикладів. Якщо відповідного значення немає або контекст недостатній, accepted=false
і sense_id=null. Поверни лише JSON за заданою схемою. confidence — це евристична оцінка,
а не калібрована ймовірність."""


def assignment_prompt(lemma: str, glosses: list[Gloss], examples: list[CandidateExample]) -> str:
    return json.dumps({
        "lemma": lemma,
        "glosses": [{"sense_id": item.sense_id, "gloss": item.gloss} for item in glosses],
        "examples": [{"example_id": item.example_id, "sentence": item.sentence} for item in examples],
    }, ensure_ascii=False)
