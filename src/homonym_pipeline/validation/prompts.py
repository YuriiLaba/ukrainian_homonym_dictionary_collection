from __future__ import annotations

import json

from homonym_pipeline.models import CandidateExample, Gloss


PROMPT_VERSION = "grac_example_assignment_v1"
GLOSS_PROMPT_VERSION = "grac_gloss_example_validation_v1"

SYSTEM_PROMPT = """Ти перевіряєш приклади для словника українських омонімів.
Для кожного прикладу визнач, чи придатний він як словникова ілюстрація, і якщо так,
признач рівно один наданий sense_id. Не вигадуй значень, не переписуй речення і не
створюй прикладів. Якщо відповідного значення немає або контекст недостатній, accepted=false
і sense_id=null. Поверни лише JSON за заданою схемою. confidence — це евристична оцінка,
а не калібрована ймовірність."""

GLOSS_SYSTEM_PROMPT = """Ти перевіряєш приклади для одного конкретного значення
українського омоніма. Для кожного прикладу визнач, чи придатний він як словникова
ілюстрація саме для наданого gloss. Не вигадуй значень, не переписуй речення і не
створюй прикладів. Якщо приклад відповідає іншому значенню, контекст недостатній,
речення пошкоджене або вживання непридатне, accepted=false і sense_id=null.
Якщо accepted=true, використовуй тільки наданий sense_id. Поверни лише JSON за
заданою схемою. confidence — це евристична оцінка, а не калібрована ймовірність."""


def assignment_prompt(lemma: str, glosses: list[Gloss], examples: list[CandidateExample]) -> str:
    return json.dumps({
        "lemma": lemma,
        "glosses": [{"sense_id": item.sense_id, "gloss": item.gloss} for item in glosses],
        "examples": [{"example_id": item.example_id, "sentence": item.sentence} for item in examples],
    }, ensure_ascii=False)


def gloss_assignment_prompt(lemma: str, gloss: Gloss, examples: list[CandidateExample]) -> str:
    return json.dumps({
        "lemma": lemma,
        "gloss": {"sense_id": gloss.sense_id, "gloss": gloss.gloss},
        "examples": [{"example_id": item.example_id, "sentence": item.sentence} for item in examples],
    }, ensure_ascii=False)
