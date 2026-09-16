from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from homonym_pipeline.config import AppConfig
from homonym_pipeline.hashing import sense_id
from homonym_pipeline.inputs.dictionary import parse_dictionary
from homonym_pipeline.inputs.lemma_list import parse_lemma_list
from homonym_pipeline.llm.client import StructuredResult
from homonym_pipeline.models import AssignmentResponse, Gloss, GlossAugmentationResponse, GlossDecision, LemmaEntry, LLMAssignment
from homonym_pipeline.models import LLMCallRecord
from homonym_pipeline.models import FinalLemmaEntry, FinalSenseEntry, SourceReference, ValidatedExample, WikipediaCandidate
from homonym_pipeline.glosses.llm_augmentation import _decisions_to_glosses
from homonym_pipeline.output.huggingface import to_huggingface_rows, write_huggingface
from homonym_pipeline.pipeline.shared import run_shared
from homonym_pipeline.retrieval.grac import FixtureGracClient
from homonym_pipeline.validation.example_validator import validate_batch


class FakeLLM:
    def __init__(self, parsed):
        self.config = AppConfig().llm
        self.parsed = parsed

    def structured(self, **kwargs):
        from homonym_pipeline.models import LLMCallRecord
        return StructuredResult(parsed=self.parsed, call_record=LLMCallRecord(model=kwargs["model"], prompt_version=kwargs["prompt_version"], temperature=0))


def test_dictionary_parsing_uses_normalized_lemma_and_definition(tmp_path: Path):
    source = tmp_path / "dictionary.json"
    source.write_text(json.dumps({"entries": [{"entry_id": 1, "lemma": "áвтомат", "lemma_normalized": "автомат", "senses": [{"sense_id": 1, "definition": "  зброя  "}]}]}, ensure_ascii=False), encoding="utf-8")
    entries = parse_dictionary(source)
    assert entries[0].lemma == "автомат"
    assert entries[0].glosses[0].gloss == "зброя"
    assert entries[0].glosses[0].source == "dictionary"


def test_lemma_list_parsing(tmp_path: Path):
    path = tmp_path / "lemmas.txt"
    path.write_text("автомат\n# comment\n автомат \n", encoding="utf-8")
    assert parse_lemma_list(path) == ["автомат"]


def test_duplicate_sense_id_is_stable():
    assert sense_id("автомат", "зброя") == sense_id("автомат", "зброя")


def test_candidate_serialization():
    client = FixtureGracClient({"автомат": [{"sentence": "Він узяв автомат до рук."}]})
    candidate = client.retrieve_examples("автомат", 10)[0]
    assert candidate.example_id.startswith("e_")
    assert candidate.source == "grac"


def test_valid_assignment_and_invalid_sense_id_are_rejected():
    gloss = Gloss(sense_id="s1", lemma="автомат", gloss="зброя", source="dictionary")
    candidate = FixtureGracClient({"автомат": [{"sentence": "Він узяв автомат до рук.", "example_id": "e1"}]}).retrieve_examples("автомат", 10)[0]
    candidate.example_id = "e1"
    fake = FakeLLM(AssignmentResponse(assignments=[LLMAssignment(example_id="e1", accepted=True, sense_id="bad", confidence=0.95, reason="bad id")]))
    validated, _ = validate_batch("автомат", [gloss], [candidate], fake)
    assert validated[0].accepted is False
    assert validated[0].validation_reason == "invalid_sense_id_returned_by_llm"


def test_valid_assignment_is_accepted():
    gloss = Gloss(sense_id="s1", lemma="автомат", gloss="зброя", source="dictionary")
    candidate = FixtureGracClient({"автомат": [{"sentence": "Він узяв автомат до рук.", "example_id": "e1"}]}).retrieve_examples("автомат", 10)[0]
    candidate.example_id = "e1"
    fake = FakeLLM(AssignmentResponse(assignments=[LLMAssignment(example_id="e1", accepted=True, sense_id="s1", confidence=0.95, reason="відповідає значенню")]))
    validated, _ = validate_batch("автомат", [gloss], [candidate], fake)
    assert validated[0].accepted is True
    assert validated[0].sense_id == "s1"


def test_rejected_example_is_preserved():
    gloss = Gloss(sense_id="s1", lemma="автомат", gloss="зброя", source="dictionary")
    candidate = FixtureGracClient({"автомат": [{"sentence": "Контекст недостатній.", "example_id": "e1"}]}).retrieve_examples("автомат", 10)[0]
    candidate.example_id = "e1"
    fake = FakeLLM(AssignmentResponse(assignments=[LLMAssignment(example_id="e1", accepted=False, sense_id=None, confidence=0.55, reason="недостатній контекст")]))
    validated, _ = validate_batch("автомат", [gloss], [candidate], fake)
    assert validated[0].accepted is False
    assert validated[0].sense_id is None


def test_shared_pipeline_aggregates_and_resumes(tmp_path: Path, caplog):
    caplog.set_level(logging.INFO, logger="homonym_pipeline.pipeline.shared")
    config = AppConfig()
    config.validation.min_confidence = 0.8
    gloss = Gloss(sense_id="s1", lemma="автомат", gloss="зброя", source="dictionary")
    entries = [LemmaEntry(lemma="автомат", glosses=[gloss])]
    fixture = FixtureGracClient({"автомат": [{"sentence": "Він узяв автомат до рук.", "example_id": "e1"}]})
    fake = FakeLLM(AssignmentResponse(assignments=[LLMAssignment(example_id="e1", accepted=True, sense_id="s1", confidence=0.95, reason="ok")]))
    first = run_shared(entries, tmp_path, config, fixture, fake, resume=True)
    assert first[0].glosses[0].examples[0].example_id == "e1"
    assert "glosses processed: 1" in caplog.text
    assert "glosses with >=1 final example: 1" in caplog.text

    class FailingGrac(FixtureGracClient):
        def retrieve_examples(self, lemma, max_examples, seed=None):
            raise AssertionError("resume should use the cached candidate")

    second = run_shared(entries, tmp_path, config, FailingGrac(fixture.examples_by_lemma), fake, resume=True)
    assert second[0].glosses[0].examples[0].example_id == "e1"


class _FakeResponse:
    def __init__(self, text):
        self.id = "resp_test"
        self.output_text = text

    def model_dump(self, mode="json"):
        return {"id": self.id, "usage": {"input_tokens": 3, "output_tokens": 4}}


class _FakeResponses:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def create(self, **kwargs):
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return _FakeResponse(output)


class _FakeOpenAI:
    def __init__(self, outputs):
        self.responses = _FakeResponses(outputs)


def test_malformed_llm_json_retries_and_fails(monkeypatch):
    from homonym_pipeline.llm.client import LLMClient
    monkeypatch.setattr("homonym_pipeline.llm.client.time.sleep", lambda _: None)
    config = AppConfig().llm
    config.max_retries = 1
    client = LLMClient(config, client=_FakeOpenAI(["not-json", "still-not-json"]))
    with pytest.raises(RuntimeError, match="failed after retries"):
        client.structured(model="gpt-5.6-luna", prompt_version="test", system="", user="", schema=AssignmentResponse)


def test_llm_retry_then_valid_response(monkeypatch):
    from homonym_pipeline.llm.client import LLMClient
    monkeypatch.setattr("homonym_pipeline.llm.client.time.sleep", lambda _: None)
    config = AppConfig().llm
    config.max_retries = 1
    valid = json.dumps({"assignments": []})
    client = LLMClient(config, client=_FakeOpenAI([ValueError("temporary"), valid]))
    result = client.structured(model="gpt-5.6-luna", prompt_version="test", system="", user="", schema=AssignmentResponse)
    assert result.parsed.assignments == []
    assert result.call_record.request_id == "resp_test"


def test_missing_assignment_is_rejected():
    gloss = Gloss(sense_id="s1", lemma="автомат", gloss="зброя", source="dictionary")
    candidate = FixtureGracClient({"автомат": [{"sentence": "Він узяв автомат до рук.", "example_id": "e1"}]}).retrieve_examples("автомат", 10)[0]
    candidate.example_id = "e1"
    fake = FakeLLM(AssignmentResponse(assignments=[]))
    validated, _ = validate_batch("автомат", [gloss], [candidate], fake)
    assert validated[0].accepted is False
    assert validated[0].validation_reason == "missing_assignment"


def test_huggingface_export_uses_only_published_three_field_schema(tmp_path: Path):
    example = ValidatedExample(
        example_id="e1",
        lemma="автомат",
        sentence="Він узяв автомат до рук.",
        sense_id="s1",
        accepted=True,
        model_confidence=0.95,
        validation_reason="відповідає значенню",
    )
    entries = [
        FinalLemmaEntry(
            lemma="автомат",
            glosses=[
                FinalSenseEntry(
                    sense_id="s1",
                    lemma="автомат",
                    gloss="автоматична вогнепальна зброя",
                    gloss_source="dictionary",
                    examples=[example],
                )
            ],
        )
    ]
    rows = to_huggingface_rows(entries)
    assert rows == [{
        "lemma": "автомат",
        "gloss": ["автоматична вогнепальна зброя"],
        "examples": ["Він узяв автомат до рук."],
    }]
    assert set(rows[0]) == {"lemma", "gloss", "examples"}

    write_huggingface(entries, tmp_path)
    exported = json.loads((tmp_path / "final" / "huggingface.json").read_text(encoding="utf-8"))
    assert exported == rows


def test_confirmed_merge_produces_one_gloss_and_preserves_references():
    candidates = [
        WikipediaCandidate(
            candidate_id="wp_1",
            lemma="автомат",
            title="Автоматична машина",
            extract="Машина або пристрій, що діє автоматично.",
            source_reference=SourceReference(source="wikipedia", title="Автоматична машина", document_id=1),
        ),
        WikipediaCandidate(
            candidate_id="wp_2",
            lemma="автомат",
            title="Автоматичний пристрій",
            extract="Пристрій, який виконує операції без безпосередньої участі людини.",
            source_reference=SourceReference(source="wikipedia", title="Автоматичний пристрій", document_id=2),
        ),
    ]
    decisions = [
        GlossDecision(
            action="keep",
            candidate_id="wp_1",
            gloss="Машина або пристрій, що діє автоматично",
            evidence_candidate_ids=["wp_1"],
            reason="Основний кандидат.",
        ),
        GlossDecision(
            action="merge",
            candidate_id="wp_2",
            gloss="Пристрій, який виконує операції автоматично",
            evidence_candidate_ids=["wp_2"],
            merge_into_candidate_id="wp_1",
            reason="Семантично еквівалентне значення.",
        ),
    ]
    glosses = _decisions_to_glosses("автомат", decisions, candidates)
    assert len(glosses) == 1
    assert glosses[0].gloss == "Машина або пристрій, що діє автоматично"
    assert glosses[0].merge_status == "merged"
    assert {reference.document_id for reference in glosses[0].source_references} == {1, 2}


def test_uncertain_merge_remains_separate_and_is_marked():
    candidates = [
        WikipediaCandidate(
            candidate_id="wp_1",
            lemma="автомат",
            title="Автомат",
            extract="Зброя.",
            source_reference=SourceReference(source="wikipedia", title="Автомат", document_id=1),
        ),
        WikipediaCandidate(
            candidate_id="wp_2",
            lemma="автомат",
            title="Автоматичний пристрій",
            extract="Пристрій.",
            source_reference=SourceReference(source="wikipedia", title="Автоматичний пристрій", document_id=2),
        ),
    ]
    decisions = [
        GlossDecision(action="keep", candidate_id="wp_1", gloss="Зброя", reason="Окреме значення."),
        GlossDecision(
            action="merge_uncertain",
            candidate_id="wp_2",
            gloss="Автоматичний пристрій",
            merge_into_candidate_id="wp_1",
            reason="Потребує ручної перевірки.",
        ),
    ]
    glosses = _decisions_to_glosses("автомат", decisions, candidates)
    assert len(glosses) == 2
    uncertain = next(item for item in glosses if item.gloss == "Автоматичний пристрій")
    assert uncertain.merge_status == "merge_uncertain"
    assert uncertain.merged_into_sense_id == glosses[0].sense_id


def test_added_sense_without_wikipedia_evidence_is_not_accepted():
    decisions = [
        GlossDecision(
            action="add",
            candidate_id=None,
            gloss="Непідтверджене значення",
            reason="Загальні знання моделі.",
        )
    ]
    assert _decisions_to_glosses("автомат", decisions, []) == []


def test_uncertain_merge_metadata_reaches_rich_final_dictionary(tmp_path: Path):
    target = Gloss(sense_id="s_target", lemma="автомат", gloss="Зброя", source="wikipedia")
    uncertain = Gloss(
        sense_id="s_uncertain",
        lemma="автомат",
        gloss="Автоматичний пристрій",
        source="wikipedia",
        merge_status="merge_uncertain",
        merged_into_sense_id="s_target",
        llm_reason="Потребує ручної перевірки.",
    )
    entries = [LemmaEntry(lemma="автомат", glosses=[target, uncertain])]
    fixture = FixtureGracClient({"автомат": []})
    fake = FakeLLM(AssignmentResponse(assignments=[]))
    final = run_shared(entries, tmp_path, AppConfig(), fixture, fake, resume=False)
    result = next(item for item in final[0].glosses if item.sense_id == "s_uncertain")
    assert result.merge_status == "merge_uncertain"
    assert result.merged_into_sense_id == "s_target"


def test_final_examples_are_capped_per_sense_by_confidence(tmp_path: Path):
    gloss = Gloss(sense_id="s1", lemma="автомат", gloss="Зброя", source="dictionary")
    fixture = FixtureGracClient({
        "автомат": [
            {"example_id": "e1", "sentence": "Приклад один."},
            {"example_id": "e2", "sentence": "Приклад два."},
            {"example_id": "e3", "sentence": "Приклад три."},
        ]
    })
    fake = FakeLLM(AssignmentResponse(assignments=[
        LLMAssignment(example_id="e1", accepted=True, sense_id="s1", confidence=0.81, reason="ok"),
        LLMAssignment(example_id="e2", accepted=True, sense_id="s1", confidence=0.95, reason="ok"),
        LLMAssignment(example_id="e3", accepted=True, sense_id="s1", confidence=0.90, reason="ok"),
    ]))
    config = AppConfig()
    config.validation.max_final_examples_per_sense = 2
    final = run_shared([LemmaEntry(lemma="автомат", glosses=[gloss])], tmp_path, config, fixture, fake, resume=False)
    assert [item.example_id for item in final[0].glosses[0].examples] == ["e2", "e3"]
