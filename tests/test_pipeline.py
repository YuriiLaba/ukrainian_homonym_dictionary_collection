from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import pytest

from homonym_pipeline.config import AppConfig
from homonym_pipeline.embeddings.client import EmbeddingBatch
from homonym_pipeline.hashing import sense_id
from homonym_pipeline.inputs.dictionary import parse_dictionary
from homonym_pipeline.inputs.lemma_list import parse_lemma_list
from homonym_pipeline.llm.client import StructuredResult
from homonym_pipeline.models import AssignmentResponse, Gloss, GlossAugmentationResponse, GlossDecision, LemmaEntry, LLMAssignment
from homonym_pipeline.models import LLMCallRecord
from homonym_pipeline.models import FinalLemmaEntry, FinalSenseEntry, SourceReference, ValidatedExample, WikipediaCandidate
from homonym_pipeline.glosses.llm_augmentation import _decisions_to_glosses
from homonym_pipeline.output.huggingface import to_huggingface_rows, write_huggingface
from homonym_pipeline.output.manifest import finish_manifest, start_manifest
from homonym_pipeline.output.statistics import calculate_statistics
from homonym_pipeline.pipeline.shared import run_shared
from homonym_pipeline.retrieval.grac import FixtureGracClient
from homonym_pipeline.retrieval.reranking import cosine_similarity, rank_candidates_by_gloss
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


def test_cosine_reranking_selects_top_candidates_per_gloss():
    class FakeEmbedder:
        config = AppConfig().embeddings
        config.model = "fake-embedder"
        dry_run = False

        def embed_texts(self, texts):
            vectors = {
                "зброя": [1.0, 0.0],
                "Він узяв автомат до рук.": [1.0, 0.0],
                "Автомат працює без оператора.": [0.0, 1.0],
            }
            return EmbeddingBatch([vectors[text] for text in texts], [])

    gloss = Gloss(sense_id="s1", lemma="автомат", gloss="зброя", source="dictionary")
    candidates = FixtureGracClient({"автомат": [
        {"example_id": "e1", "sentence": "Він узяв автомат до рук."},
        {"example_id": "e2", "sentence": "Автомат працює без оператора."},
    ]}).retrieve_examples("автомат", 1000)
    ranked = rank_candidates_by_gloss([gloss], candidates, FakeEmbedder(), top_k=1)
    assert [item.example_id for item in ranked.by_sense["s1"]] == ["e1"]
    assert ranked.by_sense["s1"][0].source_metadata["embedding_rank"] == 1
    assert ranked.by_sense["s1"][0].source_metadata["embedding_cosine_similarity"] == 1.0
    assert ranked.pairs_scored == 2


def test_mmr_prefers_diversity_after_relevance():
    class FakeEmbedder:
        config = AppConfig().embeddings
        config.model = "fake-embedder"
        dry_run = False

        def embed_texts(self, texts):
            vectors = {
                "зброя": [1.0, 0.0],
                "Перший приклад.": [1.0, 0.0],
                "Майже такий самий приклад.": [1.0, 0.0],
                "Різноманітний приклад.": [0.0, 1.0],
            }
            return EmbeddingBatch([vectors[text] for text in texts], [])

    gloss = Gloss(sense_id="s1", lemma="автомат", gloss="зброя", source="dictionary")
    candidates = FixtureGracClient({"автомат": [
        {"example_id": "e1", "sentence": "Перший приклад."},
        {"example_id": "e2", "sentence": "Майже такий самий приклад."},
        {"example_id": "e3", "sentence": "Різноманітний приклад."},
    ]}).retrieve_examples("автомат", 10)
    ranked = rank_candidates_by_gloss(
        [gloss], candidates, FakeEmbedder(), top_k=2,
        mmr_enabled=True, mmr_lambda=0.4,
    )
    selected = ranked.by_sense["s1"]
    assert [item.example_id for item in selected] == ["e1", "e3"]
    assert selected[1].source_metadata["embedding_mmr_rank"] == 2
    assert ranked.mmr_enabled is True


def test_embedding_reranking_deduplicates_normalized_sentence_text():
    class FakeEmbedder:
        config = AppConfig().embeddings
        config.model = "fake-embedder"
        dry_run = False

        def embed_texts(self, texts):
            vectors = {"зброя": [1.0, 0.0], "Однаковий приклад.": [1.0, 0.0], "Інший приклад.": [0.0, 1.0]}
            return EmbeddingBatch([vectors[text] for text in texts], [])

    gloss = Gloss(sense_id="s1", lemma="автомат", gloss="зброя", source="dictionary")
    candidates = FixtureGracClient({"автомат": [
        {"example_id": "e1", "sentence": "Однаковий приклад."},
        {"example_id": "e2", "sentence": "  Однаковий   приклад.  "},
        {"example_id": "e3", "sentence": "Інший приклад."},
    ]}).retrieve_examples("автомат", 10)
    ranked = rank_candidates_by_gloss([gloss], candidates, FakeEmbedder(), top_k=3)
    assert [item.example_id for item in ranked.by_sense["s1"]] == ["e1", "e3"]
    assert ranked.exact_duplicates_removed == 1


def test_shared_pipeline_sends_only_embedding_shortlist_to_luna(tmp_path: Path):
    class FakeEmbedder:
        config = AppConfig().embeddings
        config.model = "fake-embedder"
        dry_run = False

        def embed_texts(self, texts):
            vectors = {
                "зброя": [1.0, 0.0],
                "пристрій": [0.0, 1.0],
                "Він узяв автомат до рук.": [1.0, 0.0],
                "Автомат працює без оператора.": [0.0, 1.0],
            }
            return EmbeddingBatch([vectors[text] for text in texts], [])

    class RecordingLLM(FakeLLM):
        def __init__(self):
            super().__init__(AssignmentResponse(assignments=[]))
            self.calls = []

        def structured(self, **kwargs):
            payload = json.loads(kwargs["user"])
            self.calls.append(payload)
            item = payload["examples"][0]
            sense_id = payload["gloss"]["sense_id"]
            return StructuredResult(
                parsed=AssignmentResponse(assignments=[LLMAssignment(
                    example_id=item["example_id"], accepted=True, sense_id=sense_id,
                    confidence=0.95, reason="matches",
                )]),
                call_record=LLMCallRecord(model=kwargs["model"], prompt_version=kwargs["prompt_version"]),
            )

    config = AppConfig()
    config.validation.max_candidates_per_gloss = 1
    entries = [LemmaEntry(lemma="автомат", glosses=[
        Gloss(sense_id="s1", lemma="автомат", gloss="зброя", source="dictionary"),
        Gloss(sense_id="s2", lemma="автомат", gloss="пристрій", source="dictionary"),
    ])]
    fixture = FixtureGracClient({"автомат": [
        {"example_id": "e1", "sentence": "Він узяв автомат до рук."},
        {"example_id": "e2", "sentence": "Автомат працює без оператора."},
    ]})
    llm = RecordingLLM()
    final = run_shared(entries, tmp_path, config, fixture, llm, resume=False, embedder=FakeEmbedder())
    assert [call["examples"][0]["example_id"] for call in llm.calls] == ["e1", "e2"]
    assert [item.example_id for item in final[0].glosses[0].examples] == ["e1"]
    assert [item.example_id for item in final[0].glosses[1].examples] == ["e2"]


def test_shared_pipeline_bounds_luna_concurrency(tmp_path: Path):
    class TrackingLLM:
        def __init__(self):
            self.config = AppConfig().llm
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def structured(self, **kwargs):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self.lock:
                self.active -= 1
            return StructuredResult(
                parsed=AssignmentResponse(assignments=[]),
                call_record=LLMCallRecord(model=kwargs["model"], prompt_version=kwargs["prompt_version"]),
            )

    config = AppConfig()
    config.validation.batch_size = 1
    config.validation.max_concurrency = 2
    entries = [LemmaEntry(lemma="автомат", glosses=[
        Gloss(sense_id="s1", lemma="автомат", gloss="зброя", source="dictionary"),
        Gloss(sense_id="s2", lemma="автомат", gloss="пристрій", source="dictionary"),
    ])]
    fixture = FixtureGracClient({"автомат": [
        {"example_id": "e1", "sentence": "Він узяв автомат до рук."},
        {"example_id": "e2", "sentence": "Автомат працює без оператора."},
    ]})
    llm = TrackingLLM()
    run_shared(entries, tmp_path, config, fixture, llm, resume=False)
    assert llm.max_active == 2


def test_openai_embedding_client_batches_requests_and_records_provenance():
    class Response:
        id = "emb_test"

        def __init__(self, vectors):
            self.data = [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]

        def model_dump(self, mode="json"):
            return {"id": self.id, "model": "text-embedding-3-small",
                    "data": self.data, "usage": {"prompt_tokens": 4, "total_tokens": 4}}

    class Embeddings:
        def __init__(self):
            self.requests = []

        def create(self, **kwargs):
            self.requests.append(kwargs)
            return Response([[float(index + 1), 0.0] for index, _ in enumerate(kwargs["input"])])

    class OpenAI:
        def __init__(self):
            self.embeddings = Embeddings()

    from homonym_pipeline.embeddings.client import EmbeddingClient

    config = AppConfig().embeddings
    config.batch_size = 2
    api = OpenAI()
    result = EmbeddingClient(config, client=api).embed_texts(["один", "два", "три"])
    assert len(api.embeddings.requests) == 2
    assert [len(request["input"]) for request in api.embeddings.requests] == [2, 1]
    assert len(result.vectors) == 3
    assert result.calls[0]["request_id"] == "emb_test"
    assert result.calls[0]["input_count"] == 2


def test_openai_embedding_client_bounds_concurrency_and_preserves_order():
    class Response:
        id = "emb_concurrent"

        def __init__(self, values):
            self.data = [
                {"index": index, "embedding": [float(value), 1.0]}
                for index, value in enumerate(values)
            ]

        def model_dump(self, mode="json"):
            return {"id": self.id, "model": "text-embedding-3-small", "data": self.data}

    class Embeddings:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def create(self, **kwargs):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self.lock:
                self.active -= 1
            values = [int(text.removeprefix("text")) for text in kwargs["input"]]
            return Response(values)

    class OpenAI:
        def __init__(self):
            self.embeddings = Embeddings()

    from homonym_pipeline.embeddings.client import EmbeddingClient

    config = AppConfig().embeddings
    config.batch_size = 1
    config.max_concurrency = 2
    api = OpenAI()
    result = EmbeddingClient(config, client=api).embed_texts(["text0", "text1", "text2", "text3"])

    assert api.embeddings.max_active == 2
    assert [vector[0] for vector in result.vectors] == [0.0, 1.0, 2.0, 3.0]
    assert [call["batch_index"] for call in result.calls] == [0, 1, 2, 3]
    assert all(call["attempts"] == 1 for call in result.calls)


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
    second_gloss = Gloss(sense_id="s2", lemma="автомат", gloss="пристрій", source="dictionary")
    entries = [LemmaEntry(lemma="автомат", glosses=[gloss, second_gloss])]
    fixture = FixtureGracClient({"автомат": [{"sentence": "Він узяв автомат до рук.", "example_id": "e1"}]})
    fake = FakeLLM(AssignmentResponse(assignments=[LLMAssignment(example_id="e1", accepted=True, sense_id="s1", confidence=0.95, reason="ok")]))
    first = run_shared(entries, tmp_path, config, fixture, fake, resume=True)
    assert first[0].glosses[0].examples[0].example_id == "e1"
    assert "[input] 1 lemmas, 2 glosses; 1 lemmas have multiple glosses." in caplog.text
    assert "[1/1] автомат\n  GRAC candidates: 1" in caplog.text
    assert "  Supported senses: 1/2" in caplog.text
    assert "  Lemmas with 2+ supported senses: 0/1" in caplog.text
    assert "  Processed glosses: 2/2" in caplog.text
    assert "  GRAC retrieval time: " in caplog.text
    assert "  Embedding time: " in caplog.text
    assert "  Luna validation time: " in caplog.text
    assert "  Finalization time: " in caplog.text
    assert "  Total processing time: " in caplog.text
    assert "[stage 2/3] Complete.\n  Processed lemmas: 1/1\n  Supported glosses: 1/2" in caplog.text
    audit = json.loads((tmp_path / "audit" / "lemma_audit.jsonl").read_text(encoding="utf-8"))
    assert audit["grac_elapsed_seconds"] >= 0
    assert audit["embedding_elapsed_seconds"] >= 0
    assert audit["validation_elapsed_seconds"] >= 0
    assert audit["finalization_elapsed_seconds"] >= 0
    assert audit["elapsed_seconds"] >= audit["finalization_elapsed_seconds"]

    class FailingGrac(FixtureGracClient):
        def retrieve_examples(self, lemma, max_examples, seed=None):
            raise AssertionError("resume should use the cached candidate")

    second = run_shared(entries, tmp_path, config, FailingGrac(fixture.examples_by_lemma), fake, resume=True)
    assert second[0].glosses[0].examples[0].example_id == "e1"


def test_shared_pipeline_drops_single_gloss_lemmas(tmp_path: Path, caplog):
    caplog.set_level(logging.INFO, logger="homonym_pipeline.pipeline.shared")
    config = AppConfig()
    gloss = Gloss(sense_id="s1", lemma="однозначне", gloss="значення", source="dictionary")

    class FailingGrac(FixtureGracClient):
        def retrieve_examples(self, lemma, max_examples, seed=None):
            raise AssertionError("single-gloss lemmas must be filtered before GRAC retrieval")

    result = run_shared(
        [LemmaEntry(lemma="однозначне", glosses=[gloss])],
        tmp_path,
        config,
        FailingGrac({}),
        FakeLLM(AssignmentResponse(assignments=[])),
        resume=True,
    )
    assert result == []
    assert "[filter] Removed 1 single-gloss lemmas. Processing 0 of 1 lemmas." in caplog.text


def test_shared_pipeline_validates_each_gloss_separately(tmp_path: Path):
    class GlossSpecificLLM:
        def __init__(self):
            self.config = AppConfig().llm
            self.calls = []

        def structured(self, **kwargs):
            payload = json.loads(kwargs["user"])
            self.calls.append(payload)
            target = {"s1": "e1", "s2": "e2"}[payload["gloss"]["sense_id"]]
            assignments = [
                LLMAssignment(
                    example_id=item["example_id"],
                    accepted=item["example_id"] == target,
                    sense_id=payload["gloss"]["sense_id"] if item["example_id"] == target else None,
                    confidence=0.95 if item["example_id"] == target else 0.2,
                    reason="matches gloss" if item["example_id"] == target else "does not match gloss",
                )
                for item in payload["examples"]
            ]
            return StructuredResult(
                parsed=AssignmentResponse(assignments=assignments),
                call_record=LLMCallRecord(model=kwargs["model"], prompt_version=kwargs["prompt_version"], temperature=0),
            )

    entries = [LemmaEntry(lemma="автомат", glosses=[
        Gloss(sense_id="s1", lemma="автомат", gloss="зброя", source="dictionary"),
        Gloss(sense_id="s2", lemma="автомат", gloss="пристрій", source="dictionary"),
    ])]
    fixture = FixtureGracClient({"автомат": [
        {"example_id": "e1", "sentence": "Він узяв автомат до рук."},
        {"example_id": "e2", "sentence": "Автомат працює без оператора."},
    ]})
    llm = GlossSpecificLLM()
    final = run_shared(entries, tmp_path, AppConfig(), fixture, llm, resume=False)

    assert len(llm.calls) == 2
    assert all("gloss" in call and "glosses" not in call for call in llm.calls)
    assert [item.example_id for item in final[0].glosses[0].examples] == ["e1"]
    assert [item.example_id for item in final[0].glosses[1].examples] == ["e2"]


def test_shared_pipeline_resolves_duplicate_gloss_assignments(tmp_path: Path):
    class DuplicateAcceptingLLM:
        def __init__(self):
            self.config = AppConfig().llm

        def structured(self, **kwargs):
            payload = json.loads(kwargs["user"])
            sense_id = payload["gloss"]["sense_id"]
            confidence = 0.90 if sense_id == "s1" else 0.95
            return StructuredResult(
                parsed=AssignmentResponse(assignments=[LLMAssignment(
                    example_id="e1", accepted=True, sense_id=sense_id,
                    confidence=confidence, reason="matches gloss",
                )]),
                call_record=LLMCallRecord(model=kwargs["model"], prompt_version=kwargs["prompt_version"], temperature=0),
            )

    entries = [LemmaEntry(lemma="автомат", glosses=[
        Gloss(sense_id="s1", lemma="автомат", gloss="зброя", source="dictionary"),
        Gloss(sense_id="s2", lemma="автомат", gloss="пристрій", source="dictionary"),
    ])]
    fixture = FixtureGracClient({"автомат": [{"example_id": "e1", "sentence": "Контекст."}]})
    final = run_shared(entries, tmp_path, AppConfig(), fixture, DuplicateAcceptingLLM(), resume=False)

    assert final[0].glosses[0].examples == []
    assert [item.example_id for item in final[0].glosses[1].examples] == ["e1"]
    audit = json.loads((tmp_path / "audit" / "lemma_audit.jsonl").read_text(encoding="utf-8"))
    assert audit["rejected_by_reason"]["duplicate_assignment_conflict"] == 1


class _FakeResponse:
    def __init__(self, text):
        self.id = "resp_test"
        self.output_text = text

    def model_dump(self, mode="json"):
        return {"id": self.id, "usage": {"input_tokens": 3, "output_tokens": 4}}


class _FakeResponses:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
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


def test_reasoning_model_request_omits_temperature_when_unset():
    from homonym_pipeline.llm.client import LLMClient

    config = AppConfig().llm
    config.temperature = None
    fake_openai = _FakeOpenAI([json.dumps({"assignments": []})])
    client = LLMClient(config, client=fake_openai)
    client.structured(
        model="gpt-5.6-luna",
        prompt_version="test",
        system="",
        user="{}",
        schema=AssignmentResponse,
    )
    assert "temperature" not in fake_openai.responses.calls[0]
    assert fake_openai.responses.calls[0]["reasoning"] == {"effort": config.reasoning_effort}


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
            original_gloss=None,
            evidence_candidate_ids=["wp_1"],
            merge_into_candidate_id=None,
            reason="Основний кандидат.",
        ),
        GlossDecision(
            action="merge",
            candidate_id="wp_2",
            gloss="Пристрій, який виконує операції автоматично",
            original_gloss=None,
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
        GlossDecision(
            action="keep",
            candidate_id="wp_1",
            gloss="Зброя",
            original_gloss=None,
            evidence_candidate_ids=[],
            merge_into_candidate_id=None,
            reason="Окреме значення.",
        ),
        GlossDecision(
            action="merge_uncertain",
            candidate_id="wp_2",
            gloss="Автоматичний пристрій",
            original_gloss=None,
            evidence_candidate_ids=[],
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
            original_gloss=None,
            evidence_candidate_ids=[],
            merge_into_candidate_id=None,
            reason="Загальні знання моделі.",
        )
    ]
    assert _decisions_to_glosses("автомат", decisions, []) == []


def test_llm_response_schemas_are_strict_at_nested_object_levels():
    assignment_schema = AssignmentResponse.model_json_schema()
    assignment_item = assignment_schema["$defs"]["LLMAssignment"]
    assert assignment_schema["additionalProperties"] is False
    assert assignment_item["additionalProperties"] is False

    augmentation_schema = GlossAugmentationResponse.model_json_schema()
    decision_item = augmentation_schema["$defs"]["GlossDecision"]
    assert augmentation_schema["additionalProperties"] is False
    assert decision_item["additionalProperties"] is False


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
    second_gloss = Gloss(sense_id="s2", lemma="автомат", gloss="Пристрій", source="dictionary")
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
    final = run_shared([LemmaEntry(lemma="автомат", glosses=[gloss, second_gloss])], tmp_path, config, fixture, fake, resume=False)
    assert [item.example_id for item in final[0].glosses[0].examples] == ["e2", "e3"]


def test_run_manifest_records_input_and_completion(tmp_path: Path):
    input_path = tmp_path / "input.txt"
    input_path.write_text("автомат\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    manifest = start_manifest("baseline", input_path, output_dir, AppConfig(),
                              run_parameters={"max_lemmas": 1})
    assert manifest["status"] == "running"
    assert manifest["input_sha256"]
    assert manifest["run_parameters"]["max_lemmas"] == 1
    assert (output_dir / "config.snapshot.yaml").exists()

    finish_manifest(output_dir, status="completed", statistics={"test": True})
    saved = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved["status"] == "completed"
    assert saved["statistics"] == {"test": True}


def test_statistics_scope_llm_usage_to_run_and_calculate_cost(tmp_path: Path):
    calls_path = tmp_path / "validation" / "llm_calls.jsonl"
    calls_path.parent.mkdir(parents=True)
    calls_path.write_text(
        "\n".join([
            json.dumps({"run_id": "run-1", "stage": "example_validation", "model": "model-a",
                        "usage": {"input_tokens": 10, "output_tokens": 5}}),
            json.dumps({"run_id": "run-2", "stage": "example_validation", "model": "model-a",
                        "usage": {"input_tokens": 100, "output_tokens": 50}}),
        ]) + "\n",
        encoding="utf-8",
    )
    config = AppConfig()
    config.llm.pricing = {"model-a": {
        "input_per_million_tokens": 1.0,
        "output_per_million_tokens": 2.0,
    }}
    stats = calculate_statistics([], tmp_path, run_id="run-1", config=config)
    assert stats["llm_calls"] == 1
    assert stats["llm_token_usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert stats["llm_calls_by_stage"] == {"example_validation": 1}
    assert stats["estimated_llm_cost_usd"] == 0.00002
