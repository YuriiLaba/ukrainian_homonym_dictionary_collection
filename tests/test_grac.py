from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from homonym_pipeline.config import AppConfig, load_config
from homonym_pipeline.models import Gloss, LemmaEntry
from homonym_pipeline.pipeline.shared import run_shared
from homonym_pipeline.retrieval.grac import (
    FixtureGracClient, GracClient, GracError, JsonFileGracClient, sentence_query,
)

# Synthetic data with the real Bonito envelope; no semantic gold-label claims.
DOC_ATTRS = ["author", "title", "date", "genre", "uri"]
INFO = {
    "name": "Grac v.19", "info": "Synthetic corpus metadata",
    "attributes": [{"name": "word"}, {"name": "lemma"}],
    "structures": [{"name": "s", "attributes": []},
                   {"name": "doc", "attributes": [{"name": name} for name in DOC_ATTRS]}],
}


def line(position: int, words: list[str], *, hitlen: int | None = None) -> dict:
    return {"toknum": position, "hitlen": hitlen or len(words),
            "Left": [{"strc": "<s>"}],
            "Kwic": [{"str": word, "coll": 1} for word in words],
            "Right": [{"strc": "</s>"}],
            "Refs": [f"#{position}", "doc#7", "Автор", "Назва", "2001",
                     "===NONE===", "https://example.org/source"]}


LINES = [
    line(10, ["Він", "узяв", "автомат", "до", "рук."]),
    line(20, ["Студент", "отримав", "автомат", "з", "дисципліни."]),
    line(30, ["На", "заводі", "встановили", "новий", "автомат."]),
]


def response(rows: list[dict], *, total: int | None = None, finished: int = 1) -> dict:
    return {"Lines": rows, "concsize": len(rows) if total is None else total,
            "finished": finished, "concordance_size_limit": 0,
            "api_version": "test-api", "manatee_version": "test-manatee"}


def transport_for(rows: list[dict], calls: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/corp_info"):
            return httpx.Response(200, json=INFO)
        assert request.url.path.endswith("/concordance")
        params = request.url.params
        assert params["corpname"] == "grac19"
        assert params["structs"] == "g,s"
        assert params["viewmode"] == "sen"
        query = json.loads(params["json"])["concordance_query"]
        assert query == [{"queryselector": "cqlrow",
                          "cql": '<s/> containing [lemma="автомат"]'}]
        start = (int(params["fromp"]) - 1) * int(params["pagesize"])
        return httpx.Response(200, json=response(
            rows[start:start + int(params["pagesize"])], total=len(rows)))
    return handler


def client_for(handler, **kwargs) -> GracClient:
    return GracClient(client=httpx.Client(transport=httpx.MockTransport(handler)),
                      request_interval=0, **kwargs)


def test_pagination_metadata_raw_cache_and_offline_replay(tmp_path: Path):
    calls = []
    grac = client_for(transport_for(LINES, calls), cache_dir=tmp_path, page_size=2)
    examples = grac.retrieve_examples("автомат", 3)
    assert [x.sentence for x in examples] == [" ".join(x["str"] for x in r["Kwic"]) for r in LINES]
    assert len(calls) == 3  # corpus metadata plus two pages
    meta = examples[0].source_metadata
    assert meta["author"] == "Автор"
    assert meta["document_id"] == "doc#7"
    assert meta["year"] == "2001"
    assert "genre" not in meta  # absent source metadata is not invented
    assert meta["total_sentence_hits"] == 3
    assert meta["raw_line"] == LINES[0]
    raw = json.loads(Path(meta["raw_response_path"]).read_text())
    assert json.loads(raw["response_text"])["Lines"][0] == LINES[0]

    def offline(_):
        pytest.fail("a completed cache must make no network requests")
    replay = client_for(offline, cache_dir=tmp_path, page_size=2)
    assert replay.retrieve_examples("автомат", 3) == examples


def test_punctuation_entities_and_ukrainian_text_preserved():
    rows = [line(1, ["«Автомати»,", "п’єса", "—", "&#34;тест&#34;", "&amp;", "сло́во."])]
    examples = client_for(transport_for(rows, [])).retrieve_examples("автомат", 1)
    assert examples[0].sentence == '«Автомати», п’єса — "тест" & сло́во.'
    assert examples[0].source_metadata["raw_line"]["Kwic"][3]["str"] == "&#34;тест&#34;"


def test_long_sentences_skipped_without_returning_truncation(tmp_path: Path):
    rows = [line(5, ["truncated..."], hitlen=101), *LINES]
    examples = client_for(transport_for(rows, []), cache_dir=tmp_path, page_size=1).retrieve_examples("автомат", 2)
    assert len(examples) == 2
    assert examples[0].sentence == "Він узяв автомат до рук."
    saved = json.loads(next((tmp_path / "examples").glob("*.json")).read_text())
    assert saved["skipped"] == [{"toknum": 5, "reason": "sentence_too_long", "hitlen": 101}]


def test_empty_results_and_zero_limit():
    calls = []
    grac = client_for(transport_for([], calls))
    assert grac.retrieve_examples("автомат", 0) == []
    assert calls == []
    assert grac.retrieve_examples("автомат", 3) == []


def test_seeded_sampling_is_reproducible_and_cache_is_separate(tmp_path: Path):
    rows = [line(i, ["автомат", str(i)]) for i in range(30)]
    calls = []
    grac = client_for(transport_for(rows, calls), cache_dir=tmp_path, page_size=2)
    a = grac.retrieve_examples("автомат", 4, seed=42)
    b = grac.retrieve_examples("автомат", 4, seed=42)
    c = grac.retrieve_examples("автомат", 4, seed=13)
    assert a == b
    assert len({x.example_id for x in a}) == 4
    assert [x.example_id for x in a] != [x.example_id for x in c]
    assert all(x.source_metadata["seed"] == 42 for x in a)


def test_force_refresh_keeps_previous_raw_responses(tmp_path: Path):
    rows = list(LINES)
    calls = []
    first = client_for(transport_for(rows, calls), cache_dir=tmp_path).retrieve_examples("автомат", 1)
    old_path = Path(first[0].source_metadata["raw_response_path"])
    rows[0] = line(99, ["Інший", "автомат."])
    second = client_for(transport_for(rows, calls), cache_dir=tmp_path, resume=False).retrieve_examples("автомат", 1)
    assert first[0].example_id != second[0].example_id
    assert old_path.exists()
    assert len(calls) == 4


def test_incomplete_response_is_polled_and_raw_attempts_retained(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("homonym_pipeline.retrieval.grac.time.sleep", lambda _: None)
    count = 0
    def handler(request):
        nonlocal count
        if request.url.path.endswith("/corp_info"):
            return httpx.Response(200, json=INFO)
        count += 1
        return httpx.Response(200, json=response(LINES[:1], finished=0 if count == 1 else 1))
    grac = client_for(handler, cache_dir=tmp_path)
    assert len(grac.retrieve_examples("автомат", 1)) == 1
    assert count == 2
    assert len(list((tmp_path / "raw").glob("*.json"))) == 3


def test_retry_rate_limit_and_transport_error(monkeypatch):
    sleeps = []
    monkeypatch.setattr("homonym_pipeline.retrieval.grac.time.sleep", sleeps.append)
    count = 0
    def handler(request):
        nonlocal count
        if request.url.path.endswith("/corp_info"):
            return httpx.Response(200, json=INFO)
        count += 1
        if count == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        if count == 2:
            raise httpx.ReadTimeout("temporary")
        return httpx.Response(200, json=response(LINES[:1]))
    assert len(client_for(handler).retrieve_examples("автомат", 1)) == 1
    assert count == 3
    assert sleeps == [2.0, 2]


@pytest.mark.parametrize("bad", [
    httpx.Response(403, text="denied"),
    httpx.Response(200, text="<html>sign in</html>"),
    httpx.Response(200, json={"error": "Unknown corpus"}),
    httpx.Response(200, json={"finished": 1, "concsize": 1, "examples": []}),
    httpx.Response(200, json=response([{**LINES[0], "Refs": []}])),
])
def test_errors_are_not_silently_treated_as_empty(bad):
    def handler(request):
        return httpx.Response(200, json=INFO) if request.url.path.endswith("/corp_info") else bad
    with pytest.raises(GracError):
        client_for(handler).retrieve_examples("автомат", 1)


def test_poll_limit_and_retry_exhaustion(monkeypatch):
    monkeypatch.setattr("homonym_pipeline.retrieval.grac.time.sleep", lambda _: None)
    def pending(request):
        return httpx.Response(200, json=INFO if request.url.path.endswith("/corp_info")
                              else response([], finished=0))
    with pytest.raises(GracError, match="polling limit"):
        client_for(pending, poll_attempts=2).retrieve_examples("автомат", 1)
    calls = []
    def failed(request):
        calls.append(request)
        return httpx.Response(503)
    with pytest.raises(GracError, match="HTTP 503"):
        client_for(failed, max_retries=1).retrieve_examples("автомат", 1)
    assert len(calls) == 2


def test_literal_query_and_legacy_configuration():
    assert sentence_query("автомат") == '<s/> containing [lemma="автомат"]'
    assert sentence_query("а.*") == '<s/> containing [lemma="а\\.\\*"]'
    assert '\\"' in sentence_query('а"б')
    with pytest.raises(ValueError):
        sentence_query("автомат\n")
    grac = GracClient("https://sketch.uacorpus.org/", "Grac v.19")
    assert grac.corpus == "grac19"
    assert grac.endpoint.endswith("/bonito/run.cgi/")
    grac.close()
    assert load_config("config.yaml").grac.corpus == "grac19"


def test_export_reimports_without_losing_provenance(tmp_path: Path):
    examples = client_for(transport_for(LINES, [])).retrieve_examples("автомат", 1)
    path = tmp_path / "export.json"
    path.write_text(json.dumps([x.model_dump(mode="json") for x in examples]))
    assert JsonFileGracClient(path).retrieve_examples("автомат", 1) == examples


def test_shared_pipeline_live_adapter_dry_run_and_cache_identity(tmp_path: Path):
    calls = []
    grac = client_for(transport_for(LINES, calls))
    entries = [LemmaEntry(lemma="автомат", glosses=[
        Gloss(sense_id="s1", lemma="автомат", gloss="зброя", source="dictionary"),
        Gloss(sense_id="s2", lemma="автомат", gloss="пристрій", source="dictionary")])]
    class NoLLM:
        dry_run = True
        def structured(self, **kwargs):
            pytest.fail("dry run must not call an LLM")
    config = AppConfig()
    config.grac.max_examples_per_lemma = 1
    result = run_shared(entries, tmp_path, config, grac, NoLLM())
    assert result[0].glosses[0].examples == []
    assert not (tmp_path / "validation" / "failures.jsonl").exists()
    count = len(calls)
    assert run_shared(entries, tmp_path, config, grac, NoLLM()) == result
    assert len(calls) == count
    assert grac.cache_identity != FixtureGracClient({}).cache_identity
