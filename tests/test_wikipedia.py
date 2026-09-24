from __future__ import annotations

from homonym_pipeline.glosses.wikipedia import WikipediaClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHttpClient:
    def __init__(self):
        self.calls = []

    def get(self, endpoint, params):
        self.calls.append((endpoint, params))
        if params.get("titles") == "неіснуюча назва":
            return FakeResponse({"query": {"pages": [{"title": "Неіснуюча назва", "missing": True}]}})
        if params.get("list") == "search":
            return FakeResponse({"query": {"search": [{"title": "Знайдена сторінка"}]}})
        return FakeResponse({
            "query": {
                "pages": [{
                    "pageid": 17,
                    "title": "Знайдена сторінка",
                    "extract": "Текст знайденої сторінки.",
                    "fullurl": "https://uk.wikipedia.org/wiki/Знайдена_сторінка",
                    "revisions": [{"revid": 23}],
                }]
            }
        })


def test_missing_exact_page_activates_search_fallback():
    http = FakeHttpClient()
    candidates = WikipediaClient(client=http).retrieve_candidates("неіснуюча назва", search_fallback=True)

    assert len(candidates) == 1
    assert candidates[0].title == "Знайдена сторінка"
    assert any(params.get("list") == "search" for _, params in http.calls)


def test_missing_and_invalid_pages_are_not_candidates():
    payload = {
        "query": {
            "pages": [
                {"title": "Missing", "missing": True},
                {"title": "Invalid", "invalid": True},
                {"pageid": 1, "title": "Valid", "extract": "Text"},
            ]
        }
    }

    assert WikipediaClient._extract_pages(payload) == [
        {"pageid": 1, "title": "Valid", "extract": "Text"}
    ]
