from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx

from homonym_pipeline.hashing import content_hash, normalize_text
from homonym_pipeline.models import SourceReference, WikipediaCandidate


class WikipediaClient:
    """MediaWiki API adapter; raw API payloads are returned for immutable caching."""

    def __init__(self, endpoint: str = "https://uk.wikipedia.org/w/api.php", timeout: float = 30.0,
                 client: httpx.Client | None = None):
        self.endpoint = endpoint
        self.client = client or httpx.Client(timeout=timeout, headers={"User-Agent": "ukrainian-homonym-pipeline/0.1"})

    def retrieve_candidates(self, lemma: str, search_fallback: bool = True) -> list[WikipediaCandidate]:
        retrieved_at = datetime.now(timezone.utc)
        pages: list[dict[str, Any]] = []
        exact = self._request({"action": "query", "titles": lemma, "prop": "extracts|info|revisions",
                               "explaintext": 1, "exintro": 0, "rvprop": "ids|timestamp", "inprop": "url"})
        pages.extend(self._extract_pages(exact))
        if search_fallback and not pages:
            search = self._request({"action": "query", "list": "search", "srsearch": f'intitle:"{lemma}"',
                                    "srlimit": 10})
            for item in search.get("query", {}).get("search", []):
                page_payload = self._request({"action": "query", "titles": item.get("title"),
                                              "prop": "extracts|info|revisions", "explaintext": 1,
                                              "exintro": 0, "rvprop": "ids|timestamp", "inprop": "url"})
                pages.extend(self._extract_pages(page_payload))
        candidates: list[WikipediaCandidate] = []
        for page in pages:
            title = str(page.get("title") or "")
            extract = normalize_text(str(page.get("extract") or ""))
            if not title or not extract:
                continue
            page_id = page.get("pageid")
            candidate_key = content_hash({"lemma": lemma, "page_id": page_id, "title": title})[:20]
            revision_id = page.get("revisions", [{}])[0].get("revid") if page.get("revisions") else None
            url = page.get("fullurl") or f"https://uk.wikipedia.org/wiki/{title.replace(' ', '_')}"
            reference = SourceReference(source="wikipedia", url=url, title=title, document_id=page_id,
                                        revision_id=revision_id, retrieved_at=retrieved_at)
            candidates.append(WikipediaCandidate(candidate_id=f"wp_{candidate_key}", lemma=lemma, title=title,
                                                page_id=page_id, revision_id=revision_id, extract=extract,
                                                url=url, source_reference=reference, raw_response=page))
        unique = {candidate.candidate_id: candidate for candidate in candidates}
        return list(unique.values())

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "format": "json", "formatversion": 2}
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.client.get(self.endpoint, params=params)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                if attempt == 3:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"Wikipedia request failed after retries: {last_error}") from last_error

    @staticmethod
    def _extract_pages(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return list(payload.get("query", {}).get("pages", []))
