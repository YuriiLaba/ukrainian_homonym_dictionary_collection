"""ГРАК Bonito adapter, verified against the live Grac v.19 website backend."""
from __future__ import annotations

import html
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from homonym_pipeline.hashing import content_hash, example_id
from homonym_pipeline.models import CandidateExample
from homonym_pipeline.storage import atomic_write_gzip_json, atomic_write_json

DEFAULT_ENDPOINT = "https://sketch.uacorpus.org/bonito/run.cgi/"
ADAPTER_VERSION = "grac_bonito_sentences_v1"


class GracError(RuntimeError):
    """Access/backend/response-contract error, never interpreted as zero hits."""


def sentence_query(lemma: str) -> str:
    if not lemma or lemma != lemma.strip() or any(ord(c) < 32 for c in lemma):
        raise ValueError("lemma must be nonempty, trimmed and contain no control characters")
    # CQL quoted attributes are regexes: escape syntax, preserving literal lemma text.
    escaped = re.escape(lemma).replace('"', r'\"')
    return f'<s/> containing [lemma="{escaped}"]'


def _sentence_ranks(total: int, seed: int | None) -> Iterator[int]:
    if seed is None:
        yield from range(total)
        return
    # Lazy Fisher-Yates: uniform sampling without allocating the entire concordance.
    rng = random.Random(seed)
    swaps: dict[int, int] = {}
    for start in range(total):
        chosen = rng.randrange(start, total)
        value = swaps.get(chosen, chosen)
        swaps[chosen] = swaps.get(start, start)
        swaps.pop(start, None)
        yield value


class GracClient:
    def __init__(
        self, endpoint: str = DEFAULT_ENDPOINT, corpus: str = "grac19",
        timeout: float = 30.0, user_agent: str = "ukrainian-homonym-pipeline/0.1",
        client: httpx.Client | None = None, *, cache_dir: str | Path | None = None,
        resume: bool = True, page_size: int = 100, max_retries: int = 3,
        poll_attempts: int = 15, request_interval: float = 0.5,
        max_sentence_tokens: int = 100, seed: int | None = None,
        max_page_concurrency: int = 1, raw_cache_compression: str = "none",
        store_raw_line: bool = True,
    ):
        parts = urlsplit(endpoint)
        if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment:
            raise ValueError("endpoint must be an HTTP(S) base URL without query or fragment")
        # Accept the original config's site root and display name.
        path = parts.path if parts.path.strip("/") else "/bonito/run.cgi/"
        self.endpoint = urlunsplit((parts.scheme, parts.netloc, path.rstrip("/") + "/", "", ""))
        self.corpus = "grac19" if corpus == "Grac v.19" else corpus
        if (not self.corpus or page_size < 1 or max_retries < 0 or poll_attempts < 1
                or max_page_concurrency < 1):
            raise ValueError("invalid corpus, page_size, max_retries, poll_attempts or max_page_concurrency")
        if request_interval < 0 or not 1 <= max_sentence_tokens <= 100:
            raise ValueError("request_interval must be >= 0; max_sentence_tokens must be 1..100")
        if raw_cache_compression not in {"gzip", "none"}:
            raise ValueError("raw_cache_compression must be 'gzip' or 'none'")
        self.client = client or httpx.Client(timeout=timeout, headers={"User-Agent": user_agent})
        self._owns_client = client is None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.resume, self.page_size = resume, page_size
        self.max_retries, self.poll_attempts = max_retries, poll_attempts
        self.request_interval = request_interval
        self.max_sentence_tokens, self.seed = max_sentence_tokens, seed
        self.max_page_concurrency = max_page_concurrency
        self.raw_cache_compression = raw_cache_compression
        self.store_raw_line = store_raw_line
        self._last_request = 0.0
        self._request_start_lock = threading.Lock()
        self._retrieval_counter_lock = threading.Lock()
        self._retrieval_request_attempts = 0
        self._retrieval_retry_count = 0
        self.last_retrieval_stats: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: Any, *, cache_dir: Path, resume: bool = True) -> GracClient:
        return cls(config.endpoint, config.corpus, config.timeout_seconds, config.user_agent,
                   cache_dir=cache_dir, resume=resume, page_size=config.page_size,
                   max_page_concurrency=config.max_page_concurrency,
                   max_retries=config.max_retries, poll_attempts=config.poll_attempts,
                   request_interval=config.request_interval_seconds,
                   max_sentence_tokens=config.max_sentence_tokens, seed=config.seed,
                   raw_cache_compression=config.raw_cache_compression,
                   store_raw_line=config.store_raw_line)

    @property
    def cache_identity(self) -> dict[str, Any]:
        return {"adapter": ADAPTER_VERSION, "endpoint": self.endpoint, "corpus": self.corpus,
                "page_size": self.page_size, "max_sentence_tokens": self.max_sentence_tokens,
                "seed": self.seed, "store_raw_line": self.store_raw_line}

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> GracClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _request(self, action: str, params: dict[str, Any], *, throttle: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
        params = {"corpname": self.corpus, "format": "json", **params}
        url = self.endpoint + action
        key = content_hash({"url": url, "params": params, "adapter": ADAPTER_VERSION})
        index = self.cache_dir / "requests" / f"{key}.json" if self.cache_dir else None
        if self.resume and index and index.exists():
            record = json.loads(index.read_text(encoding="utf-8"))
            response_text = record.get("response_text")
            if response_text is None:
                raw_path = record.get("raw_response_path")
                if not raw_path:
                    raise GracError(f"ГРАК cache index has no raw response pointer: {index}")
                raw_file = Path(raw_path)
                try:
                    if raw_file.suffix == ".gz":
                        import gzip
                        with gzip.open(raw_file, "rt", encoding="utf-8") as handle:
                            raw_record = json.load(handle)
                    else:
                        raw_record = json.loads(raw_file.read_text(encoding="utf-8"))
                except (OSError, ValueError, KeyError) as error:
                    raise GracError(f"ГРАК raw cache could not be read: {raw_file}") from error
                response_text = raw_record.get("response_text")
            if not isinstance(response_text, str):
                raise GracError(f"ГРАК cache entry has no response text: {index}")
            return json.loads(response_text), record
        for attempt in range(self.max_retries + 1):
            with self._retrieval_counter_lock:
                self._retrieval_request_attempts += 1
                self._retrieval_retry_count += int(attempt > 0)
            if throttle:
                # Only sequential requests use the interval throttle. Parallel
                # page retrieval is bounded by max_page_concurrency instead.
                with self._request_start_lock:
                    delay = self.request_interval - (time.monotonic() - self._last_request)
                    if delay > 0:
                        time.sleep(delay)
                    self._last_request = time.monotonic()
            try:
                response = self.client.get(url, params=params)
            except httpx.TransportError as error:
                if attempt == self.max_retries:
                    raise GracError(f"ГРАК transport failed: {error}") from error
                time.sleep(min(2 ** attempt, 30))
                continue
            record = {"url": str(response.url), "params": params,
                      "retrieved_at": datetime.now(timezone.utc).isoformat(),
                      "status_code": response.status_code, "response_text": response.text}
            if self.cache_dir:
                suffix = ".json.gz" if self.raw_cache_compression == "gzip" else ".json"
                raw_path = self.cache_dir / "raw" / f"{content_hash(record)}{suffix}"
                raw_record = {**record, "raw_response_path": str(raw_path)}
                if self.raw_cache_compression == "gzip":
                    atomic_write_gzip_json(raw_path, raw_record)
                else:
                    atomic_write_json(raw_path, raw_record)
                # Keep the request cache as a small index.  Older caches may
                # still contain response_text and remain readable above. The
                # index is written below only after a complete valid response
                # is confirmed, so unfinished asynchronous responses are not
                # accidentally replayed as completed pages.
                record = {key: value for key, value in raw_record.items()
                          if key != "response_text"}
            if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else 2 ** attempt
                time.sleep(min(delay, 60))
                continue
            if not response.is_success:
                raise GracError(f"ГРАК {action} returned HTTP {response.status_code}; see raw cache")
            try:
                payload = response.json()
            except ValueError as error:
                raise GracError(f"ГРАК {action} returned non-JSON content; interface may have changed") from error
            if not isinstance(payload, dict) or payload.get("error"):
                raise GracError(f"ГРАК {action} error: {payload}")
            # Incomplete asynchronous responses are polled, never reused as complete.
            if index and payload.get("finished", 1) not in (0, "0", False):
                atomic_write_json(index, record)
            return payload, record
        raise AssertionError("unreachable")

    def _page(self, params: dict[str, Any], *, throttle: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
        for poll in range(self.poll_attempts):
            payload, record = self._request("concordance", params, throttle=throttle)
            if payload.get("finished") in (0, "0", False):
                time.sleep(min(1 + poll, 5))
                continue
            if (not isinstance(payload.get("Lines"), list)
                    or type(payload.get("concsize")) is not int
                    or payload["concsize"] < 0
                    or payload.get("finished") not in (1, "1", True)):
                raise GracError("Unexpected ГРАК concordance schema (Lines/concsize/finished)")
            if int(payload.get("concordance_size_limit", 0)):
                raise GracError("ГРАК capped this concordance; refusing to label it a complete result")
            return payload, record
        raise GracError("ГРАК concordance did not finish within the configured polling limit")

    def _retrieve_pages_concurrently(
        self,
        params: dict[str, Any],
        page_numbers: list[int],
    ) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
        """Retrieve distinct concordance pages with bounded parallelism.

        Page numbers are sorted before submission, and results are collected in
        that same order. This affects request scheduling only; callers still
        process sampled ranks in their original seeded order.
        """
        page_numbers = sorted(set(page_numbers))
        if not page_numbers:
            return {}
        if self.max_page_concurrency == 1:
            return {
                page: self._page({**params, "fromp": page}, throttle=True)
                for page in page_numbers
            }
        with ThreadPoolExecutor(
            max_workers=min(self.max_page_concurrency, len(page_numbers))
        ) as executor:
            futures = [
                executor.submit(self._page, {**params, "fromp": page}, throttle=False)
                for page in page_numbers
            ]
            return {
                page: future.result()
                for page, future in zip(page_numbers, futures)
            }

    def retrieve_examples(self, lemma: str, max_examples: int, seed: int | None = None) -> list[CandidateExample]:
        cql = sentence_query(lemma)
        if max_examples < 0:
            raise ValueError("max_examples must be >= 0")
        if max_examples == 0:
            self.last_retrieval_stats = {
                "total_sentence_hits": 0,
                "pages_requested": 0,
                "raw_rows_examined": 0,
                "eligible_candidates": 0,
                "skipped_by_reason": {},
                "duplicate_rows_skipped": 0,
                "request_attempts": 0,
                "retry_count": 0,
            }
            return []
        seed = self.seed if seed is None else seed
        result_key = content_hash({**self.cache_identity, "lemma": lemma,
                                   "max_examples": max_examples, "seed": seed})
        result_path = self.cache_dir / "examples" / f"{result_key}.json" if self.cache_dir else None
        if self.resume and result_path and result_path.exists():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            saved_stats = saved.get("retrieval_stats")
            if saved_stats:
                self.last_retrieval_stats = dict(saved_stats)
            else:
                skipped_rows = saved.get("skipped", [])
                skipped_by_reason: dict[str, int] = {}
                for skipped_row in skipped_rows:
                    reason = skipped_row.get("reason", "unknown")
                    skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
                self.last_retrieval_stats = {
                    "total_sentence_hits": int(saved.get("total_sentence_hits", 0)),
                    "pages_requested": 0,
                    "raw_rows_examined": len(saved.get("examples", [])) + len(skipped_rows),
                    "eligible_candidates": len(saved.get("examples", [])),
                    "skipped_by_reason": skipped_by_reason,
                    "duplicate_rows_skipped": 0,
                    "request_attempts": 0,
                    "retry_count": 0,
                }
            return [CandidateExample.model_validate(item) for item in saved["examples"]]

        with self._retrieval_counter_lock:
            self._retrieval_request_attempts = 0
            self._retrieval_retry_count = 0

        info, info_record = self._request("corp_info", {})
        if not isinstance(info.get("structures"), list) or not isinstance(info.get("attributes"), list):
            raise GracError("Unexpected ГРАК corpus-info schema")
        if "lemma" not in {item["name"] for item in info["attributes"]}:
            raise GracError(f"Corpus {self.corpus} has no lemma annotation")
        if "s" not in {item["name"] for item in info["structures"]}:
            raise GracError(f"Corpus {self.corpus} has no sentence boundaries")
        refs = ["#", "doc"]
        for struct in info["structures"]:
            if struct["name"] == "doc":
                refs.extend("=doc." + attr["name"] for attr in struct["attributes"])
        page_size = min(self.page_size, max_examples)
        params = {"json": json.dumps({"concordance_query": [
            {"queryselector": "cqlrow", "cql": cql}]}, ensure_ascii=False),
            "viewmode": "sen", "attrs": "word", "structs": "g,s", "refs": ",".join(refs),
            "pagesize": page_size, "fromp": 1}
        first, first_record = self._page(params)
        total = first["concsize"]
        # Reproducible local sampling; do not invent a server-side seed parameter.
        ranks = iter(_sentence_ranks(total, seed))
        pages = {1: (first, first_record)}
        output: list[CandidateExample] = []
        seen: set[str] = set()
        skipped: list[dict[str, Any]] = []
        skipped_by_reason: dict[str, int] = {}
        duplicate_rows_skipped = 0
        raw_rows_examined = 0
        # Prefetch a small window of sampled ranks. This preserves the exact
        # seeded rank order while allowing page requests for that window to run
        # concurrently without fetching the entire concordance up front.
        rank_window_size = 1 if seed is None else max(1, self.max_page_concurrency * 10)
        exhausted = False
        while len(output) < max_examples and not exhausted:
            rank_window = []
            for _ in range(rank_window_size):
                try:
                    rank_window.append(next(ranks))
                except StopIteration:
                    exhausted = True
                    break
            if not rank_window:
                break
            needed_pages = []
            for rank in rank_window:
                page, _ = divmod(rank, page_size)
                page += 1
                if page not in pages:
                    needed_pages.append(page)
            pages.update(self._retrieve_pages_concurrently(params, needed_pages))

            for rank in rank_window:
                page, offset = divmod(rank, page_size)
                page += 1
                payload, record = pages[page]
                if payload["concsize"] != total or offset >= len(payload["Lines"]):
                    raise GracError("ГРАК result count/page changed during pagination; retry with fresh cache")
                row = payload["Lines"][offset]
                raw_rows_examined += 1
                if not isinstance(row, dict) or type(row.get("hitlen")) is not int or row["hitlen"] < 1:
                    raise GracError("ГРАК line is missing its sentence token length")
                # Bonito displays only 100 initial tokens of long structures.
                if row["hitlen"] > self.max_sentence_tokens:
                    skipped.append({"toknum": row.get("toknum"), "reason": "sentence_too_long",
                                    "hitlen": row["hitlen"]})
                    skipped_by_reason["sentence_too_long"] = skipped_by_reason.get("sentence_too_long", 0) + 1
                    continue
                example = self._parse_line(lemma, cql, row, refs, info, payload, record,
                                           page_number=page, page_offset=offset)
                if example.example_id in seen:
                    duplicate_rows_skipped += 1
                    continue
                seen.add(example.example_id)
                example.source_metadata.update({
                    "selection_strategy": "corpus_order" if seed is None else "seeded_sentence_ranks",
                    "seed": seed, "result_rank": rank,
                    "corpus_info_raw_path": info_record.get("raw_response_path"),
                })
                output.append(example)
                if len(output) == max_examples:
                    break
        self.last_retrieval_stats = {
            "total_sentence_hits": total,
            "pages_requested": len(pages),
            "raw_rows_examined": raw_rows_examined,
            "eligible_candidates": len(output),
            "skipped_by_reason": skipped_by_reason,
            "duplicate_rows_skipped": duplicate_rows_skipped,
            "request_attempts": self._retrieval_request_attempts,
            "retry_count": self._retrieval_retry_count,
        }
        if result_path:
            atomic_write_json(result_path, {"cache_key": result_key, "lemma": lemma,
                              "total_sentence_hits": total, "skipped": skipped,
                              "retrieval_stats": self.last_retrieval_stats,
                              "examples": [item.model_dump(mode="json") for item in output]})
        return output

    def _parse_line(self, lemma: str, cql: str, row: dict[str, Any], refs: list[str],
                    info: dict[str, Any], payload: dict[str, Any], record: dict[str, Any], *,
                    page_number: int, page_offset: int) -> CandidateExample:
        if type(row.get("toknum")) is not int or not isinstance(row.get("Kwic"), list):
            raise GracError("ГРАК line is missing toknum/Kwic")
        chunks = []
        for chunk in row["Kwic"]:
            if not isinstance(chunk, dict) or "strc" in chunk:
                raise GracError("Unexpected ГРАК KWIC chunk/markup")
            if not isinstance(chunk.get("str"), str) or chunk.get("attr"):
                raise GracError("Unexpected ГРАК KWIC word/attribute schema")
            chunks.append(html.unescape(chunk["str"]))
        if not chunks or not any(chunks):
            raise GracError("Empty sentence KWIC")
        # structs=g makes Bonito apply glue within KWIC (punctuation/quotes/hyphens).
        # Join remaining chunks as the site's UI does; no Ukrainian text normalization.
        sentence = " ".join(chunks)
        values = row.get("Refs")
        if not isinstance(values, list) or len(values) != len(refs):
            raise GracError("ГРАК reference fields do not match the requested metadata")
        references = {key.lstrip("="): html.unescape(value) for key, value in zip(refs, values)
                      if isinstance(value, str) and value not in ("", "===NONE===")}
        metadata: dict[str, Any] = {
            "corpus": self.corpus, "corpus_name": info.get("name"),
            "sentence_start_token": row["toknum"], "sentence_token_count": row["hitlen"],
            "references": references, "total_sentence_hits": payload["concsize"],
            "query": cql, "endpoint": self.endpoint + "concordance",
            "page_number": page_number, "page_offset": page_offset,
            "retrieved_at": record["retrieved_at"], "raw_response_path": record.get("raw_response_path"),
            "api_version": payload.get("api_version"), "manatee_version": payload.get("manatee_version"),
            "adapter_version": ADAPTER_VERSION,
            "text_reconstruction": "Bonito g-glued KWIC chunks, HTML-decoded, joined with spaces",
        }
        if self.store_raw_line:
            metadata["raw_line"] = row
        for source, target in {"doc": "document_id", "doc.author": "author", "doc.title": "title",
                               "doc.date": "year", "doc.genre": "genre", "doc.uri": "url",
                               "doc.publication": "publication", "doc.publisher": "publisher"}.items():
            if source in references:
                metadata[target] = references[source]
        return CandidateExample(
            example_id=example_id(lemma, sentence, f"{self.endpoint}|{self.corpus}|{row['toknum']}|{row['hitlen']}"),
            lemma=lemma, sentence=sentence, query=cql, source_metadata=metadata,
            retrieved_at=datetime.fromisoformat(record["retrieved_at"]),
        )


# Preserve the earlier import name, using the now-verified response contract.
HttpJsonGracClient = GracClient


class FixtureGracClient(GracClient):
    def __init__(self, examples_by_lemma: dict[str, list[dict[str, Any]]]):
        self.examples_by_lemma = examples_by_lemma

    @property
    def cache_identity(self) -> dict[str, Any]:
        return {"adapter": "grac_fixture_v1", "data_hash": content_hash(self.examples_by_lemma)}

    def close(self) -> None:
        pass

    def retrieve_examples(self, lemma: str, max_examples: int, seed: int | None = None) -> list[CandidateExample]:
        output: list[CandidateExample] = []
        for row in self.examples_by_lemma.get(lemma, [])[:max_examples]:
            sentence = row["sentence"]
            data = {**row, "example_id": row.get("example_id") or example_id(lemma, sentence),
                    "lemma": lemma, "sentence": sentence,
                    "source_metadata": row.get("source_metadata", row.get("metadata", {})),
                    "query": row.get("query") or f'fixture:[lemma="{lemma}"]'}
            output.append(CandidateExample.model_validate(data))
        return output


class JsonFileGracClient(FixtureGracClient):
    """Offline JSON: examples_by_lemma mapping, or the retrieval CLI's example list."""

    def __init__(self, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in payload:
                grouped.setdefault(row["lemma"], []).append(row)
            payload = grouped
        super().__init__(payload.get("examples_by_lemma", payload))
