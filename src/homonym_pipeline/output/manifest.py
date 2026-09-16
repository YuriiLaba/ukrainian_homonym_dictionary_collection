from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from homonym_pipeline.config import AppConfig
from homonym_pipeline.storage import atomic_write_json, atomic_write_text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _package_versions() -> dict[str, str | None]:
    packages = ("httpx", "openai", "pydantic", "PyYAML", "pytest")
    return {
        package: _installed_version(package)
        for package in packages
    }


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def start_manifest(
    workflow: str,
    input_path: str | Path,
    output_dir: str | Path,
    config: AppConfig,
    *,
    run_id: str | None = None,
    run_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(input_path).resolve()
    started_at = datetime.now(timezone.utc)
    input_hash = _sha256(source)
    run_id = run_id or f"{started_at.strftime('%Y%m%dT%H%M%S%fZ')}_{workflow}_{input_hash[:12]}"
    root = Path(output_dir)
    manifest = {
        "run_id": run_id,
        "workflow": workflow,
        "status": "running",
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "input_file": str(source),
        "input_sha256": input_hash,
        "run_parameters": run_parameters or {},
        "config": config.model_dump(mode="json"),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git_revision": _git_revision(),
            "package_versions": _package_versions(),
        },
    }
    atomic_write_json(root / "run_manifest.json", manifest)
    atomic_write_text(
        root / "config.snapshot.yaml",
        yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
    )
    return manifest


def finish_manifest(
    output_dir: str | Path,
    *,
    status: str,
    statistics: dict[str, Any] | None = None,
) -> None:
    path = Path(output_dir) / "run_manifest.json"
    if not path.exists():
        return
    import json

    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = status
    finished_at = datetime.now(timezone.utc)
    manifest["finished_at"] = finished_at.isoformat()
    started_at = datetime.fromisoformat(manifest["started_at"])
    manifest["elapsed_seconds"] = (finished_at - started_at).total_seconds()
    if statistics is not None:
        manifest["statistics"] = statistics
    artifacts: dict[str, str] = {}
    for relative in (
        "run_input_statistics.json",
        "logs/pipeline.log",
        "audit/lemma_audit.jsonl",
        "grac/candidate_examples.jsonl",
        "grac/embedding_rankings.jsonl",
        "grac/embedding_calls.jsonl",
        "validation/llm_assignments.jsonl",
        "validation/llm_calls.jsonl",
        "validation/rejected_examples.jsonl",
        "final/homonym_dictionary.json",
        "final/homonym_dictionary.jsonl",
        "final/huggingface.json",
        "final/huggingface.jsonl",
        "statistics/pipeline_statistics.json",
    ):
        artifact = Path(output_dir) / relative
        if artifact.exists():
            artifacts[relative] = _sha256(artifact)
    manifest["artifacts_sha256"] = artifacts
    atomic_write_json(path, manifest)
