"""Retrieve ГРАК sentences independently of the LLM pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from homonym_pipeline.config import load_config
from homonym_pipeline.retrieval.grac import GracClient, GracError
from homonym_pipeline.storage import atomic_write_json, atomic_write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve ГРАК sentences containing a lemma")
    parser.add_argument("--lemma", required=True)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--output", default="outputs/grac/examples.json",
                        help="JSON list or JSONL, selected by file extension")
    parser.add_argument("--cache-dir", default="outputs/grac/cache")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--corpus", help="Backend corpus ID, e.g. grac19")
    parser.add_argument("--page-size", type=int)
    parser.add_argument("--seed", type=int, help="Sample reproducible random sentence ranks")
    parser.add_argument("--force", action="store_true", help="Refresh caches; retain raw responses")
    args = parser.parse_args()
    config = load_config(args.config)
    updates = {key: value for key, value in {
        "corpus": args.corpus, "page_size": args.page_size, "seed": args.seed,
    }.items() if value is not None}
    config.grac = type(config.grac).model_validate({**config.grac.model_dump(), **updates})
    if args.max_examples < 0:
        parser.error("--max-examples must be >= 0")
    try:
        with GracClient.from_config(config.grac, cache_dir=Path(args.cache_dir),
                                    resume=not args.force) as grac:
            examples = grac.retrieve_examples(args.lemma, args.max_examples)
    except (GracError, ValueError) as error:
        print(f"ГРАК retrieval failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    target = Path(args.output)
    if target.suffix.lower() == ".jsonl":
        atomic_write_jsonl(target, examples)
    else:
        atomic_write_json(target, [item.model_dump(mode="json") for item in examples])
    print(f"Saved {len(examples)} sentences for {args.lemma!r} to {target}")


if __name__ == "__main__":
    main()
