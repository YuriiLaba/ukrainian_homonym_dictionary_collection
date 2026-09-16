from __future__ import annotations

import argparse
import logging

from homonym_pipeline.config import load_config
from homonym_pipeline.output.huggingface import write_huggingface
from homonym_pipeline.output.manifest import finish_manifest, start_manifest
from homonym_pipeline.output.statistics import calculate_statistics
from homonym_pipeline.output.writer import write_final
from homonym_pipeline.pipeline.wikipedia_augmented import run_wikipedia_augmented


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for logger_name in ("httpx", "httpx2", "httpcore", "openai", "openai._base_client"):
        transport_logger = logging.getLogger(logger_name)
        transport_logger.setLevel(logging.WARNING)
        transport_logger.propagate = False
    parser = argparse.ArgumentParser(description="Run the Wikipedia-augmented Ukrainian homonym pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-lemmas", type=int)
    parser.add_argument("--max-grac-examples", type=int)
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-top-k", type=int)
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument("--embedding-max-concurrency", type=int)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--max-final-examples-per-sense", type=int)
    parser.add_argument("--llm-model", "--validation-model", dest="validation_model")
    parser.add_argument("--gloss-model")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--grac-fixture", help="JSON export mapping lemmas to candidate examples")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.resume is not None:
        config.pipeline.resume = args.resume
    if args.force:
        config.pipeline.resume = False
    if args.max_grac_examples is not None:
        config.grac.max_examples_per_lemma = args.max_grac_examples
    if args.embedding_model:
        config.embeddings.model = args.embedding_model
    if args.embedding_top_k is not None:
        config.validation.max_candidates_per_gloss = args.embedding_top_k
    if args.embedding_batch_size is not None:
        config.embeddings.batch_size = args.embedding_batch_size
    if args.embedding_max_concurrency is not None:
        config.embeddings.max_concurrency = args.embedding_max_concurrency
    if args.no_embeddings:
        config.embeddings.enabled = False
    if args.batch_size is not None:
        config.validation.batch_size = args.batch_size
    if args.max_concurrency is not None:
        config.validation.max_concurrency = args.max_concurrency
    if args.max_final_examples_per_sense is not None:
        config.validation.max_final_examples_per_sense = args.max_final_examples_per_sense
    if args.validation_model:
        config.llm.model_validation = args.validation_model
    if args.gloss_model:
        config.llm.model_gloss = args.gloss_model
    manifest = start_manifest("wikipedia_augmented", args.input, args.output, config)
    from homonym_pipeline.llm.client import LLMClient
    from homonym_pipeline.embeddings.client import EmbeddingClient
    grac = None
    if args.grac_fixture:
        from homonym_pipeline.retrieval.grac import JsonFileGracClient
        grac = JsonFileGracClient(args.grac_fixture)
    try:
        entries = run_wikipedia_augmented(
            args.input, args.output, config,
            llm=LLMClient(config.llm, dry_run=args.dry_run),
            embedder=EmbeddingClient(config.embeddings, dry_run=args.dry_run),
            grac=grac, max_lemmas=args.max_lemmas, resume=config.pipeline.resume,
            run_id=manifest["run_id"],
        )
        logging.info("[stage 3/3] Writing final dictionary, Hugging Face export, and statistics.")
        write_final(entries, args.output)
        write_huggingface(entries, args.output)
        stats = calculate_statistics(entries, args.output, run_id=manifest["run_id"])
        status = "completed_with_failures" if stats.get("failed_lemmas", 0) else "completed"
        finish_manifest(args.output, status=status, statistics=stats)
    except Exception:
        finish_manifest(args.output, status="failed")
        raise


if __name__ == "__main__":
    main()
