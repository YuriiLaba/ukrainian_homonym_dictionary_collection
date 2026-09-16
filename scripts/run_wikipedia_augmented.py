from __future__ import annotations

import argparse
import logging

from homonym_pipeline.config import load_config
from homonym_pipeline.output.huggingface import write_huggingface
from homonym_pipeline.output.statistics import calculate_statistics
from homonym_pipeline.output.writer import write_final
from homonym_pipeline.pipeline.wikipedia_augmented import run_wikipedia_augmented


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for logger_name in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Run the Wikipedia-augmented Ukrainian homonym pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-lemmas", type=int)
    parser.add_argument("--max-grac-examples", type=int)
    parser.add_argument("--batch-size", type=int)
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
    if args.batch_size is not None:
        config.validation.batch_size = args.batch_size
    if args.max_final_examples_per_sense is not None:
        config.validation.max_final_examples_per_sense = args.max_final_examples_per_sense
    if args.validation_model:
        config.llm.model_validation = args.validation_model
    if args.gloss_model:
        config.llm.model_gloss = args.gloss_model
    from homonym_pipeline.llm.client import LLMClient
    grac = None
    if args.grac_fixture:
        from homonym_pipeline.retrieval.grac import JsonFileGracClient
        grac = JsonFileGracClient(args.grac_fixture)
    entries = run_wikipedia_augmented(args.input, args.output, config, llm=LLMClient(config.llm, dry_run=args.dry_run), grac=grac, max_lemmas=args.max_lemmas, resume=config.pipeline.resume)
    logging.info("Stage 3/3: writing final dictionary, Hugging Face export, and statistics.")
    write_final(entries, args.output)
    write_huggingface(entries, args.output)
    calculate_statistics(entries, args.output)


if __name__ == "__main__":
    main()
