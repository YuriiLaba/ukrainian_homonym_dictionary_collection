from __future__ import annotations

import logging
from pathlib import Path


def configure_pipeline_logging(output_dir: str | Path) -> Path:
    """Send the same concise pipeline messages to stdout and a run log file."""
    log_path = Path(output_dir) / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
        force=True,
    )
    for logger_name in ("httpx", "httpx2", "httpcore", "openai", "openai._base_client"):
        transport_logger = logging.getLogger(logger_name)
        transport_logger.setLevel(logging.WARNING)
        transport_logger.propagate = False
    return log_path
