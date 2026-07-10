import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: str = "logs",
    log_file: str = "causal_gauge_field.log",
    level: str = "INFO",
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    if logger.handlers:
        return logger
    formatter = logging.Formatter(fmt)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path / log_file, encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper()))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper()))
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger