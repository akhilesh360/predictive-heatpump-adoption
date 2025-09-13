import json
import logging
from pathlib import Path

def setup_logger(name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj, path: str | Path):
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w") as f:
        json.dump(obj, f, indent=2)
