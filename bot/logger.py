"""
Structured logging for the trading bot.
Logs to both console and rotating JSON files.
"""
import os
import json
import logging
import logging.handlers
from datetime import datetime, timezone
from bot.config import LOG_DIR, LOG_LEVEL

os.makedirs(LOG_DIR, exist_ok=True)


class JsonFormatter(logging.Formatter):
    """Outputs log records as single-line JSON for easy parsing."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger that writes to console + file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Console handler — human readable
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(console)

    # File handler — JSON, rotates daily
    file_handler = logging.handlers.TimedRotatingFileHandler(
        os.path.join(LOG_DIR, "bot.jsonl"),
        when="midnight",
        backupCount=30,
        utc=True,
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    return logger


def log_trade(trade_data: dict):
    """Append a trade record to the trade journal."""
    trade_data["logged_at"] = datetime.now(timezone.utc).isoformat()
    path = os.path.join(LOG_DIR, "trades.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(trade_data) + "\n")


def log_cycle(cycle_data: dict):
    """Append a cycle snapshot (features, scores, portfolio state)."""
    cycle_data["logged_at"] = datetime.now(timezone.utc).isoformat()
    path = os.path.join(LOG_DIR, "cycles.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(cycle_data) + "\n")
