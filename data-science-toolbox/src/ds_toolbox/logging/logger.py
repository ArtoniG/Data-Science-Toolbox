"""
src/credit_toolbox/logging/logger.py

Enterprise-grade structured JSON logger.
Instead of writing flat text lines (which are impossible to query in Datadog/GCP Cloud Logging),
this outputs strictly formatted JSON. This ensures that every log event can be parsed, 
filtered, and audited by compliance and DevOps teams.
"""

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StructuredJSONFormatter(logging.Formatter):
    """
    Custom formatter that transforms standard Python LogRecords into strict JSON strings.
    Automatically captures exception stack traces and custom kwargs passed via `extra`.
    """

    # Base attributes of a Python LogRecord that we want to exclude from the dynamic 'extra' payload
    _SKIP_ATTRIBUTES = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName"
    }

    def format(self, record: logging.LogRecord) -> str:
        """Constructs the JSON payload for a single log event."""
        
        # 1. Base required payload for Cloud Logging / ELK Stack
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger_name": record.name,
            "module": record.module,
            "function": record.funcName,
            "line_number": record.lineno,
            "message": record.getMessage(),
        }

        # 2. Extract dynamically injected variables (like audit_id, session_id, duration_ms)
        for key, value in record.__dict__.items():
            if key not in self._SKIP_ATTRIBUTES:
                payload[key] = value

        # 3. Handle Exceptions cleanly
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Unknown"
            payload["exception_message"] = str(record.exc_info[1])
            payload["stack_trace"] = "".join(traceback.format_exception(*record.exc_info))

        return json.dumps(payload, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured structured JSON logger. 
    Guarantees no duplicate handlers if called multiple times in a Jupyter notebook or pipeline.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Model trained", extra={"audit_id": "A-123", "gini": 0.65})
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent log messages from propagating to the root logger (avoids double-printing)
    logger.propagate = False

    # Attach the handler only if it hasn't been attached yet
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredJSONFormatter())
        logger.addHandler(console_handler)

    return logger