"""Logging structuré (structlog).

Chaque décision de mapping doit être traçable (cf. gouvernance §4.3 / §7).
On configure un logger JSON-friendly, réutilisé par l'API, l'UI et les agents.
"""

from __future__ import annotations

import logging
from typing import cast

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog + stdlib logging de façon idempotente."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Retourne un logger structuré lié au nom donné."""
    if not _CONFIGURED:
        configure_logging()
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
