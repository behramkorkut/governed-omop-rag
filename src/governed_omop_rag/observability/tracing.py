"""Observabilité LLM (Langfuse) — traces, coûts et latences de la boucle d'agents.

Pourquoi
--------
L'architecture borne le coût par conception (router déterministe d'abord, LLM sur
le seul résidu, cache de retrieval). Cette borne était jusqu'ici une *assertion
d'architecture* : le tracing la rend **mesurable** — coût par mapping, part réelle
des requêtes atteignant le LLM, surcoût des reprises quand le Vérificateur rejette.

Contrat
-------
Le tracing est **strictement optionnel** et ne doit jamais faire échouer un mapping :

* SDK importé paresseusement (extra ``observability``) ;
* ``GOR_LANGFUSE_ENABLED=false`` (défaut en CI/tests) court-circuite tout ;
* clés absentes ou initialisation en erreur -> ``None`` + log, pas d'exception.

SDK Langfuse v4 : le client global est instancié une fois avec les clés, puis
``CallbackHandler()`` le résout — le handler ne prend plus ni secret ni host.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from governed_omop_rag.config import get_settings
from governed_omop_rag.core.logging import get_logger

logger = get_logger(__name__)


def langfuse_enabled() -> bool:
    """Vrai si le tracing est activé ET correctement configuré."""
    settings = get_settings()
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    )


@lru_cache(maxsize=1)
def _init_client() -> Any | None:
    """Instancie (une seule fois) le client Langfuse global. ``None`` si indisponible."""
    if not langfuse_enabled():
        return None

    settings = get_settings()
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning(
            "langfuse_absent",
            hint="uv sync --extra observability",
            effect="tracing désactivé, exécution normale",
        )
        return None

    try:
        assert settings.langfuse_public_key is not None
        assert settings.langfuse_secret_key is not None
        client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
            environment=str(settings.env),
        )
    except Exception as exc:  # noqa: BLE001 - le tracing ne doit jamais casser le mapping
        logger.warning("langfuse_init_echec", error=str(exc))
        return None

    logger.info("langfuse_actif", host=settings.langfuse_host, environment=str(settings.env))
    return client


@lru_cache(maxsize=1)
def get_langfuse_callback() -> Any | None:
    """Retourne le ``CallbackHandler`` LangChain/LangGraph, ou ``None``.

    Mémoïsé : un seul handler pour tout le processus. Toute erreur est absorbée
    et dégrade en ``None`` (exécution sans tracing).
    """
    if _init_client() is None:
        return None

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.warning(
            "langfuse_langchain_absent",
            hint="uv sync --extra observability",
        )
        return None

    try:
        return CallbackHandler()
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse_handler_echec", error=str(exc))
        return None


@contextmanager
def observe_generation(
    *,
    name: str,
    model: str,
    input: Any = None,  # noqa: A002 - nom imposé par l'API Langfuse
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Enveloppe un appel LLM **direct** (SDK Anthropic) dans un span ``generation``.

    Le ``CallbackHandler`` LangChain ne voit que ce qui transite par LangChain :
    nos nœuds LangGraph sont tracés, mais l'appel ``client.messages.create`` qu'ils
    contiennent reste invisible. Sans ce span, pas de tokens, pas de coût, pas de
    modèle dans Langfuse — donc pas de mesure de la borne de coût.

    Cède ``None`` quand le tracing est inactif : l'appelant garde le même code.
    Les exceptions du bloc remontent normalement (et sont enregistrées par le span).
    """
    client = _init_client()
    manager = None
    if client is not None:
        try:
            manager = client.start_as_current_observation(
                name=name,
                as_type="generation",
                model=model,
                input=input,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - le tracing ne doit jamais casser l'appel
            logger.warning("langfuse_generation_echec", error=str(exc))
            manager = None

    if manager is None:
        yield None
        return

    with manager as generation:
        yield generation


def flush() -> None:
    """Vide la file d'envoi (utile en script court : CLI, évaluation, tests manuels)."""
    client = _init_client()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("langfuse_flush_echec", error=str(exc))
