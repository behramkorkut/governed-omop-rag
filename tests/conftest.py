"""Configuration commune de la suite de tests.

Neutralise l'observabilité Langfuse pendant les tests.

Pourquoi c'est nécessaire
-------------------------
``Settings`` lit le ``.env`` du développeur. Si celui-ci contient
``GOR_LANGFUSE_ENABLED=true``, la suite de tests émet de vraies traces vers le
projet Langfuse — avec deux conséquences fâcheuses :

1. **Pollution des mesures.** ``test_graph.py`` exerce volontairement des rejets
   du Vérificateur et des boucles de reprise. Mélangées aux exécutions réelles,
   ces traces faussent le taux de rejet observé.
2. **Tests non hermétiques.** Appels réseau, latence et dépendance à un service
   externe dans une suite qui doit rester hors-ligne et déterministe.

La variable d'environnement est posée **au chargement du module** : pytest importe
``conftest.py`` avant les modules de test, donc avant toute construction de
``Settings``. Les variables d'environnement priment sur le fichier ``.env``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

# Doit précéder tout import de `Settings` par les modules de test.
os.environ["GOR_LANGFUSE_ENABLED"] = "false"


@pytest.fixture(autouse=True, scope="session")
def _tracing_desactive() -> Iterator[None]:
    """Garantit qu'aucun client ni handler Langfuse ne subsiste d'un import antérieur."""
    from governed_omop_rag.config import get_settings
    from governed_omop_rag.observability import tracing

    get_settings.cache_clear()
    tracing.get_client.cache_clear()
    tracing._init_client.cache_clear()
    tracing.get_langfuse_callback.cache_clear()

    assert not tracing.langfuse_enabled(), (
        "Le tracing Langfuse doit être désactivé pendant les tests (vérifier GOR_LANGFUSE_ENABLED)."
    )
    yield
