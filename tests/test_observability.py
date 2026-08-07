"""Tests de l'observabilité LLM (Langfuse).

Contrat vérifié ici : le tracing est **strictement optionnel**. Ces tests
tournent hors-ligne, sans clé et sans SDK installé — comme la CI.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest
from pydantic import SecretStr

from governed_omop_rag import observability
from governed_omop_rag.config import Settings
from governed_omop_rag.observability import tracing


@pytest.fixture(autouse=True)
def _clear_caches() -> Any:
    """Vide les caches mémoïsés entre les tests (client + handler)."""
    tracing._init_client.cache_clear()
    tracing.get_langfuse_callback.cache_clear()
    yield
    tracing._init_client.cache_clear()
    tracing.get_langfuse_callback.cache_clear()


def _settings(**kwargs: Any) -> Settings:
    """Settings isolés de l'environnement réel (pas de lecture du .env)."""
    base: dict[str, Any] = {
        "langfuse_enabled": False,
        "langfuse_public_key": None,
        "langfuse_secret_key": None,
    }
    base.update(kwargs)
    return Settings(_env_file=None, **base)


def test_desactive_par_defaut(monkeypatch: pytest.MonkeyPatch) -> None:
    """Défaut = tracing coupé : CI hors-ligne, aucun secret requis."""
    monkeypatch.setattr(tracing, "get_settings", lambda: _settings())
    assert tracing.langfuse_enabled() is False
    assert tracing.get_langfuse_callback() is None


def test_active_mais_sans_cles_reste_inactif(monkeypatch: pytest.MonkeyPatch) -> None:
    """`enabled=true` sans clés ne doit pas activer le tracing (ni lever)."""
    monkeypatch.setattr(tracing, "get_settings", lambda: _settings(langfuse_enabled=True))
    assert tracing.langfuse_enabled() is False
    assert tracing.get_langfuse_callback() is None


def test_sdk_absent_degrade_sans_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """SDK non installé : on renvoie None, on ne casse pas le mapping."""
    monkeypatch.setattr(
        tracing,
        "get_settings",
        lambda: _settings(
            langfuse_enabled=True,
            langfuse_public_key=SecretStr("pk-test"),
            langfuse_secret_key=SecretStr("sk-test"),
        ),
    )
    # Simule l'absence du paquet `langfuse`.
    monkeypatch.setitem(sys.modules, "langfuse", None)
    assert tracing.langfuse_enabled() is True
    assert tracing.get_langfuse_callback() is None


def test_erreur_init_client_degrade(monkeypatch: pytest.MonkeyPatch) -> None:
    """Une exception à l'init du client ne remonte pas à l'appelant."""
    monkeypatch.setattr(
        tracing,
        "get_settings",
        lambda: _settings(
            langfuse_enabled=True,
            langfuse_public_key=SecretStr("pk-test"),
            langfuse_secret_key=SecretStr("sk-test"),
        ),
    )
    monkeypatch.setattr(tracing, "_init_client", lambda: None)
    assert tracing.get_langfuse_callback() is None


def test_flush_sans_client_est_silencieux(monkeypatch: pytest.MonkeyPatch) -> None:
    """`flush()` est sûr même sans client actif."""
    monkeypatch.setattr(tracing, "get_settings", lambda: _settings())
    tracing.flush()  # ne doit pas lever


def test_api_publique_exposee() -> None:
    """Le paquet expose bien son API publique."""
    assert hasattr(observability, "get_langfuse_callback")
    assert hasattr(observability, "langfuse_enabled")


def test_trace_config_vide_si_tracing_inactif(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sans tracing, la config passée à LangGraph reste vide (comportement inchangé)."""
    from governed_omop_rag.agents.graph import LangGraphMappingAgent

    monkeypatch.setattr(tracing, "get_settings", lambda: _settings())

    agent = LangGraphMappingAgent(proposer=object(), verifier=object())  # type: ignore[arg-type]
    request = _make_request()
    assert agent._trace_config(request, [], expected_domain=None) == {}


def test_trace_config_porte_metadonnees(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avec un handler, la config porte callbacks, tags et métadonnées filtrables."""
    from governed_omop_rag.agents import graph as graph_mod
    from governed_omop_rag.agents.graph import LangGraphMappingAgent

    sentinel = object()
    monkeypatch.setattr(
        "governed_omop_rag.observability.get_langfuse_callback",
        lambda: sentinel,
    )
    assert graph_mod is not None

    agent = LangGraphMappingAgent(proposer=object(), verifier=object(), max_attempts=3)  # type: ignore[arg-type]
    config = agent._trace_config(_make_request(), [], expected_domain="Condition")

    assert config["callbacks"] == [sentinel]
    assert config["run_name"] == "mapping_agent"
    assert config["metadata"]["max_attempts"] == 3
    assert config["metadata"]["expected_domain"] == "Condition"
    assert "proposer-verifier" in config["tags"]


def _make_request() -> Any:
    """Requête minimale de mapping (évite d'importer tout le domaine ici)."""
    from governed_omop_rag.core.models import MappingRequest

    return MappingRequest(source_code="E11", source_label="diabète de type 2")
