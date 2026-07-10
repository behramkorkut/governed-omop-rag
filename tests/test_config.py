"""Tests de la configuration typée (pydantic-settings)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from governed_omop_rag.config import AppEnv, Settings, VectorBackend, get_settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isole les tests d'un éventuel .env local et de variables GOR_ héritées.

    On se place dans un répertoire temporaire (sans .env) et on purge les
    variables d'environnement GOR_* pour tester les vrais défauts.
    """
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("GOR_"):
            monkeypatch.delenv(key, raising=False)


def test_defaults_are_sane() -> None:
    s = Settings()
    assert s.env is AppEnv.DEV
    assert s.vector_backend is VectorBackend.QDRANT
    assert s.qdrant_url == "http://localhost:6333"
    assert s.top_k == 10
    assert 0.0 <= s.confidence_threshold <= 1.0
    # Aucun secret par défaut.
    assert s.anthropic_api_key is None


def test_derived_medallion_paths() -> None:
    s = Settings(data_dir=Path("data"))
    assert s.bronze_dir == Path("data/bronze")
    assert s.silver_dir == Path("data/silver")
    assert s.gold_dir == Path("data/gold")


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOR_ENV", "prod")
    monkeypatch.setenv("GOR_TOP_K", "5")
    monkeypatch.setenv("GOR_CONFIDENCE_THRESHOLD", "0.8")
    s = Settings()
    assert s.env is AppEnv.PROD
    assert s.top_k == 5
    assert s.confidence_threshold == 0.8


def test_confidence_threshold_out_of_bounds_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(confidence_threshold=1.5)


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(top_k=0)


def test_get_settings_is_singleton() -> None:
    assert get_settings() is get_settings()
