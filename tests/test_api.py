"""Tests de l'API FastAPI (skip si l'extra api n'est pas installé)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from governed_omop_rag.api.app import create_app  # noqa: E402
from governed_omop_rag.config import (  # noqa: E402
    EmbeddingBackend,
    Settings,
    VectorBackend,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
# Map officielle FIXTURE (contrôlée) : le test ne doit pas dépendre du fichier de
# production réel (data/router/…), dont le contenu évolue avec le vrai alignement ATIH.
# Contient E11.9 -> 201826 ; Z99.9 en est volontairement ABSENT (cas « unmapped »).
ROUTER_MAP = FIXTURES / "router_map.csv"


def _client() -> TestClient:
    settings = Settings(
        embedding_backend=EmbeddingBackend.HASHING,
        vector_backend=VectorBackend.MEMORY,
        embedding_dim=512,
        router_map_path=ROUTER_MAP,
        # Force le Proposer déterministe (offline) quelle que soit la clé du .env local.
        anthropic_api_key=None,
    )
    # bronze_dir est une property (data_dir/bronze) : on l'injecte à create_app.
    return TestClient(create_app(settings, bronze_dir=FIXTURES))


def test_health() -> None:
    r = _client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["concepts_indexed"] == 4


def test_map_official() -> None:
    r = _client().post("/map", json={"source_code": "E11.9", "source_vocabulary": "ICD10FR"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "official_map"
    assert body["target_concept_id"] == 201826


def test_map_rag_on_label() -> None:
    r = _client().post("/map", json={"source_label": "diabète de type 2"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "rag"
    assert body["target_concept_id"] == 201826


def test_map_deterministic_only_unknown() -> None:
    r = _client().post(
        "/map", json={"source_code": "Z99.9"}, params={"strategy": "deterministic_only"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "unmapped"
    assert body["target_concept_id"] == 0


def test_map_full_rag_bypasses_official() -> None:
    # E11.9 est dans la map officielle, mais full_rag force le passage au RAG.
    r = _client().post("/map", json={"source_code": "E11.9"}, params={"strategy": "full_rag"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] in {"rag", "unmapped"}
    assert body["source"] != "official_map"


def test_map_batch() -> None:
    r = _client().post(
        "/map/batch",
        json={
            "items": [
                {"source_code": "E11.9"},
                {"source_label": "asthme"},
            ],
            "strategy": "auto",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "auto"
    assert len(body["results"]) == 2
    assert body["results"][0]["source"] == "official_map"


def test_map_invalid_request_422() -> None:
    # Ni code ni libellé -> validation Pydantic -> 422.
    r = _client().post("/map", json={})
    assert r.status_code == 422
