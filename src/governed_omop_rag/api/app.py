"""API REST (FastAPI) — exposition du pipeline de mapping gouverné.

Deux portes d'entrée pour l'outil (CONTEXT.md §11) : cette API pour les
intégrateurs, l'UI Streamlit pour les non-devs. Le cœur métier est le
``MappingService`` **partagé** — pas de logique dupliquée.

Import de FastAPI au niveau module : ce fichier n'est chargé que lorsque l'extra
``api`` est installé (par ``gor serve`` ou les tests via importorskip).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from governed_omop_rag.config import Settings
from governed_omop_rag.core.models import MappingRequest, MappingSuggestion
from governed_omop_rag.service import MappingService, MapStrategy


class BatchRequest(BaseModel):
    """Corps de /map/batch : une liste d'entrées + une stratégie."""

    items: list[MappingRequest] = Field(default_factory=list)
    strategy: MapStrategy = MapStrategy.AUTO


class BatchResponse(BaseModel):
    """Réponse de /map/batch."""

    strategy: MapStrategy
    results: list[MappingSuggestion]


def create_app(settings: Settings | None = None, bronze_dir: Path | None = None) -> FastAPI:
    """Fabrique l'application FastAPI (pipeline construit au démarrage)."""
    service = MappingService(settings, bronze_dir)

    app = FastAPI(
        title="governed-omop-rag",
        summary="Mapping CIM-10 FR / libellés -> concepts standard OMOP (gouverné).",
        version="0.0.1",
    )
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "concepts_indexed": service.concepts_indexed}

    @app.post("/map")
    def map_one(
        request: MappingRequest, strategy: MapStrategy = MapStrategy.AUTO
    ) -> MappingSuggestion:
        return service.route(request, strategy)

    @app.post("/map/batch")
    def map_batch(body: BatchRequest) -> BatchResponse:
        results = service.map_many(body.items, body.strategy)
        return BatchResponse(strategy=body.strategy, results=results)

    return app
