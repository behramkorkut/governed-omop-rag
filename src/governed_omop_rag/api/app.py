"""API REST (FastAPI) — exposition du pipeline de mapping gouverné.

Deux portes d'entrée pour l'outil (CONTEXT.md §11) : cette API pour les
intégrateurs, l'UI Streamlit pour les non-devs. Le cœur métier (router, agent)
est **partagé** — pas de logique dupliquée.

Le pipeline (corpus -> index -> router hybride + agent) est construit **une fois**
au démarrage puis réutilisé (le rebuild par requête serait coûteux).

Import de FastAPI au niveau module : ce fichier n'est chargé que lorsque l'extra
``api`` est installé (par l'app ``gor serve`` ou les tests via importorskip).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from governed_omop_rag.agents.factory import build_proposer_llm
from governed_omop_rag.agents.orchestrator import MappingAgent
from governed_omop_rag.agents.proposer import Proposer
from governed_omop_rag.agents.verifier import Verifier
from governed_omop_rag.config import Settings, get_settings
from governed_omop_rag.core.models import MappingRequest, MappingSuggestion
from governed_omop_rag.medallion.db import connect
from governed_omop_rag.medallion.gold import fetch_gold
from governed_omop_rag.medallion.pipeline import build_corpus
from governed_omop_rag.retrieval.factory import get_embedder, get_vectorstore
from governed_omop_rag.retrieval.index import index_gold
from governed_omop_rag.retrieval.retriever import build_retriever
from governed_omop_rag.router.deterministic import DeterministicRouter, OfficialMap
from governed_omop_rag.router.hybrid import HybridRouter


class MapStrategy(StrEnum):
    """Stratégie de routage d'un batch (levier coût/qualité)."""

    AUTO = "auto"  # déterministe d'abord, RAG sur le résidu (défaut)
    DETERMINISTIC_ONLY = "deterministic_only"  # alignement officiel seulement
    FULL_RAG = "full_rag"  # tout passe par le RAG (agent)


class BatchRequest(BaseModel):
    """Corps de /map/batch : une liste d'entrées + une stratégie."""

    items: list[MappingRequest] = Field(default_factory=list)
    strategy: MapStrategy = MapStrategy.AUTO


class BatchResponse(BaseModel):
    """Réponse de /map/batch."""

    strategy: MapStrategy
    results: list[MappingSuggestion]


class _Pipeline:
    """Composants métier construits une fois, partagés par les routes."""

    def __init__(self, settings: Settings, bronze_dir: Path | None = None) -> None:
        embedder = get_embedder(settings)
        store = get_vectorstore(settings)
        con = connect(":memory:")
        try:
            build_corpus(con, bronze_dir or settings.bronze_dir)
            gold = fetch_gold(con)
        finally:
            con.close()
        self.concepts_indexed = index_gold(gold, embedder, store)

        retriever = build_retriever("hybrid", gold, embedder, store)
        official_map = OfficialMap.from_csv(settings.router_map_path)
        agent = MappingAgent(Proposer(build_proposer_llm(settings)), Verifier())

        self._deterministic = DeterministicRouter(official_map)
        self._auto = HybridRouter(
            official_map, retriever, settings.confidence_threshold, settings.top_k, agent
        )
        # Map officielle vide -> le déterministe échoue toujours -> tout part au RAG.
        self._full_rag = HybridRouter(
            OfficialMap({}), retriever, settings.confidence_threshold, settings.top_k, agent
        )

    def route(self, request: MappingRequest, strategy: MapStrategy) -> MappingSuggestion:
        if strategy is MapStrategy.DETERMINISTIC_ONLY:
            return self._deterministic.route(request)
        if strategy is MapStrategy.FULL_RAG:
            return self._full_rag.route(request)
        return self._auto.route(request)


def create_app(settings: Settings | None = None, bronze_dir: Path | None = None) -> FastAPI:
    """Fabrique l'application FastAPI (pipeline construit au démarrage)."""
    settings = settings or get_settings()
    pipeline = _Pipeline(settings, bronze_dir)

    app = FastAPI(
        title="governed-omop-rag",
        summary="Mapping CIM-10 FR / libellés -> concepts standard OMOP (gouverné).",
        version="0.0.1",
    )
    app.state.pipeline = pipeline

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "concepts_indexed": pipeline.concepts_indexed}

    @app.post("/map")
    def map_one(
        request: MappingRequest, strategy: MapStrategy = MapStrategy.AUTO
    ) -> MappingSuggestion:
        return pipeline.route(request, strategy)

    @app.post("/map/batch")
    def map_batch(body: BatchRequest) -> BatchResponse:
        results = [pipeline.route(item, body.strategy) for item in body.items]
        return BatchResponse(strategy=body.strategy, results=results)

    return app
