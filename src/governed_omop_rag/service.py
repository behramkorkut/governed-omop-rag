"""MappingService — assemble le pipeline complet, partagé par l'API et l'UI.

Source de vérité unique du cœur métier (CONTEXT.md §11) : l'API REST et l'UI
Streamlit consomment CE service, il n'y a pas de logique dupliquée. Neutre :
aucune dépendance à FastAPI ni Streamlit.

Le pipeline (corpus -> index -> router hybride + agent gouverné) est construit une
fois ; ``route`` applique la stratégie choisie.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

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
    """Stratégie de routage (levier coût/qualité)."""

    AUTO = "auto"  # déterministe d'abord, RAG sur le résidu (défaut)
    DETERMINISTIC_ONLY = "deterministic_only"  # alignement officiel seulement
    FULL_RAG = "full_rag"  # tout passe par le RAG (agent)


class MappingService:
    """Pipeline de mapping gouverné, construit une fois puis réutilisé."""

    def __init__(self, settings: Settings | None = None, bronze_dir: Path | None = None) -> None:
        s = settings or get_settings()
        embedder = get_embedder(s)
        store = get_vectorstore(s)
        con = connect(":memory:")
        try:
            build_corpus(con, bronze_dir or s.bronze_dir)
            gold = fetch_gold(con)
        finally:
            con.close()
        self.concepts_indexed = index_gold(gold, embedder, store)

        retriever = build_retriever("hybrid", gold, embedder, store)
        official_map = OfficialMap.from_csv(s.router_map_path)
        agent = MappingAgent(Proposer(build_proposer_llm(s)), Verifier())

        self._deterministic = DeterministicRouter(official_map)
        self._auto = HybridRouter(official_map, retriever, s.confidence_threshold, s.top_k, agent)
        # Map officielle vide -> le déterministe échoue toujours -> tout part au RAG.
        self._full_rag = HybridRouter(
            OfficialMap({}), retriever, s.confidence_threshold, s.top_k, agent
        )

    def route(
        self, request: MappingRequest, strategy: MapStrategy = MapStrategy.AUTO
    ) -> MappingSuggestion:
        """Route une entrée selon la stratégie."""
        if strategy is MapStrategy.DETERMINISTIC_ONLY:
            return self._deterministic.route(request)
        if strategy is MapStrategy.FULL_RAG:
            return self._full_rag.route(request)
        return self._auto.route(request)

    def map_many(
        self,
        requests: Sequence[MappingRequest],
        strategy: MapStrategy = MapStrategy.AUTO,
    ) -> list[MappingSuggestion]:
        """Route un lot d'entrées."""
        return [self.route(r, strategy) for r in requests]
