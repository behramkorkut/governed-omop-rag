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
from governed_omop_rag.core.models import (
    MappingRequest,
    MappingSource,
    MappingSuggestion,
    NoMapReason,
)
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

    def __init__(
        self,
        settings: Settings | None = None,
        bronze_dir: Path | None = None,
        domains: Sequence[str] | None = None,
        retriever_kind: str = "hybrid",
        reuse_index: bool = False,
    ) -> None:
        s = settings or get_settings()
        embedder = get_embedder(s)
        store = get_vectorstore(s)
        con = connect(":memory:")
        try:
            build_corpus(con, bronze_dir or s.bronze_dir, domains or s.corpus_domains)
            gold = fetch_gold(con)
        finally:
            con.close()
        # reuse_index : la collection vectorielle est déjà remplie (évite de
        # ré-embarquer ~10^5 concepts entre deux évaluations).
        self.concepts_indexed = store.count() if reuse_index else index_gold(gold, embedder, store)

        retriever = build_retriever(retriever_kind, gold, embedder, store)
        official_map = OfficialMap.from_csv(s.router_map_path)
        self._llm = build_proposer_llm(s)
        # L'abstention est portée par le ROUTER (min_margin), pas par l'agent, pour
        # couvrir tout orchestrateur et tout point d'entrée (audit G1).
        agent = MappingAgent(Proposer(self._llm), Verifier())

        self._deterministic = DeterministicRouter(official_map)
        self._auto = HybridRouter(
            official_map,
            retriever,
            s.confidence_threshold,
            s.top_k,
            agent,
            min_margin=s.agent_min_margin,
        )
        # Map officielle vide -> le déterministe échoue toujours -> tout part au RAG.
        self._full_rag = HybridRouter(
            OfficialMap({}),
            retriever,
            s.confidence_threshold,
            s.top_k,
            agent,
            min_margin=s.agent_min_margin,
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
        """Route un lot d'entrées, chaque item ISOLÉ des autres.

        Une exception non-parsing (API LLM en panne après retries, Qdrant down…)
        sur UN item ne doit pas faire échouer tout le lot : on la capture et on
        rend cet item `UNMAPPED/ERREUR_AGENT`, puis on continue (fin de G2 —
        les erreurs de parsing sont déjà dégradées en amont par l'orchestrateur)."""
        results: list[MappingSuggestion] = []
        for r in requests:
            try:
                results.append(self.route(r, strategy))
            except Exception:  # noqa: BLE001 — dégradation propre, item isolé
                results.append(
                    MappingSuggestion(
                        request=r,
                        source=MappingSource.UNMAPPED,
                        no_map_reason=NoMapReason.ERREUR_AGENT,
                        justification="Échec de l'agent sur cet item (isolé du lot).",
                    )
                )
        return results

    def token_usage(self) -> tuple[int, int]:
        """Tokens LLM cumulés (input, output) — 0 avec le Proposer hors-ligne."""
        return (
            int(getattr(self._llm, "input_tokens", 0)),
            int(getattr(self._llm, "output_tokens", 0)),
        )
