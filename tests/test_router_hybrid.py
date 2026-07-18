"""Tests du Router hybride (déterministe -> RAG sur le résidu)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from governed_omop_rag.agents.llm import FakeProposerLLM
from governed_omop_rag.agents.orchestrator import MappingAgent
from governed_omop_rag.agents.proposer import Proposer
from governed_omop_rag.agents.verifier import Verifier
from governed_omop_rag.core.models import (
    UNMAPPED_CONCEPT_ID,
    ConceptCandidate,
    MappingRequest,
    MappingSource,
    MappingSuggestion,
    NoMapReason,
)
from governed_omop_rag.medallion.db import connect
from governed_omop_rag.medallion.gold import fetch_gold
from governed_omop_rag.medallion.pipeline import build_corpus
from governed_omop_rag.retrieval.embeddings import HashingEmbedder
from governed_omop_rag.retrieval.index import index_gold
from governed_omop_rag.retrieval.retriever import DenseRetriever
from governed_omop_rag.retrieval.vectorstore import MemoryVectorStore
from governed_omop_rag.router.deterministic import OfficialMap
from governed_omop_rag.router.hybrid import HybridRouter

FIXTURES = Path(__file__).parent / "fixtures"


class StubRetriever:
    """Retriever de test : renvoie des candidats préconfigurés et compte les appels."""

    def __init__(self, candidates: list[ConceptCandidate]) -> None:
        self.candidates = candidates
        self.calls = 0

    def retrieve(self, query: str, top_k: int = 10) -> list[ConceptCandidate]:
        self.calls += 1
        return self.candidates[:top_k]


def _cand(concept_id: int, score: float, name: str = "x") -> ConceptCandidate:
    return ConceptCandidate(
        concept_id=concept_id,
        concept_name=name,
        vocabulary_id="SNOMED",
        domain_id="Condition",
        standard_concept="S",
        score=score,
    )


# --------------------------------------------------------------------------- #
# 1. Le match officiel court-circuite le retrieval
# --------------------------------------------------------------------------- #
def test_official_match_short_circuits_retriever() -> None:
    stub = StubRetriever([_cand(999, 0.99)])
    router = HybridRouter(OfficialMap({"E11.9": 201826}), stub)
    sugg = router.route(MappingRequest(source_code="E11.9"))
    assert sugg.source is MappingSource.OFFICIAL_MAP
    assert sugg.target_concept_id == 201826
    assert sugg.confidence == 1.0
    assert stub.calls == 0  # le retriever n'a PAS été appelé


# --------------------------------------------------------------------------- #
# 2. Le résidu passe au retrieval
# --------------------------------------------------------------------------- #
def test_residue_uses_retriever_above_threshold() -> None:
    stub = StubRetriever([_cand(201826, 0.9), _cand(4048098, 0.2)])
    router = HybridRouter(OfficialMap({"E11.9": 201826}), stub, confidence_threshold=0.5)
    sugg = router.route(MappingRequest(source_code="Z99.9"))  # code hors map
    assert stub.calls == 1
    assert sugg.source is MappingSource.RAG
    assert sugg.target_concept_id == 201826
    assert sugg.confidence == pytest.approx(0.9)
    assert len(sugg.candidates) == 2


def test_residue_below_threshold_is_unmapped_but_keeps_candidates() -> None:
    stub = StubRetriever([_cand(201826, 0.3)])
    router = HybridRouter(OfficialMap({}), stub, confidence_threshold=0.5)
    sugg = router.route(MappingRequest(source_label="libellé ambigu"))
    assert sugg.source is MappingSource.UNMAPPED
    assert sugg.target_concept_id == UNMAPPED_CONCEPT_ID
    assert sugg.no_map_reason is NoMapReason.CONFIDENCE_INSUFFISANTE
    assert sugg.confidence == pytest.approx(0.3)
    assert len(sugg.candidates) == 1  # candidats exposés au steward


def test_no_candidates_is_aucun_candidat() -> None:
    stub = StubRetriever([])
    router = HybridRouter(OfficialMap({}), stub)
    sugg = router.route(MappingRequest(source_label="inconnu total"))
    assert sugg.source is MappingSource.UNMAPPED
    assert sugg.no_map_reason is NoMapReason.AUCUN_CANDIDAT
    assert sugg.candidates == []


def test_label_only_skips_deterministic_and_uses_retriever() -> None:
    stub = StubRetriever([_cand(201826, 0.8)])
    router = HybridRouter(OfficialMap({"E11.9": 201826}), stub, confidence_threshold=0.5)
    sugg = router.route(MappingRequest(source_label="diabète de type 2"))
    assert stub.calls == 1
    assert sugg.source is MappingSource.RAG
    assert sugg.target_concept_id == 201826


# --------------------------------------------------------------------------- #
# 3. Intégration avec un vrai DenseRetriever (hashing + memory, offline)
# --------------------------------------------------------------------------- #
def _dense_router(official: OfficialMap, threshold: float = 0.5) -> HybridRouter:
    con = connect(":memory:")
    try:
        build_corpus(con, FIXTURES)
        gold = fetch_gold(con)
    finally:
        con.close()
    embedder = HashingEmbedder(512)
    store = MemoryVectorStore()
    index_gold(gold, embedder, store)
    return HybridRouter(official, DenseRetriever(embedder, store), confidence_threshold=threshold)


def test_integration_official_then_dense() -> None:
    router = _dense_router(OfficialMap({"E11.9": 201826}))
    # Code couvert -> déterministe.
    s1 = router.route(MappingRequest(source_code="E11.9"))
    assert s1.source is MappingSource.OFFICIAL_MAP
    assert s1.target_concept_id == 201826
    # Libellé libre -> RAG dense.
    s2 = router.route(MappingRequest(source_label="diabète de type 2"))
    assert s2.source is MappingSource.RAG
    assert s2.target_concept_id == 201826
    assert s2.candidates


# --------------------------------------------------------------------------- #
# 4. Router branché sur l'agent gouverné (Proposer -> Vérificateur)
# --------------------------------------------------------------------------- #
def _governed_agent() -> MappingAgent:
    return MappingAgent(Proposer(FakeProposerLLM()), Verifier())


def test_router_with_agent_maps_via_governance() -> None:
    stub = StubRetriever([_cand(201826, 0.9)])
    router = HybridRouter(OfficialMap({}), stub, agent=_governed_agent())
    sugg = router.route(MappingRequest(source_label="diabète de type 2"))
    assert sugg.source is MappingSource.RAG
    assert sugg.target_concept_id == 201826


def test_router_with_agent_official_still_short_circuits() -> None:
    stub = StubRetriever([_cand(999, 0.9)])
    router = HybridRouter(OfficialMap({"E11.9": 201826}), stub, agent=_governed_agent())
    sugg = router.route(MappingRequest(source_code="E11.9"))
    assert sugg.source is MappingSource.OFFICIAL_MAP
    assert stub.calls == 0  # ni retriever ni agent sur un match officiel


def test_router_with_agent_rejects_non_standard() -> None:
    # Candidat non-standard -> le Vérificateur FAIL -> sortie non mappée.
    non_std = ConceptCandidate(
        concept_id=1,
        concept_name="x",
        vocabulary_id="SNOMED",
        domain_id="Condition",
        standard_concept=None,
        score=0.9,
    )
    router = HybridRouter(OfficialMap({}), StubRetriever([non_std]), agent=_governed_agent())
    sugg = router.route(MappingRequest(source_label="x"))
    assert sugg.source is MappingSource.UNMAPPED
    assert sugg.no_map_reason is NoMapReason.CONFIDENCE_INSUFFISANTE


# --------------------------------------------------------------------------- #
# G1 — l'abstention est portée par le ROUTER : elle s'applique quel que soit
# l'agent (elle n'appelle même pas l'agent), donc aucun point d'entrée ni
# orchestrateur ne peut la contourner.
# --------------------------------------------------------------------------- #
class _MapEverythingAgent:
    """Agent factice qui mappe TOUJOURS le 1er candidat (aucune abstention)."""

    def run(
        self,
        request: MappingRequest,
        candidates: Sequence[ConceptCandidate],
        expected_domain: str | None = None,
    ) -> MappingSuggestion:
        c = candidates[0]
        return MappingSuggestion(
            request=request,
            target_concept_id=c.concept_id,
            candidates=list(candidates),
            confidence=c.score,
            source=MappingSource.RAG,
            justification="stub",
        )


def test_router_abstains_before_agent_on_ambiguous_retrieval() -> None:
    # Deux candidats quasi ex aequo (marge 0.02) < seuil 0.5 : le router s'abstient
    # AVANT d'appeler l'agent — même un agent « mappe-tout » ne peut pas passer.
    stub = StubRetriever([_cand(1, 0.81), _cand(2, 0.79)])
    router = HybridRouter(OfficialMap({}), stub, agent=_MapEverythingAgent(), min_margin=0.5)
    sugg = router.route(MappingRequest(source_label="cas ambigu"))
    assert sugg.source is MappingSource.UNMAPPED
    assert sugg.no_map_reason is NoMapReason.CONFIDENCE_INSUFFISANTE


def test_router_maps_when_margin_is_wide_enough() -> None:
    # Marge large (0.6) > seuil : l'agent décide normalement.
    stub = StubRetriever([_cand(1, 0.9), _cand(2, 0.3)])
    router = HybridRouter(OfficialMap({}), stub, agent=_MapEverythingAgent(), min_margin=0.5)
    sugg = router.route(MappingRequest(source_label="cas net"))
    assert sugg.source is MappingSource.RAG
    assert sugg.target_concept_id == 1
