"""Tests d'indexation & recherche dense (Gold -> vecteurs -> candidats)."""

from __future__ import annotations

from pathlib import Path

from governed_omop_rag.core.models import ConceptCandidate
from governed_omop_rag.medallion.db import connect
from governed_omop_rag.medallion.gold import GoldConcept, fetch_gold
from governed_omop_rag.medallion.pipeline import build_corpus
from governed_omop_rag.retrieval.embeddings import HashingEmbedder
from governed_omop_rag.retrieval.index import index_gold, search_concepts
from governed_omop_rag.retrieval.vectorstore import MemoryVectorStore

FIXTURES = Path(__file__).parent / "fixtures"


def _gold() -> list[GoldConcept]:
    con = connect(":memory:")
    try:
        build_corpus(con, FIXTURES)
        return fetch_gold(con)
    finally:
        con.close()


def test_index_and_search_finds_diabetes() -> None:
    gold = _gold()
    embedder = HashingEmbedder(512)
    store = MemoryVectorStore()

    n = index_gold(gold, embedder, store)
    assert n == 4
    assert store.count() == 4

    candidates = search_concepts("diabète de type 2", embedder, store, top_k=3)
    assert candidates
    assert isinstance(candidates[0], ConceptCandidate)
    assert candidates[0].concept_id == 201826  # Type 2 diabetes mellitus
    assert candidates[0].standard_concept == "S"
    assert 0.0 <= candidates[0].score <= 1.0


def test_synonyms_are_transferred_to_candidate() -> None:
    """Les synonymes du concept doivent remonter jusqu'au ConceptCandidate
    (contexte utile à l'agent Proposer en Phase 3)."""
    gold = _gold()
    embedder = HashingEmbedder(512)
    store = MemoryVectorStore()
    index_gold(gold, embedder, store)
    [top] = search_concepts("diabète de type 2", embedder, store, top_k=1)
    assert top.concept_id == 201826
    assert set(top.synonyms) == {"diabète de type 2", "T2DM"}


def test_index_empty_returns_zero() -> None:
    store = MemoryVectorStore()
    assert index_gold([], HashingEmbedder(64), store) == 0
    assert store.count() == 0


def test_search_on_empty_store_returns_empty() -> None:
    embedder = HashingEmbedder(64)
    store = MemoryVectorStore()
    store.ensure_collection(embedder.dimension)
    assert search_concepts("diabète", embedder, store, top_k=5) == []


def test_search_top_k_respected() -> None:
    gold = _gold()
    embedder = HashingEmbedder(256)
    store = MemoryVectorStore()
    index_gold(gold, embedder, store)
    assert len(search_concepts("asthme", embedder, store, top_k=2)) == 2


def test_search_top_k_greater_than_corpus() -> None:
    gold = _gold()  # 4 concepts
    embedder = HashingEmbedder(256)
    store = MemoryVectorStore()
    index_gold(gold, embedder, store)
    assert len(search_concepts("diabète", embedder, store, top_k=10)) == 4
