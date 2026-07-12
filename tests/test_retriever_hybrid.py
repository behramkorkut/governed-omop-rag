"""Tests BM25Retriever + HybridRetriever (fusion RRF) sur le corpus de fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from governed_omop_rag.core.models import ConceptCandidate
from governed_omop_rag.medallion.db import connect
from governed_omop_rag.medallion.gold import GoldConcept, fetch_gold
from governed_omop_rag.medallion.pipeline import build_corpus
from governed_omop_rag.retrieval.embeddings import HashingEmbedder
from governed_omop_rag.retrieval.index import index_gold
from governed_omop_rag.retrieval.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    LexicalBaselineRetriever,
    build_retriever,
)
from governed_omop_rag.retrieval.vectorstore import MemoryVectorStore

FIXTURES = Path(__file__).parent / "fixtures"


def _gold() -> list[GoldConcept]:
    con = connect(":memory:")
    try:
        build_corpus(con, FIXTURES)
        return fetch_gold(con)
    finally:
        con.close()


def _dense(gold: list[GoldConcept]) -> DenseRetriever:
    embedder = HashingEmbedder(512)
    store = MemoryVectorStore()
    index_gold(gold, embedder, store)
    return DenseRetriever(embedder, store)


# --------------------------------------------------------------------------- #
# BM25Retriever
# --------------------------------------------------------------------------- #
def test_bm25_retriever_ranks_and_normalizes() -> None:
    bm25 = BM25Retriever(_gold())
    cands = bm25.retrieve("asthme", top_k=3)
    assert cands[0].concept_id == 4048098
    assert cands[0].score == pytest.approx(1.0)  # normalisé par le meilleur score
    assert all(0.0 <= c.score <= 1.0 for c in cands)


def test_bm25_retriever_empty_query() -> None:
    assert BM25Retriever(_gold()).retrieve("   ", top_k=5) == []


def test_bm25_retriever_carries_synonyms() -> None:
    [top] = BM25Retriever(_gold()).retrieve("diabète de type 2", top_k=1)
    assert top.concept_id == 201826
    assert set(top.synonyms) == {"diabète de type 2", "T2DM"}


# --------------------------------------------------------------------------- #
# HybridRetriever (RRF)
# --------------------------------------------------------------------------- #
def test_hybrid_retriever_fuses_and_ranks() -> None:
    gold = _gold()
    hybrid = HybridRetriever([BM25Retriever(gold), _dense(gold)])
    cands = hybrid.retrieve("diabète de type 2", top_k=3)
    assert cands
    assert isinstance(cands[0], ConceptCandidate)
    assert cands[0].concept_id == 201826
    assert all(0.0 <= c.score <= 1.0 for c in cands)


def test_hybrid_retriever_empty_query() -> None:
    gold = _gold()
    hybrid = HybridRetriever([BM25Retriever(gold), _dense(gold)])
    assert hybrid.retrieve("", top_k=5) == []


def test_hybrid_retriever_requires_at_least_one() -> None:
    with pytest.raises(ValueError):
        HybridRetriever([])


# --------------------------------------------------------------------------- #
# build_retriever
# --------------------------------------------------------------------------- #
def test_build_retriever_kinds() -> None:
    gold = _gold()
    embedder = HashingEmbedder(256)
    store = MemoryVectorStore()
    index_gold(gold, embedder, store)
    for kind in ("dense", "bm25", "hybrid", "baseline"):
        retriever = build_retriever(kind, gold, embedder, store)
        cands = retriever.retrieve("asthme", top_k=1)
        assert cands and cands[0].concept_id == 4048098


# --------------------------------------------------------------------------- #
# Baseline lexicale (proxy Usagi)
# --------------------------------------------------------------------------- #
def test_baseline_exact_match_scores_one() -> None:
    base = LexicalBaselineRetriever(_gold())
    cands = base.retrieve("asthme", top_k=3)  # synonyme exact d'Asthma
    assert cands[0].concept_id == 4048098
    assert cands[0].score == pytest.approx(1.0)


def test_baseline_empty_query() -> None:
    assert LexicalBaselineRetriever(_gold()).retrieve("   ", top_k=3) == []


def test_build_retriever_unknown() -> None:
    gold = _gold()
    embedder = HashingEmbedder(64)
    store = MemoryVectorStore()
    index_gold(gold, embedder, store)
    with pytest.raises(ValueError):
        build_retriever("magique", gold, embedder, store)
