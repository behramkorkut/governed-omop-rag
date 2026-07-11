"""Tests du cache de retrieval (Memory + DuckDB + CachedRetriever)."""

from __future__ import annotations

from pathlib import Path

from governed_omop_rag.core.models import ConceptCandidate
from governed_omop_rag.retrieval.cache import (
    CachedRetriever,
    DuckDBCandidateCache,
    MemoryCandidateCache,
)


def _cand(concept_id: int, name: str = "x", score: float = 0.9) -> ConceptCandidate:
    return ConceptCandidate(
        concept_id=concept_id,
        concept_name=name,
        vocabulary_id="SNOMED",
        domain_id="Condition",
        standard_concept="S",
        score=score,
    )


class CountingRetriever:
    """Retriever de test : compte les appels réels (pour prouver les hits)."""

    def __init__(self, candidates: list[ConceptCandidate]) -> None:
        self.candidates = candidates
        self.calls = 0

    def retrieve(self, query: str, top_k: int = 10) -> list[ConceptCandidate]:
        self.calls += 1
        return list(self.candidates[:top_k])


# --------------------------------------------------------------------------- #
# MemoryCandidateCache
# --------------------------------------------------------------------------- #
def test_memory_cache_roundtrip() -> None:
    cache = MemoryCandidateCache()
    assert cache.get("k") is None
    cache.set("k", [_cand(1)])
    got = cache.get("k")
    assert got is not None
    assert [c.concept_id for c in got] == [1]


def test_memory_cache_is_defensive() -> None:
    cache = MemoryCandidateCache()
    cache.set("k", [_cand(1)])
    got = cache.get("k")
    assert got is not None
    got.append(_cand(2))  # muter la valeur retournée ne doit pas polluer le cache
    again = cache.get("k")
    assert again is not None
    assert len(again) == 1


# --------------------------------------------------------------------------- #
# DuckDBCandidateCache
# --------------------------------------------------------------------------- #
def test_duckdb_cache_roundtrip_and_persist(tmp_path: Path) -> None:
    path = tmp_path / "cache.duckdb"
    cache = DuckDBCandidateCache(path)
    cache.set("k", [_cand(201826, "Type 2 diabetes mellitus")])
    cache.close()

    reopened = DuckDBCandidateCache(path)
    got = reopened.get("k")
    reopened.close()
    assert got is not None
    assert got[0].concept_id == 201826
    assert got[0].concept_name == "Type 2 diabetes mellitus"


def test_duckdb_cache_overwrite(tmp_path: Path) -> None:
    cache = DuckDBCandidateCache(tmp_path / "c.duckdb")
    cache.set("k", [_cand(1)])
    cache.set("k", [_cand(2)])
    got = cache.get("k")
    cache.close()
    assert got is not None
    assert [c.concept_id for c in got] == [2]


# --------------------------------------------------------------------------- #
# CachedRetriever
# --------------------------------------------------------------------------- #
def test_cached_retriever_hit_and_miss() -> None:
    inner = CountingRetriever([_cand(1)])
    cr = CachedRetriever(inner, MemoryCandidateCache())

    r1 = cr.retrieve("diabète", top_k=5)
    assert inner.calls == 1 and cr.misses == 1 and cr.hits == 0

    r2 = cr.retrieve("diabète", top_k=5)
    assert inner.calls == 1  # PAS de second appel : servi par le cache
    assert cr.hits == 1
    assert [c.concept_id for c in r1] == [c.concept_id for c in r2]


def test_cached_retriever_key_is_case_insensitive() -> None:
    inner = CountingRetriever([_cand(1)])
    cr = CachedRetriever(inner, MemoryCandidateCache())
    cr.retrieve("Diabète", top_k=5)
    cr.retrieve("  diabète ", top_k=5)  # même clé normalisée
    assert inner.calls == 1
    assert cr.hits == 1


def test_cached_retriever_topk_changes_key() -> None:
    inner = CountingRetriever([_cand(1)])
    cr = CachedRetriever(inner, MemoryCandidateCache())
    cr.retrieve("x", top_k=5)
    cr.retrieve("x", top_k=10)  # top_k différent -> autre clé
    assert inner.calls == 2


def test_cached_retriever_namespace_isolation() -> None:
    cache = MemoryCandidateCache()
    inner = CountingRetriever([_cand(1)])
    a = CachedRetriever(inner, cache, namespace="modelA")
    b = CachedRetriever(inner, cache, namespace="modelB")
    a.retrieve("x", top_k=5)
    b.retrieve("x", top_k=5)  # namespace différent -> pas de collision
    assert inner.calls == 2
