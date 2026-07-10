"""Tests du VectorStore (MemoryVectorStore + cosinus)."""

from __future__ import annotations

import pytest

from governed_omop_rag.retrieval.vectorstore import (
    MemoryVectorStore,
    VectorItem,
    cosine_similarity,
)


def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # vecteur nul


def _store() -> MemoryVectorStore:
    store = MemoryVectorStore()
    store.ensure_collection(3)
    store.upsert(
        [
            VectorItem(1, [1.0, 0.0, 0.0], {"concept_name": "A"}),
            VectorItem(2, [0.0, 1.0, 0.0], {"concept_name": "B"}),
            VectorItem(3, [0.9, 0.1, 0.0], {"concept_name": "C"}),
        ]
    )
    return store


def test_upsert_and_count() -> None:
    store = _store()
    assert store.count() == 3


def test_search_orders_by_similarity() -> None:
    store = _store()
    hits = store.search([1.0, 0.0, 0.0], top_k=3)
    assert [h.concept_id for h in hits] == [1, 3, 2]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].payload["concept_name"] == "A"


def test_search_top_k_limits() -> None:
    store = _store()
    assert len(store.search([1.0, 0.0, 0.0], top_k=2)) == 2
    assert store.search([1.0, 0.0, 0.0], top_k=0) == []


def test_upsert_overwrites_same_id() -> None:
    store = MemoryVectorStore()
    store.ensure_collection(2)
    store.upsert([VectorItem(1, [1.0, 0.0])])
    store.upsert([VectorItem(1, [0.0, 1.0])])
    assert store.count() == 1
    hit = store.search([0.0, 1.0], top_k=1)[0]
    assert hit.concept_id == 1
    assert hit.score == pytest.approx(1.0)


def test_dimension_mismatch_raises() -> None:
    store = MemoryVectorStore()
    store.ensure_collection(3)
    with pytest.raises(ValueError):
        store.upsert([VectorItem(1, [1.0, 0.0])])  # dim 2 != 3
