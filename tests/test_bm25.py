"""Tests de l'index BM25 (pur, hors-ligne)."""

from __future__ import annotations

from governed_omop_rag.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_normalizes() -> None:
    assert tokenize("Diabète  de Type 2") == ["diabete", "de", "type", "2"]
    assert tokenize("") == []


def _index() -> BM25Index:
    docs = [
        (1, ["diabete", "type", "2", "mellitus"]),
        (2, ["asthme"]),
        (3, ["hypertension", "arterielle", "essentielle"]),
    ]
    return BM25Index(docs)


def test_score_matches_relevant_doc() -> None:
    idx = _index()
    scores = idx.score(["diabete"])
    assert scores[1] > 0.0
    assert scores[2] == 0.0
    assert scores[3] == 0.0


def test_top_k_orders_and_filters_zero() -> None:
    idx = _index()
    hits = idx.top_k(["asthme"], 5)
    assert hits[0][0] == 2  # doc pertinent en tête
    assert all(score > 0 for _, score in hits)  # scores nuls exclus
    assert len(hits) == 1  # un seul doc contient 'asthme'


def test_top_k_empty_query() -> None:
    idx = _index()
    assert idx.top_k([], 5) == []
    assert idx.top_k(["inconnu"], 5) == []


def test_top_k_zero() -> None:
    assert _index().top_k(["diabete"], 0) == []


def test_idf_positive_and_rarer_terms_weigh_more() -> None:
    # 'commun' dans les 3 docs, 'rare' dans 1 seul -> idf(rare) > idf(commun).
    docs = [
        (1, ["commun", "rare"]),
        (2, ["commun"]),
        (3, ["commun"]),
    ]
    idx = BM25Index(docs)
    assert idx.idf["rare"] > idx.idf["commun"]
    assert idx.idf["commun"] >= 0.0
