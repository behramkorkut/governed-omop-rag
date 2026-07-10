"""Test de régression : le retrieval doit rester bon sur le gold set."""

from __future__ import annotations

from pathlib import Path

from governed_omop_rag.eval.gold_set import load_gold_set
from governed_omop_rag.eval.runner import evaluate
from governed_omop_rag.medallion.db import connect
from governed_omop_rag.medallion.gold import fetch_gold
from governed_omop_rag.medallion.pipeline import build_corpus
from governed_omop_rag.retrieval.embeddings import HashingEmbedder
from governed_omop_rag.retrieval.index import index_gold
from governed_omop_rag.retrieval.retriever import DenseRetriever
from governed_omop_rag.retrieval.vectorstore import MemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
GOLD_SET = ROOT / "data" / "eval" / "gold_set.csv"


def _dense_retriever() -> DenseRetriever:
    con = connect(":memory:")
    try:
        build_corpus(con, FIXTURES)
        gold_concepts = fetch_gold(con)
    finally:
        con.close()
    embedder = HashingEmbedder(512)
    store = MemoryVectorStore()
    index_gold(gold_concepts, embedder, store)
    return DenseRetriever(embedder, store)


def test_regression_gate_on_fixtures() -> None:
    gold = load_gold_set(GOLD_SET)
    report = evaluate(gold, _dense_retriever())
    assert report.n == len(gold) >= 5
    # Gate de régression : si le retrieval se dégrade, le test échoue.
    assert report.top1 >= 0.8
    assert report.recall_at_k[3] == 1.0
    assert report.mrr >= 0.8


def test_evaluate_empty_gold() -> None:
    report = evaluate([], _dense_retriever())
    assert report.n == 0
    assert report.top1 == 0.0
