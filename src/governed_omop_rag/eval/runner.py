"""Exécuteur d'évaluation : lance le retrieval sur le gold set et agrège.

Boucle de feedback reproductible : à brancher en test de régression (chaque PR)
et pour le benchmark vs Usagi (Phase 5).
"""

from __future__ import annotations

from collections.abc import Sequence

from governed_omop_rag.eval.gold_set import GoldItem
from governed_omop_rag.eval.metrics import EvalReport, aggregate
from governed_omop_rag.retrieval.retriever import Retriever


def evaluate(
    gold: Sequence[GoldItem],
    retriever: Retriever,
    ks: Sequence[int] = (1, 3, 5),
) -> EvalReport:
    """Évalue un Retriever sur le gold set (Top-1, recall@k, MRR)."""
    max_k = max(ks) if ks else 1
    per_query: list[tuple[int, list[int]]] = []
    for item in gold:
        candidates = retriever.retrieve(item.query, top_k=max_k)
        ranked_ids = [c.concept_id for c in candidates]
        per_query.append((item.expected_concept_id, ranked_ids))
    return aggregate(per_query, ks)
