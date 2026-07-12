"""Évaluation & observabilité : Top-1/Top-5, recall@k, faithfulness,
taux d'hallucination, coût/latence, benchmark vs Usagi.

On n'exporte ici que les briques légères (metrics, gold_set). ``runner``
(evaluate) s'importe explicitement (il dépend de retrieval/core).
"""

from governed_omop_rag.eval.gold_set import GoldItem, load_gold_set
from governed_omop_rag.eval.metrics import (
    EvalReport,
    MappingReport,
    aggregate,
    aggregate_mapping,
    hit_at_k,
    rank_of,
    reciprocal_rank,
)

__all__ = [
    "EvalReport",
    "GoldItem",
    "MappingReport",
    "aggregate",
    "aggregate_mapping",
    "hit_at_k",
    "load_gold_set",
    "rank_of",
    "reciprocal_rank",
]
