"""Exécuteur d'évaluation : lance le retrieval sur le gold set et agrège.

Boucle de feedback reproductible : à brancher en test de régression (chaque PR)
et pour le benchmark vs Usagi (Phase 5).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from governed_omop_rag.core.models import MappingRequest, MappingSuggestion
from governed_omop_rag.eval.gold_set import GoldItem
from governed_omop_rag.eval.metrics import (
    EvalReport,
    MappingReport,
    aggregate,
    aggregate_mapping,
)
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


def evaluate_mapping(
    gold: Sequence[GoldItem],
    route: Callable[[MappingRequest], MappingSuggestion],
    token_usage: Callable[[], tuple[int, int]] | None = None,
) -> MappingReport:
    """Évalue le mapping FINAL (pas seulement le retrieval).

    ``route`` prend une requête et renvoie la suggestion (ex. MappingService.route
    partiellement appliqué, ou un routeur). Mesure Top-1, couverture, précision.
    ``token_usage`` : rappel optionnel renvoyant le cumul LLM (in, out) après le
    lot, pour reporter le coût par entrée (P5-5).
    """
    outcomes: list[tuple[bool, bool]] = []
    total_ms = 0.0
    for item in gold:
        request = MappingRequest(
            source_code=item.source_code,
            source_label=item.source_label,
        )
        start = time.perf_counter()
        suggestion = route(request)
        total_ms += (time.perf_counter() - start) * 1000.0
        mapped = suggestion.is_mapped
        correct = mapped and suggestion.target_concept_id == item.expected_concept_id
        outcomes.append((mapped, correct))
    avg_latency = total_ms / len(gold) if gold else 0.0
    in_tokens, out_tokens = token_usage() if token_usage is not None else (0, 0)
    return aggregate_mapping(
        outcomes,
        avg_latency_ms=avg_latency,
        total_input_tokens=in_tokens,
        total_output_tokens=out_tokens,
    )
