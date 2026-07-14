"""Métriques de retrieval : Top-1, recall@k, MRR.

Volontairement sans dépendance au reste (types simples) : on peut les tester
isolément et les réutiliser pour le benchmark vs Usagi (Phase 5).

Définitions :
- hit@k : le concept attendu est-il dans les k premiers candidats ?
- recall@k (ici, un attendu par requête) : moyenne des hit@k.
- Top-1 accuracy : recall@1.
- MRR : moyenne des inverses du rang du bon concept (0 si absent).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def rank_of(expected: int, ranked_ids: Sequence[int]) -> int | None:
    """Rang (1-indexé) du concept attendu, ou None s'il est absent."""
    for i, cid in enumerate(ranked_ids, start=1):
        if cid == expected:
            return i
    return None


def hit_at_k(expected: int, ranked_ids: Sequence[int], k: int) -> bool:
    """True si le concept attendu figure dans les k premiers."""
    if k <= 0:
        return False
    return expected in list(ranked_ids)[:k]


def reciprocal_rank(expected: int, ranked_ids: Sequence[int]) -> float:
    """1/rang du bon concept, 0.0 s'il est absent."""
    r = rank_of(expected, ranked_ids)
    return 1.0 / r if r is not None else 0.0


@dataclass(frozen=True)
class EvalReport:
    """Rapport agrégé sur un gold set."""

    n: int
    top1: float
    mrr: float
    recall_at_k: dict[int, float]

    def as_table(self) -> str:
        """Rendu texte compact (pour CLI / docs)."""
        lines = [
            f"n concepts évalués : {self.n}",
            f"Top-1 accuracy     : {self.top1:.3f}",
            f"MRR                : {self.mrr:.3f}",
        ]
        for k in sorted(self.recall_at_k):
            lines.append(f"recall@{k:<11}: {self.recall_at_k[k]:.3f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class MappingReport:
    """Métriques au niveau du mapping final (au-delà du recall de retrieval).

    - ``top1`` : part des entrées mappées ET correctes (exactitude globale) ;
    - ``coverage`` : part des entrées effectivement mappées (concept_id != 0) ;
    - ``unmapped_rate`` : 1 - coverage (l'outil sait dire « je ne sais pas ») ;
    - ``precision_mapped`` : exactitude parmi les seules entrées mappées.
    """

    n: int
    top1: float
    coverage: float
    unmapped_rate: float
    precision_mapped: float
    avg_latency_ms: float = 0.0
    # Coût LLM par entrée (P5-5) : 0 avec le Proposer hors-ligne / le déterministe.
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0

    def as_table(self) -> str:
        lines = [
            f"n entrées          : {self.n}",
            f"Top-1 (global)     : {self.top1:.3f}",
            f"couverture         : {self.coverage:.3f}",
            f"taux non-mappé     : {self.unmapped_rate:.3f}",
            f"précision (mappés) : {self.precision_mapped:.3f}",
            f"latence moyenne    : {self.avg_latency_ms:.1f} ms/entrée",
        ]
        if self.avg_input_tokens or self.avg_output_tokens:
            lines.append(
                f"tokens/entrée      : {self.avg_input_tokens:.1f} in / "
                f"{self.avg_output_tokens:.1f} out"
            )
        return "\n".join(lines)


def aggregate_mapping(
    outcomes: Sequence[tuple[bool, bool]],
    avg_latency_ms: float = 0.0,
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
) -> MappingReport:
    """Agrège des (mappé, correct) en MappingReport.

    ``total_*_tokens`` : cumul LLM sur tout le lot, réparti par entrée (P5-5).
    """
    n = len(outcomes)
    if n == 0:
        return MappingReport(0, 0.0, 0.0, 0.0, 0.0, avg_latency_ms)
    mapped = sum(1 for m, _ in outcomes if m)
    correct = sum(1 for m, c in outcomes if m and c)
    coverage = mapped / n
    return MappingReport(
        n=n,
        top1=correct / n,
        coverage=coverage,
        unmapped_rate=1.0 - coverage,
        precision_mapped=(correct / mapped) if mapped else 0.0,
        avg_latency_ms=avg_latency_ms,
        avg_input_tokens=total_input_tokens / n,
        avg_output_tokens=total_output_tokens / n,
    )


def aggregate(
    per_query: Sequence[tuple[int, Sequence[int]]],
    ks: Sequence[int] = (1, 3, 5),
) -> EvalReport:
    """Agrège les métriques sur une liste de (concept_attendu, ids_classés)."""
    n = len(per_query)
    if n == 0:
        return EvalReport(n=0, top1=0.0, mrr=0.0, recall_at_k=dict.fromkeys(ks, 0.0))

    top1 = sum(hit_at_k(exp, ranked, 1) for exp, ranked in per_query) / n
    mrr = sum(reciprocal_rank(exp, ranked) for exp, ranked in per_query) / n
    recall_at_k = {k: sum(hit_at_k(exp, ranked, k) for exp, ranked in per_query) / n for k in ks}
    return EvalReport(n=n, top1=top1, mrr=mrr, recall_at_k=recall_at_k)
