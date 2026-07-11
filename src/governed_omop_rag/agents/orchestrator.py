"""Orchestrateur agentique — Proposer -> Vérificateur, boucle de correction bornée.

Multi-agent employé UNIQUEMENT là où Anthropic le justifie (CONTEXT.md §4.1) :
spécialisation (Proposer vs Vérificateur, prompts/contraintes incompatibles s'ils
étaient fusionnés) + vérification (sous-agent boîte noire). Le reste (retrieval)
reste du code déterministe.

Flux :
1. Proposer choisit un candidat (sortie fermée) ;
2. Vérificateur applique les règles OMOP -> PASS/FAIL ;
3. si FAIL, on exclut ce candidat et on retente (boucle **bornée**, max_attempts) ;
4. si aucun candidat validé, sortie **non mappée explicite** (concept_id=0 + raison).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from governed_omop_rag.agents.proposer import ClosedOutputViolation, Proposer
from governed_omop_rag.agents.verifier import Verifier
from governed_omop_rag.core.models import (
    ConceptCandidate,
    MappingRequest,
    MappingSource,
    MappingSuggestion,
    NoMapReason,
)


@runtime_checkable
class Agent(Protocol):
    """Contrat d'un orchestrateur de mapping (implémentation simple ou LangGraph)."""

    def run(
        self,
        request: MappingRequest,
        candidates: Sequence[ConceptCandidate],
        expected_domain: str | None = None,
    ) -> MappingSuggestion:
        """Produit une suggestion validée (ou une sortie non mappée explicite)."""
        ...


class MappingAgent:
    """Coordonne Proposer et Vérificateur avec une boucle de correction bornée."""

    def __init__(self, proposer: Proposer, verifier: Verifier, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts doit être >= 1")
        self.proposer = proposer
        self.verifier = verifier
        self.max_attempts = max_attempts

    def run(
        self,
        request: MappingRequest,
        candidates: Sequence[ConceptCandidate],
        expected_domain: str | None = None,
    ) -> MappingSuggestion:
        """Produit une suggestion validée, ou une sortie non mappée explicite."""
        candidates = list(candidates)
        if not candidates:
            return MappingSuggestion(
                request=request,
                source=MappingSource.UNMAPPED,
                no_map_reason=NoMapReason.AUCUN_CANDIDAT,
                justification="Aucun candidat à soumettre à l'agent.",
            )

        query = request.source_label or request.source_code or ""
        excluded: set[int] = set()

        for _attempt in range(self.max_attempts):
            try:
                proposal = self.proposer.propose(query, candidates, excluded)
            except ClosedOutputViolation:
                # Garde-fou : proposition hors-liste -> rejet structurel, on arrête.
                return MappingSuggestion(
                    request=request,
                    candidates=candidates,
                    source=MappingSource.UNMAPPED,
                    no_map_reason=NoMapReason.HORS_VOCABULAIRE,
                    justification="Proposition hors-vocabulaire rejetée (sortie fermée).",
                )
            if proposal is None:
                break

            chosen = next(c for c in candidates if c.concept_id == proposal.concept_id)
            verdict = self.verifier.verify(chosen, expected_domain)
            if verdict.passed:
                return MappingSuggestion(
                    request=request,
                    target_concept_id=chosen.concept_id,
                    candidates=candidates,
                    confidence=chosen.score,
                    source=MappingSource.RAG,
                    justification=proposal.justification,
                )
            # Correction : on écarte le candidat rejeté et on retente.
            excluded.add(chosen.concept_id)

        return MappingSuggestion(
            request=request,
            candidates=candidates,
            source=MappingSource.UNMAPPED,
            no_map_reason=NoMapReason.CONFIDENCE_INSUFFISANTE,
            justification=(
                f"Aucun candidat validé par le vérificateur après {self.max_attempts} essai(s)."
            ),
        )
