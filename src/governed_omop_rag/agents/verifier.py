"""Sous-agent Vérificateur — règles OMOP dures, en boîte noire.

Pattern « verification subagent » d'Anthropic (CONTEXT.md §4.1/§4.2) : un agent
dont l'unique rôle est de valider le travail d'un autre. Il n'a besoin QUE du
candidat + des règles (pas de l'historique de raisonnement) → frontière propre.

Règles (garde-fous §4.3) :
- ``standard_concept == 'S'`` (concept standard uniquement) ;
- ``domain_id`` cohérent avec le domaine attendu (si fourni).

Volontairement **déterministe** (pas d'appel LLM) : les règles OMOP sont dures,
un LLM n'apporterait rien et coûterait des tokens.
"""

from __future__ import annotations

from governed_omop_rag.agents.schemas import Verdict, VerdictStatus
from governed_omop_rag.core.models import ConceptCandidate


class Verifier:
    """Valide un candidat contre les règles OMOP. Sans état, réutilisable."""

    def verify(self, candidate: ConceptCandidate, expected_domain: str | None = None) -> Verdict:
        """PASS si le candidat est standard et (si demandé) dans le bon domaine."""
        if not candidate.is_standard:
            return Verdict(
                status=VerdictStatus.FAIL,
                reason=(
                    f"concept_id={candidate.concept_id} non standard "
                    f"(standard_concept={candidate.standard_concept!r})."
                ),
            )
        if expected_domain is not None and candidate.domain_id != expected_domain:
            return Verdict(
                status=VerdictStatus.FAIL,
                reason=(f"domaine {candidate.domain_id!r} != attendu {expected_domain!r}."),
            )
        return Verdict(
            status=VerdictStatus.PASS,
            reason=(
                f"concept_id={candidate.concept_id} standard, "
                f"domaine {candidate.domain_id!r} cohérent."
            ),
        )
