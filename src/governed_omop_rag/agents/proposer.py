"""Agent Proposer — choisit un concept parmi les candidats, sortie fermée.

Garde-fou structurel (CONTEXT.md §4.3) : le Proposer ne peut retourner qu'un
``concept_id`` **présent dans les candidats fournis**. Toute proposition
hors-liste lève ``ClosedOutputViolation`` — l'anti-hallucination n'est pas qu'une
consigne de prompt, c'est vérifié dans le code.
"""

from __future__ import annotations

from collections.abc import Sequence

from governed_omop_rag.agents.llm import ProposerLLM
from governed_omop_rag.agents.schemas import ProposerOutput
from governed_omop_rag.core.models import ConceptCandidate


class ClosedOutputViolation(Exception):
    """Levée quand le LLM propose un concept_id hors de la liste des candidats."""


class Proposer:
    """Enveloppe un ProposerLLM et impose la contrainte de sortie fermée."""

    def __init__(self, llm: ProposerLLM) -> None:
        self.llm = llm

    def propose(
        self,
        query: str,
        candidates: Sequence[ConceptCandidate],
        excluded: set[int] | None = None,
    ) -> ProposerOutput | None:
        """Propose un concept parmi les candidats non exclus.

        Retourne None s'il ne reste aucun candidat. Lève ClosedOutputViolation
        si le LLM sort de la liste (proposition rejetée structurellement).
        """
        excluded = excluded or set()
        available = [c for c in candidates if c.concept_id not in excluded]
        if not available:
            return None
        output = self.llm.propose(query, available)
        allowed = {c.concept_id for c in available}
        if output.concept_id not in allowed:
            raise ClosedOutputViolation(
                f"concept_id={output.concept_id} hors des candidats {sorted(allowed)}"
            )
        return output
