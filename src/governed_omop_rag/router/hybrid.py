"""Router hybride (P2-1) — déterministe d'abord, RAG sur le résidu.

Réalise la stratégie directrice du projet (CONTEXT.md §5.5) :

1. **Match déterministe** via l'alignement officiel (DeterministicRouter). Gratuit,
   instantané, fiable. S'il réussit, on s'arrête là.
2. **Retrieval sur le résidu** : pour les codes non couverts et les libellés en
   texte libre, on interroge le Retriever (dense en v1) et on renvoie les
   candidats. Le meilleur candidat n'est retenu (source RAG) que si sa confiance
   dépasse le seuil ; sinon on reste **non mappé** mais on expose les candidats
   au steward (human-in-the-loop).

Bénéfice : on **borne le coût** (le retrieval/LLM ne voit que la queue difficile)
et on est *par construction* au niveau de l'alignement officiel sur les cas faciles.

NB : en Phase 2, le « RAG » se limite au retrieval (candidats). L'agent Proposer +
sous-agent Vérificateur (Phase 3) viendront décider/justifier sur ces candidats.
"""

from __future__ import annotations

from governed_omop_rag.agents.orchestrator import Agent
from governed_omop_rag.core.models import (
    MappingRequest,
    MappingSource,
    MappingSuggestion,
    NoMapReason,
)
from governed_omop_rag.retrieval.retriever import Retriever
from governed_omop_rag.router.deterministic import DeterministicRouter, OfficialMap

JUSTIFICATION_RAG = (
    "Suggestion par recherche dense sur le résidu (hors alignement officiel) "
    "— à valider par un steward."
)
JUSTIFICATION_LOW_CONF = (
    "Candidats trouvés mais confiance insuffisante — validation humaine requise."
)
JUSTIFICATION_NO_CANDIDATE = "Aucun candidat trouvé pour cette entrée."


class HybridRouter:
    """Compose le match déterministe et le retrieval pour router une requête."""

    def __init__(
        self,
        official_map: OfficialMap,
        retriever: Retriever,
        confidence_threshold: float = 0.5,
        top_k: int = 10,
        agent: Agent | None = None,
    ) -> None:
        self._deterministic = DeterministicRouter(official_map)
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        # Si fourni, l'agent (Proposer -> Vérificateur, boucle bornée) décide sur
        # le résidu à la place du simple seuil de confiance.
        self.agent = agent

    def route(self, request: MappingRequest) -> MappingSuggestion:
        # 1. Déterministe d'abord (uniquement si un code est fourni).
        if request.source_code:
            deterministic = self._deterministic.route(request)
            if deterministic.is_mapped:
                return deterministic

        # 2. RAG sur le résidu : requête = libellé sinon code.
        query = request.source_label or request.source_code or ""
        candidates = self.retriever.retrieve(query, self.top_k) if query.strip() else []

        # 2a. Si un agent gouverné est branché, il décide (avec garde-fous).
        if self.agent is not None:
            return self.agent.run(request, candidates)

        # 2b. Sinon, décision par simple seuil de confiance.
        if not candidates:
            return MappingSuggestion(
                request=request,
                source=MappingSource.UNMAPPED,
                no_map_reason=NoMapReason.AUCUN_CANDIDAT,
                justification=JUSTIFICATION_NO_CANDIDATE,
            )

        best = candidates[0]
        if best.score >= self.confidence_threshold:
            return MappingSuggestion(
                request=request,
                target_concept_id=best.concept_id,
                candidates=candidates,
                confidence=best.score,
                source=MappingSource.RAG,
                justification=JUSTIFICATION_RAG,
            )

        # Confiance insuffisante : non mappé, mais on expose les candidats.
        return MappingSuggestion(
            request=request,
            candidates=candidates,
            confidence=best.score,
            source=MappingSource.UNMAPPED,
            no_map_reason=NoMapReason.CONFIDENCE_INSUFFISANTE,
            justification=JUSTIFICATION_LOW_CONF,
        )
