"""Orchestrateur LangGraph (P3-1) — même logique que MappingAgent, en StateGraph.

On exprime le flux Proposer -> Vérificateur -> boucle bornée sous forme de graphe
d'états LangGraph (nodes ``propose`` / ``verify`` + arêtes conditionnelles). C'est
le runtime d'orchestration décidé au CONTEXT.md (§5.4).

``langgraph`` est importé **paresseusement** (extra ``agents``) pour ne pas alourdir
l'installation/CI par défaut. ``LangGraphMappingAgent`` satisfait le protocole
``Agent`` : il est interchangeable avec ``MappingAgent`` dans le HybridRouter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

from governed_omop_rag.agents.proposer import ClosedOutputViolation, Proposer
from governed_omop_rag.agents.verifier import Verifier
from governed_omop_rag.core.models import (
    ConceptCandidate,
    MappingRequest,
    MappingSource,
    MappingSuggestion,
    NoMapReason,
)

JUSTIFICATION_HALLUCINATION = "Proposition hors-vocabulaire rejetée (sortie fermée)."


class GraphState(TypedDict, total=False):
    """État circulant dans le graphe."""

    query: str
    candidates: list[ConceptCandidate]
    excluded: list[int]
    attempts_left: int
    outcome: str  # "mapped" | "hallucinated" | "exhausted"
    chosen_id: int
    justification: str


def build_mapping_graph(
    proposer: Proposer,
    verifier: Verifier,
    max_attempts: int = 3,
    expected_domain: str | None = None,
) -> Any:
    """Construit et compile le StateGraph (import paresseux de langgraph)."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover - dépend de l'extra agents
        raise ImportError("langgraph requis pour ce backend. uv sync --extra agents") from exc

    def propose(state: GraphState) -> dict[str, Any]:
        attempts_left = state["attempts_left"]
        if attempts_left <= 0:
            return {"outcome": "exhausted"}
        excluded = set(state.get("excluded", []))
        available = [c for c in state["candidates"] if c.concept_id not in excluded]
        if not available:
            return {"outcome": "exhausted"}
        try:
            out = proposer.propose(state["query"], state["candidates"], excluded)
        except ClosedOutputViolation:
            return {"outcome": "hallucinated", "attempts_left": attempts_left - 1}
        assert out is not None
        return {
            "chosen_id": out.concept_id,
            "justification": out.justification,
            "attempts_left": attempts_left - 1,
        }

    def verify(state: GraphState) -> dict[str, Any]:
        chosen_id = state["chosen_id"]
        chosen = next(c for c in state["candidates"] if c.concept_id == chosen_id)
        verdict = verifier.verify(chosen, expected_domain)
        if verdict.passed:
            return {"outcome": "mapped"}
        return {"excluded": [*state.get("excluded", []), chosen_id]}

    def after_propose(state: GraphState) -> str:
        return "end" if state.get("outcome") else "verify"

    def after_verify(state: GraphState) -> str:
        return "end" if state.get("outcome") else "propose"

    graph = StateGraph(GraphState)
    graph.add_node("propose", propose)
    graph.add_node("verify", verify)
    graph.set_entry_point("propose")
    graph.add_conditional_edges("propose", after_propose, {"verify": "verify", "end": END})
    graph.add_conditional_edges("verify", after_verify, {"propose": "propose", "end": END})
    return graph.compile()


class LangGraphMappingAgent:
    """Orchestrateur basé LangGraph — interchangeable avec MappingAgent."""

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
        candidates = list(candidates)
        if not candidates:
            return MappingSuggestion(
                request=request,
                source=MappingSource.UNMAPPED,
                no_map_reason=NoMapReason.AUCUN_CANDIDAT,
                justification="Aucun candidat à soumettre à l'agent.",
            )

        compiled = build_mapping_graph(
            self.proposer, self.verifier, self.max_attempts, expected_domain
        )
        final: dict[str, Any] = compiled.invoke(
            {
                "query": request.source_label or request.source_code or "",
                "candidates": candidates,
                "excluded": [],
                "attempts_left": self.max_attempts,
            }
        )
        return self._to_suggestion(request, candidates, final)

    def _to_suggestion(
        self,
        request: MappingRequest,
        candidates: list[ConceptCandidate],
        final: dict[str, Any],
    ) -> MappingSuggestion:
        outcome = final.get("outcome", "exhausted")
        if outcome == "mapped":
            chosen = next(c for c in candidates if c.concept_id == final["chosen_id"])
            return MappingSuggestion(
                request=request,
                target_concept_id=chosen.concept_id,
                candidates=candidates,
                confidence=chosen.score,
                source=MappingSource.RAG,
                justification=str(final.get("justification", "")),
            )
        if outcome == "hallucinated":
            return MappingSuggestion(
                request=request,
                candidates=candidates,
                source=MappingSource.UNMAPPED,
                no_map_reason=NoMapReason.HORS_VOCABULAIRE,
                justification=JUSTIFICATION_HALLUCINATION,
            )
        return MappingSuggestion(
            request=request,
            candidates=candidates,
            source=MappingSource.UNMAPPED,
            no_map_reason=NoMapReason.CONFIDENCE_INSUFFISANTE,
            justification=(
                f"Aucun candidat validé par le vérificateur après {self.max_attempts} essai(s)."
            ),
        )
