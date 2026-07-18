"""Tests de l'orchestrateur LangGraph (skip si l'extra agents n'est pas installé).

Vérifie que ``LangGraphMappingAgent`` produit les MÊMES décisions que
``MappingAgent`` (parité) — le graphe n'est qu'un runtime alternatif.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

pytest.importorskip("langgraph")

from governed_omop_rag.agents.graph import LangGraphMappingAgent  # noqa: E402
from governed_omop_rag.agents.llm import FakeProposerLLM  # noqa: E402
from governed_omop_rag.agents.orchestrator import MappingAgent  # noqa: E402
from governed_omop_rag.agents.proposer import Proposer  # noqa: E402
from governed_omop_rag.agents.schemas import ProposerOutput  # noqa: E402
from governed_omop_rag.agents.verifier import Verifier  # noqa: E402
from governed_omop_rag.core.models import (  # noqa: E402
    UNMAPPED_CONCEPT_ID,
    ConceptCandidate,
    MappingRequest,
    MappingSource,
    NoMapReason,
)


def _cand(
    cid: int, score: float = 0.9, standard: str | None = "S", domain: str = "Condition"
) -> ConceptCandidate:
    return ConceptCandidate(
        concept_id=cid,
        concept_name=f"c-{cid}",
        vocabulary_id="SNOMED",
        domain_id=domain,
        standard_concept=standard,
        score=score,
    )


def _lg() -> LangGraphMappingAgent:
    return LangGraphMappingAgent(Proposer(FakeProposerLLM()), Verifier())


class _RogueLLM:
    def propose(self, query: str, candidates: Sequence[ConceptCandidate]) -> ProposerOutput:
        return ProposerOutput(concept_id=999999, justification="inventé")


class _BrokenLLM:
    """LLM qui lève une réponse illisible (JSON cassé / ValueError)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def propose(self, query: str, candidates: Sequence[ConceptCandidate]) -> ProposerOutput:
        raise self._exc


def test_langgraph_maps_on_pass() -> None:
    sugg = _lg().run(MappingRequest(source_label="diabète"), [_cand(201826, 0.8)])
    assert sugg.source is MappingSource.RAG
    assert sugg.target_concept_id == 201826
    assert sugg.confidence == pytest.approx(0.8)


def test_langgraph_correction_loop() -> None:
    candidates = [_cand(1, 0.9, standard=None), _cand(2, 0.7, standard="S")]
    sugg = _lg().run(MappingRequest(source_label="x"), candidates)
    assert sugg.source is MappingSource.RAG
    assert sugg.target_concept_id == 2


def test_langgraph_exhausts() -> None:
    candidates = [_cand(1, standard=None), _cand(2, standard=None)]
    sugg = _lg().run(MappingRequest(source_label="x"), candidates)
    assert sugg.source is MappingSource.UNMAPPED
    assert sugg.no_map_reason is NoMapReason.CONFIDENCE_INSUFFISANTE
    assert sugg.target_concept_id == UNMAPPED_CONCEPT_ID


def test_langgraph_empty() -> None:
    sugg = _lg().run(MappingRequest(source_label="x"), [])
    assert sugg.no_map_reason is NoMapReason.AUCUN_CANDIDAT


def test_langgraph_rejects_hallucination() -> None:
    agent = LangGraphMappingAgent(Proposer(_RogueLLM()), Verifier())
    sugg = agent.run(MappingRequest(source_label="x"), [_cand(10), _cand(20)])
    assert sugg.source is MappingSource.UNMAPPED
    assert sugg.no_map_reason is NoMapReason.HORS_VOCABULAIRE


def test_langgraph_agent_error_on_unreadable_llm() -> None:
    """G2 : une réponse LLM illisible (JSONDecodeError, sous-classe de ValueError)
    dégrade proprement en UNMAPPED/ERREUR_AGENT — jamais d'exception propagée."""
    import json

    exc = json.JSONDecodeError("Expecting value", "garbage{{{", 0)
    agent = LangGraphMappingAgent(Proposer(_BrokenLLM(exc)), Verifier())
    sugg = agent.run(MappingRequest(source_label="x"), [_cand(1, 0.9)])
    assert sugg.source is MappingSource.UNMAPPED
    assert sugg.no_map_reason is NoMapReason.ERREUR_AGENT
    assert sugg.target_concept_id == UNMAPPED_CONCEPT_ID


@pytest.mark.parametrize(
    "candidates",
    [
        [_cand(201826, 0.8)],
        [_cand(1, 0.9, standard=None), _cand(2, 0.7)],
        [_cand(1, standard=None)],
        [],
    ],
)
def test_parity_with_simple_agent(candidates: list[ConceptCandidate]) -> None:
    request = MappingRequest(source_label="diabète de type 2")
    simple = MappingAgent(Proposer(FakeProposerLLM()), Verifier()).run(request, candidates)
    graph = _lg().run(request, candidates)
    assert graph.source is simple.source
    assert graph.target_concept_id == simple.target_concept_id
    assert graph.no_map_reason is simple.no_map_reason
