"""Tests des agents : Vérificateur, Proposer (sortie fermée), orchestrateur."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import SecretStr

from governed_omop_rag.agents.factory import build_proposer_llm
from governed_omop_rag.agents.llm import FakeProposerLLM
from governed_omop_rag.agents.orchestrator import MappingAgent
from governed_omop_rag.agents.proposer import ClosedOutputViolation, Proposer
from governed_omop_rag.agents.schemas import ProposerOutput, VerdictStatus
from governed_omop_rag.agents.verifier import Verifier
from governed_omop_rag.config import Settings
from governed_omop_rag.core.models import (
    UNMAPPED_CONCEPT_ID,
    ConceptCandidate,
    MappingRequest,
    MappingSource,
    NoMapReason,
)


def _cand(
    concept_id: int,
    score: float = 0.9,
    standard: str | None = "S",
    domain: str = "Condition",
) -> ConceptCandidate:
    return ConceptCandidate(
        concept_id=concept_id,
        concept_name=f"concept-{concept_id}",
        vocabulary_id="SNOMED",
        domain_id=domain,
        standard_concept=standard,
        score=score,
    )


# --------------------------------------------------------------------------- #
# Vérificateur
# --------------------------------------------------------------------------- #
def test_verifier_pass_on_standard() -> None:
    v = Verifier().verify(_cand(1, standard="S"))
    assert v.status is VerdictStatus.PASS
    assert v.passed is True


def test_verifier_fail_on_non_standard() -> None:
    assert Verifier().verify(_cand(1, standard=None)).status is VerdictStatus.FAIL


def test_verifier_fail_on_domain_mismatch() -> None:
    v = Verifier().verify(_cand(1, domain="Drug"), expected_domain="Condition")
    assert v.status is VerdictStatus.FAIL


def test_verifier_pass_on_domain_match() -> None:
    v = Verifier().verify(_cand(1, domain="Condition"), expected_domain="Condition")
    assert v.passed is True


# --------------------------------------------------------------------------- #
# Proposer (sortie fermée)
# --------------------------------------------------------------------------- #
def test_proposer_picks_a_candidate() -> None:
    p = Proposer(FakeProposerLLM())
    out = p.propose("diabète", [_cand(10), _cand(20)])
    assert out is not None
    assert out.concept_id == 10


def test_proposer_respects_exclusion() -> None:
    p = Proposer(FakeProposerLLM())
    out = p.propose("diabète", [_cand(10), _cand(20)], excluded={10})
    assert out is not None
    assert out.concept_id == 20


def test_proposer_returns_none_when_all_excluded() -> None:
    p = Proposer(FakeProposerLLM())
    assert p.propose("x", [_cand(10)], excluded={10}) is None


class _RogueLLM:
    """LLM qui hallucine un concept_id hors de la liste."""

    def propose(self, query: str, candidates: Sequence[ConceptCandidate]) -> ProposerOutput:
        return ProposerOutput(concept_id=999999, justification="inventé")


def test_proposer_rejects_out_of_list() -> None:
    p = Proposer(_RogueLLM())
    with pytest.raises(ClosedOutputViolation):
        p.propose("x", [_cand(10), _cand(20)])


# --------------------------------------------------------------------------- #
# Orchestrateur
# --------------------------------------------------------------------------- #
def _agent(llm: object, max_attempts: int = 3) -> MappingAgent:
    return MappingAgent(Proposer(llm), Verifier(), max_attempts=max_attempts)  # type: ignore[arg-type]


def test_agent_maps_on_first_pass() -> None:
    agent = _agent(FakeProposerLLM())
    sugg = agent.run(MappingRequest(source_label="diabète"), [_cand(201826, 0.8)])
    assert sugg.source is MappingSource.RAG
    assert sugg.target_concept_id == 201826
    assert sugg.confidence == pytest.approx(0.8)


def test_agent_correction_loop_skips_failed_candidate() -> None:
    # Le 1er candidat est non-standard (FAIL) -> l'agent doit passer au 2e.
    candidates = [_cand(1, 0.9, standard=None), _cand(2, 0.7, standard="S")]
    sugg = _agent(FakeProposerLLM()).run(MappingRequest(source_label="x"), candidates)
    assert sugg.source is MappingSource.RAG
    assert sugg.target_concept_id == 2


def test_agent_exhausts_and_returns_unmapped() -> None:
    candidates = [_cand(1, standard=None), _cand(2, standard=None)]
    sugg = _agent(FakeProposerLLM()).run(MappingRequest(source_label="x"), candidates)
    assert sugg.source is MappingSource.UNMAPPED
    assert sugg.target_concept_id == UNMAPPED_CONCEPT_ID
    assert sugg.no_map_reason is NoMapReason.CONFIDENCE_INSUFFISANTE
    assert len(sugg.candidates) == 2  # candidats conservés pour le steward


def test_agent_empty_candidates() -> None:
    sugg = _agent(FakeProposerLLM()).run(MappingRequest(source_label="x"), [])
    assert sugg.source is MappingSource.UNMAPPED
    assert sugg.no_map_reason is NoMapReason.AUCUN_CANDIDAT


def test_agent_rejects_hallucinated_output() -> None:
    sugg = _agent(_RogueLLM()).run(MappingRequest(source_label="x"), [_cand(10), _cand(20)])
    assert sugg.source is MappingSource.UNMAPPED
    assert sugg.no_map_reason is NoMapReason.HORS_VOCABULAIRE


def test_agent_requires_positive_attempts() -> None:
    with pytest.raises(ValueError):
        MappingAgent(Proposer(FakeProposerLLM()), Verifier(), max_attempts=0)


# --------------------------------------------------------------------------- #
# Fabrique du Proposer LLM (dégradation gracieuse)
# --------------------------------------------------------------------------- #
def test_build_proposer_llm_without_key_is_fake() -> None:
    s = Settings(anthropic_api_key=None)
    assert isinstance(build_proposer_llm(s), FakeProposerLLM)


def test_build_proposer_llm_key_but_no_package_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Clé configurée mais paquet 'anthropic' absent -> fallback Fake (pas de crash).
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    s = Settings(anthropic_api_key=SecretStr("sk-test"))
    assert isinstance(build_proposer_llm(s), FakeProposerLLM)


def test_agent_respects_expected_domain() -> None:
    # Bon concept mais mauvais domaine -> FAIL -> pas de candidat validé.
    candidates = [_cand(1, domain="Drug")]
    sugg = _agent(FakeProposerLLM()).run(
        MappingRequest(source_label="x"), candidates, expected_domain="Condition"
    )
    assert sugg.source is MappingSource.UNMAPPED
    assert sugg.no_map_reason is NoMapReason.CONFIDENCE_INSUFFISANTE
