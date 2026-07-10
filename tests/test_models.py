"""Tests des schémas domaine (garde-fous de gouvernance)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from governed_omop_rag.core.models import (
    UNMAPPED_CONCEPT_ID,
    ConceptCandidate,
    MappingRequest,
    MappingSource,
    MappingSuggestion,
    NoMapReason,
    StewardDecision,
    StewardFeedback,
)


def test_request_requires_code_or_label() -> None:
    with pytest.raises(ValidationError):
        MappingRequest()  # ni code ni libellé


def test_request_accepts_label_only() -> None:
    req = MappingRequest(source_label="diabète type 2")
    assert req.source_label == "diabète type 2"
    assert req.source_code is None


def test_candidate_is_standard() -> None:
    std = ConceptCandidate(
        concept_id=201826,
        concept_name="Type 2 diabetes mellitus",
        vocabulary_id="SNOMED",
        domain_id="Condition",
        standard_concept="S",
        score=0.9,
    )
    non_std = std.model_copy(update={"standard_concept": None})
    assert std.is_standard is True
    assert non_std.is_standard is False


def test_candidate_score_bounds() -> None:
    with pytest.raises(ValidationError):
        ConceptCandidate(
            concept_id=1,
            concept_name="x",
            vocabulary_id="SNOMED",
            domain_id="Condition",
            score=1.2,
        )


def test_suggestion_unmapped_by_default() -> None:
    sugg = MappingSuggestion(
        request=MappingRequest(source_label="libellé inconnu"),
        no_map_reason=NoMapReason.AUCUN_CANDIDAT,
    )
    assert sugg.is_mapped is False
    assert sugg.target_concept_id == UNMAPPED_CONCEPT_ID
    assert sugg.source is MappingSource.UNMAPPED


def test_suggestion_mapped_with_no_map_reason_is_incoherent() -> None:
    with pytest.raises(ValidationError):
        MappingSuggestion(
            request=MappingRequest(source_label="diabète type 2"),
            target_concept_id=201826,
            source=MappingSource.RAG,
            no_map_reason=NoMapReason.AMBIGU,  # incohérent avec un mapping retenu
        )


def test_suggestion_unmapped_must_not_claim_rag_source() -> None:
    with pytest.raises(ValidationError):
        MappingSuggestion(
            request=MappingRequest(source_label="x"),
            target_concept_id=UNMAPPED_CONCEPT_ID,
            source=MappingSource.RAG,  # non mappé mais prétend venir du RAG
        )


def test_valid_mapped_suggestion() -> None:
    sugg = MappingSuggestion(
        request=MappingRequest(source_code="E11.9", source_vocabulary="ICD10FR"),
        target_concept_id=201826,
        source=MappingSource.OFFICIAL_MAP,
        confidence=1.0,
        justification="Match exact via alignement officiel CIM-10 <-> SNOMED-CT.",
    )
    assert sugg.is_mapped is True
    assert sugg.confidence == 1.0


def test_steward_feedback_roundtrip() -> None:
    sugg = MappingSuggestion(
        request=MappingRequest(source_label="diabète"),
        no_map_reason=NoMapReason.AMBIGU,
    )
    fb = StewardFeedback(
        suggestion=sugg,
        decision=StewardDecision.EDIT,
        corrected_concept_id=201826,
        reason="Le steward précise : diabète de type 2.",
    )
    assert fb.decision is StewardDecision.EDIT
    assert fb.corrected_concept_id == 201826
