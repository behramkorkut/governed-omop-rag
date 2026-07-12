"""Tests du feedback steward (classification + persistance DuckDB)."""

from __future__ import annotations

from pathlib import Path

from governed_omop_rag.core.models import (
    ConceptCandidate,
    MappingRequest,
    MappingSource,
    MappingSuggestion,
    NoMapReason,
    StewardDecision,
)
from governed_omop_rag.feedback import (
    FeedbackStore,
    feedback_records_from_decisions,
)


def _cand(cid: int, vocab: str = "SNOMED") -> ConceptCandidate:
    return ConceptCandidate(
        concept_id=cid,
        concept_name=f"c-{cid}",
        vocabulary_id=vocab,
        domain_id="Condition",
        standard_concept="S",
        score=0.9,
    )


def _mapped() -> MappingSuggestion:
    return MappingSuggestion(
        request=MappingRequest(source_code="E11.9", source_vocabulary="ICD10FR"),
        target_concept_id=201826,
        candidates=[_cand(201826), _cand(320128)],
        confidence=0.9,
        source=MappingSource.RAG,
    )


def _unmapped() -> MappingSuggestion:
    return MappingSuggestion(
        request=MappingRequest(source_label="ambigu"),
        source=MappingSource.UNMAPPED,
        no_map_reason=NoMapReason.CONFIDENCE_INSUFFISANTE,
        candidates=[_cand(999)],
    )


# --------------------------------------------------------------------------- #
# Classification des décisions
# --------------------------------------------------------------------------- #
def test_decisions_classification() -> None:
    mapped = _mapped()
    records = feedback_records_from_decisions(
        [
            (mapped, 201826),  # accept
            (mapped, 320128),  # edit
            (mapped, None),  # reject
        ]
    )
    assert [r.decision for r in records] == [
        StewardDecision.ACCEPT,
        StewardDecision.EDIT,
        StewardDecision.REJECT,
    ]
    assert records[0].final_concept_id == 201826
    assert records[1].final_concept_id == 320128
    assert records[2].final_concept_id == 0


def test_unmapped_then_chosen_is_edit() -> None:
    # Suggestion non mappée, mais le steward choisit un candidat -> EDIT.
    [rec] = feedback_records_from_decisions([(_unmapped(), 999)])
    assert rec.decision is StewardDecision.EDIT
    assert rec.proposed_concept_id == 0
    assert rec.final_concept_id == 999


# --------------------------------------------------------------------------- #
# Persistance DuckDB
# --------------------------------------------------------------------------- #
def test_store_roundtrip_and_persist(tmp_path: Path) -> None:
    db = tmp_path / "fb.duckdb"
    store = FeedbackStore(db)
    n = store.record(feedback_records_from_decisions([(_mapped(), 201826)]))
    store.close()
    assert n == 1

    reopened = FeedbackStore(db)
    try:
        assert reopened.count() == 1
    finally:
        reopened.close()


def test_to_gold_records_keeps_accept_edit_only(tmp_path: Path) -> None:
    mapped = _mapped()
    store = FeedbackStore(tmp_path / "fb.duckdb")
    try:
        store.record(
            feedback_records_from_decisions(
                [
                    (mapped, 201826),  # accept -> gardé
                    (mapped, 320128),  # edit -> gardé
                    (mapped, None),  # reject -> exclu
                ]
            )
        )
        gold = store.to_gold_records()
    finally:
        store.close()
    ids = sorted(str(g["expected_concept_id"]) for g in gold)
    assert ids == ["201826", "320128"]
    assert all("expected_concept_id" in g for g in gold)
