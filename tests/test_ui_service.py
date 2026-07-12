"""Tests de la logique UI (parsing, mise en forme, export source_to_concept_map)."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from governed_omop_rag.core.models import (
    ConceptCandidate,
    MappingRequest,
    MappingSource,
    MappingSuggestion,
    NoMapReason,
)
from governed_omop_rag.ui.service import (
    STCM_COLUMNS,
    collect_validated,
    requests_from_records,
    suggestion_to_row,
    to_source_to_concept_map,
    validated_from_suggestion,
    write_source_to_concept_map_csv,
)


def _cand(cid: int, name: str, vocab: str = "SNOMED", score: float = 0.9) -> ConceptCandidate:
    return ConceptCandidate(
        concept_id=cid,
        concept_name=name,
        vocabulary_id=vocab,
        domain_id="Condition",
        standard_concept="S",
        score=score,
    )


def _mapped() -> MappingSuggestion:
    return MappingSuggestion(
        request=MappingRequest(source_code="E11.9", source_vocabulary="ICD10FR"),
        target_concept_id=201826,
        candidates=[_cand(201826, "Type 2 diabetes mellitus"), _cand(320128, "Hypertension")],
        confidence=0.87,
        source=MappingSource.RAG,
        justification="ok",
    )


def _unmapped() -> MappingSuggestion:
    return MappingSuggestion(
        request=MappingRequest(source_label="libellé inconnu"),
        source=MappingSource.UNMAPPED,
        no_map_reason=NoMapReason.CONFIDENCE_INSUFFISANTE,
    )


# --------------------------------------------------------------------------- #
# Parsing des entrées
# --------------------------------------------------------------------------- #
def test_requests_from_records_skips_empty_and_nan() -> None:
    records: list[Mapping[str, object]] = [
        {"source_code": "E11.9", "source_vocabulary": "ICD10FR"},
        {"source_label": "asthme"},
        {"source_code": "", "source_label": ""},  # ignorée
        {"source_code": "nan", "source_label": "nan"},  # NaN texte -> ignorée
        {"source_code": float("nan"), "source_label": float("nan")},  # vrai NaN pandas
    ]
    reqs = requests_from_records(records)
    assert len(reqs) == 2
    assert reqs[0].source_code == "E11.9"
    assert reqs[1].source_label == "asthme"


# --------------------------------------------------------------------------- #
# Mise en forme
# --------------------------------------------------------------------------- #
def test_suggestion_to_row_mapped() -> None:
    row = suggestion_to_row(_mapped())
    assert row["target_concept_id"] == 201826
    assert row["target_concept_name"] == "Type 2 diabetes mellitus"
    assert row["source"] == "rag"
    assert row["n_candidates"] == 2


def test_suggestion_to_row_unmapped() -> None:
    row = suggestion_to_row(_unmapped())
    assert row["target_concept_id"] == 0
    assert row["target_concept_name"] == ""
    assert row["no_map_reason"] == "confidence_insuffisante"


# --------------------------------------------------------------------------- #
# Validation + export
# --------------------------------------------------------------------------- #
def test_validated_from_suggestion_default() -> None:
    v = validated_from_suggestion(_mapped())
    assert v.target_concept_id == 201826
    assert v.target_vocabulary_id == "SNOMED"
    assert v.source_code == "E11.9"


def test_validated_from_suggestion_edit_override() -> None:
    # Le steward corrige vers un autre candidat de la liste.
    v = validated_from_suggestion(_mapped(), target_concept_id=320128)
    assert v.target_concept_id == 320128
    assert v.target_vocabulary_id == "SNOMED"


def test_to_source_to_concept_map_shape() -> None:
    rows = to_source_to_concept_map([validated_from_suggestion(_mapped())])
    assert list(rows[0].keys()) == STCM_COLUMNS
    assert rows[0]["source_concept_id"] == 0
    assert rows[0]["target_concept_id"] == 201826
    assert rows[0]["source_vocabulary_id"] == "ICD10FR"


def test_collect_validated_skips_rejects() -> None:
    mapped = _mapped()
    decisions = [
        (mapped, 201826),  # accepté
        (mapped, None),  # rejeté -> ignoré
        (mapped, 320128),  # corrigé vers un autre candidat
    ]
    validated = collect_validated(decisions)
    assert [v.target_concept_id for v in validated] == [201826, 320128]


def test_collect_validated_all_rejected_is_empty() -> None:
    assert collect_validated([(_mapped(), None), (_unmapped(), None)]) == []


def test_write_source_to_concept_map_csv(tmp_path: Path) -> None:
    out = tmp_path / "stcm.csv"
    n = write_source_to_concept_map_csv([validated_from_suggestion(_mapped())], out)
    assert n == 1
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["target_concept_id"] == "201826"
    assert rows[0]["source_code"] == "E11.9"
