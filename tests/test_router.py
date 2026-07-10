"""Tests du Router déterministe v1 (match officiel CIM-10 <-> SNOMED-CT)."""

from __future__ import annotations

from pathlib import Path

import pytest

from governed_omop_rag.core.models import (
    UNMAPPED_CONCEPT_ID,
    MappingRequest,
    MappingSource,
    NoMapReason,
)
from governed_omop_rag.router.deterministic import (
    JUSTIFICATION_MATCH,
    JUSTIFICATION_NO_MATCH,
    DeterministicRouter,
    OfficialMap,
    normalize_code,
    route_deterministic,
)


@pytest.fixture
def official_map() -> OfficialMap:
    return OfficialMap(
        {"E11.9": 201826, "J45": 4048098, "I10": 320128},
        names={"E11.9": "Type 2 diabetes mellitus"},
    )


# --------------------------------------------------------------------------- #
# normalize_code (unitaire)
# --------------------------------------------------------------------------- #
def test_normalize_code() -> None:
    assert normalize_code(" e11.9 ") == "E11.9"
    assert normalize_code("J45") == "J45"
    assert normalize_code(None) == ""
    assert normalize_code("") == ""


# --------------------------------------------------------------------------- #
# Match exact
# --------------------------------------------------------------------------- #
def test_exact_match(official_map: OfficialMap) -> None:
    sugg = route_deterministic(MappingRequest(source_code="E11.9"), official_map)
    assert sugg.target_concept_id == 201826
    assert sugg.source is MappingSource.OFFICIAL_MAP
    assert sugg.confidence == 1.0
    assert sugg.justification == JUSTIFICATION_MATCH
    assert sugg.candidates == []
    assert sugg.no_map_reason is None
    assert sugg.is_mapped is True


def test_router_class_equivalent(official_map: OfficialMap) -> None:
    router = DeterministicRouter(official_map)
    sugg = router.route(MappingRequest(source_code="J45"))
    assert sugg.target_concept_id == 4048098
    assert sugg.source is MappingSource.OFFICIAL_MAP


# --------------------------------------------------------------------------- #
# Code inconnu
# --------------------------------------------------------------------------- #
def test_unknown_code_is_unmapped(official_map: OfficialMap) -> None:
    sugg = route_deterministic(MappingRequest(source_code="Z99.9"), official_map)
    assert sugg.target_concept_id == UNMAPPED_CONCEPT_ID
    assert sugg.source is MappingSource.UNMAPPED
    assert sugg.no_map_reason is NoMapReason.HORS_VOCABULAIRE
    assert sugg.justification == JUSTIFICATION_NO_MATCH
    assert sugg.is_mapped is False


# --------------------------------------------------------------------------- #
# Libellé sans code (v1 ne gère que les codes)
# --------------------------------------------------------------------------- #
def test_label_only_is_unmapped(official_map: OfficialMap) -> None:
    sugg = route_deterministic(MappingRequest(source_label="diabète type 2"), official_map)
    assert sugg.target_concept_id == UNMAPPED_CONCEPT_ID
    assert sugg.source is MappingSource.UNMAPPED
    assert sugg.no_map_reason is NoMapReason.AUCUN_CANDIDAT


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_case_and_whitespace_insensitive(official_map: OfficialMap) -> None:
    for code in ("e11.9", "E11.9", "  e11.9  ", "E11.9\t"):
        sugg = route_deterministic(MappingRequest(source_code=code), official_map)
        assert sugg.target_concept_id == 201826, code


def test_empty_string_code_does_not_crash(official_map: OfficialMap) -> None:
    # source_code vide mais libellé présent (le modèle exige l'un ou l'autre).
    req = MappingRequest(source_code="", source_label="placeholder")
    sugg = route_deterministic(req, official_map)
    assert sugg.target_concept_id == UNMAPPED_CONCEPT_ID
    assert sugg.no_map_reason is NoMapReason.AUCUN_CANDIDAT


def test_lookup_none_and_empty(official_map: OfficialMap) -> None:
    assert official_map.lookup(None) is None
    assert official_map.lookup("") is None
    assert official_map.lookup("nope") is None


def test_map_container_helpers(official_map: OfficialMap) -> None:
    assert len(official_map) == 3
    assert "E11.9" in official_map
    assert "e11.9" in official_map  # insensible à la casse
    assert "X00" not in official_map
    assert official_map.name_of("E11.9") == "Type 2 diabetes mellitus"


# --------------------------------------------------------------------------- #
# Chargement CSV (substituable par le vrai alignement ATIH)
# --------------------------------------------------------------------------- #
def test_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "align.csv"
    csv_path.write_text(
        "source_code,target_concept_id,target_concept_name\n"
        "E11.9,201826,Type 2 diabetes mellitus\n"
        "J45,4048098,Asthma\n",
        encoding="utf-8",
    )
    m = OfficialMap.from_csv(csv_path)
    assert len(m) == 2
    assert m.lookup("e11.9") == 201826  # insensible à la casse après chargement
    assert m.name_of("J45") == "Asthma"


def test_from_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OfficialMap.from_csv(tmp_path / "absent.csv")


def test_from_csv_skips_incomplete_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "align.csv"
    csv_path.write_text(
        "source_code,target_concept_id,target_concept_name\n"
        "E11.9,201826,Type 2 diabetes mellitus\n"
        ",999,ligne sans code\n"
        "J45,,ligne sans concept_id\n",
        encoding="utf-8",
    )
    m = OfficialMap.from_csv(csv_path)
    assert len(m) == 1
    assert m.lookup("E11.9") == 201826
