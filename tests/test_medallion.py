"""Tests du corpus médaillon Bronze -> Silver -> Gold."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from pathlib import Path

import duckdb
import pytest

from governed_omop_rag.medallion.bronze import load_bronze
from governed_omop_rag.medallion.db import (
    BRONZE_CONCEPT,
    GOLD_CONCEPT,
    SILVER_CONCEPT,
    connect,
)
from governed_omop_rag.medallion.gold import build_gold, fetch_gold
from governed_omop_rag.medallion.normalize import (
    expand_ligatures,
    normalize_ascii,
    normalize_text,
    strip_accents,
)
from governed_omop_rag.medallion.pipeline import build_corpus, run_pipeline
from governed_omop_rag.medallion.silver import build_silver

FIXTURES = Path(__file__).parent / "fixtures"

_CONCEPT_COLS = [
    "concept_id",
    "concept_name",
    "domain_id",
    "vocabulary_id",
    "concept_class_id",
    "standard_concept",
    "concept_code",
    "valid_start_date",
    "valid_end_date",
    "invalid_reason",
]
_SYN_COLS = ["concept_id", "concept_synonym_name", "language_concept_id"]


def _write_bronze(
    d: Path,
    concepts: Sequence[Sequence[object]],
    synonyms: Sequence[Sequence[object]] = (),
    encoding: str = "utf-8",
) -> Path:
    """Écrit des fichiers Bronze OHDSI (tab-delimited) dans le répertoire d."""
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "CONCEPT.csv", "w", newline="", encoding=encoding) as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(_CONCEPT_COLS)
        w.writerows(concepts)
    with open(d / "CONCEPT_SYNONYM.csv", "w", newline="", encoding=encoding) as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(_SYN_COLS)
        w.writerows(synonyms)
    return d


def _std_condition(concept_id: int, name: str) -> list[object]:
    """Fabrique une ligne CONCEPT standard/valide (Condition SNOMED)."""
    return [
        concept_id,
        name,
        "Condition",
        "SNOMED",
        "Clinical Finding",
        "S",
        str(concept_id),
        "1970-01-01",
        "2099-12-31",
        "",
    ]


def _scalar(
    con: duckdb.DuckDBPyConnection, sql: str, params: Sequence[object] | None = None
) -> object:
    """Exécute une requête et retourne la 1re colonne de la 1re ligne (non-None)."""
    row = con.execute(sql, list(params) if params is not None else []).fetchone()
    assert row is not None, "requête sans résultat"
    return row[0]


# concept_ids attendus après filtrage Silver (standard='S' ET valides).
STANDARD_VALID_IDS = {201826, 4048098, 320128, 1503297}
NON_STANDARD_ID = 45542735  # standard_concept vide
INVALID_ID = 111111  # invalid_reason='D'


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    c = connect(":memory:")
    load_bronze(c, FIXTURES)
    yield c
    c.close()


# --------------------------------------------------------------------------- #
# Normalisation (unitaire)
# --------------------------------------------------------------------------- #
def test_normalize_text_lowercases_and_collapses() -> None:
    assert normalize_text("  Diabète   de\tType 2 ") == "diabète de type 2"


def test_normalize_text_handles_none_and_empty() -> None:
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_strip_accents() -> None:
    assert strip_accents("diabète") == "diabete"
    assert strip_accents("hémorragie çà") == "hemorragie ca"
    assert strip_accents("hypertension artérielle") == "hypertension arterielle"


def test_normalize_ascii() -> None:
    assert normalize_ascii("Diabète  Type 2") == "diabete type 2"


def test_ligatures_expanded_in_ascii_form() -> None:
    # Politique : la forme ASCII étend œ -> oe, æ -> ae (matching FR robuste).
    assert expand_ligatures("œæŒÆ") == "oeaeOEAE"
    assert normalize_ascii("Œdème") == "oedeme"
    assert strip_accents("cœur") == "coeur"
    assert normalize_ascii("nævus") == "naevus"
    # La forme accentuée conserve la ligature (casefold seulement).
    assert normalize_text("Œdème") == "œdème"


# --------------------------------------------------------------------------- #
# Bronze
# --------------------------------------------------------------------------- #
def test_bronze_loads_all_rows(con: duckdb.DuckDBPyConnection) -> None:
    n = _scalar(con, f"SELECT COUNT(*) FROM {BRONZE_CONCEPT}")
    assert n == 6


def test_bronze_empty_fields_become_null(con: duckdb.DuckDBPyConnection) -> None:
    # Le concept non-standard a standard_concept vide -> doit être NULL.
    val = _scalar(
        con,
        f"SELECT standard_concept FROM {BRONZE_CONCEPT} WHERE concept_id = ?",
        [NON_STANDARD_ID],
    )
    assert val is None
    # Un concept valide a invalid_reason vide -> NULL.
    inv = _scalar(
        con,
        f"SELECT invalid_reason FROM {BRONZE_CONCEPT} WHERE concept_id = ?",
        [201826],
    )
    assert inv is None


# --------------------------------------------------------------------------- #
# Silver
# --------------------------------------------------------------------------- #
def test_silver_keeps_only_standard_and_valid(con: duckdb.DuckDBPyConnection) -> None:
    build_silver(con)
    ids = {r[0] for r in con.execute(f"SELECT concept_id FROM {SILVER_CONCEPT}").fetchall()}
    assert ids == STANDARD_VALID_IDS
    assert NON_STANDARD_ID not in ids  # non-standard écarté
    assert INVALID_ID not in ids  # invalide écarté


def test_silver_domain_filter(con: duckdb.DuckDBPyConnection) -> None:
    build_silver(con, domains={"Condition"})
    ids = {r[0] for r in con.execute(f"SELECT concept_id FROM {SILVER_CONCEPT}").fetchall()}
    assert 1503297 not in ids  # metformin (Drug) exclu
    assert ids == {201826, 4048098, 320128}


def test_silver_adds_normalized_name(con: duckdb.DuckDBPyConnection) -> None:
    build_silver(con)
    norm = _scalar(con, f"SELECT concept_name_norm FROM {SILVER_CONCEPT} WHERE concept_id = 201826")
    assert norm == "type 2 diabetes mellitus"


# --------------------------------------------------------------------------- #
# Gold
# --------------------------------------------------------------------------- #
def test_gold_doc_text_structure(con: duckdb.DuckDBPyConnection) -> None:
    build_silver(con)
    build_gold(con)
    doc = _scalar(con, f"SELECT doc_text FROM {GOLD_CONCEPT} WHERE concept_id = 201826")
    assert isinstance(doc, str)
    assert "Type 2 diabetes mellitus" in doc
    assert "synonymes:" in doc
    assert "diabète de type 2" in doc  # synonyme FR avec accents conservé
    assert "domaine: Condition" in doc
    assert "vocabulaire: SNOMED" in doc


def test_gold_excludes_synonym_identical_to_name(con: duckdb.DuckDBPyConnection) -> None:
    build_silver(con)
    build_gold(con)
    [gc] = [c for c in fetch_gold(con) if c.concept_id == 201826]
    # "Type 2 diabetes mellitus" (== nom) ne doit pas figurer dans les synonymes.
    assert "Type 2 diabetes mellitus" not in gc.synonyms
    assert set(gc.synonyms) == {"diabète de type 2", "T2DM"}


def test_gold_concept_without_synonym(con: duckdb.DuckDBPyConnection) -> None:
    build_silver(con)
    build_gold(con)
    [metformin] = [c for c in fetch_gold(con) if c.concept_id == 1503297]
    assert metformin.synonyms == []
    assert "synonymes:" not in metformin.doc_text
    assert "domaine: Drug" in metformin.doc_text


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def test_build_corpus_stats() -> None:
    c = connect(":memory:")
    try:
        stats = build_corpus(c, FIXTURES)
    finally:
        c.close()
    assert stats.bronze_concepts == 6
    assert stats.bronze_synonyms == 5
    assert stats.silver_concepts == 4
    assert stats.gold_concepts == 4


def test_run_pipeline_persists_to_file(tmp_path: Path) -> None:
    db = tmp_path / "corpus.duckdb"
    stats = run_pipeline(FIXTURES, db)
    assert stats.gold_concepts == 4
    # Rouvrir la base persistée et vérifier que Gold est bien là.
    c = connect(db)
    try:
        n = _scalar(c, f"SELECT COUNT(*) FROM {GOLD_CONCEPT}")
    finally:
        c.close()
    assert n == 4


def test_missing_bronze_file_raises(tmp_path: Path) -> None:
    c = connect(":memory:")
    try:
        with pytest.raises(FileNotFoundError):
            load_bronze(c, tmp_path)  # répertoire vide
    finally:
        c.close()


# --------------------------------------------------------------------------- #
# Edge cases médicaux français
# --------------------------------------------------------------------------- #
def test_gold_preserves_double_negation(tmp_path: Path) -> None:
    """Un libellé à négation ('Absence de fièvre') ne doit pas être altéré :
    le doc_text conserve le sens (aucune suppression de la négation)."""
    _write_bronze(tmp_path, [_std_condition(900001, "Absence de fièvre")])
    c = connect(":memory:")
    try:
        load_bronze(c, tmp_path)
        build_silver(c)
        build_gold(c)
        [gc] = fetch_gold(c)
        assert gc.concept_name == "Absence de fièvre"
        assert "Absence de fièvre" in gc.doc_text
        # La normalisation lexicale conserve elle aussi la négation.
        norm = _scalar(
            c, f"SELECT concept_name_norm FROM {SILVER_CONCEPT} WHERE concept_id = 900001"
        )
        assert norm == "absence de fièvre"
    finally:
        c.close()


def test_bronze_reads_latin1_encoding(tmp_path: Path) -> None:
    """Un export FR encodé en Latin-1 se charge correctement avec encoding='latin-1'.

    NB : Latin-1 strict ne contient pas la ligature œ (elle est en cp1252 /
    ISO-8859-15) ; on teste ici des accents Latin-1 valides (è, é, à).
    """
    _write_bronze(
        tmp_path,
        [
            _std_condition(1, "diabète de type 2"),
            _std_condition(2, "hypertension artérielle"),
        ],
        encoding="latin-1",
    )
    c = connect(":memory:")
    try:
        load_bronze(c, tmp_path, encoding="latin-1")
        name = _scalar(c, f"SELECT concept_name FROM {BRONZE_CONCEPT} WHERE concept_id = 1")
        assert name == "diabète de type 2"
        name2 = _scalar(c, f"SELECT concept_name FROM {BRONZE_CONCEPT} WHERE concept_id = 2")
        assert name2 == "hypertension artérielle"
    finally:
        c.close()


def test_pipeline_propagates_encoding(tmp_path: Path) -> None:
    """L'encodage passé à run_pipeline atteint bien le loader Bronze (Latin-1)."""
    bronze = tmp_path / "bronze"
    _write_bronze(bronze, [_std_condition(1, "diabète")], encoding="latin-1")
    db = tmp_path / "corpus.duckdb"
    stats = run_pipeline(bronze, db, encoding="latin-1")
    assert stats.gold_concepts == 1
    c = connect(db)
    try:
        name = _scalar(c, f"SELECT concept_name FROM {BRONZE_CONCEPT} WHERE concept_id = 1")
        assert name == "diabète"
    finally:
        c.close()
