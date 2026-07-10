"""Couche Bronze : ingestion brute des fichiers OHDSI (format Athena).

Charge ``CONCEPT.csv`` et ``CONCEPT_SYNONYM.csv`` (tab-delimited, convention
OHDSI/Athena) dans DuckDB, en typant a minima et en normalisant les champs
« vides » en NULL (OHDSI code l'absence par une chaîne vide).

Aucune transformation métier ici : on reflète fidèlement la source. Déposer les
vrais exports Athena dans ``data/bronze/`` ; un échantillon est fourni pour les
tests et la démo « 2 minutes ».
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from governed_omop_rag.medallion.db import BRONZE_CONCEPT, BRONZE_SYNONYM

CONCEPT_FILE = "CONCEPT.csv"
SYNONYM_FILE = "CONCEPT_SYNONYM.csv"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier OHDSI introuvable : {path}. "
            "Déposez les exports Athena dans le répertoire Bronze."
        )
    return path


def load_concept(con: duckdb.DuckDBPyConnection, path: Path, encoding: str = "utf-8") -> int:
    """Charge CONCEPT.csv dans la table Bronze. Retourne le nb de lignes.

    ``encoding`` : utf-8 par défaut ; passer 'latin-1' pour un export FR classique.
    """
    _require(path)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {BRONZE_CONCEPT} AS
        SELECT
            CAST(concept_id AS BIGINT)              AS concept_id,
            concept_name                            AS concept_name,
            domain_id                               AS domain_id,
            vocabulary_id                           AS vocabulary_id,
            concept_class_id                        AS concept_class_id,
            NULLIF(TRIM(standard_concept), '')      AS standard_concept,
            concept_code                            AS concept_code,
            valid_start_date                        AS valid_start_date,
            valid_end_date                          AS valid_end_date,
            NULLIF(TRIM(invalid_reason), '')        AS invalid_reason
        FROM read_csv(?, delim='\t', header=true, all_varchar=true, encoding=?)
        """,
        [str(path), encoding],
    )
    return _count(con, BRONZE_CONCEPT)


def load_synonym(con: duckdb.DuckDBPyConnection, path: Path, encoding: str = "utf-8") -> int:
    """Charge CONCEPT_SYNONYM.csv dans la table Bronze. Retourne le nb de lignes."""
    _require(path)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {BRONZE_SYNONYM} AS
        SELECT
            CAST(concept_id AS BIGINT)  AS concept_id,
            concept_synonym_name        AS concept_synonym_name,
            language_concept_id         AS language_concept_id
        FROM read_csv(?, delim='\t', header=true, all_varchar=true, encoding=?)
        """,
        [str(path), encoding],
    )
    return _count(con, BRONZE_SYNONYM)


def load_bronze(
    con: duckdb.DuckDBPyConnection, bronze_dir: str | Path, encoding: str = "utf-8"
) -> dict[str, int]:
    """Charge concepts + synonymes depuis un répertoire Bronze.

    Retourne le nombre de lignes chargées par table.
    """
    d = Path(bronze_dir)
    return {
        BRONZE_CONCEPT: load_concept(con, d / CONCEPT_FILE, encoding),
        BRONZE_SYNONYM: load_synonym(con, d / SYNONYM_FILE, encoding),
    }


def _count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    row = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0
