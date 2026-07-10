"""Orchestration du médaillon : Bronze -> Silver -> Gold.

Transformations enchaînées sur une même connexion DuckDB (pas d'ETL orchestré).
Utilisable en mémoire (tests) ou sur fichier (persistance / CLI).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import duckdb

from governed_omop_rag.medallion.bronze import load_bronze
from governed_omop_rag.medallion.db import (
    BRONZE_CONCEPT,
    BRONZE_SYNONYM,
    connect,
)
from governed_omop_rag.medallion.gold import build_gold
from governed_omop_rag.medallion.silver import build_silver


@dataclass(frozen=True)
class CorpusStats:
    """Décompte des lignes par couche (traçabilité / observabilité)."""

    bronze_concepts: int
    bronze_synonyms: int
    silver_concepts: int
    gold_concepts: int


def build_corpus(
    con: duckdb.DuckDBPyConnection,
    bronze_dir: str | Path,
    domains: Iterable[str] | None = None,
    encoding: str = "utf-8",
) -> CorpusStats:
    """Exécute Bronze -> Silver -> Gold sur la connexion fournie.

    ``encoding`` : encodage des CSV OHDSI ('latin-1'/'cp1252' pour un export FR).
    """
    counts = load_bronze(con, bronze_dir, encoding)
    silver_n = build_silver(con, domains)
    gold_n = build_gold(con)
    return CorpusStats(
        bronze_concepts=counts[BRONZE_CONCEPT],
        bronze_synonyms=counts[BRONZE_SYNONYM],
        silver_concepts=silver_n,
        gold_concepts=gold_n,
    )


def run_pipeline(
    bronze_dir: str | Path,
    duckdb_path: str | Path = ":memory:",
    domains: Iterable[str] | None = None,
    encoding: str = "utf-8",
) -> CorpusStats:
    """Ouvre une connexion (fichier persistant ou mémoire), construit, ferme.

    Pour ``:memory:`` la base disparaît à la fermeture : utiliser un chemin
    fichier pour persister le corpus destiné à l'indexation.
    """
    con = connect(duckdb_path)
    try:
        return build_corpus(con, bronze_dir, domains, encoding)
    finally:
        con.close()
