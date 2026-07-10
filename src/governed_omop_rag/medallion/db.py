"""Connexion DuckDB — staging local du vocabulaire (couche data).

DuckDB héberge les tables/vues Bronze/Silver/Gold. Volontairement léger :
transformations SQL/Python, pas de framework ETL orchestré (cf. CONTEXT.md §5.2).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

# Nom des tables/vues du médaillon (une seule source de vérité).
BRONZE_CONCEPT = "bronze_concept"
BRONZE_SYNONYM = "bronze_concept_synonym"
SILVER_CONCEPT = "silver_concept"
GOLD_CONCEPT = "gold_concept"


def connect(path: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    """Ouvre une connexion DuckDB (fichier ou en mémoire).

    Le répertoire parent est créé si nécessaire pour un chemin fichier.
    """
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


@contextmanager
def connection(path: str | Path = ":memory:") -> Iterator[duckdb.DuckDBPyConnection]:
    """Context manager fermant proprement la connexion."""
    con = connect(path)
    try:
        yield con
    finally:
        con.close()
