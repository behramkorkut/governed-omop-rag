"""Couche Silver : filtrage OMOP + normalisation.

Règles dures (gouvernance, cf. CONTEXT.md §4.3) :
- ``standard_concept = 'S'`` (concepts standard uniquement) ;
- ``invalid_reason IS NULL`` (concepts valides uniquement) ;
- filtre optionnel sur ``domain_id`` (Condition d'abord, cf. §3).

On ajoute une colonne ``concept_name_norm`` (minuscules + espaces réduits),
calculée en SQL natif — équivalent lexical de ``normalize.normalize_text``,
sans dépendre d'une UDF (robustesse inter-versions DuckDB).
"""

from __future__ import annotations

from collections.abc import Iterable

import duckdb

from governed_omop_rag.medallion.db import BRONZE_CONCEPT, SILVER_CONCEPT

# Normalisation lexicale : minuscules + collapse des espaces multiples.
_NORM_SQL = "lower(trim(regexp_replace(concept_name, '\\s+', ' ', 'g')))"


def build_silver(
    con: duckdb.DuckDBPyConnection,
    domains: Iterable[str] | None = None,
) -> int:
    """Construit la table Silver depuis Bronze. Retourne le nb de lignes retenues.

    ``domains`` : ensemble de ``domain_id`` à conserver (ex. {"Condition"}).
    None = tous les domaines (mais toujours standard + valide).
    """
    domain_clause = ""
    params: list[str] = []
    domain_list = list(domains) if domains is not None else []
    if domain_list:
        placeholders = ", ".join(["?"] * len(domain_list))
        domain_clause = f"AND domain_id IN ({placeholders})"
        params = domain_list

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {SILVER_CONCEPT} AS
        SELECT
            concept_id,
            concept_name,
            {_NORM_SQL} AS concept_name_norm,
            domain_id,
            vocabulary_id,
            concept_class_id,
            concept_code
        FROM {BRONZE_CONCEPT}
        WHERE standard_concept = 'S'
          AND invalid_reason IS NULL
          {domain_clause}
        """,
        params,
    )
    row = con.execute(f"SELECT COUNT(*) FROM {SILVER_CONCEPT}").fetchone()
    return int(row[0]) if row else 0
