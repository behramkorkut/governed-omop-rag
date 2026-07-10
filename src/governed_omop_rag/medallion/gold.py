"""Couche Gold : documents « embedding-ready ».

Un concept = ``nom + synonymes + domaine + vocabulaire`` concaténés en un
``doc_text`` unique — c'est ce texte qui sera vectorisé (BioLORD) et indexé
(cf. CONTEXT.md §5.2). Les synonymes strictement identiques au nom sont écartés.
"""

from __future__ import annotations

import duckdb
from pydantic import BaseModel, Field

from governed_omop_rag.medallion.db import (
    BRONZE_SYNONYM,
    GOLD_CONCEPT,
    SILVER_CONCEPT,
)

_SYNONYM_SEP = " ; "


class GoldConcept(BaseModel):
    """Concept prêt à vectoriser (sortie de la couche Gold)."""

    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_code: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    doc_text: str


def build_gold(con: duckdb.DuckDBPyConnection) -> int:
    """Construit la table Gold depuis Silver + synonymes Bronze. Retourne le nb de lignes."""
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {GOLD_CONCEPT} AS
        WITH syn AS (
            SELECT concept_id, TRIM(concept_synonym_name) AS syn
            FROM {BRONZE_SYNONYM}
            WHERE TRIM(concept_synonym_name) <> ''
        ),
        agg AS (
            SELECT
                s.concept_id,
                s.concept_name,
                s.concept_name_norm,
                s.domain_id,
                s.vocabulary_id,
                s.concept_code,
                COALESCE(
                    string_agg(DISTINCT syn.syn, '{_SYNONYM_SEP}' ORDER BY syn.syn),
                    ''
                ) AS synonyms
            FROM {SILVER_CONCEPT} s
            LEFT JOIN syn
                ON syn.concept_id = s.concept_id
               AND lower(syn.syn) <> lower(TRIM(s.concept_name))
            GROUP BY ALL
        )
        SELECT
            *,
            concept_name
            || CASE WHEN synonyms <> '' THEN ' | synonymes: ' || synonyms ELSE '' END
            || ' | domaine: ' || domain_id
            || ' | vocabulaire: ' || vocabulary_id
            AS doc_text
        FROM agg
        """
    )
    row = con.execute(f"SELECT COUNT(*) FROM {GOLD_CONCEPT}").fetchone()
    return int(row[0]) if row else 0


def fetch_gold(con: duckdb.DuckDBPyConnection) -> list[GoldConcept]:
    """Retourne les concepts Gold comme objets typés (prêts pour l'embedding)."""
    rows = con.execute(
        f"""
        SELECT concept_id, concept_name, domain_id, vocabulary_id,
               concept_code, synonyms, doc_text
        FROM {GOLD_CONCEPT}
        ORDER BY concept_id
        """
    ).fetchall()
    concepts: list[GoldConcept] = []
    for r in rows:
        synonyms = [s for s in (r[5] or "").split(_SYNONYM_SEP) if s]
        concepts.append(
            GoldConcept(
                concept_id=r[0],
                concept_name=r[1],
                domain_id=r[2],
                vocabulary_id=r[3],
                concept_code=r[4],
                synonyms=synonyms,
                doc_text=r[6],
            )
        )
    return concepts
