"""Corpus en médaillon Bronze -> Silver -> Gold (couche data).

Bronze : fichiers bruts OHDSI (CONCEPT.csv, CONCEPT_SYNONYM.csv, ...) + CIM-10 FR.
Silver : concepts filtrés (standard='S', valides), domaines pertinents, normalisés.
Gold   : documents « embedding-ready » (nom + synonymes + domaine + vocabulaire).

Implémenté en transformations Python + DuckDB (pas de framework ETL orchestré).
"""

from governed_omop_rag.medallion.gold import GoldConcept, build_gold, fetch_gold
from governed_omop_rag.medallion.pipeline import CorpusStats, build_corpus, run_pipeline

__all__ = [
    "GoldConcept",
    "CorpusStats",
    "build_corpus",
    "build_gold",
    "fetch_gold",
    "run_pipeline",
]
