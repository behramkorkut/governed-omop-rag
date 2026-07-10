"""Corpus en médaillon Bronze -> Silver -> Gold (couche data).

Bronze : fichiers bruts OHDSI (CONCEPT.csv, CONCEPT_SYNONYM.csv, ...) + CIM-10 FR.
Silver : concepts filtrés (standard='S', valides), domaines pertinents, normalisés.
Gold   : documents « embedding-ready » (nom + synonymes + domaine + vocabulaire).

Implémenté en transformations Python + DuckDB (pas de framework ETL orchestré).
"""
