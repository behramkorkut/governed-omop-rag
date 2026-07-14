"""Construit un gold set réel CIM-10 (ICD10) -> SNOMED à partir des vocabulaires OHDSI (Athena).

Entrée : le répertoire décompressé du bundle Athena (fichiers tab-delimited)
    - CONCEPT.csv
    - CONCEPT_RELATIONSHIP.csv

Méthode (vérité terrain non ambiguë) :
    1. On prend les concepts source de vocabulaire ICD10 / ICD10CM.
    2. On suit la relation officielle « Maps to » vers un concept cible.
    3. On ne garde que les cibles STANDARD (standard_concept = 'S') et VALIDES
       (invalid_reason IS NULL), dans les vocabulaires standard (SNOMED en tête).
    4. On ne garde que les codes source qui mappent vers EXACTEMENT UNE cible
       standard (mappings 1-à-1) : ambiguïté = exclu du gold set.
    5. Échantillonnage déterministe (seed) équilibré par domaine.

Sortie : ``data/eval/gold_set.csv`` au format attendu par le loader :
    ``source_code,source_label,expected_concept_id``

Usage :
    uv run python scripts/build_gold_set.py --athena-dir /chemin/vers/athena \\
        --n 80 --domains Condition --out data/eval/gold_set.csv

Le même bundle Athena sert à construire le corpus (``gor build-corpus``) : gold set
et corpus proviennent ainsi du MÊME snapshot de vocabulaire (indispensable pour
que le recall@k ait un sens).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

# Vocabulaire source par défaut : CIM10 = édition française réelle (ATIH, vocab OHDSI 130).
# Fallback possible via --source-vocab si CIM10 n'a pas de « Maps to » direct vers SNOMED :
#   ICD10 (OMS, id 34) ou ICD10CM (NCHS, id 70).
DEFAULT_SOURCE_VOCAB = "CIM10"
# Priorité des vocabulaires cible standard (SNOMED d'abord ; RxNorm/LOINC ensuite).
TARGET_VOCABS = ("SNOMED", "RxNorm", "LOINC")


def build_gold_set(
    athena_dir: Path,
    out_path: Path,
    n: int,
    domains: tuple[str, ...],
    source_vocabs: tuple[str, ...] = (DEFAULT_SOURCE_VOCAB,),
    seed: int = 42,
) -> int:
    """Extrait le gold set et l'écrit en CSV. Retourne le nombre de lignes écrites."""
    concept = athena_dir / "CONCEPT.csv"
    rel = athena_dir / "CONCEPT_RELATIONSHIP.csv"
    for f in (concept, rel):
        if not f.exists():
            raise FileNotFoundError(
                f"Fichier Athena manquant : {f}. Décompresse le bundle Athena "
                f"et pointe --athena-dir vers le dossier contenant CONCEPT.csv."
            )

    con = duckdb.connect()
    # Lecture directe des CSV tab-delimited d'Athena (pas de chargement en RAM).
    # CREATE VIEW n'accepte pas de paramètre préparé -> chemin inliné (échappé).
    concept_sql = str(concept).replace("'", "''")
    rel_sql = str(rel).replace("'", "''")
    con.execute(
        f"CREATE VIEW concept AS "
        f"SELECT * FROM read_csv('{concept_sql}', delim='\t', header=true, all_varchar=true)"
    )
    con.execute(
        f"CREATE VIEW rel AS "
        f"SELECT * FROM read_csv('{rel_sql}', delim='\t', header=true, all_varchar=true)"
    )

    src_list = ", ".join(f"'{v}'" for v in source_vocabs)
    tgt_list = ", ".join(f"'{v}'" for v in TARGET_VOCABS)
    dom_list = ", ".join(f"'{d}'" for d in domains)

    # Paires (source ICD10) -> (cible standard) via « Maps to », cibles standard/valides.
    # On calcule le nombre de cibles distinctes par code source pour ne garder que le 1-à-1.
    query = f"""
    WITH mapping AS (
        SELECT
            s.concept_code                          AS source_code,
            s.concept_name                          AS source_label,
            CAST(t.concept_id AS BIGINT)            AS expected_concept_id,
            t.domain_id                             AS domain_id,
            t.vocabulary_id                         AS target_vocabulary_id
        FROM rel r
        JOIN concept s ON CAST(r.concept_id_1 AS BIGINT) = CAST(s.concept_id AS BIGINT)
        JOIN concept t ON CAST(r.concept_id_2 AS BIGINT) = CAST(t.concept_id AS BIGINT)
        WHERE r.relationship_id = 'Maps to'
          AND s.vocabulary_id IN ({src_list})
          AND t.vocabulary_id IN ({tgt_list})
          AND t.standard_concept = 'S'
          AND (t.invalid_reason IS NULL OR t.invalid_reason = '')
          AND (s.invalid_reason IS NULL OR s.invalid_reason = '')
          AND t.domain_id IN ({dom_list})
    ),
    one_to_one AS (
        SELECT source_code
        FROM mapping
        GROUP BY source_code
        HAVING COUNT(DISTINCT expected_concept_id) = 1
    ),
    clean AS (
        SELECT DISTINCT m.source_code, m.source_label, m.expected_concept_id, m.domain_id
        FROM mapping m
        JOIN one_to_one o USING (source_code)
    ),
    ranked AS (
        SELECT
            source_code, source_label, expected_concept_id, domain_id,
            ROW_NUMBER() OVER (
                PARTITION BY domain_id
                ORDER BY hash(source_code || '{seed}')
            ) AS rn
        FROM clean
    )
    SELECT source_code, source_label, expected_concept_id
    FROM ranked
    ORDER BY domain_id, rn
    LIMIT {int(n)};
    """
    rows = con.execute(query).fetchall()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("source_code,source_label,expected_concept_id\n")
        for code, label, cid in rows:
            safe_label = (label or "").replace('"', "").replace(",", " ")
            fh.write(f'{code},"{safe_label}",{cid}\n')

    con.close()
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--athena-dir",
        required=True,
        type=Path,
        help="Dossier du bundle Athena décompressé (CONCEPT.csv + CONCEPT_RELATIONSHIP.csv).",
    )
    p.add_argument("--out", type=Path, default=Path("data/eval/gold_set.csv"))
    p.add_argument("--n", type=int, default=80, help="Nombre de mappings (50-100 recommandé).")
    p.add_argument(
        "--domains",
        nargs="+",
        default=["Condition"],
        help="Domaines cible (ex. Condition Drug Measurement).",
    )
    p.add_argument(
        "--source-vocab",
        nargs="+",
        default=[DEFAULT_SOURCE_VOCAB],
        help="Vocabulaire(s) source. Défaut CIM10 (ATIH). Fallback : ICD10, ICD10CM.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    written = build_gold_set(
        athena_dir=args.athena_dir,
        out_path=args.out,
        n=args.n,
        domains=tuple(args.domains),
        source_vocabs=tuple(args.source_vocab),
        seed=args.seed,
    )
    print(
        f"Gold set écrit : {args.out}  ({written} mappings 1-à-1, "
        f"source={args.source_vocab}, domaines={args.domains})"
    )
    if written < args.n:
        print(
            f"  Note : {written} < {args.n} demandés. Pistes : "
            f"élargis --domains (ex. 'Condition Drug Measurement'), baisse --n, "
            f"ou change --source-vocab (ex. ICD10CM si CIM10 n'a pas de 'Maps to' direct)."
        )


if __name__ == "__main__":
    main()
