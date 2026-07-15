"""Construit l'alignement officiel CIM-10 FR (ATIH) -> SNOMED pour le router déterministe.

Entrée : le bundle Athena décompressé (CONCEPT.csv + CONCEPT_RELATIONSHIP.csv).
Sortie : ``data/router/cim10_snomed_official.csv`` au format attendu par
``OfficialMap.from_csv`` : ``source_code,target_concept_id,target_concept_name``.

Méthode (identique au gold set, pour cohérence) : relation officielle « Maps to »
de CIM10 (édition française ATIH) vers un concept SNOMED **standard** et **valide**,
en ne gardant que les mappings **1-à-1** (un code -> exactement un concept standard).

IMPORTANT — anti-fuite (held-out) : on **exclut** par défaut les codes présents dans
le gold set (``--exclude-gold``). Sans ça, le router déterministe « réussirait » le
gold set en recopiant la table de lookup (évaluation circulaire). Les codes du gold
set restent donc du **résidu** routé vers le RAG : leur évaluation reste honnête.

Usage :
    uv run python scripts/build_official_map.py --athena-dir bundle/athena_bundle \\
        --exclude-gold data/eval/gold_set_atih.csv \\
        --out data/router/cim10_snomed_official.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import duckdb


def _read_gold_codes(path: Path | None) -> set[str]:
    """Codes source du gold set à exclure de la map (held-out)."""
    if path is None or not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {(r.get("source_code") or "").strip() for r in csv.DictReader(fh)} - {""}


def build_official_map(
    athena_dir: Path,
    out_path: Path,
    exclude_codes: set[str],
    source_vocab: str = "CIM10",
    target_vocab: str = "SNOMED",
) -> tuple[int, int]:
    """Écrit la map officielle. Retourne (paires écrites, paires exclues held-out)."""
    concept = athena_dir / "CONCEPT.csv"
    rel = athena_dir / "CONCEPT_RELATIONSHIP.csv"
    for f in (concept, rel):
        if not f.exists():
            raise FileNotFoundError(f"Fichier Athena manquant : {f}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute(
        f"CREATE VIEW concept AS SELECT * FROM "
        f"read_csv('{str(concept).replace(chr(39), chr(39) * 2)}', "
        f"delim='\t', header=true, all_varchar=true)"
    )
    con.execute(
        f"CREATE VIEW rel AS SELECT * FROM "
        f"read_csv('{str(rel).replace(chr(39), chr(39) * 2)}', "
        f"delim='\t', header=true, all_varchar=true)"
    )

    rows = con.execute(
        f"""
        WITH mapping AS (
            SELECT
                s.concept_code                 AS source_code,
                CAST(t.concept_id AS BIGINT)   AS target_concept_id,
                t.concept_name                 AS target_concept_name
            FROM rel r
            JOIN concept s ON CAST(r.concept_id_1 AS BIGINT) = CAST(s.concept_id AS BIGINT)
            JOIN concept t ON CAST(r.concept_id_2 AS BIGINT) = CAST(t.concept_id AS BIGINT)
            WHERE r.relationship_id = 'Maps to'
              AND s.vocabulary_id = '{source_vocab}'
              AND t.vocabulary_id = '{target_vocab}'
              AND t.standard_concept = 'S'
              AND (t.invalid_reason IS NULL OR t.invalid_reason = '')
              AND (s.invalid_reason IS NULL OR s.invalid_reason = '')
        ),
        one_to_one AS (
            SELECT source_code FROM mapping
            GROUP BY source_code HAVING COUNT(DISTINCT target_concept_id) = 1
        )
        SELECT DISTINCT m.source_code, m.target_concept_id, m.target_concept_name
        FROM mapping m JOIN one_to_one o USING (source_code)
        ORDER BY m.source_code
        """
    ).fetchall()
    con.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    excluded = 0
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source_code", "target_concept_id", "target_concept_name"])
        for code, cid, name in rows:
            if code in exclude_codes:
                excluded += 1
                continue
            w.writerow([code, cid, (name or "").replace("\n", " ")])
            written += 1
    return written, excluded


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--athena-dir", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("data/router/cim10_snomed_official.csv"))
    p.add_argument(
        "--exclude-gold",
        type=Path,
        default=Path("data/eval/gold_set_atih.csv"),
        help="Gold set dont les codes sont exclus de la map (anti-fuite, held-out).",
    )
    p.add_argument("--source-vocab", default="CIM10")
    p.add_argument("--target-vocab", default="SNOMED")
    args = p.parse_args()

    exclude = _read_gold_codes(args.exclude_gold)
    written, excluded = build_official_map(
        athena_dir=args.athena_dir,
        out_path=args.out,
        exclude_codes=exclude,
        source_vocab=args.source_vocab,
        target_vocab=args.target_vocab,
    )
    print(
        f"Map officielle écrite : {args.out}\n"
        f"  paires 1-à-1 {args.source_vocab}->{args.target_vocab} : {written}\n"
        f"  exclues (held-out, gold set)                : {excluded}"
    )


if __name__ == "__main__":
    main()
