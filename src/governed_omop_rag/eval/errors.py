"""Taxonomie des échecs de mapping : *pourquoi* le pipeline se trompe.

Un Top-1 de 0,65 dit qu'un tiers des cas est faux. Il ne dit pas **comment**. Or les
erreurs n'ont pas la même valeur pour un steward : proposer l'enfant direct du concept
attendu (« Liquid paint » pour « Paint ») n'a rien à voir avec proposer un concept sans
rapport. Ce module classe chaque échec en s'appuyant sur la hiérarchie **OMOP
CONCEPT_ANCESTOR** — la référence du domaine, pas une heuristique maison.

Catégories
----------
* ``descendant`` : le concept proposé est un enfant de l'attendu (sur-spécification) ;
* ``ancetre``    : il en est un parent (sous-spécification) ;
* ``sans_lien``  : aucune relation ancêtre/descendant directe.

Aucun appel LLM : l'analyse relit un export de scores et les tables OHDSI.

⚠️ Cette taxonomie **complète** le Top-1 exact, elle ne le remplace pas. Annoncer un
« Top-1 à un niveau près » comme métrique principale reviendrait à déplacer les poteaux
après le match.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

COMMENT_PATTERN = re.compile(r"attendu\s+(\d+),\s*obtenu\s+(\S+)")

CONCEPT_FILE = "CONCEPT.csv"
ANCESTOR_FILE = "CONCEPT_ANCESTOR.csv"


@dataclass(frozen=True)
class Failure:
    """Un échec de mapping, enrichi de sa relation hiérarchique."""

    expected_id: int
    obtained_id: int | None
    expected_name: str = ""
    obtained_name: str = ""
    category: str = "sans_lien"
    levels: int | None = None


@dataclass(frozen=True)
class ErrorReport:
    """Distribution des catégories d'échec."""

    total_cases: int
    failures: list[Failure]

    @property
    def counts(self) -> Counter[str]:
        return Counter(f.category for f in self.failures)

    @property
    def related(self) -> int:
        """Échecs reliés hiérarchiquement (ancêtre ou descendant)."""
        return self.counts["descendant"] + self.counts["ancetre"]

    def as_table(self) -> str:
        n = len(self.failures)
        if n == 0:
            return f"cas analysés : {self.total_cases}\naucun échec."
        lignes = [
            f"cas analysés : {self.total_cases}",
            f"échecs       : {n} ({n / self.total_cases:.1%})",
            "",
            "taxonomie des échecs :",
        ]
        libelles = {
            "descendant": "sur-spécification (enfant de l'attendu)",
            "ancetre": "sous-spécification (parent de l'attendu)",
            "sans_lien": "aucun lien hiérarchique direct",
        }
        for cat, compte in self.counts.most_common():
            lignes.append(f"  {libelles.get(cat, cat):<44} {compte:>3}  ({compte / n:.1%})")
        lignes.append("")
        lignes.append(
            f"  reliés hiérarchiquement {'':<21} {self.related:>3}  ({self.related / n:.1%})"
        )
        sauts = Counter(f.levels for f in self.failures if f.levels)
        if sauts:
            detail = ", ".join(f"{k} niveau(x) : {v}" for k, v in sorted(sauts.items()))
            lignes.append(f"  distance hiérarchique : {detail}")
        return "\n".join(lignes)


def parse_scores_csv(path: str | Path) -> tuple[list[tuple[int, int | None]], int]:
    """Extrait les couples (attendu, obtenu) des échecs d'un export de scores Langfuse.

    Renvoie aussi le nombre total de cas (lignes de score ``top1``), pour rapporter les
    échecs à l'ensemble.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Export de scores introuvable : {p}")

    paires: list[tuple[int, int | None]] = []
    total = 0
    with p.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("name") != "top1":
                continue
            total += 1
            if str(row.get("value", "")).strip() not in {"0", "0.0"}:
                continue
            m = COMMENT_PATTERN.search(row.get("comment") or "")
            if not m:
                continue
            brut = m.group(2).rstrip(".,")
            paires.append((int(m.group(1)), None if brut == "None" else int(brut)))
    return paires, total


def _load_names(bronze_dir: Path, wanted: set[str]) -> dict[int, str]:
    """Libellés des concepts recherchés (lecture en flux, corpus volumineux)."""
    path = bronze_dir / CONCEPT_FILE
    if not path.exists():
        return {}
    noms: dict[int, str] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            # rstrip avant split : sans cela, un libellé en dernière colonne
            # embarquerait le saut de ligne (cohérent avec _load_links).
            champs = line.rstrip("\n").split("\t")
            if len(champs) < 2:
                continue
            if champs[0] in wanted:
                noms[int(champs[0])] = champs[1]
                if len(noms) == len(wanted):
                    break
    return noms


def _load_links(ancestor_path: Path, wanted: set[str]) -> dict[tuple[int, int], int]:
    """Relations ancêtre -> descendant restreintes aux concepts recherchés."""
    if not ancestor_path.exists():
        raise FileNotFoundError(
            f"Table CONCEPT_ANCESTOR introuvable : {ancestor_path}. "
            "Elle fait partie de l'export Athena."
        )
    liens: dict[tuple[int, int], int] = {}
    with ancestor_path.open(encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            champs = line.rstrip("\n").split("\t")
            if len(champs) < 3:
                continue
            a, d = champs[0], champs[1]
            if a in wanted and d in wanted and a != d:
                liens[(int(a), int(d))] = int(champs[2])
    return liens


def classify(
    paires: list[tuple[int, int | None]],
    *,
    bronze_dir: Path,
    ancestor_path: Path | None = None,
    total_cases: int | None = None,
) -> ErrorReport:
    """Classe chaque échec selon sa relation hiérarchique OMOP."""
    ancestor_path = ancestor_path or (bronze_dir / ANCESTOR_FILE)
    wanted = {str(x) for a, b in paires for x in (a, b) if x is not None}

    noms = _load_names(bronze_dir, wanted) if wanted else {}
    liens = _load_links(ancestor_path, wanted) if wanted else {}

    failures: list[Failure] = []
    for attendu, obtenu in paires:
        categorie, niveaux = "sans_lien", None
        if obtenu is not None:
            if (attendu, obtenu) in liens:
                categorie, niveaux = "descendant", liens[(attendu, obtenu)]
            elif (obtenu, attendu) in liens:
                categorie, niveaux = "ancetre", liens[(obtenu, attendu)]
        failures.append(
            Failure(
                expected_id=attendu,
                obtained_id=obtenu,
                expected_name=noms.get(attendu, ""),
                obtained_name=noms.get(obtenu, "") if obtenu is not None else "",
                category=categorie,
                levels=niveaux,
            )
        )
    return ErrorReport(total_cases=total_cases or len(paires), failures=failures)
