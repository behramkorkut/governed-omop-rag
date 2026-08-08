"""Tests de la taxonomie des échecs de mapping (hiérarchie OMOP)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from governed_omop_rag.eval.errors import classify, parse_scores_csv

# Hiérarchie de test : 100 -> 101 (1 niveau), 200 -> 201 (2 niveaux), 300 isolé.
ANCESTOR_ROWS = [
    ("100", "101", "1", "1"),
    ("200", "201", "2", "2"),
]
CONCEPT_ROWS = [
    ("100", "Concept parent"),
    ("101", "Concept enfant"),
    ("200", "Concept general"),
    ("201", "Concept precis"),
    ("300", "Concept isole"),
    ("301", "Autre concept isole"),
]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Mini export OHDSI tabulé (CONCEPT + CONCEPT_ANCESTOR)."""
    concept = tmp_path / "CONCEPT.csv"
    with concept.open("w", encoding="utf-8") as f:
        f.write("concept_id\tconcept_name\n")
        for cid, nom in CONCEPT_ROWS:
            f.write(f"{cid}\t{nom}\n")

    ancestor = tmp_path / "CONCEPT_ANCESTOR.csv"
    with ancestor.open("w", encoding="utf-8") as f:
        f.write("ancestor_concept_id\tdescendant_concept_id\tmin_levels\tmax_levels\n")
        for row in ANCESTOR_ROWS:
            f.write("\t".join(row) + "\n")
    return tmp_path


def _scores_csv(path: Path, lignes: list[tuple[str, str, str]]) -> Path:
    """Écrit un export de scores minimal (name, value, comment)."""
    p = path / "scores.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "value", "comment"])
        w.writerows(lignes)
    return p


def test_parse_ne_retient_que_les_echecs_top1(tmp_path: Path) -> None:
    p = _scores_csv(
        tmp_path,
        [
            ("top1", "1", "attendu 100, obtenu 100"),
            ("top1", "0", "attendu 100, obtenu 101"),
            ("mapped", "0", "ignoré : mauvais score"),
            ("source", "rag", ""),
        ],
    )
    paires, total = parse_scores_csv(p)
    assert paires == [(100, 101)]
    assert total == 2  # deux lignes top1, un seul échec


def test_parse_gere_un_concept_obtenu_absent(tmp_path: Path) -> None:
    p = _scores_csv(tmp_path, [("top1", "0", "attendu 100, obtenu None")])
    paires, _ = parse_scores_csv(p)
    assert paires == [(100, None)]


def test_parse_fichier_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Export de scores introuvable"):
        parse_scores_csv(tmp_path / "absent.csv")


def test_classification_descendant_ancetre_et_sans_lien(corpus: Path) -> None:
    report = classify(
        [(100, 101), (201, 200), (300, 301), (300, None)],
        bronze_dir=corpus,
        total_cases=10,
    )
    categories = [f.category for f in report.failures]
    assert categories == ["descendant", "ancetre", "sans_lien", "sans_lien"]

    assert report.failures[0].levels == 1
    assert report.failures[1].levels == 2
    assert report.related == 2
    assert report.counts["sans_lien"] == 2


def test_libelles_resolus_depuis_le_corpus(corpus: Path) -> None:
    report = classify([(100, 101)], bronze_dir=corpus)
    f = report.failures[0]
    assert f.expected_name == "Concept parent"
    assert f.obtained_name == "Concept enfant"


def test_table_lisible_et_proportions(corpus: Path) -> None:
    report = classify([(100, 101), (300, 301)], bronze_dir=corpus, total_cases=4)
    table = report.as_table()
    assert "cas analysés : 4" in table
    assert "échecs       : 2" in table
    assert "sur-spécification" in table


def test_aucun_echec(corpus: Path) -> None:
    report = classify([], bronze_dir=corpus, total_cases=12)
    assert "aucun échec" in report.as_table()


def test_table_ancestor_manquante(tmp_path: Path) -> None:
    (tmp_path / "CONCEPT.csv").write_text("concept_id\tconcept_name\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="CONCEPT_ANCESTOR"):
        classify([(1, 2)], bronze_dir=tmp_path)
