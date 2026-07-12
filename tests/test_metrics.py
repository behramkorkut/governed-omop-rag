"""Tests des métriques d'évaluation + du chargeur de gold set (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from governed_omop_rag.eval.gold_set import GoldItem, load_gold_set
from governed_omop_rag.eval.metrics import (
    aggregate,
    aggregate_mapping,
    hit_at_k,
    rank_of,
    reciprocal_rank,
)


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #
def test_rank_of() -> None:
    assert rank_of(42, [7, 42, 9]) == 2
    assert rank_of(42, [1, 2, 3]) is None


def test_hit_at_k() -> None:
    assert hit_at_k(42, [7, 42, 9], 2) is True
    assert hit_at_k(42, [7, 42, 9], 1) is False
    assert hit_at_k(42, [7, 42, 9], 0) is False


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(42, [42, 1, 2]) == pytest.approx(1.0)
    assert reciprocal_rank(42, [1, 42, 2]) == pytest.approx(0.5)
    assert reciprocal_rank(42, [1, 2, 3]) == 0.0


def test_aggregate_known_values() -> None:
    per_query = [
        (10, [10, 20, 30]),  # rang 1
        (20, [10, 20, 30]),  # rang 2
        (99, [10, 20, 30]),  # absent
    ]
    report = aggregate(per_query, ks=(1, 3))
    assert report.n == 3
    assert report.top1 == pytest.approx(1 / 3)
    assert report.recall_at_k[1] == pytest.approx(1 / 3)
    assert report.recall_at_k[3] == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_aggregate_empty() -> None:
    report = aggregate([], ks=(1, 5))
    assert report.n == 0
    assert report.top1 == 0.0
    assert report.recall_at_k == {1: 0.0, 5: 0.0}


def test_report_as_table() -> None:
    table = aggregate([(1, [1])], ks=(1,)).as_table()
    assert "Top-1" in table
    assert "recall@1" in table


# --------------------------------------------------------------------------- #
# Métriques niveau-mapping
# --------------------------------------------------------------------------- #
def test_aggregate_mapping() -> None:
    # (mappé, correct) : 3 mappés / 4, 2 corrects / 4.
    report = aggregate_mapping([(True, True), (True, False), (False, False), (True, True)])
    assert report.n == 4
    assert report.coverage == pytest.approx(0.75)
    assert report.unmapped_rate == pytest.approx(0.25)
    assert report.top1 == pytest.approx(0.5)
    assert report.precision_mapped == pytest.approx(2 / 3)


def test_aggregate_mapping_empty() -> None:
    report = aggregate_mapping([])
    assert report.n == 0
    assert report.coverage == 0.0
    assert "couverture" in report.as_table()


# --------------------------------------------------------------------------- #
# Gold set
# --------------------------------------------------------------------------- #
def test_gold_item_requires_code_or_label() -> None:
    with pytest.raises(ValidationError):
        GoldItem(expected_concept_id=1)


def test_gold_item_query_prefers_label() -> None:
    assert GoldItem(source_code="E11.9", source_label="diabète", expected_concept_id=1).query == (
        "diabète"
    )
    assert GoldItem(source_code="E11.9", expected_concept_id=1).query == "E11.9"


def test_load_gold_set(tmp_path: Path) -> None:
    p = tmp_path / "gold.csv"
    p.write_text(
        "source_code,source_label,expected_concept_id\n"
        ",diabète de type 2,201826\n"
        "E11.9,,201826\n"
        ",ligne sans concept,\n",  # ignorée (pas de concept attendu)
        encoding="utf-8",
    )
    gold = load_gold_set(p)
    assert len(gold) == 2
    assert gold[0].expected_concept_id == 201826
    assert gold[1].source_code == "E11.9"


def test_load_gold_set_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gold_set(tmp_path / "absent.csv")
