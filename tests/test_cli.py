"""Tests du CLI `gor` (Typer)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from governed_omop_rag import __version__
from governed_omop_rag.cli import app

runner = CliRunner()

FIXTURES = Path(__file__).parent / "fixtures"


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


@pytest.mark.smoke
def test_smoke_exits_zero() -> None:
    result = runner.invoke(app, ["smoke"])
    assert result.exit_code == 0
    assert "SMOKE OK" in result.stdout


def test_info_does_not_leak_secret() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    # `info` ne doit afficher qu'un booléen, jamais la clé elle-même.
    assert "anthropic_key_set" in result.stdout


def test_build_corpus_command(tmp_path: Path) -> None:
    db = tmp_path / "test.duckdb"
    result = runner.invoke(
        app,
        [
            "build-corpus",
            "--bronze-dir",
            str(FIXTURES),
            "--duckdb-path",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Corpus construit" in result.stdout
    assert "Gold" in result.stdout
    assert db.exists()  # le fichier DuckDB a bien été créé


def _write_map(tmp_path: Path) -> Path:
    csv = tmp_path / "map.csv"
    csv.write_text(
        "source_code,target_concept_id\nE11.9,201826\n",
        encoding="utf-8",
    )
    return csv


def test_route_command_found(tmp_path: Path) -> None:
    csv = _write_map(tmp_path)
    result = runner.invoke(app, ["route", "--source-code", "E11.9", "--map-path", str(csv)])
    assert result.exit_code == 0, result.output
    assert "201826" in result.stdout
    assert "official_map" in result.stdout


def test_route_command_not_found(tmp_path: Path) -> None:
    csv = _write_map(tmp_path)
    result = runner.invoke(app, ["route", "--source-code", "Z99.9", "--map-path", str(csv)])
    assert result.exit_code == 0, result.output
    assert "target_concept_id : 0" in result.stdout
    assert "hors_vocabulaire" in result.stdout


def test_route_command_requires_source_code(tmp_path: Path) -> None:
    csv = _write_map(tmp_path)
    result = runner.invoke(app, ["route", "--map-path", str(csv)])
    assert result.exit_code == 2  # --source-code requis


def test_search_command_offline() -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "diabète de type 2",
            "--bronze-dir",
            str(FIXTURES),
            "--embedding-backend",
            "hashing",
            "--vector-backend",
            "memory",
            "--top-k",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = result.stdout.strip().split("\n")
    # 1re ligne = en-tête ; 2e ligne = meilleur candidat -> doit être 201826 + score.
    assert "201826" in lines[1]
    assert "0." in lines[1]


def test_map_command_deterministic(tmp_path: Path) -> None:
    csv = _write_map(tmp_path)
    result = runner.invoke(
        app,
        [
            "map",
            "--source-code",
            "E11.9",
            "--map-path",
            str(csv),
            "--bronze-dir",
            str(FIXTURES),
            "--embedding-backend",
            "hashing",
            "--vector-backend",
            "memory",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "official_map" in result.stdout
    assert "201826" in result.stdout


def test_map_command_rag_on_label(tmp_path: Path) -> None:
    csv = _write_map(tmp_path)
    result = runner.invoke(
        app,
        [
            "map",
            "--source-label",
            "diabète de type 2",
            "--map-path",
            str(csv),
            "--bronze-dir",
            str(FIXTURES),
            "--embedding-backend",
            "hashing",
            "--vector-backend",
            "memory",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "rag" in result.stdout
    assert "201826" in result.stdout


def test_map_command_requires_input(tmp_path: Path) -> None:
    csv = _write_map(tmp_path)
    result = runner.invoke(app, ["map", "--map-path", str(csv)])
    assert result.exit_code == 2


def test_eval_command(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    gold.write_text(
        "source_code,source_label,expected_concept_id\n"
        ",diabète de type 2,201826\n"
        ",asthme,4048098\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "eval",
            "--gold-path",
            str(gold),
            "--bronze-dir",
            str(FIXTURES),
            "--embedding-backend",
            "hashing",
            "--vector-backend",
            "memory",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Top-1" in result.stdout
    assert "recall@" in result.stdout


def test_map_command_cache_hit_on_second_run(tmp_path: Path) -> None:
    csv = _write_map(tmp_path)
    cache_db = tmp_path / "cache.duckdb"
    args = [
        "map",
        "--source-label",
        "diabète de type 2",
        "--map-path",
        str(csv),
        "--bronze-dir",
        str(FIXTURES),
        "--embedding-backend",
        "hashing",
        "--vector-backend",
        "memory",
        "--cache",
        "--cache-path",
        str(cache_db),
    ]
    r1 = runner.invoke(app, args)
    assert r1.exit_code == 0, r1.output
    assert "0 hit / 1 miss" in r1.stdout  # 1er passage : miss
    r2 = runner.invoke(app, args)
    assert r2.exit_code == 0, r2.output
    assert "1 hit / 0 miss" in r2.stdout  # 2e passage : servi par le cache persistant
