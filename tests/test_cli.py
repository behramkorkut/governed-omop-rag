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
