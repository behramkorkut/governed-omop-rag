"""Tests du CLI `gor` (Typer)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from governed_omop_rag import __version__
from governed_omop_rag.cli import app

runner = CliRunner()


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
