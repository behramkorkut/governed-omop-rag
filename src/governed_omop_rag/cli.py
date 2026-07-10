"""Interface en ligne de commande (Typer).

Point d'entrée ``gor`` (défini dans pyproject). Sert aussi au smoke-test du
harness : ``gor smoke`` doit sortir en code 0 sur un environnement sain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from governed_omop_rag import __version__
from governed_omop_rag.config import get_settings
from governed_omop_rag.core.logging import get_logger

app = typer.Typer(
    add_completion=False,
    help="governed-omop-rag — mapping CIM-10 FR / libellés -> concepts standard OMOP.",
)


@app.command()
def version() -> None:
    """Affiche la version du paquet."""
    typer.echo(__version__)


@app.command()
def info() -> None:
    """Affiche un résumé de la configuration active (sans secrets)."""
    s = get_settings()
    typer.echo(f"env               : {s.env.value}")
    typer.echo(f"vector_backend    : {s.vector_backend.value}")
    typer.echo(f"qdrant_url        : {s.qdrant_url}")
    typer.echo(f"qdrant_collection : {s.qdrant_collection}")
    typer.echo(f"embedding_model   : {s.embedding_model}")
    typer.echo(f"top_k             : {s.top_k}")
    typer.echo(f"confidence_thresh : {s.confidence_threshold}")
    typer.echo(f"anthropic_key_set : {s.anthropic_api_key is not None}")


@app.command()
def smoke() -> None:
    """Smoke-test : la config se charge et le logging fonctionne. Exit 0 si OK."""
    log = get_logger("smoke")
    settings = get_settings()
    log.info("smoke_ok", version=__version__, env=settings.env.value)
    typer.echo("SMOKE OK")


@app.command("build-corpus")
def build_corpus_cmd(
    bronze_dir: Annotated[
        Path | None,
        typer.Option(help="Répertoire des fichiers OHDSI bruts (défaut: config)."),
    ] = None,
    duckdb_path: Annotated[
        Path | None,
        typer.Option(help="Chemin du fichier DuckDB de sortie (défaut: config)."),
    ] = None,
    domain: Annotated[
        list[str] | None,
        typer.Option(help="Filtre domain_id (répétable), ex. --domain Condition."),
    ] = None,
) -> None:
    """Construit le corpus médaillon Bronze -> Silver -> Gold (couche data)."""
    from governed_omop_rag.medallion.pipeline import run_pipeline

    settings = get_settings()
    src = bronze_dir or settings.bronze_dir
    out = duckdb_path or settings.duckdb_path
    stats = run_pipeline(src, out, domains=domain or None)

    log = get_logger("build-corpus")
    log.info(
        "corpus_built",
        bronze_concepts=stats.bronze_concepts,
        bronze_synonyms=stats.bronze_synonyms,
        silver_concepts=stats.silver_concepts,
        gold_concepts=stats.gold_concepts,
        duckdb=str(out),
    )
    typer.echo(
        f"Corpus construit -> {out}\n"
        f"  Bronze : {stats.bronze_concepts} concepts, {stats.bronze_synonyms} synonymes\n"
        f"  Silver : {stats.silver_concepts} concepts (standard + valides)\n"
        f"  Gold   : {stats.gold_concepts} documents embedding-ready"
    )


if __name__ == "__main__":
    app()
