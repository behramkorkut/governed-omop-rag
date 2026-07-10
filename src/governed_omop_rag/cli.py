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
    encoding: Annotated[
        str,
        typer.Option(help="Encodage des CSV OHDSI (latin-1/cp1252 pour un export FR)."),
    ] = "utf-8",
) -> None:
    """Construit le corpus médaillon Bronze -> Silver -> Gold (couche data)."""
    from governed_omop_rag.medallion.pipeline import run_pipeline

    settings = get_settings()
    src = bronze_dir or settings.bronze_dir
    out = duckdb_path or settings.duckdb_path
    stats = run_pipeline(src, out, domains=domain or None, encoding=encoding)

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


@app.command()
def route(
    source_code: Annotated[
        str | None,
        typer.Option(help="Code source à mapper, ex. E11.9 (CIM-10 FR)."),
    ] = None,
    source_vocabulary: Annotated[
        str | None,
        typer.Option(help="Vocabulaire source, ex. ICD10FR."),
    ] = None,
    map_path: Annotated[
        Path | None,
        typer.Option(help="Chemin de l'alignement officiel CSV (défaut: config)."),
    ] = None,
) -> None:
    """Route un code source via l'alignement officiel (match déterministe, v1)."""
    from governed_omop_rag.core.models import MappingRequest
    from governed_omop_rag.router.deterministic import OfficialMap, route_deterministic

    if not source_code:
        typer.echo("Erreur : --source-code est requis.", err=True)
        raise typer.Exit(code=2)

    settings = get_settings()
    official_map = OfficialMap.from_csv(map_path or settings.router_map_path)
    request = MappingRequest(source_code=source_code, source_vocabulary=source_vocabulary)
    suggestion = route_deterministic(request, official_map)

    typer.echo(f"source_code       : {source_code}")
    typer.echo(f"target_concept_id : {suggestion.target_concept_id}")
    typer.echo(f"source            : {suggestion.source.value}")
    typer.echo(f"confidence        : {suggestion.confidence}")
    if suggestion.no_map_reason is not None:
        typer.echo(f"no_map_reason     : {suggestion.no_map_reason.value}")
    typer.echo(f"justification     : {suggestion.justification}")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Libellé ou code à rechercher.")],
    bronze_dir: Annotated[
        Path | None,
        typer.Option(help="Répertoire des fichiers OHDSI bruts (défaut: config)."),
    ] = None,
    top_k: Annotated[int, typer.Option(help="Nombre de candidats.")] = 5,
    embedding_backend: Annotated[
        str | None,
        typer.Option(help="hashing (offline) ou sentence_transformers (défaut: config)."),
    ] = None,
    vector_backend: Annotated[
        str | None,
        typer.Option(help="memory (offline) ou qdrant (défaut: config)."),
    ] = None,
) -> None:
    """Recherche dense end-to-end : construit le corpus, l'indexe, cherche les top-k.

    Astuce démo hors-ligne : --embedding-backend hashing --vector-backend memory.

    NB : cette commande reconstruit le corpus en mémoire à CHAQUE appel (pratique
    pour la démo « essayez en 2 minutes »). En production, on réutilisera le
    DuckDB persisté et un index Qdrant déjà peuplé plutôt que de tout reconstruire.
    """
    from governed_omop_rag.medallion.db import connect
    from governed_omop_rag.medallion.gold import fetch_gold
    from governed_omop_rag.medallion.pipeline import build_corpus
    from governed_omop_rag.retrieval.embeddings import (
        HashingEmbedder,
        SentenceTransformerEmbedder,
    )
    from governed_omop_rag.retrieval.factory import get_vectorstore
    from governed_omop_rag.retrieval.index import index_gold, search_concepts
    from governed_omop_rag.retrieval.vectorstore import MemoryVectorStore

    settings = get_settings()
    src = bronze_dir or settings.bronze_dir
    emb = embedding_backend or settings.embedding_backend.value
    vec = vector_backend or settings.vector_backend.value

    embedder = (
        HashingEmbedder(settings.embedding_dim)
        if emb == "hashing"
        else SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_device)
    )
    store = MemoryVectorStore() if vec == "memory" else get_vectorstore(settings)

    con = connect(":memory:")
    try:
        build_corpus(con, src)
        gold = fetch_gold(con)
    finally:
        con.close()

    n = index_gold(gold, embedder, store)
    candidates = search_concepts(query, embedder, store, top_k=top_k)

    typer.echo(f"Requête : {query!r}  (indexés: {n}, backend: {emb}/{vec})")
    if not candidates:
        typer.echo("Aucun candidat.")
        return
    for rank, c in enumerate(candidates, start=1):
        typer.echo(
            f"  {rank}. [{c.score:.3f}] concept_id={c.concept_id} "
            f"{c.concept_name} ({c.vocabulary_id}/{c.domain_id})"
        )


@app.command("map")
def map_cmd(
    source_code: Annotated[
        str | None, typer.Option(help="Code source, ex. E11.9 (CIM-10 FR).")
    ] = None,
    source_label: Annotated[
        str | None, typer.Option(help="Libellé clinique, ex. 'diabète de type 2'.")
    ] = None,
    source_vocabulary: Annotated[
        str | None, typer.Option(help="Vocabulaire source, ex. ICD10FR.")
    ] = None,
    bronze_dir: Annotated[
        Path | None, typer.Option(help="Répertoire OHDSI (défaut: config).")
    ] = None,
    map_path: Annotated[
        Path | None, typer.Option(help="Alignement officiel CSV (défaut: config).")
    ] = None,
    top_k: Annotated[int, typer.Option(help="Nombre de candidats RAG.")] = 5,
    embedding_backend: Annotated[
        str | None, typer.Option(help="hashing (offline) ou sentence_transformers.")
    ] = None,
    vector_backend: Annotated[str | None, typer.Option(help="memory (offline) ou qdrant.")] = None,
) -> None:
    """Mapping hybride : match officiel déterministe, sinon RAG (retrieval) sur le résidu.

    Démo offline : --embedding-backend hashing --vector-backend memory.
    """
    from governed_omop_rag.core.models import MappingRequest
    from governed_omop_rag.medallion.db import connect
    from governed_omop_rag.medallion.gold import fetch_gold
    from governed_omop_rag.medallion.pipeline import build_corpus
    from governed_omop_rag.retrieval.embeddings import (
        HashingEmbedder,
        SentenceTransformerEmbedder,
    )
    from governed_omop_rag.retrieval.factory import get_vectorstore
    from governed_omop_rag.retrieval.index import index_gold
    from governed_omop_rag.retrieval.retriever import DenseRetriever
    from governed_omop_rag.retrieval.vectorstore import MemoryVectorStore
    from governed_omop_rag.router.deterministic import OfficialMap
    from governed_omop_rag.router.hybrid import HybridRouter

    if not (source_code or source_label):
        typer.echo("Erreur : fournir --source-code et/ou --source-label.", err=True)
        raise typer.Exit(code=2)

    settings = get_settings()
    emb = embedding_backend or settings.embedding_backend.value
    vec = vector_backend or settings.vector_backend.value
    embedder = (
        HashingEmbedder(settings.embedding_dim)
        if emb == "hashing"
        else SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_device)
    )
    store = MemoryVectorStore() if vec == "memory" else get_vectorstore(settings)

    con = connect(":memory:")
    try:
        build_corpus(con, bronze_dir or settings.bronze_dir)
        gold = fetch_gold(con)
    finally:
        con.close()
    index_gold(gold, embedder, store)

    official_map = OfficialMap.from_csv(map_path or settings.router_map_path)
    router = HybridRouter(
        official_map,
        DenseRetriever(embedder, store),
        confidence_threshold=settings.confidence_threshold,
        top_k=top_k,
    )
    request = MappingRequest(
        source_code=source_code,
        source_label=source_label,
        source_vocabulary=source_vocabulary,
    )
    suggestion = router.route(request)

    typer.echo(f"source            : {suggestion.source.value}")
    typer.echo(f"target_concept_id : {suggestion.target_concept_id}")
    typer.echo(f"confidence        : {suggestion.confidence:.3f}")
    if suggestion.no_map_reason is not None:
        typer.echo(f"no_map_reason     : {suggestion.no_map_reason.value}")
    typer.echo(f"justification     : {suggestion.justification}")
    if suggestion.candidates:
        typer.echo("candidats :")
        for rank, c in enumerate(suggestion.candidates, start=1):
            typer.echo(
                f"  {rank}. [{c.score:.3f}] concept_id={c.concept_id} "
                f"{c.concept_name} ({c.vocabulary_id}/{c.domain_id})"
            )


@app.command()
def eval(
    gold_path: Annotated[Path | None, typer.Option(help="Gold set CSV (défaut: config).")] = None,
    bronze_dir: Annotated[
        Path | None, typer.Option(help="Répertoire OHDSI (défaut: config).")
    ] = None,
    embedding_backend: Annotated[
        str | None, typer.Option(help="hashing (offline) ou sentence_transformers.")
    ] = None,
    vector_backend: Annotated[str | None, typer.Option(help="memory (offline) ou qdrant.")] = None,
) -> None:
    """Évalue le retrieval sur le gold set (Top-1, recall@k, MRR)."""
    from governed_omop_rag.eval.gold_set import load_gold_set
    from governed_omop_rag.eval.runner import evaluate
    from governed_omop_rag.medallion.db import connect
    from governed_omop_rag.medallion.gold import fetch_gold
    from governed_omop_rag.medallion.pipeline import build_corpus
    from governed_omop_rag.retrieval.embeddings import (
        HashingEmbedder,
        SentenceTransformerEmbedder,
    )
    from governed_omop_rag.retrieval.factory import get_vectorstore
    from governed_omop_rag.retrieval.index import index_gold
    from governed_omop_rag.retrieval.retriever import DenseRetriever
    from governed_omop_rag.retrieval.vectorstore import MemoryVectorStore

    settings = get_settings()
    emb = embedding_backend or settings.embedding_backend.value
    vec = vector_backend or settings.vector_backend.value
    embedder = (
        HashingEmbedder(settings.embedding_dim)
        if emb == "hashing"
        else SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_device)
    )
    store = MemoryVectorStore() if vec == "memory" else get_vectorstore(settings)

    con = connect(":memory:")
    try:
        build_corpus(con, bronze_dir or settings.bronze_dir)
        gold_concepts = fetch_gold(con)
    finally:
        con.close()
    index_gold(gold_concepts, embedder, store)

    gold = load_gold_set(gold_path or settings.gold_set_path)
    report = evaluate(gold, DenseRetriever(embedder, store))
    typer.echo(f"Backend : {emb}/{vec}")
    typer.echo(report.as_table())


if __name__ == "__main__":
    app()
