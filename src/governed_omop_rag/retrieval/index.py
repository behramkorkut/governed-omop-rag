"""Indexation & recherche dense : Gold -> vecteurs -> VectorStore -> candidats.

Relie les embeddings (P1-5) et l'index vectoriel (P1-6). ``search_concepts``
produit des ``ConceptCandidate`` (top-k) — le contexte minimal à haut signal qui
sera injecté à l'agent Proposer aux phases suivantes (CONTEXT.md §4.4).
"""

from __future__ import annotations

from collections.abc import Sequence

from governed_omop_rag.core.models import ConceptCandidate
from governed_omop_rag.medallion.gold import GoldConcept
from governed_omop_rag.retrieval.embeddings import Embedder
from governed_omop_rag.retrieval.vectorstore import SearchHit, VectorItem, VectorStore


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def index_gold(
    concepts: Sequence[GoldConcept],
    embedder: Embedder,
    store: VectorStore,
) -> int:
    """Vectorise les documents Gold et les insère dans le VectorStore.

    Retourne le nombre de concepts indexés.
    """
    store.ensure_collection(embedder.dimension)
    if not concepts:
        return 0
    vectors = embedder.embed([c.doc_text for c in concepts])
    items = [
        VectorItem(
            concept_id=c.concept_id,
            vector=vector,
            payload={
                "concept_name": c.concept_name,
                "domain_id": c.domain_id,
                "vocabulary_id": c.vocabulary_id,
                "concept_code": c.concept_code,
                "synonyms": list(c.synonyms),
                "doc_text": c.doc_text,
            },
        )
        for c, vector in zip(concepts, vectors, strict=True)
    ]
    return store.upsert(items)


def _hit_to_candidate(hit: SearchHit) -> ConceptCandidate:
    payload = hit.payload
    synonyms = [str(s) for s in (payload.get("synonyms") or [])]
    return ConceptCandidate(
        concept_id=hit.concept_id,
        concept_name=str(payload.get("concept_name", "")),
        vocabulary_id=str(payload.get("vocabulary_id", "")),
        domain_id=str(payload.get("domain_id", "")),
        # Le corpus Silver ne contient que des concepts standard (garde-fou OMOP).
        standard_concept="S",
        score=_clamp01(hit.score),
        synonyms=synonyms,
    )


def search_concepts(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    top_k: int = 10,
) -> list[ConceptCandidate]:
    """Recherche dense : renvoie les top-k concepts candidats pour une requête."""
    vector = embedder.embed_one(query)
    hits = store.search(vector, top_k)
    return [_hit_to_candidate(h) for h in hits]
