"""Retriever — abstraction de la recherche de candidats pour une requête.

Le ``HybridRouter`` (Phase 2) dépend d'un ``Retriever`` sans connaître son
implémentation : dense aujourd'hui (``DenseRetriever``), hybride BM25+dense
demain (P2-2). Frontière propre = on pourra enrichir le retrieval sans toucher
au router.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from governed_omop_rag.core.models import ConceptCandidate
from governed_omop_rag.retrieval.embeddings import Embedder
from governed_omop_rag.retrieval.index import search_concepts
from governed_omop_rag.retrieval.vectorstore import VectorStore


@runtime_checkable
class Retriever(Protocol):
    """Contrat : produire des candidats classés pour une requête textuelle."""

    def retrieve(self, query: str, top_k: int = 10) -> list[ConceptCandidate]:
        """Retourne les top_k concepts candidats (score décroissant)."""
        ...


class DenseRetriever:
    """Recherche dense : embed la requête puis interroge le VectorStore."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 10) -> list[ConceptCandidate]:
        if not query.strip():
            return []
        return search_concepts(query, self.embedder, self.store, top_k=top_k)
