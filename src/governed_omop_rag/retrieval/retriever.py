"""Retriever — abstraction de la recherche de candidats pour une requête.

Le ``HybridRouter`` (Phase 2) dépend d'un ``Retriever`` sans connaître son
implémentation : dense aujourd'hui (``DenseRetriever``), hybride BM25+dense
demain (P2-2). Frontière propre = on pourra enrichir le retrieval sans toucher
au router.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from governed_omop_rag.core.models import ConceptCandidate
from governed_omop_rag.medallion.gold import GoldConcept
from governed_omop_rag.medallion.normalize import normalize_ascii
from governed_omop_rag.retrieval.bm25 import BM25Index, tokenize
from governed_omop_rag.retrieval.embeddings import Embedder
from governed_omop_rag.retrieval.index import search_concepts
from governed_omop_rag.retrieval.vectorstore import VectorStore


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


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


class BM25Retriever:
    """Recherche lexicale BM25 sur les documents Gold."""

    def __init__(self, concepts: Sequence[GoldConcept], k1: float = 1.5, b: float = 0.75) -> None:
        self._meta: dict[int, GoldConcept] = {c.concept_id: c for c in concepts}
        docs = [(c.concept_id, tokenize(c.doc_text)) for c in concepts]
        self._index = BM25Index(docs, k1=k1, b=b)

    def retrieve(self, query: str, top_k: int = 10) -> list[ConceptCandidate]:
        if not query.strip():
            return []
        scored = self._index.top_k(tokenize(query), top_k)
        if not scored:
            return []
        # Normalise par le meilleur score de la requête -> score dans [0, 1].
        best = scored[0][1] or 1.0
        return [self._to_candidate(cid, s / best) for cid, s in scored]

    def _to_candidate(self, concept_id: int, score: float) -> ConceptCandidate:
        c = self._meta[concept_id]
        return ConceptCandidate(
            concept_id=c.concept_id,
            concept_name=c.concept_name,
            vocabulary_id=c.vocabulary_id,
            domain_id=c.domain_id,
            standard_concept="S",
            score=_clamp01(score),
            synonyms=list(c.synonyms),
        )


class LexicalBaselineRetriever:
    """Baseline lexicale par appariement de chaînes — proxy reproductible d'Usagi.

    Usagi (OHDSI) fait du string-matching semi-automatique. On reproduit ici un
    appariement lexical simple (match exact du libellé, sinon Jaccard sur les
    tokens du nom + synonymes) pour disposer d'un **point de comparaison honnête**
    à notre retrieval hybride. Ce n'est PAS Usagi lui-même (outil Java), mais une
    baseline lexicale du même esprit, exécutable partout.
    """

    def __init__(self, concepts: Sequence[GoldConcept]) -> None:
        self._entries: list[tuple[GoldConcept, set[str], set[str]]] = []
        for c in concepts:
            tokens = set(tokenize(c.concept_name))
            for syn in c.synonyms:
                tokens |= set(tokenize(syn))
            exacts = {normalize_ascii(c.concept_name)} | {normalize_ascii(s) for s in c.synonyms}
            self._entries.append((c, tokens, exacts))

    def retrieve(self, query: str, top_k: int = 10) -> list[ConceptCandidate]:
        q_tokens = set(tokenize(query))
        q_norm = normalize_ascii(query)
        if not q_tokens:
            return []
        scored: list[tuple[GoldConcept, float]] = []
        for concept, tokens, exacts in self._entries:
            if not tokens:
                continue
            if q_norm in exacts:
                score = 1.0
            else:
                inter = len(q_tokens & tokens)
                union = len(q_tokens | tokens)
                score = inter / union if union else 0.0
            if score > 0.0:
                scored.append((concept, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            ConceptCandidate(
                concept_id=c.concept_id,
                concept_name=c.concept_name,
                vocabulary_id=c.vocabulary_id,
                domain_id=c.domain_id,
                standard_concept="S",
                score=_clamp01(score),
                synonyms=list(c.synonyms),
            )
            for c, score in scored[:top_k]
        ]


class HybridRetriever:
    """Fusionne plusieurs Retriever par Reciprocal Rank Fusion (RRF).

    RRF est robuste car il fusionne des **rangs**, pas des scores d'échelles
    différentes (BM25 non borné vs cosinus). Score(d) = Σ 1/(k + rang_d).
    """

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever requiert au moins un retriever.")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 10) -> list[ConceptCandidate]:
        if not query.strip():
            return []
        # On récupère plus large que top_k pour bien fusionner les classements.
        fetch = max(top_k * 2, 20)
        fused: dict[int, float] = defaultdict(float)
        meta: dict[int, ConceptCandidate] = {}
        for retriever in self.retrievers:
            for rank, cand in enumerate(retriever.retrieve(query, fetch), start=1):
                fused[cand.concept_id] += 1.0 / (self.rrf_k + rank)
                meta.setdefault(cand.concept_id, cand)

        if not fused:
            return []
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        # Normalise : le max possible est d'être 1er dans TOUS les retrievers.
        max_possible = len(self.retrievers) / (self.rrf_k + 1)
        return [
            meta[cid].model_copy(update={"score": _clamp01(score / max_possible)})
            for cid, score in ranked
        ]


def build_retriever(
    kind: str,
    concepts: Sequence[GoldConcept],
    embedder: Embedder,
    store: VectorStore,
    rrf_k: int = 60,
) -> Retriever:
    """Fabrique un Retriever : 'dense', 'bm25', 'hybrid' ou 'baseline' (lexical)."""
    dense = DenseRetriever(embedder, store)
    if kind == "dense":
        return dense
    if kind == "bm25":
        return BM25Retriever(concepts)
    if kind == "hybrid":
        return HybridRetriever([BM25Retriever(concepts), dense], rrf_k=rrf_k)
    if kind == "baseline":
        return LexicalBaselineRetriever(concepts)
    raise ValueError(f"retriever inconnu : {kind!r} (dense|bm25|hybrid|baseline)")
