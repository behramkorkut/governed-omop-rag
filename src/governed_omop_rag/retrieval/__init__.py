"""Retrieval déterministe : recherche hybride (BM25 + dense) + reranking.

Interface VectorStore abstraite ; implémentation par défaut Qdrant (souverain).
Embeddings BioLORD calculés en local. Sortie : top-k candidats à haut signal.

NB : on n'exporte ici que les briques « légères » (embeddings, vectorstore),
sans couplage à la config. ``factory`` (get_embedder/get_vectorstore) et
``index`` (index_gold/search_concepts) s'importent explicitement depuis leur
module — ils dépendent de config/core.
"""

from governed_omop_rag.retrieval.bm25 import BM25Index, tokenize
from governed_omop_rag.retrieval.embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)
from governed_omop_rag.retrieval.vectorstore import (
    MemoryVectorStore,
    QdrantVectorStore,
    SearchHit,
    VectorItem,
    VectorStore,
    cosine_similarity,
)

__all__ = [
    "BM25Index",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "MemoryVectorStore",
    "QdrantVectorStore",
    "SearchHit",
    "VectorItem",
    "VectorStore",
    "cosine_similarity",
    "tokenize",
]
