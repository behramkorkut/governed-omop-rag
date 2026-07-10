"""Fabriques : sélectionnent l'implémentation d'embeddings / de VectorStore
selon la configuration (isolé ici car dépend de config, gardant embeddings.py et
vectorstore.py sans couplage au reste).
"""

from __future__ import annotations

from governed_omop_rag.config import (
    EmbeddingBackend,
    Settings,
    VectorBackend,
    get_settings,
)
from governed_omop_rag.retrieval.embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)
from governed_omop_rag.retrieval.vectorstore import (
    MemoryVectorStore,
    QdrantVectorStore,
    VectorStore,
)


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Retourne l'Embedder configuré (hashing hors-ligne ou BioLORD réel)."""
    s = settings or get_settings()
    if s.embedding_backend is EmbeddingBackend.HASHING:
        return HashingEmbedder(s.embedding_dim)
    return SentenceTransformerEmbedder(s.embedding_model, s.embedding_device)


def get_vectorstore(settings: Settings | None = None) -> VectorStore:
    """Retourne le VectorStore configuré (Qdrant souverain ou mémoire)."""
    s = settings or get_settings()
    if s.vector_backend is VectorBackend.MEMORY:
        return MemoryVectorStore()
    api_key = s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None
    return QdrantVectorStore(url=s.qdrant_url, collection=s.qdrant_collection, api_key=api_key)
