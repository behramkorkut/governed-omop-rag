"""Embeddings — vectorisation locale des concepts et des requêtes.

Décision (CONTEXT.md §3/§5.3) : embeddings **biomédicaux locaux** (BioLORD via
sentence-transformers), calculés hors-ligne, rien n'est envoyé à un tiers.

Deux implémentations derrière un protocole commun :
- ``SentenceTransformerEmbedder`` : le vrai modèle (BioLORD), import paresseux ;
- ``HashingEmbedder`` : embedding déterministe sans dépendance (hashlib), pour
  tester le RAG et développer hors-ligne sans télécharger de modèle. Ce n'est PAS
  un mock : c'est un vrai vectoriseur reproductible (bag-of-words hashé + L2).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

# Réutilise la normalisation FR du médaillon (casse, espaces, ligatures).
from governed_omop_rag.medallion.normalize import normalize_ascii


@runtime_checkable
class Embedder(Protocol):
    """Contrat minimal d'un vectoriseur."""

    model_name: str

    @property
    def dimension(self) -> int:
        """Dimension des vecteurs produits."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Vectorise un lot de textes."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Vectorise un texte unique."""
        ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class HashingEmbedder:
    """Embedding déterministe et hors-ligne (feature hashing bag-of-words).

    Chaque token FR normalisé est projeté dans un bucket via BLAKE2b (hash
    **stable** entre processus, contrairement à ``hash()``), avec un signe.
    Résultat L2-normalisé -> le cosinus reflète le recouvrement lexical.

    ATTENTION — ce backend n'est PAS sémantique : il matche des mots, pas du
    sens. « diabète » ~ « diabète », mais « diabète » et « glycémie élevée » ne
    matchent pas (mêmes concepts, mots différents). Il sert aux tests et au dev
    hors-ligne. La sémantique réelle vient de ``SentenceTransformerEmbedder``
    (BioLORD), à utiliser en production.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension doit être > 0")
        self.dimension = dimension
        self.model_name = f"hashing-{dimension}"

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        bucket = h % self.dimension
        sign = 1.0 if (h >> 1) & 1 else -1.0
        return bucket, sign

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in normalize_ascii(text).split():
            bucket, sign = self._hash(token)
            vector[bucket] += sign
        return _l2_normalize(vector)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class SentenceTransformerEmbedder:
    """Vectoriseur biomédical réel (BioLORD / sentence-transformers).

    Le modèle est chargé paresseusement au premier appel afin de ne pas imposer
    la dépendance lourde (torch) à ceux qui utilisent le backend hashing.
    """

    def __init__(
        self,
        model_name: str = "FremyCompany/BioLORD-2023",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None
        self._dimension: int | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - dépend de l'install
                raise ImportError(
                    "sentence-transformers requis pour ce backend. "
                    "Installer l'extra : uv sync --extra retrieval"
                ) from exc
            model = SentenceTransformer(self.model_name, device=self.device)
            self._model = model
            self._dimension = int(model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dimension(self) -> int:
        self._ensure_model()
        assert self._dimension is not None
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        # normalize_embeddings=True -> vecteurs unitaires (cosinus direct).
        vectors = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return [[float(x) for x in row] for row in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
