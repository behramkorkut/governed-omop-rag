"""VectorStore — index vectoriel derrière une interface (souveraineté / switch).

Décision (CONTEXT.md §3/§5.3) : Qdrant par défaut (souverain, européen), isolé
derrière ``VectorStore`` pour pouvoir changer de backend sans toucher au reste.

- ``MemoryVectorStore`` : implémentation en mémoire (cosinus), hors-ligne, pour
  les tests et le développement sans Docker ;
- ``QdrantVectorStore`` : implémentation Qdrant (import paresseux).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VectorItem:
    """Point à indexer : identifiant concept + vecteur + métadonnées (payload)."""

    concept_id: int
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    """Résultat de recherche : concept + score de similarité + payload."""

    concept_id: int
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosinus entre deux vecteurs. 0.0 si l'un est nul."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class VectorStore(ABC):
    """Contrat d'un index vectoriel."""

    @abstractmethod
    def ensure_collection(self, dimension: int) -> None:
        """Crée la collection si besoin (idempotent)."""

    @abstractmethod
    def upsert(self, items: Sequence[VectorItem]) -> int:
        """Insère/maj des points. Retourne le nombre de points traités."""

    @abstractmethod
    def search(self, vector: Sequence[float], top_k: int = 10) -> list[SearchHit]:
        """Retourne les top_k voisins par similarité décroissante."""

    @abstractmethod
    def count(self) -> int:
        """Nombre de points indexés."""


class MemoryVectorStore(VectorStore):
    """Index en mémoire (cosinus exact). Suffisant pour tests et petits corpus."""

    def __init__(self) -> None:
        self._items: dict[int, VectorItem] = {}
        self._dimension: int | None = None

    def ensure_collection(self, dimension: int) -> None:
        self._dimension = dimension

    def upsert(self, items: Sequence[VectorItem]) -> int:
        for item in items:
            if self._dimension is not None and len(item.vector) != self._dimension:
                raise ValueError(
                    f"Dimension {len(item.vector)} != collection {self._dimension} "
                    f"(concept_id={item.concept_id})."
                )
            self._items[item.concept_id] = item
        return len(items)

    def search(self, vector: Sequence[float], top_k: int = 10) -> list[SearchHit]:
        if top_k <= 0:
            return []
        scored = [
            SearchHit(
                concept_id=item.concept_id,
                score=cosine_similarity(vector, item.vector),
                payload=item.payload,
            )
            for item in self._items.values()
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._items)


class QdrantVectorStore(VectorStore):
    """Index Qdrant (souverain). Le client est importé paresseusement."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "ohdsi_concepts",
        api_key: str | None = None,
    ) -> None:
        self.url = url
        self.collection = collection
        self.api_key = api_key
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:  # pragma: no cover - dépend de l'install
                raise ImportError(
                    "qdrant-client requis pour ce backend. "
                    "Installer l'extra : uv sync --extra retrieval"
                ) from exc
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    def ensure_collection(self, dimension: int) -> None:  # pragma: no cover - I/O réseau
        from qdrant_client.models import Distance, VectorParams

        client = self._get_client()
        if not client.collection_exists(self.collection):
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    # Qdrant borne le payload JSON d'une requête (~32 Mo). On upsert par lots
    # pour supporter un corpus réel (~10^5 concepts) sans dépasser la limite.
    _UPSERT_BATCH = 1000

    def upsert(self, items: Sequence[VectorItem]) -> int:  # pragma: no cover - I/O réseau
        from qdrant_client.models import PointStruct

        client = self._get_client()
        total = 0
        for start in range(0, len(items), self._UPSERT_BATCH):
            batch = items[start : start + self._UPSERT_BATCH]
            points = [
                PointStruct(id=item.concept_id, vector=item.vector, payload=item.payload)
                for item in batch
            ]
            client.upsert(collection_name=self.collection, points=points)
            total += len(points)
        return total

    def search(
        self, vector: Sequence[float], top_k: int = 10
    ) -> list[SearchHit]:  # pragma: no cover - I/O réseau
        client = self._get_client()
        result = client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchHit(
                concept_id=int(p.id),
                score=float(p.score),
                payload=dict(p.payload or {}),
            )
            for p in result.points
        ]

    def count(self) -> int:  # pragma: no cover - I/O réseau
        client = self._get_client()
        return int(client.count(collection_name=self.collection).count)
