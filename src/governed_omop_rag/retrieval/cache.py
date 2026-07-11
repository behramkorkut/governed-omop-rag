"""Cache de retrieval — borne le coût et la latence (P2-6).

Motivation (CONTEXT.md §7 / reco Kimi) : si le même libellé/source revient, on ne
refait pas la recherche (et, en Phase 3, on ne rappelle pas le LLM). ``CachedRetriever``
enveloppe n'importe quel ``Retriever`` sans le modifier (décorateur), et expose des
compteurs hits/misses pour l'observabilité.

Deux backends de cache :
- ``MemoryCandidateCache`` : dict en mémoire (tests, process unique) ;
- ``DuckDBCandidateCache`` : persistant (souverain, réutilisé entre exécutions).

Clé de cache = ``namespace | top_k | requête normalisée``. Le ``namespace`` permet
d'invalider proprement (ex. y encoder le modèle d'embeddings et la version des
vocabulaires) pour éviter de servir des résultats périmés.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import TypeAdapter

from governed_omop_rag.core.models import ConceptCandidate
from governed_omop_rag.medallion.db import connect
from governed_omop_rag.medallion.normalize import normalize_ascii
from governed_omop_rag.retrieval.retriever import Retriever

_ADAPTER: TypeAdapter[list[ConceptCandidate]] = TypeAdapter(list[ConceptCandidate])
_CACHE_TABLE = "candidate_cache"


@runtime_checkable
class CandidateCache(Protocol):
    """Contrat d'un cache de candidats (clé -> liste de candidats)."""

    def get(self, key: str) -> list[ConceptCandidate] | None:
        """Retourne les candidats mémorisés, ou None si absent."""
        ...

    def set(self, key: str, value: list[ConceptCandidate]) -> None:
        """Mémorise des candidats pour une clé."""
        ...


class MemoryCandidateCache:
    """Cache en mémoire (non persistant)."""

    def __init__(self) -> None:
        self._store: dict[str, list[ConceptCandidate]] = {}

    def get(self, key: str) -> list[ConceptCandidate] | None:
        cached = self._store.get(key)
        # Copie défensive : le cache ne doit pas être muté par l'appelant.
        return list(cached) if cached is not None else None

    def set(self, key: str, value: list[ConceptCandidate]) -> None:
        self._store[key] = list(value)


class DuckDBCandidateCache:
    """Cache persistant DuckDB (sérialisation JSON via Pydantic)."""

    def __init__(self, path: str | Path) -> None:
        self._con = connect(path)
        self._con.execute(
            f"CREATE TABLE IF NOT EXISTS {_CACHE_TABLE} (key VARCHAR PRIMARY KEY, value VARCHAR)"
        )

    def get(self, key: str) -> list[ConceptCandidate] | None:
        row = self._con.execute(f"SELECT value FROM {_CACHE_TABLE} WHERE key = ?", [key]).fetchone()
        if row is None:
            return None
        return _ADAPTER.validate_json(row[0])

    def set(self, key: str, value: list[ConceptCandidate]) -> None:
        payload = _ADAPTER.dump_json(value).decode("utf-8")
        self._con.execute(f"INSERT OR REPLACE INTO {_CACHE_TABLE} VALUES (?, ?)", [key, payload])

    def close(self) -> None:
        self._con.close()


class CachedRetriever:
    """Décore un Retriever d'un cache. Satisfait lui-même le protocole Retriever."""

    def __init__(
        self,
        inner: Retriever,
        cache: CandidateCache,
        namespace: str = "default",
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.namespace = namespace
        self.hits = 0
        self.misses = 0

    def _key(self, query: str, top_k: int) -> str:
        return f"{self.namespace}|{top_k}|{normalize_ascii(query)}"

    def retrieve(self, query: str, top_k: int = 10) -> list[ConceptCandidate]:
        key = self._key(query, top_k)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        result = self.inner.retrieve(query, top_k)
        self.cache.set(key, result)
        return result
