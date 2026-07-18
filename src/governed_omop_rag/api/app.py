"""API REST (FastAPI) — exposition du pipeline de mapping gouverné.

Deux portes d'entrée pour l'outil (CONTEXT.md §11) : cette API pour les
intégrateurs, l'UI Streamlit pour les non-devs. Le cœur métier est le
``MappingService`` **partagé** — pas de logique dupliquée.

Import de FastAPI au niveau module : ce fichier n'est chargé que lorsque l'extra
``api`` est installé (par ``gor serve`` ou les tests via importorskip).
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from governed_omop_rag.config import Settings, get_settings
from governed_omop_rag.core.models import MappingRequest, MappingSuggestion
from governed_omop_rag.service import MappingService, MapStrategy


class BatchRequest(BaseModel):
    """Corps de /map/batch : une liste d'entrées + une stratégie."""

    items: list[MappingRequest] = Field(default_factory=list)
    strategy: MapStrategy = MapStrategy.AUTO


class _RateLimiter:
    """Garde de coût par IP : au plus `max_requests` par fenêtre glissante de
    `window_seconds` secondes.

    L'API est volontairement PUBLIQUE (démo : recruteurs / techniciens peuvent
    tester sans clé). On n'authentifie pas, mais on borne le coût : chaque mapping
    peut déclencher un appel LLM. L'authentification sera ajoutée plus tard ;
    ce limiteur est le garde-fou d'ici là (audit G3).

    Limites assumées : compteur EN MÉMOIRE (par-processus, remis à zéro au
    redémarrage, non partagé entre réplicas). Un quota durable nécessiterait un
    store partagé (Redis). L'identification se fait par IP réelle (X-Forwarded-For
    honoré derrière un reverse-proxy) ; une IP partagée (NAT) partage son quota.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def _purge(self, cutoff: float) -> None:
        """Retire les entrées expirées et les buckets vides — évite la fuite
        mémoire lente (une IP inactive ne doit pas rester en mémoire pour
        toujours)."""
        for ip in list(self._hits):
            bucket = self._hits[ip]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if not bucket:
                del self._hits[ip]

    def check(self, client: str) -> None:
        if self.max_requests <= 0:
            return  # désactivé
        now = time.time()
        # Purge globale à chaque appel : borne la taille du dict dans le temps.
        self._purge(now - self.window_seconds)
        bucket = self._hits.setdefault(client, deque())
        if len(bucket) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded (public demo: {self.max_requests} "
                    f"requests / {self.window_seconds}s per IP)."
                ),
            )
        bucket.append(now)


class BatchResponse(BaseModel):
    """Réponse de /map/batch."""

    strategy: MapStrategy
    results: list[MappingSuggestion]


def create_app(settings: Settings | None = None, bronze_dir: Path | None = None) -> FastAPI:
    """Fabrique l'application FastAPI (pipeline construit au démarrage)."""
    s = settings or get_settings()
    service = MappingService(s, bronze_dir)
    limiter = _RateLimiter(s.api_rate_limit_max, s.api_rate_limit_window_seconds)
    max_batch = s.api_max_batch_size

    app = FastAPI(
        title="governed-omop-rag",
        summary="Mapping CIM-10 FR / libellés -> concepts standard OMOP (gouverné).",
        version="0.0.1",
    )
    app.state.service = service

    def _client(request: Request) -> str:
        # Derrière un reverse-proxy (docker-compose, nginx), request.client.host
        # est l'IP du proxy → sans ceci, tous les utilisateurs partageraient un
        # seul quota. On prend la 1re IP de X-Forwarded-For (l'IP réelle du
        # client) quand l'en-tête est présent.
        # str(...) explicite : selon la version de Starlette, headers.get() et
        # client.host peuvent être typés Any → warn_return_any (mypy strict) le
        # refuserait sinon.
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return str(xff).split(",")[0].strip()
        client = request.client
        return str(client.host) if client is not None else "unknown"

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "concepts_indexed": service.concepts_indexed}

    @app.post("/map")
    def map_one(
        request: Request,
        body: MappingRequest,
        strategy: MapStrategy = MapStrategy.AUTO,
    ) -> MappingSuggestion:
        limiter.check(_client(request))
        return service.route(body, strategy)

    @app.post("/map/batch")
    def map_batch(request: Request, body: BatchRequest) -> BatchResponse:
        limiter.check(_client(request))
        # Borne le coût d'un seul appel : un lot trop gros = autant d'appels LLM.
        if len(body.items) > max_batch:
            raise HTTPException(
                status_code=413,
                detail=f"Batch too large: {len(body.items)} items (max {max_batch}).",
            )
        results = service.map_many(body.items, body.strategy)
        return BatchResponse(strategy=body.strategy, results=results)

    return app
