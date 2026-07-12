# ---------------------------------------------------------------------------
# Image applicative (api + ui). On installe les extras api/ui/agents + qdrant-client
# (recherche vectorielle) SANS torch : l'image reste légère, backend embeddings
# = hashing (déterministe, hors-ligne) par défaut dans le docker-compose.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

# uv (gestionnaire de paquets) depuis l'image officielle.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Couche deps (cache) : on copie d'abord les métadonnées de projet.
COPY pyproject.toml README.md ./
COPY src ./src

# Installe le paquet + extras exposition/agents + client Qdrant (pas de torch).
RUN uv pip install --system ".[api,ui,agents]" "qdrant-client>=1.9"

# Le code restant (config, scripts) est monté ou copié selon le service.
COPY . .

# Smoke par défaut ; les services surchargent la commande.
CMD ["gor", "info"]
