# ---------------------------------------------------------------------------
# Image applicative de base (Phase 0). Les services (API/UI) seront ajoutés
# au docker-compose phase par phase. On installe les deps cœur + retrieval/api.
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

# Installe le paquet (deps cœur uniquement pour l'image de base).
RUN uv pip install --system .

# Le code restant (config, scripts) est monté ou copié selon le service.
COPY . .

# Smoke par défaut ; les services surchargent la commande.
CMD ["gor", "info"]
