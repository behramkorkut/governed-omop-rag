#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# init.sh — setup idempotent de l'environnement de dev + smoke-test.
# Peut être relancé plusieurs fois sans casser (harness Anthropic).
#
# Usage : ./init.sh
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

echo "==> governed-omop-rag : initialisation de l'environnement"

# 1. Vérifier / installer uv --------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "==> uv introuvable — installation..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Rendre uv disponible dans le shell courant.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
echo "==> uv : $(uv --version)"

# 2. Synchroniser l'environnement (deps cœur + groupe dev) --------------------
echo "==> uv sync (deps cœur + dev)"
uv sync --group dev

# 3. Préparer .env si absent --------------------------------------------------
if [ ! -f .env ]; then
    echo "==> création de .env depuis .env.example"
    cp .env.example .env
fi

# 4. Lint + typage (non bloquant en scaffold, informatif) ---------------------
echo "==> ruff check"
uv run ruff check . || echo "    (ruff a signalé des points — non bloquant en Phase 0)"

# 5. Smoke-test : la config se charge et le CLI répond ------------------------
echo "==> smoke-test (gor smoke)"
uv run gor smoke

# 6. Suite de tests -----------------------------------------------------------
echo "==> pytest"
uv run pytest

echo "==> OK : environnement prêt."
