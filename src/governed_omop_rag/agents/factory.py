"""Fabrique du LLM Proposer avec dégradation gracieuse.

Objectif « utilisable par un tiers » : ne jamais crasher au moment de la requête.
Si une clé Anthropic est configurée mais que le paquet ``anthropic`` n'est pas
installé (extra ``agents`` manquant), on retombe sur le Proposer déterministe
hors-ligne avec un avertissement, plutôt que de lever une erreur 500.
"""

from __future__ import annotations

import importlib.util

from governed_omop_rag.agents.llm import (
    ClaudeProposerLLM,
    FakeProposerLLM,
    ProposerLLM,
)
from governed_omop_rag.config import Settings, get_settings
from governed_omop_rag.core.logging import get_logger


def build_proposer_llm(settings: Settings | None = None) -> ProposerLLM:
    """Retourne Claude si (clé configurée ET paquet anthropic présent), sinon Fake."""
    s = settings or get_settings()
    if s.anthropic_api_key is not None:
        if importlib.util.find_spec("anthropic") is not None:
            return ClaudeProposerLLM(s.anthropic_api_key.get_secret_value(), s.llm_model)
        get_logger("agents").warning(
            "claude_indisponible_fallback_fake",
            raison="paquet 'anthropic' non installé (uv sync --extra agents)",
        )
    return FakeProposerLLM()
