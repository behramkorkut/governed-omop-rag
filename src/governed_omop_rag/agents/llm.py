"""LLM du Proposer — abstraction pour rester testable et swappable.

L'agent Proposer ne dépend pas de Claude directement mais d'un protocole
``ProposerLLM``. Deux implémentations :
- ``FakeProposerLLM`` : déterministe, hors-ligne (tests / dev sans API). Ce n'est
  pas un mock du système testé : c'est un double contrôlable du LLM, le reste de
  l'orchestration (sortie fermée, vérification, boucle) est bien exercé.
- ``ClaudeProposerLLM`` : appelle Claude (import paresseux d'``anthropic``).

Context engineering (CONTEXT.md §4.4) : on n'injecte que le top-k reclassé
(nom + synonymes + domaine + vocabulaire), pas tout le vocabulaire.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from governed_omop_rag.agents.schemas import ProposerOutput
from governed_omop_rag.core.models import ConceptCandidate


@runtime_checkable
class ProposerLLM(Protocol):
    """Contrat : choisir un concept parmi les candidats et le justifier."""

    def propose(self, query: str, candidates: Sequence[ConceptCandidate]) -> ProposerOutput:
        """Retourne un ProposerOutput (concept_id choisi + justification)."""
        ...


class FakeProposerLLM:
    """LLM déterministe : choisit le meilleur candidat (ou un id préféré).

    ``prefer`` permet aux tests de forcer un choix précis (ex. pour déclencher un
    FAIL du Vérificateur et exercer la boucle de correction).
    """

    def __init__(self, prefer: int | None = None) -> None:
        self.prefer = prefer

    def propose(self, query: str, candidates: Sequence[ConceptCandidate]) -> ProposerOutput:
        chosen = None
        if self.prefer is not None:
            chosen = next((c for c in candidates if c.concept_id == self.prefer), None)
        if chosen is None:
            chosen = candidates[0]  # candidats triés par score décroissant
        return ProposerOutput(
            concept_id=chosen.concept_id,
            justification=(
                f"Candidat le plus proche de « {query} » : "
                f"{chosen.concept_name} ({chosen.vocabulary_id})."
            ),
        )


_SYSTEM_PROMPT = (
    "Tu es un assistant de terminologie médicale. On te donne un libellé source "
    "et une liste de concepts candidats OHDSI. Choisis le concept_id le PLUS "
    "pertinent PARMI LA LISTE UNIQUEMENT. Réponds STRICTEMENT en JSON : "
    '{"concept_id": <int>, "justification": "<courte phrase>"}. '
    "N'invente jamais de concept_id hors de la liste."
)


class ClaudeProposerLLM:  # pragma: no cover - nécessite une clé API + réseau
    """Proposer basé sur Claude (import paresseux d'anthropic)."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-5") -> None:
        self.api_key = api_key
        self.model = model
        self._client: Any | None = None
        # Compteurs de coût (observabilité §7) : tokens cumulés sur la session.
        self.input_tokens = 0
        self.output_tokens = 0

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "anthropic requis pour ce backend. uv sync --extra agents"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    @staticmethod
    def _render_candidates(candidates: Sequence[ConceptCandidate]) -> str:
        lines = []
        for c in candidates:
            syn = f" ; synonymes: {', '.join(c.synonyms)}" if c.synonyms else ""
            lines.append(
                f"- concept_id={c.concept_id} | {c.concept_name} "
                f"({c.vocabulary_id}/{c.domain_id}){syn}"
            )
        return "\n".join(lines)

    def propose(self, query: str, candidates: Sequence[ConceptCandidate]) -> ProposerOutput:
        client = self._get_client()
        user = f"Libellé source : {query}\n\nCandidats :\n{self._render_candidates(candidates)}"
        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        usage = getattr(message, "usage", None)
        if usage is not None:
            self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

        data = json.loads(self._extract_json(message.content))
        return ProposerOutput(
            concept_id=int(data["concept_id"]),
            justification=str(data.get("justification", "")),
        )

    @staticmethod
    def _extract_json(content: Sequence[object]) -> str:
        """Extrait le JSON de la réponse.

        Les modèles à raisonnement renvoient un ``ThinkingBlock`` avant le texte :
        on prend le **premier bloc texte** (attribut ``text``), pas ``content[0]``.
        On tolère aussi un JSON entouré de prose ou de barrières Markdown.
        """
        text = ""
        for block in content:
            candidate = getattr(block, "text", None)
            if isinstance(candidate, str) and candidate.strip():
                text = candidate
                break
        if not text:
            raise ValueError("Réponse LLM sans bloc texte exploitable.")
        stripped = text.strip()
        if stripped.startswith("```"):  # retire ```json ... ```
            stripped = stripped.strip("`")
            newline = stripped.find("\n")
            if newline != -1:
                stripped = stripped[newline + 1 :]
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            start, end = stripped.find("{"), stripped.rfind("}")
            if start != -1 and end > start:
                return stripped[start : end + 1]
            raise
