"""Robustesse du parsing de la réponse Claude (ClaudeProposerLLM._extract_json).

Régression : Sonnet 5 renvoie un ``ThinkingBlock`` avant le bloc texte -> il ne
faut pas prendre ``content[0]`` mais le premier bloc porteur d'un attribut ``text``.
On tolère aussi le JSON entouré de prose ou de barrières Markdown.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from governed_omop_rag.agents.llm import ClaudeProposerLLM


class _Thinking:
    """Simule un bloc de raisonnement : PAS d'attribut ``text``."""

    def __init__(self, thinking: str = "…") -> None:
        self.thinking = thinking


class _Text:
    def __init__(self, text: str) -> None:
        self.text = text


def _cid(blocks: Sequence[object]) -> int:
    return int(json.loads(ClaudeProposerLLM._extract_json(blocks))["concept_id"])


def test_skips_thinking_block() -> None:
    blocks = [_Thinking(), _Text('{"concept_id": 201826, "justification": "diabète"}')]
    assert _cid(blocks) == 201826


def test_json_wrapped_in_prose() -> None:
    blocks = [_Text('Mon choix :\n{"concept_id": 4048098, "justification": "asthme"}\nVoilà.')]
    assert _cid(blocks) == 4048098


def test_json_in_markdown_fence() -> None:
    blocks = [_Text('```json\n{"concept_id": 320128, "justification": "HTA"}\n```')]
    assert _cid(blocks) == 320128


def test_plain_json_first_block() -> None:
    blocks = [_Text('{"concept_id": 1503297, "justification": "metformine"}')]
    assert _cid(blocks) == 1503297


def test_no_text_block_raises() -> None:
    try:
        ClaudeProposerLLM._extract_json([_Thinking()])
    except ValueError:
        return
    raise AssertionError("Une réponse sans bloc texte doit lever ValueError.")
