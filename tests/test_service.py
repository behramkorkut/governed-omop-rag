"""Tests du MappingService — isolation de lot (résilience G2, action 3).

Un item dont l'agent échoue sur une erreur NON-parsing (API LLM down, Qdrant
indisponible — simulé par RuntimeError) ne doit pas faire échouer tout le lot :
`map_many` rend cet item UNMAPPED/ERREUR_AGENT et traite les autres.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

pytest.importorskip("rank_bm25")

from governed_omop_rag.agents.schemas import ProposerOutput  # noqa: E402
from governed_omop_rag.config import (  # noqa: E402
    EmbeddingBackend,
    Settings,
    VectorBackend,
)
from governed_omop_rag.core.models import (  # noqa: E402
    ConceptCandidate,
    MappingRequest,
    MappingSource,
    NoMapReason,
)
from governed_omop_rag.service import MappingService  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _service() -> MappingService:
    return MappingService(
        Settings(
            embedding_backend=EmbeddingBackend.HASHING,
            vector_backend=VectorBackend.MEMORY,
            embedding_dim=512,
            router_map_path=FIXTURES / "router_map.csv",
            anthropic_api_key=None,
        ),
        FIXTURES,
    )


def test_map_many_isolates_a_broken_item(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _service()
    real_propose = svc._llm.propose

    def boom_propose(query: str, candidates: Sequence[ConceptCandidate]) -> ProposerOutput:
        if "BOOM" in query:
            raise RuntimeError("API down (simulé)")
        return real_propose(query, candidates)

    monkeypatch.setattr(svc._llm, "propose", boom_propose)

    batch = [
        MappingRequest(source_label="grippe"),
        MappingRequest(source_label="BOOM cas qui plante"),
        MappingRequest(source_label="diabète"),
    ]
    results = svc.map_many(batch)

    assert len(results) == 3
    # L'item cassé est isolé et rendu explicitement en erreur d'agent.
    assert results[1].source is MappingSource.UNMAPPED
    assert results[1].no_map_reason is NoMapReason.ERREUR_AGENT
    # Les autres items sont traités normalement (pas d'erreur d'agent).
    assert results[0].no_map_reason is not NoMapReason.ERREUR_AGENT
    assert results[2].no_map_reason is not NoMapReason.ERREUR_AGENT
