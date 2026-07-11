"""Schémas internes des agents (Proposer / Vérificateur)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ProposerOutput(BaseModel):
    """Décision de l'agent Proposer : un concept choisi + sa justification."""

    concept_id: int
    justification: str


class VerdictStatus(StrEnum):
    """Résultat du sous-agent Vérificateur."""

    PASS = "pass"
    FAIL = "fail"


class Verdict(BaseModel):
    """Verdict du Vérificateur : PASS/FAIL + raison lisible (traçabilité)."""

    status: VerdictStatus
    reason: str

    @property
    def passed(self) -> bool:
        return self.status is VerdictStatus.PASS
