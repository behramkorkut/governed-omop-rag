"""Feedback du steward — persistance des décisions (amélioration continue).

Quand un steward valide/corrige/rejette une suggestion, cette information est de
l'or (CONTEXT.md — reco d'amélioration continue) : elle sert à (a) enrichir le
gold set d'évaluation avec des corrections réelles, (b) plus tard, améliorer le
reranking / les prompts. On la journalise dans une table DuckDB ``steward_feedback``.

Traçabilité (gouvernance §4.3) : entrée, proposition initiale, décision, correction.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from pydantic import BaseModel, Field

from governed_omop_rag.core.models import (
    UNMAPPED_CONCEPT_ID,
    MappingSuggestion,
    StewardDecision,
)
from governed_omop_rag.medallion.db import connect

_TABLE = "steward_feedback"


class FeedbackRecord(BaseModel):
    """Une décision de steward, traçable."""

    source_code: str | None = None
    source_label: str | None = None
    source_vocabulary: str | None = None
    proposed_concept_id: int  # ce que l'outil a proposé
    proposed_source: str  # official_map / rag / unmapped
    decision: StewardDecision
    final_concept_id: int  # concept retenu (0 si rejet)
    reason: str | None = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def feedback_records_from_decisions(
    decisions: Sequence[tuple[MappingSuggestion, int | None]],
) -> list[FeedbackRecord]:
    """Classe les décisions du steward en accept / edit / reject.

    ``(suggestion, target)`` : target = concept_id retenu, ou None pour un rejet.
    - target None                    -> REJECT (final = 0) ;
    - target == proposition mappée   -> ACCEPT ;
    - target différent               -> EDIT.
    """
    records: list[FeedbackRecord] = []
    for suggestion, target in decisions:
        proposed = suggestion.target_concept_id
        if target is None:
            decision, final = StewardDecision.REJECT, UNMAPPED_CONCEPT_ID
        elif target == proposed and suggestion.is_mapped:
            decision, final = StewardDecision.ACCEPT, target
        else:
            decision, final = StewardDecision.EDIT, target
        records.append(
            FeedbackRecord(
                source_code=suggestion.request.source_code,
                source_label=suggestion.request.source_label,
                source_vocabulary=suggestion.request.source_vocabulary,
                proposed_concept_id=proposed,
                proposed_source=suggestion.source.value,
                decision=decision,
                final_concept_id=final,
            )
        )
    return records


class FeedbackStore:
    """Journal DuckDB des décisions steward."""

    def __init__(self, path: str | Path) -> None:
        self._con: duckdb.DuckDBPyConnection = connect(path)
        self._con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                source_code VARCHAR,
                source_label VARCHAR,
                source_vocabulary VARCHAR,
                proposed_concept_id BIGINT,
                proposed_source VARCHAR,
                decision VARCHAR,
                final_concept_id BIGINT,
                reason VARCHAR,
                decided_at VARCHAR
            )
            """
        )

    def record(self, records: Sequence[FeedbackRecord]) -> int:
        """Insère des décisions. Retourne le nombre de lignes ajoutées."""
        for r in records:
            self._con.execute(
                f"INSERT INTO {_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    r.source_code,
                    r.source_label,
                    r.source_vocabulary,
                    r.proposed_concept_id,
                    r.proposed_source,
                    r.decision.value,
                    r.final_concept_id,
                    r.reason,
                    r.decided_at.isoformat(),
                ],
            )
        return len(records)

    def count(self) -> int:
        row = self._con.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()
        return int(row[0]) if row else 0

    def to_gold_records(self) -> list[dict[str, object]]:
        """Dérive des entrées gold-set des décisions retenues (accept/edit).

        Format compatible ``data/eval/gold_set.csv`` :
        ``source_code, source_label, expected_concept_id``.
        """
        rows = self._con.execute(
            f"""
            SELECT source_code, source_label, final_concept_id
            FROM {_TABLE}
            WHERE decision IN ('accept', 'edit') AND final_concept_id <> 0
            """
        ).fetchall()
        return [
            {
                "source_code": r[0] or "",
                "source_label": r[1] or "",
                "expected_concept_id": int(r[2]),
            }
            for r in rows
        ]

    def close(self) -> None:
        self._con.close()
