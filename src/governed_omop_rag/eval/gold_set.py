"""Gold set : vérité terrain pour l'évaluation reproductible.

Chaque entrée associe une requête source (code CIM-10 FR et/ou libellé) au
``concept_id`` standard attendu. Idéalement dérivé de l'alignement officiel
CIM-10 <-> SNOMED-CT et/ou d'annotations manuelles (CONTEXT.md §7).

Format CSV : ``source_code,source_label,expected_concept_id``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel, model_validator


class GoldItem(BaseModel):
    """Une entrée de vérité terrain."""

    source_code: str | None = None
    source_label: str | None = None
    expected_concept_id: int

    @model_validator(mode="after")
    def _require_code_or_label(self) -> GoldItem:
        if not (self.source_code or self.source_label):
            raise ValueError("GoldItem : 'source_code' ou 'source_label' requis.")
        return self

    @property
    def query(self) -> str:
        """Requête à soumettre au retrieval (libellé prioritaire sinon code)."""
        return self.source_label or self.source_code or ""


def load_gold_set(path: str | Path) -> list[GoldItem]:
    """Charge un gold set depuis un CSV. Ignore les lignes sans concept attendu."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Gold set introuvable : {p}.")
    items: list[GoldItem] = []
    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = (row.get("expected_concept_id") or "").strip()
            if not raw_id:
                continue
            items.append(
                GoldItem(
                    source_code=(row.get("source_code") or "").strip() or None,
                    source_label=(row.get("source_label") or "").strip() or None,
                    expected_concept_id=int(raw_id),
                )
            )
    return items
