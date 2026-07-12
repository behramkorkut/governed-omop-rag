"""Logique de l'UI de revue steward — testable, sans dépendance à Streamlit.

Sépare le métier (parsing des entrées, mise en forme des suggestions, export au
format OMOP ``source_to_concept_map``) de la couche d'affichage (``ui/app.py``),
afin de pouvoir tester la logique sans lancer Streamlit.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from governed_omop_rag.core.models import MappingRequest, MappingSuggestion

# Colonnes standard de la table OMOP source_to_concept_map.
STCM_COLUMNS = [
    "source_code",
    "source_concept_id",
    "source_vocabulary_id",
    "source_code_description",
    "target_concept_id",
    "target_vocabulary_id",
    "valid_start_date",
    "valid_end_date",
    "invalid_reason",
]


def _cell(value: object) -> str:
    """Nettoie une cellule (gère None et les NaN pandas rendus en 'nan')."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def requests_from_records(records: Sequence[Mapping[str, object]]) -> list[MappingRequest]:
    """Construit des MappingRequest depuis des lignes (dicts) CSV/Excel.

    Colonnes reconnues : ``source_code``, ``source_label``, ``source_vocabulary``.
    Les lignes sans code ni libellé sont ignorées.
    """
    requests: list[MappingRequest] = []
    for row in records:
        code = _cell(row.get("source_code")) or None
        label = _cell(row.get("source_label")) or None
        vocab = _cell(row.get("source_vocabulary")) or None
        if not (code or label):
            continue
        requests.append(
            MappingRequest(source_code=code, source_label=label, source_vocabulary=vocab)
        )
    return requests


def suggestion_to_row(suggestion: MappingSuggestion) -> dict[str, object]:
    """Aplati une suggestion en ligne d'affichage (tableau de revue)."""
    target_name = ""
    if suggestion.is_mapped:
        match = next(
            (c for c in suggestion.candidates if c.concept_id == suggestion.target_concept_id),
            None,
        )
        target_name = match.concept_name if match else ""
    return {
        "source_code": suggestion.request.source_code or "",
        "source_label": suggestion.request.source_label or "",
        "target_concept_id": suggestion.target_concept_id,
        "target_concept_name": target_name,
        "confidence": round(suggestion.confidence, 3),
        "source": suggestion.source.value,
        "no_map_reason": suggestion.no_map_reason.value if suggestion.no_map_reason else "",
        "justification": suggestion.justification or "",
        "n_candidates": len(suggestion.candidates),
    }


@dataclass(frozen=True)
class ValidatedMapping:
    """Mapping validé par le steward, prêt pour l'export."""

    source_code: str | None
    source_label: str | None
    source_vocabulary: str | None
    target_concept_id: int
    target_vocabulary_id: str


def validated_from_suggestion(
    suggestion: MappingSuggestion, target_concept_id: int | None = None
) -> ValidatedMapping:
    """Construit un mapping validé (avec correction éventuelle du concept_id)."""
    tid = target_concept_id if target_concept_id is not None else suggestion.target_concept_id
    match = next((c for c in suggestion.candidates if c.concept_id == tid), None)
    return ValidatedMapping(
        source_code=suggestion.request.source_code,
        source_label=suggestion.request.source_label,
        source_vocabulary=suggestion.request.source_vocabulary,
        target_concept_id=tid,
        target_vocabulary_id=match.vocabulary_id if match else "",
    )


def to_source_to_concept_map(
    mappings: Sequence[ValidatedMapping],
    valid_start_date: str = "1970-01-01",
    valid_end_date: str = "2099-12-31",
) -> list[dict[str, object]]:
    """Convertit des mappings validés au format OMOP source_to_concept_map."""
    rows: list[dict[str, object]] = []
    for m in mappings:
        rows.append(
            {
                "source_code": m.source_code or "",
                "source_concept_id": 0,
                "source_vocabulary_id": m.source_vocabulary or "",
                "source_code_description": m.source_label or "",
                "target_concept_id": m.target_concept_id,
                "target_vocabulary_id": m.target_vocabulary_id,
                "valid_start_date": valid_start_date,
                "valid_end_date": valid_end_date,
                "invalid_reason": "",
            }
        )
    return rows


def write_source_to_concept_map_csv(mappings: Sequence[ValidatedMapping], path: str | Path) -> int:
    """Écrit le source_to_concept_map en CSV. Retourne le nombre de lignes."""
    rows = to_source_to_concept_map(mappings)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STCM_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
