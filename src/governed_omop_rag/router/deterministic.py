"""Router déterministe v1 — match exact via alignement officiel CIM-10 <-> SNOMED-CT.

Cœur de la stratégie hybride (CONTEXT.md §5.5) : on tente d'abord un match
**déterministe** sur l'alignement officiel (publié 2×/an par l'ATIH). En v1 on
ne fait QUE ce match ; le RAG agentique traitera le résidu (phases suivantes).

Bénéfices : gratuit, instantané, 100 % fiable sur les codes couverts, et cela
**borne le coût** du LLM (qui ne verra que la queue difficile).

La table de correspondance est rechargeable depuis un CSV
(``source_code,target_concept_id[,target_concept_name]``) pour que l'utilisateur
substitue le vrai alignement ATIH sans toucher au code.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from governed_omop_rag.core.models import (
    MappingRequest,
    MappingSource,
    MappingSuggestion,
    NoMapReason,
)

JUSTIFICATION_MATCH = "Match exact via alignement officiel CIM-10 <-> SNOMED-CT."
JUSTIFICATION_NO_MATCH = "Aucun match déterministe trouvé dans l'alignement officiel."


def normalize_code(code: str | None) -> str:
    """Normalise un code source pour un matching insensible à la casse.

    CIM-10 FR : lettres/chiffres avec point (E11.9). On retire les espaces de
    bord et on met en majuscules. Retourne '' pour None/vide.

    Limite v1 (documentée) : les extensions/suffixes ATIH (ex. ``E11.9+``,
    dagger/astérisque) ne sont PAS normalisés — le code est comparé tel quel.
    Pour les couvrir, ajouter les variantes voulues directement dans le CSV
    d'alignement. Une normalisation de suffixes pourra être ajoutée en Phase 2.
    """
    if not code:
        return ""
    return code.strip().upper()


class OfficialMap:
    """Table de correspondance code CIM-10 FR (normalisé) -> concept_id SNOMED."""

    def __init__(
        self,
        mapping: Mapping[str, int],
        names: Mapping[str, str] | None = None,
    ) -> None:
        # Ré-indexe sur le code normalisé (robustesse casse/espaces).
        self._map: dict[str, int] = {}
        self._names: dict[str, str] = {}
        names = names or {}
        for raw_code, concept_id in mapping.items():
            key = normalize_code(raw_code)
            if not key:
                continue
            self._map[key] = int(concept_id)
            if raw_code in names:
                self._names[key] = names[raw_code]

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, int]]) -> OfficialMap:
        """Construit une map depuis des paires (code, concept_id)."""
        return cls(dict(pairs))

    @classmethod
    def from_csv(cls, path: str | Path, delimiter: str = ",") -> OfficialMap:
        """Charge la map depuis un CSV.

        Colonnes attendues : ``source_code``, ``target_concept_id`` et
        optionnellement ``target_concept_name``.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Fichier d'alignement introuvable : {p}. "
                "Fournir l'alignement officiel CIM-10 <-> SNOMED-CT (ATIH)."
            )
        mapping: dict[str, int] = {}
        names: dict[str, str] = {}
        with p.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                code = (row.get("source_code") or "").strip()
                raw_id = (row.get("target_concept_id") or "").strip()
                if not code or not raw_id:
                    continue
                mapping[code] = int(raw_id)
                name = (row.get("target_concept_name") or "").strip()
                if name:
                    names[code] = name
        return cls(mapping, names)

    def lookup(self, code: str | None) -> int | None:
        """Retourne le concept_id pour un code (insensible à la casse), sinon None."""
        return self._map.get(normalize_code(code))

    def name_of(self, code: str | None) -> str | None:
        """Retourne le libellé cible si connu."""
        return self._names.get(normalize_code(code))

    def __contains__(self, code: object) -> bool:
        return isinstance(code, str) and normalize_code(code) in self._map

    def __len__(self) -> int:
        return len(self._map)


class DeterministicRouter:
    """Route une requête via la table de correspondance officielle (déterministe)."""

    def __init__(self, official_map: OfficialMap) -> None:
        self.official_map = official_map

    def route(self, request: MappingRequest) -> MappingSuggestion:
        """Applique le match déterministe.

        - code trouvé      -> OFFICIAL_MAP, confidence 1.0 ;
        - code non trouvé  -> UNMAPPED, HORS_VOCABULAIRE ;
        - aucun code fourni -> UNMAPPED, AUCUN_CANDIDAT (v1 ne gère que les codes).
        """
        if not request.source_code:
            return MappingSuggestion(
                request=request,
                source=MappingSource.UNMAPPED,
                no_map_reason=NoMapReason.AUCUN_CANDIDAT,
                justification=JUSTIFICATION_NO_MATCH,
            )

        concept_id = self.official_map.lookup(request.source_code)
        if concept_id is not None:
            return MappingSuggestion(
                request=request,
                target_concept_id=concept_id,
                candidates=[],
                confidence=1.0,
                source=MappingSource.OFFICIAL_MAP,
                justification=JUSTIFICATION_MATCH,
            )

        return MappingSuggestion(
            request=request,
            source=MappingSource.UNMAPPED,
            no_map_reason=NoMapReason.HORS_VOCABULAIRE,
            justification=JUSTIFICATION_NO_MATCH,
        )


def route_deterministic(request: MappingRequest, official_map: OfficialMap) -> MappingSuggestion:
    """Wrapper fonctionnel : route une requête avec la map fournie."""
    return DeterministicRouter(official_map).route(request)
