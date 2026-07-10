"""Schémas Pydantic du domaine — source de vérité unique.

Partagés par l'API (FastAPI), l'UI (Streamlit) et les agents (LangGraph).
Ces modèles incarnent des décisions de gouvernance du CONTEXT.md :

- sortie fermée : un candidat DOIT référencer un ``concept_id`` réel du vocabulaire ;
- « non mappé » explicite : ``concept_id = 0`` + raison typée (jamais de mapping forcé) ;
- traçabilité : chaque suggestion porte sa source, ses candidats et un score.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

# Convention OMOP : un mapping absent est représenté par concept_id = 0.
UNMAPPED_CONCEPT_ID = 0


class MappingSource(StrEnum):
    """D'où vient la suggestion (transparence pour le steward)."""

    OFFICIAL_MAP = "official_map"  # alignement officiel CIM-10 <-> SNOMED-CT
    RAG = "rag"  # retrieval hybride + agents
    MANUAL = "manual"  # saisi/corrigé par le steward
    UNMAPPED = "unmapped"  # aucun mapping fiable


class NoMapReason(StrEnum):
    """Raison typée d'une absence de mapping (concept_id = 0)."""

    HORS_VOCABULAIRE = "hors_vocabulaire"
    AMBIGU = "ambigu"
    CONFIDENCE_INSUFFISANTE = "confidence_insuffisante"
    AUCUN_CANDIDAT = "aucun_candidat"


class MappingRequest(BaseModel):
    """Entrée à mapper : un code source, un libellé, ou les deux."""

    source_code: str | None = Field(default=None, description="Code source, ex. CIM-10 FR 'E11.9'.")
    source_label: str | None = Field(
        default=None, description="Libellé clinique en texte libre, ex. 'diabète type 2'."
    )
    source_vocabulary: str | None = Field(
        default=None, description="Vocabulaire source, ex. 'ICD10FR'."
    )

    @model_validator(mode="after")
    def _require_code_or_label(self) -> MappingRequest:
        if not (self.source_code or self.source_label):
            raise ValueError("Il faut au moins 'source_code' ou 'source_label'.")
        return self


class ConceptCandidate(BaseModel):
    """Un concept standard OHDSI candidat (issu du vocabulaire chargé)."""

    concept_id: int = Field(description="concept_id OHDSI (doit exister dans le vocabulaire).")
    concept_name: str
    vocabulary_id: str = Field(description="ex. 'SNOMED', 'RxNorm', 'LOINC'.")
    domain_id: str = Field(description="ex. 'Condition', 'Drug', 'Measurement'.")
    standard_concept: str | None = Field(
        default=None, description="'S' si standard, sinon None/'C'."
    )
    score: float = Field(ge=0.0, le=1.0, description="Score de similarité/confiance du candidat.")
    synonyms: list[str] = Field(default_factory=list)

    @property
    def is_standard(self) -> bool:
        """Garde-fou OMOP : n'est mappable que si standard_concept == 'S'."""
        return self.standard_concept == "S"


class MappingSuggestion(BaseModel):
    """Suggestion complète et traçable pour une entrée."""

    request: MappingRequest
    target_concept_id: int = Field(
        default=UNMAPPED_CONCEPT_ID,
        description="concept_id retenu ; 0 si non mappé.",
    )
    candidates: list[ConceptCandidate] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: MappingSource = MappingSource.UNMAPPED
    no_map_reason: NoMapReason | None = None
    justification: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_mapped(self) -> bool:
        """True si un concept a été retenu (différent de la valeur non-mappé)."""
        return self.target_concept_id != UNMAPPED_CONCEPT_ID

    @model_validator(mode="after")
    def _coherence(self) -> MappingSuggestion:
        # Non mappé => raison obligatoire ; mappé => pas de raison de non-map.
        if self.is_mapped and self.no_map_reason is not None:
            raise ValueError("Un mapping retenu ne peut pas porter de no_map_reason.")
        if not self.is_mapped and self.source not in (
            MappingSource.UNMAPPED,
            MappingSource.MANUAL,
        ):
            raise ValueError("Une entrée non mappée doit avoir source UNMAPPED (ou MANUAL).")
        return self


class StewardDecision(StrEnum):
    """Décision humaine sur une suggestion (human-in-the-loop)."""

    ACCEPT = "accept"
    REJECT = "reject"
    EDIT = "edit"


class StewardFeedback(BaseModel):
    """Retour steward : matière première pour l'amélioration continue.

    (cf. recommandation « feedback steward » — table d'or pour reranking/prompts.)
    """

    suggestion: MappingSuggestion
    decision: StewardDecision
    corrected_concept_id: int | None = None
    reason: str | None = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
