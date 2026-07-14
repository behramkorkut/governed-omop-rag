"""Métriques de qualité gouvernance : faithfulness (P5-3) et hallucination (P5-4).

Volontairement SANS dépendance au reste du paquet (types simples) : fonctions
pures, testables isolément et réutilisables dans le benchmark (P5-6).

- Faithfulness (style RAGAS, version légère et déterministe) : la justification du
  Proposer s'appuie-t-elle sur les CANDIDATS fournis, ou invente-t-elle du
  contexte externe ? On mesure la part des tokens de contenu de la justification
  qui apparaissent effectivement dans le texte des candidats retournés.

- Taux d'hallucination : part des concepts proposés qui sont hors-vocabulaire ou
  non-standard. Grâce au garde-fou de sortie fermée (ClosedOutputViolation), ce
  taux doit tendre vers 0 : la métrique le VÉRIFIE, elle ne le suppose pas.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence

# Mots-outils FR/EN ignorés : ils ne portent pas de signal clinique et fausseraient
# la mesure de faithfulness (« the », « de », « with »... sont partout).
_STOPWORDS = frozenset(
    {
        # français
        "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "à", "au",
        "aux", "en", "dans", "sur", "sous", "par", "pour", "avec", "sans", "ce",
        "cet", "cette", "ces", "est", "sont", "qui", "que", "quoi", "dont", "se",
        "sa", "son", "ses", "plus", "moins", "type",
        # anglais
        "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
        "without", "is", "are", "this", "that", "these", "those", "as", "by",
        "at", "from", "not", "other", "unspecified", "due",
    }
)

_MIN_TOKEN_LEN = 3


def _fold(text: str) -> str:
    """Minuscule + suppression des accents (comparaison robuste FR)."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def content_tokens(text: str) -> set[str]:
    """Tokens de contenu : alphanumériques, sans accents, hors mots-outils et courts."""
    folded = _fold(text)
    raw = "".join(c if c.isalnum() else " " for c in folded).split()
    return {t for t in raw if len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS}


def faithfulness_score(justification: str, candidate_texts: Sequence[str]) -> float:
    """Part des tokens de contenu de la justification présents dans les candidats.

    1.0 = chaque mot de contenu de la justification est ancré dans un candidat
    (fidèle au contexte). 0.0 = justification entièrement « hors-contexte ».
    Une justification vide renvoie 1.0 (rien d'infidèle).
    """
    just_tokens = content_tokens(justification)
    if not just_tokens:
        return 1.0
    grounded: set[str] = set()
    for text in candidate_texts:
        grounded |= content_tokens(text)
    supported = len(just_tokens & grounded)
    return supported / len(just_tokens)


def mean_faithfulness(samples: Iterable[tuple[str, Sequence[str]]]) -> float:
    """Faithfulness moyen sur des (justification, candidats). 0.0 si aucun échantillon."""
    scores = [faithfulness_score(j, c) for j, c in samples]
    return sum(scores) / len(scores) if scores else 0.0


def is_hallucinated(concept_id: int, valid_concept_ids: frozenset[int] | set[int]) -> bool:
    """True si un concept PROPOSÉ (concept_id != 0) n'est pas un concept standard connu.

    concept_id == 0 = « je ne sais pas » : ce n'est PAS une hallucination (abstention).
    """
    return concept_id != 0 and concept_id not in valid_concept_ids


def hallucination_rate(
    concept_ids: Sequence[int], valid_concept_ids: frozenset[int] | set[int]
) -> float:
    """Part des concepts PROPOSÉS (non nuls) qui sont hors-vocabulaire/non-standard.

    Dénominateur = nombre d'entrées effectivement mappées (concept_id != 0).
    Renvoie 0.0 si aucune entrée n'a été mappée.
    """
    mapped = [c for c in concept_ids if c != 0]
    if not mapped:
        return 0.0
    hallucinated = sum(1 for c in mapped if c not in valid_concept_ids)
    return hallucinated / len(mapped)
