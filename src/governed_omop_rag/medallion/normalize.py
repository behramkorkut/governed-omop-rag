"""Normalisation de texte (libellés cliniques FR).

Objectif : produire une forme comparable (casse, espaces, accents) pour la
recherche lexicale, sans détruire l'information utile aux embeddings.
Les accents FR sont fréquents (diabète, œdème, hémorragie) : on fournit une
variante avec et sans accents.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")

# Ligatures françaises fréquentes en médecine (œdème, cœur, æquo...).
# NFKD ne les décompose pas : on les étend explicitement pour la forme ASCII.
_LIGATURES = str.maketrans({"œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE"})


def collapse_whitespace(text: str) -> str:
    """Réduit toute séquence d'espaces (tabs, retours ligne) à une seule espace."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def expand_ligatures(text: str) -> str:
    """Étend les ligatures (œ -> oe, æ -> ae) — utile pour un matching robuste."""
    return text.translate(_LIGATURES)


def normalize_text(text: str | None) -> str:
    """Forme normalisée : minuscules + espaces réduits. Conserve les accents.

    Retourne une chaîne vide pour None (jamais d'exception sur entrée nulle).
    """
    if not text:
        return ""
    return collapse_whitespace(text).casefold()


def strip_accents(text: str) -> str:
    """Retire les diacritiques (é -> e, ç -> c) et étend les ligatures (œ -> oe).

    Les ligatures sont étendues *avant* la décomposition car NFKD ne les traite
    pas (œdème resterait « œdeme »).
    """
    decomposed = unicodedata.normalize("NFKD", expand_ligatures(text))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_ascii(text: str | None) -> str:
    """Forme normalisée sans accents (utile pour un matching lexical robuste)."""
    return strip_accents(normalize_text(text))
