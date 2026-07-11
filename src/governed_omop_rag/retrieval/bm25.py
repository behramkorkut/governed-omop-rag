"""BM25 Okapi — recherche lexicale (pur Python, sans dépendance).

Implémenté maison plutôt qu'avec ``rank-bm25`` afin de rester **testable
hors-ligne** (aucune dépendance optionnelle requise en CI) et de montrer la
maîtrise de l'algorithme. C'est le versant *lexical* de la recherche hybride
(P2-2) : il capte le recouvrement exact de tokens là où l'embedding capte le sens.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from governed_omop_rag.medallion.normalize import normalize_ascii


def tokenize(text: str) -> list[str]:
    """Tokenisation simple : normalisation FR (accents, casse) puis split espaces."""
    return normalize_ascii(text).split()


class BM25Index:
    """Index BM25 Okapi sur des documents identifiés par un entier.

    ``docs`` : séquence de (doc_id, tokens). Paramètres standard k1=1.5, b=0.75.
    """

    def __init__(
        self,
        docs: Sequence[tuple[int, Sequence[str]]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[int] = []
        self.doc_tokens: list[Counter[str]] = []
        self.doc_len: list[int] = []
        df: Counter[str] = Counter()
        for doc_id, tokens in docs:
            counts: Counter[str] = Counter(tokens)
            self.doc_ids.append(doc_id)
            self.doc_tokens.append(counts)
            self.doc_len.append(sum(counts.values()))
            df.update(counts.keys())

        self.n_docs = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        # idf lissé (toujours positif) : ln(1 + (N - df + 0.5)/(df + 0.5)).
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def _score_doc(self, query_tokens: Sequence[str], idx: int) -> float:
        counts = self.doc_tokens[idx]
        dl = self.doc_len[idx]
        score = 0.0
        for term in query_tokens:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = self.idf.get(term, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def score(self, query_tokens: Sequence[str]) -> dict[int, float]:
        """Score BM25 de chaque document pour la requête."""
        return {
            self.doc_ids[i]: self._score_doc(query_tokens, i)
            for i in range(self.n_docs)
        }

    def top_k(self, query_tokens: Sequence[str], k: int) -> list[tuple[int, float]]:
        """Top-k (doc_id, score) par score décroissant, en excluant les scores nuls."""
        if k <= 0:
            return []
        scored = [(doc_id, s) for doc_id, s in self.score(query_tokens).items() if s > 0.0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
