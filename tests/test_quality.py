"""Tests des métriques de qualité gouvernance : faithfulness (P5-3), hallucination
(P5-4) et coût par entrée (P5-5)."""

from __future__ import annotations

from governed_omop_rag.eval.metrics import aggregate_mapping
from governed_omop_rag.eval.quality import (
    content_tokens,
    faithfulness_score,
    hallucination_rate,
    is_hallucinated,
    mean_faithfulness,
)

# --- P5-3 : faithfulness ---------------------------------------------------


def test_content_tokens_drops_stopwords_accents_and_short() -> None:
    toks = content_tokens("Diabète de type 2 non insulino-dépendant")
    assert "diabete" in toks  # accent retiré
    assert "insulino" in toks
    assert "dependant" in toks
    assert "de" not in toks  # mot-outil
    assert "type" not in toks  # mot-outil
    assert "2" not in toks  # trop court (< 3)


def test_faithfulness_full_when_grounded() -> None:
    # Chaque mot de contenu de la justification figure dans un candidat.
    just = "diabète non insulino-dépendant sucré"
    candidates = ["Diabète non insulino-dépendant", "diabète sucré de type 2"]
    assert faithfulness_score(just, candidates) == 1.0


def test_faithfulness_penalises_external_context() -> None:
    # « pancréas » et « génétique » n'apparaissent dans aucun candidat.
    just = "diabète lié au pancréas et à une prédisposition génétique héréditaire"
    candidates = ["Diabète non insulino-dépendant"]
    score = faithfulness_score(just, candidates)
    assert 0.0 < score < 1.0


def test_faithfulness_empty_justification_is_one() -> None:
    assert faithfulness_score("", ["quoi que ce soit"]) == 1.0
    assert faithfulness_score("de la le les", ["x"]) == 1.0  # que des mots-outils


def test_mean_faithfulness() -> None:
    samples = [
        ("asthme", ["asthme bronchique"]),  # 1.0
        ("tumeur inventée ailleurs", ["asthme"]),  # < 1.0
    ]
    m = mean_faithfulness(samples)
    assert 0.0 < m < 1.0
    assert mean_faithfulness([]) == 0.0


# --- P5-4 : taux d'hallucination -------------------------------------------


def test_is_hallucinated() -> None:
    valid = {201826, 4048098}
    assert is_hallucinated(999999, valid) is True  # hors-vocabulaire
    assert is_hallucinated(201826, valid) is False  # connu/standard
    assert is_hallucinated(0, valid) is False  # abstention, pas une hallucination


def test_hallucination_rate_zero_when_all_valid() -> None:
    valid = {1, 2, 3}
    assert hallucination_rate([1, 2, 0, 3], valid) == 0.0  # le 0 est ignoré


def test_hallucination_rate_counts_only_mapped() -> None:
    valid = {1, 2}
    # 4 mappés (1,2,99,98), 2 hors-vocabulaire -> 0.5 ; les 0 ne comptent pas.
    assert hallucination_rate([1, 2, 99, 98, 0, 0], valid) == 0.5


def test_hallucination_rate_no_mapping() -> None:
    assert hallucination_rate([0, 0, 0], {1, 2}) == 0.0
    assert hallucination_rate([], {1, 2}) == 0.0


# --- P5-5 : tokens par entrée dans le rapport -------------------------------


def test_mapping_report_average_tokens() -> None:
    outcomes = [(True, True), (True, False), (False, False), (True, True)]
    report = aggregate_mapping(
        outcomes, avg_latency_ms=12.0, total_input_tokens=400, total_output_tokens=80
    )
    assert report.n == 4
    assert report.avg_input_tokens == 100.0
    assert report.avg_output_tokens == 20.0
    assert "tokens/entrée" in report.as_table()


def test_mapping_report_hides_tokens_when_offline() -> None:
    outcomes = [(True, True), (False, False)]
    report = aggregate_mapping(outcomes)  # pas de tokens (Proposer hors-ligne)
    assert report.avg_input_tokens == 0.0
    assert "tokens/entrée" not in report.as_table()
