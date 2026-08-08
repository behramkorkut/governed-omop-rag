"""Tests de la publication du gold set comme dataset Langfuse.

Hors-ligne : aucune clé, aucun appel réseau. Le client est remplacé par un double.
"""

from __future__ import annotations

from typing import Any

import pytest

from governed_omop_rag.eval.gold_set import GoldItem
from governed_omop_rag.observability import datasets


def _items() -> list[GoldItem]:
    return [
        GoldItem(source_code="E11.9", source_label="diabète de type 2", expected_concept_id=201826),
        GoldItem(source_code=None, source_label="asthme", expected_concept_id=4048098),
    ]


class _ClientDouble:
    """Double du client Langfuse : enregistre les appels au lieu de les émettre."""

    def __init__(self) -> None:
        self.datasets: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []
        self.flushed = False

    def create_dataset(self, **kwargs: Any) -> None:
        self.datasets.append(kwargs)

    def create_dataset_item(self, **kwargs: Any) -> None:
        self.items.append(kwargs)

    def flush(self) -> None:
        self.flushed = True


def test_identifiant_stable_et_discriminant() -> None:
    """Même contenu -> même id (idempotence) ; contenu différent -> id différent."""
    a, b = _items()
    assert datasets.item_id(a) == datasets.item_id(a)
    assert datasets.item_id(a) != datasets.item_id(b)
    # Une modification du concept attendu doit changer l'identifiant.
    modifie = GoldItem(
        source_code=a.source_code, source_label=a.source_label, expected_concept_id=999
    )
    assert datasets.item_id(a) != datasets.item_id(modifie)


def test_payload_separe_entree_et_attendu() -> None:
    entree, attendu = datasets.to_payload(_items()[0])
    assert entree == {"source_code": "E11.9", "source_label": "diabète de type 2"}
    assert attendu == {"expected_concept_id": 201826}


def test_dry_run_n_appelle_pas_le_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """La simulation ne doit toucher ni le réseau ni le client."""

    def _interdit() -> Any:  # pragma: no cover - ne doit jamais être appelé
        raise AssertionError("get_client ne doit pas être appelé en dry-run")

    monkeypatch.setattr("governed_omop_rag.observability.tracing.get_client", _interdit)
    report = datasets.push_gold_set(_items(), dry_run=True)
    assert report.dry_run is True
    assert report.items == 2
    assert "SIMULATION" in report.as_table()


def test_publication_cree_dataset_et_items(monkeypatch: pytest.MonkeyPatch) -> None:
    double = _ClientDouble()
    monkeypatch.setattr("governed_omop_rag.observability.tracing.get_client", lambda: double)

    report = datasets.push_gold_set(
        _items(), dataset_name="jeu-test", source_file="gold.csv", dry_run=False
    )

    assert report.items == 2
    assert report.dry_run is False
    assert len(double.datasets) == 1
    assert double.datasets[0]["name"] == "jeu-test"
    assert len(double.items) == 2
    assert double.flushed is True

    premier = double.items[0]
    assert premier["dataset_name"] == "jeu-test"
    assert premier["input"]["source_label"] == "diabète de type 2"
    assert premier["expected_output"]["expected_concept_id"] == 201826
    assert premier["metadata"]["index"] == 0
    # Identifiants stables et distincts entre items.
    assert {i["id"] for i in double.items} == {datasets.item_id(i) for i in _items()}


def test_client_indisponible_leve_une_erreur_explicite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Une publication demandée explicitement ne doit pas échouer en silence."""
    monkeypatch.setattr("governed_omop_rag.observability.tracing.get_client", lambda: None)
    with pytest.raises(RuntimeError, match="Client Langfuse indisponible"):
        datasets.push_gold_set(_items(), dry_run=False)


# --------------------------------------------------------------------------- #
# Évaluateurs du run
# --------------------------------------------------------------------------- #


def _scores(scores: list[Any]) -> dict[str, Any]:
    return {s.name: s.value for s in scores}


def test_evaluateur_top1_correct() -> None:
    s = _scores(
        datasets.score_mapping(
            output={"target_concept_id": 201826, "is_mapped": True, "source": "router"},
            expected_output={"expected_concept_id": 201826},
        )
    )
    assert s["top1"] == 1.0
    assert s["mapped"] == 1.0
    assert s["source"] == "router"


def test_evaluateur_top1_incorrect() -> None:
    s = _scores(
        datasets.score_mapping(
            output={"target_concept_id": 999, "is_mapped": True, "source": "rag"},
            expected_output={"expected_concept_id": 201826},
        )
    )
    assert s["top1"] == 0.0
    assert s["mapped"] == 1.0
    assert s["source"] == "rag"


def test_evaluateur_non_mappe_ne_compte_pas_comme_correct() -> None:
    """Une abstention n'est jamais un top-1, même si l'identifiant coïncide."""
    s = _scores(
        datasets.score_mapping(
            output={"target_concept_id": 201826, "is_mapped": False, "source": "unmapped"},
            expected_output={"expected_concept_id": 201826},
        )
    )
    assert s["top1"] == 0.0
    assert s["mapped"] == 0.0


def test_evaluateur_tolere_des_sorties_vides() -> None:
    """Sortie absente (tâche en erreur) : score nul, pas d'exception."""
    s = _scores(datasets.score_mapping(output=None, expected_output=None))
    assert s["top1"] == 0.0
    assert s["mapped"] == 0.0
    assert s["source"] == "inconnu"


def test_run_sans_client_leve_une_erreur(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("governed_omop_rag.observability.tracing.get_client", lambda: None)
    with pytest.raises(RuntimeError, match="Client Langfuse indisponible"):
        datasets.run_gold_set_experiment(dataset_name="x", route=lambda _r: None)


def test_adaptateur_sdk_si_langfuse_installe() -> None:
    """L'adaptateur produit de vrais objets Evaluation — ignoré si le SDK est absent.

    La CI n'installe pas l'extra `observability` : ce test y est *skipped*, tandis que
    la logique pure (`score_mapping`) reste couverte partout.
    """
    pytest.importorskip("langfuse", reason="extra observability non installé")

    evals = datasets.evaluate_top1(
        output={"target_concept_id": 201826, "is_mapped": True, "source": "rag"},
        expected_output={"expected_concept_id": 201826},
    )
    assert [e.name for e in evals] == ["top1", "mapped", "source"]
    assert evals[0].value == 1.0
    assert "attendu 201826" in evals[0].comment
