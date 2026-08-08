"""Publication du gold set comme **dataset Langfuse** (versionné, comparable).

Pourquoi
--------
Le gold set vit aujourd'hui dans un CSV et ses résultats dans un tableau de terminal :
comparer deux versions du pipeline revient à comparer deux copies d'écran. Publié comme
dataset, il devient le référentiel partagé auquel des *runs* successifs peuvent être
rattachés — avec, pour chaque cas, la trace de son exécution.

Idempotence
-----------
Chaque item reçoit un identifiant **dérivé de son contenu** (empreinte SHA-256 tronquée).
Republier le même gold set mettra donc à jour les items existants au lieu de les
dupliquer ; seules les lignes réellement modifiées changent d'identifiant.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from governed_omop_rag.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - imports de typage uniquement
    from collections.abc import Sequence

    from governed_omop_rag.eval.gold_set import GoldItem

logger = get_logger(__name__)

DEFAULT_DATASET_NAME = "gold-set-atih-conditions"


@dataclass(frozen=True)
class PushReport:
    """Résultat d'une publication de dataset."""

    dataset_name: str
    items: int
    dry_run: bool

    def as_table(self) -> str:
        mode = "SIMULATION (aucun envoi)" if self.dry_run else "publié"
        return f"dataset : {self.dataset_name}\nitems   : {self.items}\nstatut  : {mode}"


def item_id(item: GoldItem) -> str:
    """Identifiant stable dérivé du contenu — garantit l'idempotence des republications."""
    empreinte = f"{item.source_code or ''}|{item.source_label or ''}|{item.expected_concept_id}"
    return hashlib.sha256(empreinte.encode("utf-8")).hexdigest()[:32]


def to_payload(item: GoldItem) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convertit un `GoldItem` en couple (entrée, sortie attendue) pour Langfuse."""
    entree = {"source_code": item.source_code, "source_label": item.source_label}
    attendu = {"expected_concept_id": item.expected_concept_id}
    return entree, attendu


def push_gold_set(
    items: Sequence[GoldItem],
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    description: str | None = None,
    source_file: str | None = None,
    dry_run: bool = False,
) -> PushReport:
    """Publie le gold set comme dataset Langfuse.

    ``dry_run`` valide la transformation sans aucun appel réseau : utile pour vérifier
    le contenu avant publication, et pour tester la logique sans clés.

    Lève ``RuntimeError`` si le client Langfuse est indisponible (SDK absent ou clés
    manquantes) : contrairement au tracing, une publication demandée explicitement ne
    doit pas échouer en silence.
    """
    if dry_run:
        logger.info("dataset_dry_run", dataset=dataset_name, items=len(items))
        return PushReport(dataset_name=dataset_name, items=len(items), dry_run=True)

    from governed_omop_rag.observability.tracing import get_client

    client = get_client()
    if client is None:
        raise RuntimeError(
            "Client Langfuse indisponible : vérifier l'extra `observability` "
            "(uv sync --extra observability) et les clés GOR_LANGFUSE_PUBLIC_KEY / "
            "GOR_LANGFUSE_SECRET_KEY."
        )

    client.create_dataset(
        name=dataset_name,
        description=description
        or "Gold set ATIH (résidu held-out) : libellés CIM-10 FR et concept OHDSI attendu.",
        metadata={"source_file": source_file, "items": len(items)},
    )

    for index, item in enumerate(items):
        entree, attendu = to_payload(item)
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=item_id(item),
            input=entree,
            expected_output=attendu,
            metadata={"source_file": source_file, "index": index},
        )

    client.flush()
    logger.info("dataset_publie", dataset=dataset_name, items=len(items))
    return PushReport(dataset_name=dataset_name, items=len(items), dry_run=False)


# --------------------------------------------------------------------------- #
# Exécution d'un run évalué sur le dataset
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Score:
    """Un score, indépendant du SDK — cœur testable sans dépendance."""

    name: str
    value: float | str
    data_type: str
    comment: str | None = None


def score_mapping(output: Any, expected_output: Any) -> list[Score]:
    """Calcule les scores d'un item : correction du top-1, couverture, voie de résolution.

    Logique **pure** : ni réseau, ni SDK. C'est elle que couvrent les tests, et elle
    reste exécutable dans une CI qui n'installe pas l'extra ``observability``.

    La voie (`router` déterministe vs `rag`) est un score catégoriel : c'est elle qui
    montre, run après run, quelle part des cas est résolue **sans** appel LLM — la borne
    de coût, observée plutôt qu'affirmée.
    """
    attendu = (expected_output or {}).get("expected_concept_id")
    obtenu = (output or {}).get("target_concept_id")
    mappe = bool((output or {}).get("is_mapped"))
    source = str((output or {}).get("source") or "inconnu")

    # Une abstention n'est jamais un top-1, même si l'identifiant coïncide.
    correct = 1.0 if (mappe and obtenu == attendu) else 0.0
    return [
        Score("top1", correct, "NUMERIC", f"attendu {attendu}, obtenu {obtenu}"),
        Score("mapped", 1.0 if mappe else 0.0, "NUMERIC"),
        Score("source", source, "CATEGORICAL"),
    ]


def evaluate_top1(
    *, input: Any = None, output: Any = None, expected_output: Any = None, **_: Any
) -> list[Any]:
    """Adaptateur vers le type ``Evaluation`` du SDK (consommé par ``run_experiment``).

    Fine couche au-dessus de :func:`score_mapping` : le SDK lit ses évaluations par
    accès attribut, on ne peut donc pas lui passer de simples dictionnaires. L'import
    reste paresseux — cette fonction n'est appelée que pendant un run, qui exige de
    toute façon le SDK et un client.
    """
    from langfuse import Evaluation

    return [
        Evaluation(
            name=s.name,
            value=s.value,
            data_type=s.data_type,  # type: ignore[arg-type]
            comment=s.comment,
        )
        for s in score_mapping(output, expected_output)
    ]


def run_gold_set_experiment(
    *,
    dataset_name: str,
    route: Any,
    run_name: str | None = None,
    description: str | None = None,
    limit: int | None = None,
    concurrency: int = 1,
) -> Any:
    """Exécute le pipeline sur les items du dataset et enregistre un run évalué.

    ``concurrency`` vaut **1 par défaut**, à l'inverse du défaut 50 du SDK : des appels
    LLM parallèles déclencheraient le rate limit et rendraient le coût difficile à
    borner. Au-delà de 1, les compteurs de tokens du Proposer (incréments non atomiques)
    peuvent légèrement sous-compter.
    """
    from governed_omop_rag.core.models import MappingRequest
    from governed_omop_rag.observability.tracing import get_client

    client = get_client()
    if client is None:
        raise RuntimeError(
            "Client Langfuse indisponible : vérifier l'extra `observability` et les clés."
        )

    dataset = client.get_dataset(dataset_name)
    items = list(dataset.items)
    if limit is not None:
        items = items[:limit]

    def task(*, item: Any, **_: Any) -> dict[str, Any]:
        entree = item.input or {}
        suggestion = route(
            MappingRequest(
                source_code=entree.get("source_code"),
                source_label=entree.get("source_label"),
            )
        )
        return {
            "target_concept_id": suggestion.target_concept_id,
            "is_mapped": suggestion.is_mapped,
            "source": suggestion.source.value,
            "confidence": suggestion.confidence,
            "justification": suggestion.justification,
        }

    result = client.run_experiment(
        name=run_name or f"gold-set — {len(items)} cas",
        description=description or "Évaluation du pipeline sur le gold set ATIH.",
        data=items,
        task=task,
        evaluators=[evaluate_top1],
        max_concurrency=concurrency,
    )
    client.flush()
    logger.info("dataset_run_termine", dataset=dataset_name, items=len(items))
    return result
