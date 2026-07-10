# Évaluation

Le projet mesure honnêtement la qualité du retrieval sur un **gold set**
reproductible plutôt que de prétendre battre l'état de l'art (cf. CONTEXT.md §7).

## Métriques

- **Top-1 accuracy** : le bon `concept_id` est-il le 1er candidat ?
- **recall@k** : le bon `concept_id` est-il dans les k premiers candidats ?
  (c'est la métrique prioritaire : maximiser le débit du steward — le bon concept
  est presque toujours proposé, l'humain tranche.)
- **MRR** (Mean Reciprocal Rank) : qualité moyenne du classement.

## Gold set

`data/eval/gold_set.csv` — format `source_code,source_label,expected_concept_id`.
Vérité terrain à dériver de l'alignement officiel CIM-10 ↔ SNOMED-CT (ATIH) et/ou
d'annotations manuelles. Le jeu livré est un **échantillon** aligné sur le corpus
de démonstration (`tests/fixtures`) ; le remplacer par un gold set réel (50–100
mappings) pour un benchmark sérieux.

## Reproduire

```bash
# baseline hors-ligne (embedding lexical déterministe)
uv run gor eval --bronze-dir tests/fixtures \
  --embedding-backend hashing --vector-backend memory
```

## Résultats (baseline)

Backend `hashing` (lexical, hors-ligne) sur le gold set de démonstration
(5 requêtes, corpus de 4 concepts) :

| Backend | n | Top-1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| hashing (lexical) | 5 | 1.000 | 1.000 | 1.000 | 1.000 |

> ⚠️ Corpus de démonstration minuscule : ces chiffres valident la **tuyauterie**
> d'évaluation, pas la performance réelle. Le backend `hashing` est purement
> **lexical** ; le backend `sentence_transformers` (BioLORD) capturera la
> **sémantique** (ex. « glycémie élevée » ↔ diabète) et sera évalué sur un gold
> set réel. Un **benchmark comparatif vs Usagi** est prévu en Phase 5.

## Régression

Un test (`tests/test_eval.py`) exécute cette évaluation à chaque `pytest` et
échoue si la qualité chute sous un seuil — « chaque PR validée contre un
benchmark reproductible ».
