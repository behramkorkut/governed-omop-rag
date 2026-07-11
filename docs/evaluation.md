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

# comparer les stratégies de retrieval sur le gold set
uv run gor eval --bronze-dir tests/fixtures --embedding-backend hashing \
  --vector-backend memory --retriever dense    # dense seul
uv run gor eval ... --retriever bm25            # lexical BM25 seul
uv run gor eval ... --retriever hybrid          # fusion RRF (BM25 + dense)
```

Le retrieval hybride (`--retriever hybrid`) fusionne BM25 (lexical) et dense
(embeddings) par **Reciprocal Rank Fusion**. Sur le corpus de **démonstration**
(4 concepts, requêtes lexicalement proches), les trois stratégies obtiennent le
même Top-1 : le corpus est trop petit et trop « lexical » pour les départager.
L'intérêt de la fusion apparaît sur un **corpus réel** (recouvrement lexical
partiel, synonymie, fautes) et surtout avec l'embedding **sémantique** BioLORD.

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
