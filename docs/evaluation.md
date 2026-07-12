# Évaluation

Le projet mesure honnêtement la qualité du retrieval sur un **gold set**
reproductible plutôt que de prétendre battre l'état de l'art (cf. CONTEXT.md §7).

## Deux niveaux de mesure

**1. Qualité du retrieval** (le bon concept est-il proposé ?)

- **Top-1 accuracy** : le bon `concept_id` est-il le 1er candidat ?
- **recall@k** : le bon `concept_id` est-il dans les k premiers candidats ?
  (métrique prioritaire : maximiser le débit du steward — le bon concept est
  presque toujours proposé, l'humain tranche.)
- **MRR** (Mean Reciprocal Rank) : qualité moyenne du classement.

**2. Qualité du mapping final** (au bout du pipeline complet : router + agent)

- **Top-1 (global)** : part des entrées mappées **et** correctes.
- **couverture** : part des entrées effectivement mappées (`concept_id != 0`).
- **taux non-mappé** : l'outil sait dire « je ne sais pas » (garde-fou).
- **précision (mappés)** : exactitude parmi les seules entrées mappées.

```bash
uv run gor eval-map --bronze-dir tests/fixtures \
  --embedding-backend hashing --vector-backend memory --strategy auto
```

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
uv run gor eval ... --retriever baseline        # baseline lexicale (proxy Usagi)
```

## Benchmark vs Usagi (proxy honnête)

Usagi (OHDSI) fait du **string-matching semi-automatique** (précision indicative
~44 % sur du mapping de médicaments informels, cf. CONTEXT.md §2). Comme Usagi est
un outil Java difficile à scripter en CI, on fournit une **baseline lexicale
reproductible du même esprit** (`--retriever baseline` : match exact + Jaccard sur
nom/synonymes) pour comparer, à armes égales et partout, notre retrieval hybride.

| Stratégie | n | Top-1 |
|---|---|---|
| baseline lexicale (proxy Usagi) | 5 | 1.000 |
| dense (embeddings) | 5 | 1.000 |
| BM25 | 5 | 1.000 |
| hybride (RRF) | 5 | 1.000 |

> Sur le corpus de démonstration (4 concepts), toutes les stratégies obtiennent
> le même Top-1 : le corpus est trop petit/lexical pour les départager. L'écart
> attendu (fusion hybride + embedding **sémantique** BioLORD > baseline lexicale)
> se mesurera sur un **corpus réel** et un **gold set** conséquent.

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

## Obtenir des chiffres réels (à faire)

Pour un benchmark sérieux et citable :

1. **Corpus réel** : déposer les exports Athena (`CONCEPT.csv`,
   `CONCEPT_SYNONYM.csv`) dans `data/bronze/`, puis `gor build-corpus`.
2. **Alignement officiel** : remplacer `data/router/cim10_snomed_official.csv` par
   le vrai alignement CIM-10 FR ↔ SNOMED-CT (ATIH, publié 2×/an).
3. **Gold set étendu** (50–100) : dérivé de l'alignement officiel et **enrichi par
   les corrections réelles du steward** — `FeedbackStore.to_gold_records()`
   produit directement des entrées au format `gold_set.csv`.
4. **Embedding sémantique** : `--embedding-backend sentence_transformers` (BioLORD).

## Régression

Un test (`tests/test_eval.py`) exécute cette évaluation à chaque `pytest` et
échoue si la qualité chute sous un seuil — « chaque PR validée contre un
benchmark reproductible ». Il vérifie aussi le mapping final (`evaluate_mapping`).
