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

**3. Coût & latence** (observabilité §7)

`gor eval-map` reporte la **latence moyenne par entrée** (mesurée partout) et les
**tokens moyens par entrée** (input/output) quand le Proposer est Claude — ligne
`tokens/entrée` du rapport. Hors-ligne (Proposer déterministe), les tokens valent 0
et la ligne est masquée ; c'est avec la clé Anthropic (`uv sync --extra agents`) que
le coût réel apparaît. La stratégie hybride **borne ce coût** : le LLM ne voit que le
résidu, pas les cas couverts par l'alignement officiel.

## Gold set réel (ATIH)

`data/eval/gold_set_atih.csv` — **80 mappings CIM-10 FR → SNOMED**, format
`source_code,source_label,expected_concept_id`. Construit automatiquement
(`scripts/build_gold_set.py`) à partir des vocabulaires OHDSI (Athena) :

- **source** = vocabulaire **CIM10** (édition française réelle de l'ATIH, `vocabulary_id='CIM10'`) ;
- **cible** = concept **SNOMED standard** (`standard_concept='S'`, valide) ;
- via la relation officielle **« Maps to »** (19 191 paires CIM10→SNOMED dans le bundle) ;
- **uniquement les mappings 1-à-1** : un code source qui mappe vers exactement un
  concept standard (les ambiguïtés 1-à-N sont exclues = vérité terrain nette).

Un gold set de démonstration minimal (`data/eval/gold_set.csv`, 5 entrées) reste
utilisé par le test de régression `tests/test_eval.py`.

> **Deux limites assumées.** (1) Les `concept_name` de la CIM10 dans OHDSI sont en
> **anglais** ; le test réel porte sur le **code** CIM-10 FR (ex. `J44.1`), pas sur
> le libellé. (2) On **restreint le corpus au domaine `Condition`** (~105 k concepts
> SNOMED standard sur 831 k) : c'est cohérent avec le gold set (mapper un diagnostic
> → candidats conditions) et rend l'évaluation tractable.

## Reproduire

```bash
docker compose up -d qdrant   # base vectorielle (pour les backends dense/qdrant)

# Baselines lexicales (rapides, sans embedding réel) :
uv run gor eval --gold-path data/eval/gold_set_atih.csv --bronze-dir bundle/athena_bundle \
  --domain Condition --embedding-backend hashing --vector-backend memory --retriever baseline
uv run gor eval ... --retriever bm25            # BM25 lexical seul

# Retrieval sémantique (BioLORD) — LOURD (~10^5 concepts embarqués une fois) :
GOR_QDRANT_COLLECTION=ohdsi_biolord uv run gor eval \
  --gold-path data/eval/gold_set_atih.csv --bronze-dir bundle/athena_bundle --domain Condition \
  --embedding-backend sentence_transformers --vector-backend qdrant --retriever dense
# puis, sans ré-embarquer le corpus :
GOR_QDRANT_COLLECTION=ohdsi_biolord uv run gor eval ... --retriever hybrid --reuse-index
```

## Benchmark vs Usagi (proxy honnête)

Usagi (OHDSI) fait du **string-matching semi-automatique** (précision indicative
~44 % sur du mapping de médicaments informels, cf. CONTEXT.md §2). Comme Usagi est
un outil Java difficile à scripter en CI, on fournit une **baseline lexicale
reproductible du même esprit** (`--retriever baseline` : match exact + Jaccard sur
nom/synonymes) pour comparer, à armes égales et partout, notre retrieval.

**Gold set ATIH (80 conditions CIM-10 FR → SNOMED), corpus Condition (~105 k) :**

| Stratégie | n | Top-1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| baseline lexicale (proxy Usagi) | 80 | 0.325 | 0.438 | 0.487 | 0.380 |
| BM25 (lexical) | 80 | 0.300 | 0.562 | 0.613 | 0.433 |
| dense — `hashing` (plancher non-sémantique) | 80 | 0.113 | 0.212 | 0.237 | 0.160 |
| dense — **BioLORD** (sémantique) | 80 | _à compléter_ | _—_ | _—_ | _—_ |
| hybride RRF — **BioLORD** | 80 | _à compléter_ | _—_ | _—_ | _—_ |

**Lecture.** Sur un vrai corpus, les stratégies se départagent enfin. Le **BM25**
dépasse déjà la baseline proxy-Usagi au rappel (recall@5 : **0.613 vs 0.487**) : un
lexical bien pondéré (TF-IDF/BM25) bat le simple match+Jaccard. La ligne `hashing`
est un **plancher** : embedding purement lexical par hachage, non sémantique — il
n'est là que pour valider la chaîne. Les lignes **BioLORD** (embedding biomédical)
et **hybride** (fusion BM25 + dense par Reciprocal Rank Fusion) sont laissées à
compléter : leur intérêt — synonymie, reformulations, « glycémie élevée » ↔ diabète
— ne se mesure qu'avec l'embedding sémantique, dont le calcul sur ~105 k concepts
est coûteux (une passe de plusieurs dizaines de minutes sur CPU).

> Honnêteté méthodologique : on **ne prétend pas** battre l'état de l'art. On publie
> un protocole reproductible, un gold set réel et des chiffres bruts — y compris le
> plancher. « On ne dit pas que c'est mieux, on le mesure. »

## Fidélité & hallucination (garde-fous mesurés)

Au-delà de « le bon concept est-il trouvé ? », on mesure **la gouvernance elle-même**
(`governed_omop_rag.eval.quality`, fonctions pures, testées dans `tests/test_quality.py`).

**Faithfulness** (P5-3, style RAGAS léger) — `faithfulness_score(justification, candidats)` :
part des mots de contenu de la justification du Proposer qui apparaissent
effectivement dans les **candidats fournis**. Un score < 1 signale une justification
qui s'appuie sur du contexte **externe** (risque d'hallucination de raisonnement).
Mesure déterministe, sans appel LLM supplémentaire.

**Taux d'hallucination** (P5-4) — `hallucination_rate(concepts_proposés, vocabulaire_valide)` :
part des `target_concept_id` proposés qui sont **hors-vocabulaire ou non-standard**.
Le point clé : `concept_id = 0` (abstention, « je ne sais pas ») **n'est pas** une
hallucination. Grâce au **garde-fou de sortie fermée** (`ClosedOutputViolation` : le
Proposer ne peut choisir qu'un `concept_id` réellement présent dans les candidats),
ce taux est **structurellement ≈ 0**. La métrique le **vérifie**, elle ne le suppose
pas — c'est la différence entre « on a mis un garde-fou » et « on prouve qu'il tient ».

## Obtenir des chiffres réels — statut

| # | Étape | Statut |
|---|---|---|
| 1 | **Corpus réel** — bundle Athena → `gor build-corpus` (831 k concepts, ~105 k Condition) | ✅ fait |
| 2 | **Gold set réel** — 80 mappings CIM-10 FR → SNOMED (1-à-1, `scripts/build_gold_set.py`) | ✅ fait |
| 3 | **Baselines chiffrées** — proxy Usagi + BM25 sur le gold set réel | ✅ fait |
| 4 | **Embedding sémantique BioLORD** — `--embedding-backend sentence_transformers` | ⏳ à lancer (coûteux) |
| 5 | **Alignement officiel enrichi** — remplacer `cim10_snomed_official.csv` par l'alignement ATIH complet | ⏳ |
| 6 | **Gold set enrichi steward** — `FeedbackStore.to_gold_records()` produit des entrées `gold_set.csv` | ⏳ (boucle continue) |

Il ne reste, pour un benchmark pleinement citable, qu'à lancer la passe **BioLORD**
(étape 4) et à reporter ses deux lignes dans le tableau ci-dessus.

## Régression

Un test (`tests/test_eval.py`) exécute cette évaluation à chaque `pytest` et
échoue si la qualité chute sous un seuil — « chaque PR validée contre un
benchmark reproductible ». Il vérifie aussi le mapping final (`evaluate_mapping`).
