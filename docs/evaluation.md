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

## Couche déterministe (alignement officiel) — la vraie force

Avant tout RAG, le router résout les codes couverts par l'**alignement officiel
CIM-10 FR → SNOMED** (`data/router/cim10_snomed_official.csv`, construit par
`scripts/build_official_map.py` depuis le bundle Athena). Sur les 42 886 codes
CIM-10 FR du vocabulaire :

| Couche | Codes | Part | Précision | Coût LLM |
|---|---|---|---|---|
| **Alignement officiel 1-à-1** (déterministe) | 13 651 | **31.8 %** | 100 % (par construction) | **0** |
| Résidu (rare, ambigu, non aligné) → RAG gouverné + steward | reste | 68.2 % | mesurée sur held-out | borné |

C'est le cœur de la proposition : **on n'utilise l'IA que là où elle apporte**.
Près d'un tiers des codes sont mappés **gratuitement, sans erreur, sans LLM** ;
le RAG ne travaille que sur ce qui reste.

> **Anti-fuite (held-out).** Le gold set (80 codes) est **exclu** de la map
> officielle (`--exclude-gold`). Sans ça, le déterministe recopierait les réponses
> du gold set (évaluation circulaire). Les 80 codes restent donc du **résidu** : ils
> partent au RAG, et le benchmark de retrieval ci-dessous reste un vrai test à
> l'aveugle. C'est la différence entre « mon outil a 100 % » (faux, table de lookup)
> et « ma couche officielle couvre 31.8 %, mesuré, et je teste l'IA sur le reste ».

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
| dense — `hashing` (plancher non-sémantique) | 80 | 0.113 | 0.150 | 0.188 | 0.138 |
| dense — **BioLORD** (sémantique) | 80 | 0.188 | 0.438 | 0.525 | 0.321 |
| **hybride RRF — BM25 + BioLORD** | 80 | **0.412** | **0.588** | **0.700** | **0.520** |

> Le plancher `hashing` est un **embedding par hachage** (256 dim) : sur 105 k
> concepts, les **collisions** produisent des scores ex æquo. Le Top-1 est stable
> (0.113) mais les recall@3/@5 **fluctuent légèrement** d'un run à l'autre selon le
> tie-break — c'est attendu d'un plancher non-sémantique, et sans incidence sur les
> lignes lexicales/BioLORD (déterministes).

**Lecture.** Sur un vrai corpus, les stratégies se départagent enfin, et le résultat
est instructif — ni le lexical seul, ni le sémantique seul ne gagne, c'est **leur
fusion** :

- Le **BM25** (0.300 Top-1 / 0.613 recall@5) dépasse la baseline proxy-Usagi (0.325 /
  0.487) au rappel : un lexical bien pondéré bat le simple match exact + Jaccard.
- Le **dense BioLORD seul** (0.188 Top-1) **ne bat pas** le lexical : les libellés du
  gold set sont des descriptions CIM-10 **administratives et verbeuses** (« Toxic
  effect: Paints and dyes… »), où le recouvrement de mots (BM25) prime sur la
  proximité sémantique. C'est un phénomène **connu en mapping de terminologies** — et
  le mesurer, plutôt que de supposer que « l'embedding est toujours meilleur », est
  précisément ce qui rend le protocole crédible. (Le dense double tout de même le
  plancher `hashing`, 0.113 → 0.188 : le modèle a bien chargé.)
- L'**hybride RRF (BM25 + BioLORD)** remporte tout : **0.412 Top-1** et **0.700
  recall@5** — nettement au-dessus de chaque composant pris isolément. La fusion
  récupère les cas où le lexical échoue mais le sémantique aide, et inversement. Le
  bon concept SNOMED est dans le **top-5 pour 70 %** des codes du résidu — le steward
  tranche vite.

**Coût de l'amélioration** (argument d'ingénierie honnête) : indexer les 105 324
concepts `Condition` avec BioLORD prend **~43 min sur CPU** (une passe, réutilisable
via `--reuse-index`). La qualité se paie ; l'hybride justifie ce coût, le dense seul
non.

> Honnêteté méthodologique : on **ne prétend pas** battre l'état de l'art. On publie
> un protocole reproductible, un gold set réel et des chiffres bruts — y compris le
> plancher. « On ne dit pas que c'est mieux, on le mesure. »

### Mapping de bout en bout (pipeline complet, résidu held-out)

`gor eval-map --strategy auto` sur les 80 codes du gold set (tous **held-out** de
l'alignement officiel → routés vers le RAG gouverné, retrieval **hybride**,
Proposer hors-ligne) :

| n | Top-1 (global) | couverture | précision (mappés) | latence |
|---|---|---|---|---|
| 80 | **0.412** | 1.000 | **0.412** | **198 ms/entrée** |

Deux lectures utiles. (1) La **précision de bout en bout** (0.412) reflète le
retrieval **hybride** (0.412 Top-1, meilleur que BM25 seul 0.300) suivi du choix du
Proposer — cohérent : le Proposer hors-ligne (`FakeProposerLLM`) retourne le 1er
candidat du retriever, donc le Top-1 mapping = Top-1 retrieval. (2) La **couverture
= 1.000** vient du Proposer **hors-ligne** (`FakeProposerLLM`), qui choisit
toujours un candidat et ne s'abstient jamais. Avec le **vrai** Proposer (Claude) +
le Vérificateur + le seuil de confiance, les cas douteux **s'abstiennent**
(`concept_id = 0`) : la couverture baisse et la précision-sur-mappés monte —
l'outil « sait dire je ne sais pas ». La décision finale reste **toujours** au
steward.

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
| 4 | **Embedding sémantique BioLORD** — `--embedding-backend sentence_transformers` | ✅ fait |
| 5 | **Alignement officiel enrichi** — `cim10_snomed_official.csv` depuis le bundle ATIH (13 651 paires, held-out propre) | ✅ fait |
| 6 | **Gold set enrichi steward** — `FeedbackStore.to_gold_records()` produit des entrées `gold_set.csv` | ⏳ (boucle continue) |

Le benchmark est désormais **pleinement mesuré** : le tableau ci-dessus contient les
chiffres bruts (baselines, BioLORD, hybride) sur un gold set réel et un corpus réel.
La ligne « BioLORD » a été obtenue après ~43 min d'indexation CPU ; l'hybride est
réutilisable instantanément (`--reuse-index`).

## Régression

Un test (`tests/test_eval.py`) exécute cette évaluation à chaque `pytest` et
échoue si la qualité chute sous un seuil — « chaque PR validée contre un
benchmark reproductible ». Il vérifie aussi le mapping final (`evaluate_mapping`).
