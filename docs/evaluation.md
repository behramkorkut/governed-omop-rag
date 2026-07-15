# Évaluation

Le projet mesure honnêtement la qualité du retrieval sur un **gold set**
réproductible plutôt que de prétendre battre l'état de l'art (cf. CONTEXT.md §7).

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
l'alignement officiel → routés vers le RAG gouverné, retrieval **hybride BioLORD**),
selon le Proposer :

| Proposer | Top-1 (global) | couverture | précision (mappés) | latence | tokens/entrée |
|---|---|---|---|---|---|
| hors-ligne (`FakeProposerLLM`) | 0.412 | 1.000 | 0.412 | 198 ms | — |
| **Claude Sonnet 5** | **0.650** | 1.000 | **0.650** | 2718 ms | 1507 in / 65 out |

**C'est le résultat central du projet.** Le Proposer LLM ne se contente pas de
prendre le 1er candidat : il **choisit dans le top-k** le bon concept. Résultat, le
mapping passe de **0.412 → 0.650 Top-1** (+23.8 points) sur le résidu le plus
difficile — la valeur ajoutée du « l'IA propose » est **mesurée**, pas supposée. Le
retrieval hybride met le bon concept dans le top-5 (recall@5 0.700) ; le jugement de
Claude le remonte souvent en tête.

Lectures complémentaires. (1) **C'est du RAG authentique** : le prompt envoyé à
Claude ne contient que le **libellé source** (description anglaise CIM-10, ex.
« Chronic obstructive pulmonary disease with acute exacerbation »), **pas le code**
(`J44.1`). La sortie fermée (`ClosedOutputViolation`) l'empêche de choisir hors de la
liste des candidats retrouvés. Le 65 % n'est pas de la mémorisation — c'est Claude qui
juge lequel des 5 candidats est le plus pertinent. (2) **Coût borné** : ~1507 tokens
in / 65 out par entrée, soit **≈ 0,005 $/entrée** aux tarifs Sonnet publics (à vérifier
sur la facturation) — et le LLM ne voit **que le résidu**, jamais les 31.8 % couverts
par l'alignement officiel. (3) **Latence** : 198 → 2718 ms/entrée (l'appel réseau au
LLM domine) — un coût assumé pour du mapping par lots supervisé, pas du temps réel.
(4) **Couverture = 1.000** : sur ces 80 conditions, chaque candidat du retriever est
un concept SNOMED standard valide, donc le Vérificateur les accepte et l'agent ne
s'abstient jamais. L'abstention (`concept_id = 0`) reste possible par construction
(aucun candidat, ou violation de sortie fermée) ; un **seuil de confiance sur la voie
agent** pour provoquer l'abstention sur cas douteux est une amélioration identifiée.
La décision finale reste **toujours** au steward.

### ⚠️ Bug identifié : couverture artificiellement à 100 % (voie agent)

**Constat.** Quand l'agent Proposer est branché (`--agent`), le `HybridRouter`
court-circuite le seuil de confiance (`hybrid.py:73-74`) : l'agent décide sur tous
les résidus, même les plus douteux. Le Vérificateur ne vérifie que les règles OMOP
(standard='S', domaine) — tous les candidats du retriever étant des SNOMED standard,
il passe toujours → **aucune abstention**.

**Impact.** La couverture = 1.000 est artificielle. En production, on veut que
l'outil dise « je ne sais pas » quand le match est ambigu — sinon le steward valide
mécaniquement et la précision apparente cache du bruit.

**Piste de correction.** Un signal d'abstention adapté à la voie agent :

| Signal | Description | Seuil |
|---|---|---|
| **Marge top-1 / top-2** | Si le score RRF du 1er est très proche du 2ème → ambigu | marge < 0.01 |
| **Rang du candidat choisi** | Si Claude choisit le 4ème ou 5ème → peu confiant | rang ≥ 3 |
| **Score cosinus dense** | Score BioLORD du candidat choisi (indépendant du RRF) | < 0.3 |

Implémentation : dans `MappingAgent.run()`, après le choix de Claude, évaluer le
signal. Si le signal est sous le seuil → retourner `concept_id=0` + `NoMapReason.AMBIGU`,
les candidats restant exposés au steward. La métrique F1 (couverture × précision)
permettra de calibrer le seuil.

> C'est une amélioration planifiée, pas un blocage. Le chiffre 0.650 reste valide :
>c'est la précision **quand l'agent propose** (ce qu'il fait aujourd'hui sur 100 %
>des résidus). Avec le fix, la couverture baissera (ex. 0.85) et la
>précision-sur-mappés montera (ex. 0.75) — l'outil deviendra plus **utilisable**
>en production.

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
