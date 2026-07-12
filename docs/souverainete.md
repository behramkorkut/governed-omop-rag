# Souveraineté

Le projet vise l'**interopérabilité des données de santé françaises**. Les choix
techniques sont pensés pour un déploiement **souverain**.

## Embeddings calculés en local

Les embeddings biomédicaux (**BioLORD** / sentence-transformers) sont calculés
**en local** (`retrieval/embeddings.py`). Aucun texte n'est envoyé à un tiers pour
la vectorisation. C'est un choix de qualité (embeddings spécialisés) **et** de
souveraineté.

## Base vectorielle

- **Décision par défaut : Qdrant** — base vectorielle **européenne** (allemande),
  auto-hébergeable, avec recherche hybride native.
- Isolée derrière l'interface `VectorStore` (`retrieval/vectorstore.py`) : on peut
  changer de backend (ou passer en `memory` pour les tests) **sans toucher au
  reste du code**.
- **Ce qui est stocké** : uniquement des **vecteurs de vocabulaire de référence
  public** (noms de concepts OHDSI). Aucune donnée patient. Un service managé
  serait donc acceptable, mais le défaut auto-hébergeable garde la maîtrise
  complète.

## LLM des agents

Le Proposer utilise Claude (API). C'est le **seul** appel à un service tiers, et
il est **borné** : la stratégie hybride n'envoie au LLM que le **résidu** (les
libellés non couverts par l'alignement officiel), et un **cache** évite de
recalculer. Un **Proposer déterministe hors-ligne** (`FakeProposerLLM`) permet de
faire tourner tout le pipeline **sans aucun appel réseau** (démo, tests, dev).

## Déploiement pleinement souverain

Pour un déploiement 100 % souverain : Qdrant auto-hébergé (défaut), embeddings
locaux (défaut), et — si l'usage d'un LLM tiers n'est pas souhaité — remplacer le
Proposer par un modèle **local** derrière le même protocole `ProposerLLM`. Le
reste (retrieval, router, garde-fous, UI) est déjà local.
