# governed-omop-rag

> RAG agentique **gouverné** pour le mapping de terminologies FR
> (**CIM-10 FR**, libellés cliniques) vers les **concepts standard OHDSI** (OMOP CDM),
> sous **supervision humaine** (human-in-the-loop).

[![CI](https://github.com/OWNER/governed-omop-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/governed-omop-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> **Statut : Phase 0 (scaffold).** Le produit se construit phase par phase
> (cf. [`feature_list.json`](feature_list.json) et [`CONTEXT.md`](CONTEXT.md)).

---

## Le problème

Un Entrepôt de Données de Santé français reçoit des codes en nomenclatures
locales (CIM-10 FR / ATIH, CCAM, NABM…) ou en texte libre. Pour faire de la
recherche reproductible, il faut les **traduire vers les vocabulaires standard
OHDSI** (SNOMED-CT, RxNorm, LOINC…). Ce mapping est aujourd'hui **manuel et
coûteux** ; l'outil de référence (Usagi) fait du string-matching limité.

## L'approche

Un pipeline **hybride** qui n'utilise l'IA que là où elle apporte :

1. **Match déterministe d'abord** — via l'alignement officiel CIM-10 ↔ SNOMED-CT
   (100 % fiable, gratuit, instantané sur les codes couverts).
2. **RAG agentique sur le résidu** — retrieval hybride (BM25 + embeddings
   biomédicaux **BioLORD** locaux) → agent **Proposer** (Claude) →
   sous-agent **Vérificateur** (règles OMOP strictes) → suggestion tracée.
3. **Human-in-the-loop** — l'outil **propose**, un **data steward valide**.
   Sortie fermée (anti-hallucination structurel), traçabilité, conformité IA Act.

Livrable : une table **`source_to_concept_map`** validée, au format OMOP.

## Décisions techniques

| Sujet | Choix |
|---|---|
| Entrée | CIM-10 FR / libellés FR → OMOP standard |
| Embeddings | BioLORD / sentence-transformers, **en local** |
| Base vectorielle | **Qdrant** (souverain, européen) derrière une interface `VectorStore` |
| Orchestration agentique | **LangGraph** (Proposer + Vérificateur) |
| LLM | Claude |
| Data | Corpus **médaillon Bronze → Silver → Gold** (DuckDB) |
| Outillage | Python 3.11, uv, ruff, mypy, pytest, Docker, GitHub Actions |
| Données | 100 % publiques / synthétiques (zéro RGPD) |

## Démarrage rapide

```bash
# 1. Cloner puis initialiser (installe uv si absent, sync, smoke-test, tests)
./init.sh

# 2. Vérifier que le CLI répond
uv run gor info
uv run gor smoke

# 3. Lancer la suite de tests
uv run pytest
```

Copier `.env.example` en `.env` pour surcharger la configuration (clé Anthropic,
URL Qdrant, modèle d'embeddings…). Toutes les variables sont préfixées `GOR_`.

### Qdrant local (dev)

```bash
docker compose up -d qdrant   # base vectorielle sur http://localhost:6333
```

### Démo vs production

Les commandes `gor search` / `gor map` / `gor eval` **reconstruisent le corpus en
mémoire à chaque appel** — pratique pour la démo « essayez en 2 minutes », mais
non-optimal. Le workflow **production** visé (couches à venir) est :

1. `gor build-corpus` une fois → persiste le Gold en DuckDB ;
2. indexation une fois → upsert des vecteurs dans Qdrant (persistant) ;
3. `gor map` lit l'index Qdrant existant, sans rien reconstruire.

Backends **hors-ligne** (`--embedding-backend hashing --vector-backend memory`)
pour tester sans Docker ni téléchargement ; backends **réels** (BioLORD + Qdrant)
en production.

## Structure

```
src/governed_omop_rag/
├── config.py        # configuration typée (pydantic-settings)
├── cli.py           # CLI `gor` (version / info / smoke)
├── core/            # schémas domaine + logging/traçabilité
├── medallion/       # corpus Bronze → Silver → Gold (DuckDB)
├── retrieval/       # recherche hybride + embeddings + VectorStore (Qdrant)
├── router/          # match officiel déterministe puis RAG sur le résidu
├── agents/          # orchestration LangGraph : Proposer + Vérificateur
├── api/             # FastAPI (/map, /map/batch)
├── ui/              # Streamlit (revue steward)
└── eval/            # métriques : Top-k, faithfulness, hallucination, coût
```

## Feuille de route

Voir [`feature_list.json`](feature_list.json). Phases : **0** scaffold → **1** corpus
médaillon + embeddings + Router v1 → **2** retrieval avancé + gold set → **3**
multi-agent LangGraph → **4** API + UI steward → **5** évaluation + benchmark
Usagi → **6** MLOps + démo publique.

## Documentation

- [Guide utilisateur (non-technique, FR)](docs/guide_utilisateur.md) — « essayez en 2 minutes ».
- [Évaluation](docs/evaluation.md) — métriques, gold set, benchmark.

## Gouvernance & conformité

Outil d'**aide à la décision** avec validation humaine. Données publiques/
synthétiques, sortie contrainte au vocabulaire réel, traçabilité complète,
**aucune décision clinique automatisée** (cadrage IA Act). Voir `docs/` (à venir).

## Licence

MIT.
