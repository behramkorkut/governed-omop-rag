# Guide utilisateur

Ce guide s'adresse aux **utilisateurs non-techniques** : data steward d'un
entrepôt de données de santé, médecin, chercheur en épidémiologie. Aucune ligne
de code à écrire.

## À quoi sert cet outil ?

Il **traduit** des codes ou des libellés médicaux français (par exemple le code
CIM-10 FR « E11.9 » ou le texte « diabète de type 2 ») vers les **concepts
standard OMOP** utilisés pour la recherche (SNOMED-CT, RxNorm, LOINC…).

L'outil **propose**, avec un score de confiance et une justification ; **vous
décidez**. Rien n'est écrit sans votre validation. Les données utilisées pour la
démonstration sont **publiques et synthétiques**.

## Essayez en 2 minutes

1. Installez l'application (une seule fois) :

   ```bash
   uv sync --extra ui --extra agents
   ```

2. Lancez l'écran de revue :

   ```bash
   GOR_EMBEDDING_BACKEND=hashing GOR_VECTOR_BACKEND=memory uv run gor ui
   ```

   Votre navigateur ouvre `http://localhost:8501`. Le répertoire de données se
   règle dans la barre latérale (par défaut `tests/fixtures` pour la démo).

3. Dans l'onglet **« Libellés (texte) »**, collez par exemple :

   ```
   diabète de type 2
   asthme
   hypertension artérielle
   ```

4. Cliquez **« Mapper »**. L'outil affiche une suggestion par ligne.
5. Pour chaque suggestion : dépliez-la, lisez la justification, puis **acceptez**,
   **corrigez** (choisissez un autre candidat) ou **rejetez**.
6. Cliquez **« Exporter source_to_concept_map (CSV) »** pour récupérer le résultat,
   et **« Enregistrer le feedback »** pour conserver vos décisions.

> Vous pouvez aussi importer un **fichier CSV/Excel** (onglet « Fichier CSV/Excel »).
> Un exemple prêt à l'emploi est fourni : `data/examples/exemple_entrees.csv`.

## Format d'entrée

Un fichier CSV ou Excel avec, au choix, les colonnes suivantes (au moins un code
ou un libellé par ligne) :

| Colonne | Description | Exemple |
|---|---|---|
| `source_code` | Code source (CIM-10 FR…) | `E11.9` |
| `source_label` | Libellé en texte libre | `diabète de type 2` |
| `source_vocabulary` | Vocabulaire source (optionnel) | `ICD10FR` |

## Comprendre une suggestion

- **`target_concept_id`** : l'identifiant du concept standard proposé (0 = non mappé).
- **Score de confiance** : de 0 à 1. Plus il est haut, plus la proposition est sûre.
- **Source** :
  - `official_map` = correspondance **officielle exacte** (CIM-10 ↔ SNOMED-CT), la plus fiable ;
  - `rag` = proposition trouvée par **recherche intelligente** (à valider) ;
  - `unmapped` = aucune proposition fiable (à traiter manuellement).
- **Justification** : la raison de la proposition.
- **Candidats** : les autres concepts possibles, si vous devez corriger.

## Valider, corriger, rejeter

- **Accepter** : vous retenez le concept proposé.
- **Corriger** : vous choisissez un autre candidat dans la liste.
- **Rejeter** : aucune correspondance ne convient (rien n'est exporté pour cette ligne).

Vos décisions peuvent être **enregistrées** : elles servent à améliorer l'outil et
peuvent enrichir le jeu d'évaluation.

## Format de sortie

Le fichier exporté est au format OMOP **`source_to_concept_map`**, directement
réutilisable dans un entrepôt OMOP. Colonnes principales : `source_code`,
`source_vocabulary_id`, `source_code_description`, `target_concept_id`,
`target_vocabulary_id`, dates de validité.

## Choisir une stratégie

Dans la barre latérale :

- **`auto`** (recommandé) : correspondance officielle d'abord, recherche
  intelligente seulement pour le reste. Économique et fiable.
- **`deterministic_only`** : uniquement les correspondances officielles exactes
  (rapide, sans IA).
- **`full_rag`** : tout passe par la recherche intelligente.

## Autres façons d'utiliser l'outil

- **Intégrateurs** : une API REST (`gor serve`) expose `/map` et `/map/batch`,
  avec une documentation interactive sur `/docs`.
- **Ligne de commande** : `gor map --source-label "diabète de type 2"`,
  `gor eval` (mesures de qualité).

## Bon à savoir

Cet outil est une **aide à la décision sous supervision humaine**. Il ne pose
aucun diagnostic et ne prend aucune décision automatisée sur une personne. Les
données de démonstration sont publiques/synthétiques. Voir aussi la documentation
de gouvernance et de conformité (IA Act) du projet.
