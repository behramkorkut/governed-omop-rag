# Gouvernance

`governed-omop-rag` est un outil d'**aide à la décision sous supervision humaine**.
La gouvernance n'est pas un ajout : elle est **incarnée dans le code**.

## Données

- **100 % publiques / synthétiques** : vocabulaires de référence OHDSI (noms de
  concepts) et données Synthea. **Aucune donnée patient réelle** n'est utilisée.
- **Zéro RGPD** sur le périmètre du projet : les vecteurs stockés ne contiennent
  que du vocabulaire de référence public (cf. `souverainete.md`).

## Garde-fous (dans le code, pas seulement en consigne)

- **Sortie fermée** : l'agent Proposer ne peut retourner qu'un `concept_id`
  **présent dans les candidats fournis** ; toute proposition hors-liste lève
  `ClosedOutputViolation` et la suggestion devient « non mappée »
  (`agents/proposer.py`). Anti-hallucination **structurel**.
- **Règles OMOP dures** : le sous-agent Vérificateur exige `standard_concept = 'S'`
  et un domaine cohérent, sinon `FAIL` (`agents/verifier.py`). Déterministe,
  sans LLM.
- **Human-in-the-loop obligatoire** : l'outil **propose**, le steward **dispose**.
  Aucun mapping n'est écrit sans validation (UI de revue, `ui/`).
- **Non-mappé explicite** : jamais de mapping forcé ; `concept_id = 0` + raison
  typée (`HORS_VOCABULAIRE`, `AMBIGU`, `CONFIDENCE_INSUFFISANTE`, `AUCUN_CANDIDAT`).
- **Moindre privilège** : aucune écriture de SQL libre ; la sortie est contrainte
  au vocabulaire réel.

## Traçabilité

Chaque suggestion porte : l'entrée, les candidats, les scores, la **source**
(`official_map` vs `rag`), la justification, l'horodatage. Les décisions du steward
sont journalisées (`feedback.py`, table `steward_feedback`) et peuvent enrichir le
gold set d'évaluation — **amélioration continue tracée**.

## Observabilité

Logging structuré (`structlog`), métriques d'évaluation reproductibles (Top-k,
recall@k, couverture, taux d'hallucination cible ≈ 0, coût en tokens, latence).
Voir `evaluation.md`.

## Non-buts

Pas de diagnostic, pas de décision clinique automatisée, aucun remplacement du
steward humain, aucune donnée patient réelle.
