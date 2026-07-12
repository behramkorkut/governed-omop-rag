# Conformité — Règlement IA (IA Act)

Ce document acte la **classification** de l'outil au regard du Règlement européen
sur l'intelligence artificielle et les principes de conception associés.

## Finalité

`governed-omop-rag` est un **outil d'aide à la décision** pour la standardisation
de terminologies médicales (mapping vers OMOP), destiné à des professionnels
(data steward, chercheur, médecin). Il **ne pose pas de diagnostic** et **ne prend
aucune décision automatisée concernant une personne physique**.

## Supervision humaine (human oversight)

La validation humaine est **obligatoire et intégrée** : l'outil propose, un steward
valide/corrige/rejette avant toute intégration. Aucun mapping n'est appliqué sans
décision humaine (`ui/`, human-in-the-loop). L'humain peut **toujours** ignorer la
suggestion : score de confiance, candidats alternatifs, justification et source
sont affichés pour permettre la vérification.

## Gestion des risques & robustesse

- **Anti-hallucination structurel** : sortie contrainte au vocabulaire réel
  (sortie fermée), règles OMOP dures vérifiées par un sous-agent.
- **Incertitude explicite** : l'outil sait dire « je ne sais pas »
  (`concept_id = 0` + raison typée) plutôt que de forcer un mapping.
- **Évaluation & suivi** : métriques reproductibles, taux d'hallucination mesuré,
  coût et latence suivis (`evaluation.md`).

## Traçabilité & journalisation

Chaque suggestion et chaque décision sont journalisées (entrée, candidats, scores,
source, décision, horodatage). Voir `governance.md`.

## Données

Données **publiques / synthétiques**, aucune donnée patient réelle. Traitement de
données à caractère personnel : **hors périmètre** de l'outil.

## Classification indicative

Au regard de sa finalité (aide à la décision terminologique, supervision humaine
systématique, absence de décision automatisée sur une personne), l'outil se situe
**hors des systèmes à haut risque** visant l'évaluation ou la décision médicale
directe. Cette classification doit être **réévaluée** en fonction du contexte de
déploiement réel (intégration dans un dispositif médical, usage clinique, etc.),
en lien avec le responsable de traitement.
