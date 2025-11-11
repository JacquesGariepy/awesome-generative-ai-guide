# Partie 2 : Les 4 types de systèmes agentiques (et quand utiliser quoi)

Bonjour,

Dans la partie précédente, nous avons examiné ce qui rend l'IA agentique — il ne s'agit pas seulement de comprendre ou de générer du contenu, il s'agit d'effectuer des actions et de gérer des tâches de bout en bout.

Mais alors que les équipes se précipitent pour "ajouter des agents" à leur stack, voici le problème :
Tous les agents ne sont pas construits de la même manière, et tous les problèmes ne nécessitent pas de systèmes hautement autonomes.

Dans cette leçon, nous allons parcourir quatre types de systèmes agentiques (comme discuté hier), en utilisant une lentille simple mais puissante :

- Combien d'autonomie l'agent a-t-il ?
- Combien de contrôle l'humain ou le système conserve-t-il ?

Cet équilibre impacte la façon dont le système se comporte, comment vous l'évaluez et quelle infrastructure vous devez construire.

---
<img width="1216" height="413" alt="image" src="https://github.com/user-attachments/assets/58097651-e6d0-4835-9c13-042f647cf437" />


## Le LLM augmenté d'outils

Au cœur de la plupart des agents modernes se trouve un **LLM (Large Language Model)** agissant comme le cerveau du système.
Tout au long de ce cours, nous utilisons le terme LLM pour désigner largement les modèles d'IA générative — pas seulement les modèles textuels.

Seul, il peut générer du contenu, mais pour en faire un agent, vous l'augmentez avec :
- **Outils** → API, fonctions, bases de données qu'il peut appeler
- **Planification** → La capacité de décomposer un objectif en plusieurs étapes
- **Mémoire** → Pour qu'il puisse suivre les actions et résultats passés
- **État et logique de contrôle** → Pour savoir ce qui est fait, ce qui a échoué et quoi faire ensuite

Lorsqu'il est connecté à ces composants, le LLM devient plus qu'un chatbot.
Il devient un système orienté objectif qui peut raisonner, agir et s'adapter.

Mais selon le degré de confiance que vous lui accordez pour agir sans supervision, vous obtenez différents types d'agents.
Passons-les en revue, en commençant par le moins autonome.

---

## 1. Systèmes/Agents basés sur des règles
**Faible autonomie, faible contrôle**

Ces systèmes n'utilisent pas du tout de LLM. Ils sont construits avec une logique traditionnelle *si-ceci-alors-cela*. Chaque chemin de décision est scripté manuellement. Il n'y a pas de raisonnement ni d'apprentissage. Les agents basés sur des règles existent bien avant l'ère des LLM.

> Attendez, ne parlons-nous pas d'agents IA ?
> Oui — mais tous les problèmes n'ont pas besoin d'un modèle IA. Commencez par le problème, pas par l'IA. Si vous pouvez le résoudre sans IA, ne le compliquez pas.

**Quels problèmes résolvent-ils ?**
Tâches bien structurées et répétitives avec des entrées et sorties fixes.

**Exemples :**
- Approuver automatiquement les remboursements inférieurs à un montant fixe
- Renommer les fichiers dans un dossier en fonction de modèles de noms de fichiers
- Copier des données de feuilles Excel dans des champs de formulaire

**Avantages :** Rapide, auditable, prévisible
**Inconvénients :** Fragile aux changements, ne peut pas gérer l'ambiguïté
**Meilleur usage :** Lorsque vous connaissez toutes les conditions à l'avance et qu'il n'y a pas besoin de flexibilité.

---

## 2. Agents de workflow
**Faible autonomie, contrôle élevé**

C'est souvent la première étape pour les entreprises qui introduisent des LLM dans leurs workflows.
Ici, le LLM améliore un workflow existant mais n'exécute pas d'actions de manière indépendante. Un humain reste aux commandes.

**Quels problèmes résolvent-ils ?**
Tâches répétitives qui bénéficient de la compréhension du langage naturel, de la synthèse ou de la génération, mais nécessitent toujours une prise de décision humaine.

**Exemples :**
- Suggérer des brouillons de réponses dans un outil de support comme Zendesk
- Générer des résumés de transcriptions de réunions
- Traduire des requêtes en langage naturel en entrées de recherche structurées pour les tableaux de bord BI

**Comment le LLM est utilisé :**
Il lit l'entrée (texte, tickets, documents), comprend le contexte et génère du contenu utile, mais n'agit pas dessus.
Un humain décide toujours quoi faire.

**Avantages :** Facile à déployer, faible risque, valeur rapide
**Inconvénients :** Ne peut pas exécuter ou planifier, valeur de bout en bout limitée
**Meilleur usage :** Lorsque vous voulez augmenter la productivité de votre équipe sans abandonner la supervision.

---

## 3. Agents semi-autonomes
**Autonomie modérée à élevée, contrôle modéré**

Ce sont de véritables systèmes agentiques. Ils comprennent non seulement les tâches mais peuvent planifier des actions en plusieurs étapes, invoquer des outils et accomplir des objectifs avec une supervision minimale. Cependant, ils fonctionnent souvent avec certaines contraintes ou une surveillance intégrée.

**Quels problèmes résolvent-ils ?**
Workflows en plusieurs étapes qui sont bien compris mais trop fastidieux ou chronophages pour les humains.

**Exemples :**
- Un agent de suivi de prospects qui rédige, personnalise et envoie des e-mails basés sur les données CRM, tout en enregistrant les résultats
- Un agent d'automatisation de documents qui extrait les détails des contrats et met à jour les systèmes internes
- Un agent de recherche qui récupère des données de plusieurs sources, compare les résultats et envoie un rapport structuré

**Comment le LLM est utilisé :**
Le LLM planifie les étapes, appelle des API pour récupérer ou pousser des données, suit la progression et s'adapte si quelque chose ne va pas.
Il inclut souvent des chemins de secours ou des points de contrôle pour la révision humaine.

**Avantages :** Automatise les workflows complexes, économise du temps, ROI plus élevé
**Inconvénients :** Nécessite de l'infrastructure (planification, mémoire, appel d'outils), plus difficile à tester
**Meilleur usage :** Lorsque vous voulez automatiser des workflows métier bien délimités tout en conservant un certain contrôle.

---

## 4. Agents autonomes
**Autonomie élevée, faible contrôle**

Ces agents sont entièrement orientés objectif. Vous leur donnez un objectif général, et ils déterminent quoi faire, comment le faire, quand réessayer et quand escalader. Ils agissent de manière indépendante, souvent à travers les systèmes et au fil du temps.

**Quels problèmes résolvent-ils ?**
Tâches à fort effort, asynchrones ou de longue durée qui couvrent plusieurs systèmes ou étapes et ne nécessitent pas d'entrée humaine constante.

**Exemples :**
- Un agent de recherche concurrentielle qui récupère des données sur plusieurs jours, résume les mises à jour et génère des résumés d'insights hebdomadaires
- Un agent d'automatisation des opérations qui détecte les problèmes dans les pipelines, diagnostique les causes racines et crée des tickets avec des correctifs suggérés
- Un agent de test qui exécute de manière autonome les flux de produits, enregistre les résultats et suggère de nouveaux scénarios de cas limites

**Comment le LLM est utilisé :**
Le LLM est le planificateur, le décideur, l'utilisateur d'outils, le traceur de mémoire et le communicateur. Il gère les réessais, évalue si les objectifs sont atteints et décide quand s'arrêter ou s'adapter.

**Avantages :** Extrêmement évolutif, peut gérer des tâches complexes
**Inconvénients :** Risque élevé s'il n'est pas surveillé, difficile à évaluer ou à tracer, lourd en infrastructure
**Meilleur usage :** Lorsque la tâche est à fort effet de levier, asynchrone et ne nécessite pas de retour humain à chaque étape.

---
<img width="683" height="316" alt="image" src="https://github.com/user-attachments/assets/cbea3f8f-b2c5-4dbc-8d0f-5b102430d675" />


## Comment décider quoi construire

Pas en choisissant votre architecture préférée.
Vous commencez par le **problème**.

Demandez-vous :
- Est-ce répétitif et structuré ?
- Cela implique-t-il la compréhension ou la génération de langage ?
- Est-ce une tâche en plusieurs étapes qui nécessite une prise de décision ?
- Faites-vous confiance à un système IA pour exécuter la tâche entière, ou voulez-vous un humain dans la boucle ?

Voici le point clé :
- Ces approches ne s'excluent pas mutuellement.
- Un seul système peut les mélanger — certaines parties peuvent nécessiter un contrôle élevé, d'autres peuvent bénéficier d'une autonomie élevée.
- Chaque type de problème peut être abordé soit par un seul agent, soit par un groupe d'agents collaborant.

Nous approfondirons la **conception mono-agent vs. multi-agents** plus tard dans le cours.
Pour l'instant, souvenez-vous :
> Ne commencez pas par "Comment puis-je construire un système multi-agents ?"
> Commencez par "Quel est le problème que je résous, et quel type d'autonomie nécessite-t-il ?"

Laissez le problème façonner la conception agentique, pas l'inverse.

---

Dans la partie suivante, nous plongerons plus profondément dans le **rôle des outils** dans les systèmes agentiques. Ce sont la raison pour laquelle l'IA est devenue beaucoup plus utilisable — et nous décomposerons exactement comment et pourquoi dans notre analyse approfondie.

PS : Nous enseignons également un cours très apprécié sur la façon de construire réellement des systèmes IA dans cet environnement en évolution rapide, en utilisant une approche axée sur les problèmes. Il est conçu pour les PM, les leaders, les ingénieurs, les décideurs, etc. qui travaillent dans des contraintes du monde réel. Les anciens élèves viennent de Google, Meta, Apple, Netflix, AWS, Spotify, Snapchat, Deloitte et bien d'autres. Notre prochaine cohorte commence bientôt. Les tarifs early bird sont en ligne : utilisez le code "GITHUB" pour obtenir 300 $ de réduction (valide uniquement pour août 2025) pour [vous inscrire ici](https://maven.com/aishwarya-kiriti/genai-system-design) !!

