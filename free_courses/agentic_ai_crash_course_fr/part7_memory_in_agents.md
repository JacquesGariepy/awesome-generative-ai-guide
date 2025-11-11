# Partie 7 : Mémoire dans les agents


---

Au cours des dernières parties, nous avons exploré ce qui fait agir les agents — des **outils** et **RAG**, à **MCP** et aux **modèles de raisonnement**.

Aujourd'hui, nous changeons de vitesse vers quelque chose qui détermine **à quel point** ils agissent au fil du temps : la **mémoire**.

Parce que voici la base :
Les modèles IA **n'ont pas de mémoire de manière inhérente**. Ils sont **sans état** par conception. Chaque entrée est traitée de manière indépendante à moins que vous **n'architecturiez la mémoire dans le système**.

---

## Pourquoi la mémoire est importante
<img width="438" height="164" alt="image" src="https://github.com/user-attachments/assets/1d3fccea-8f9a-42b8-baa4-b3fe9f57dad1" />


_Source de l'image : https://arxiv.org/html/2502.12110v1_

Si un agent vous aide à rédiger des e-mails, résumer de longs fils de discussion, ou gérer des workflows sur des jours ou des semaines — il doit se souvenir de :
- Le format de l'e-mail
- Le nom de l'utilisateur
- Le ton à utiliser

Bien sûr, vous pourriez passer ces informations encore et encore avec chaque prompt…
Mais ne serait-il pas mieux si l'agent pouvait récupérer les bonnes informations **par lui-même**, au bon moment, depuis une **base de données externe** ?

C'est exactement là que la **mémoire** entre en jeu.

---

## "Attendez… n'est-ce pas comme RAG agentique (Jour 4) ?"

Bonne question — et vous n'avez pas tort. Gérer la mémoire ressemble souvent beaucoup à faire du RAG agentique.

Vous :
1. Écrivez des mémoires structurées ou non structurées (faits, journaux, sorties passées)
2. Les stockez avec des métadonnées, des tags ou des embeddings
3. Récupérez la tranche pertinente quand nécessaire
4. Fondez la prochaine action du modèle en utilisant ce contexte

**La différence :**
- **RAG** → Aide à répondre aux questions avec des connaissances.
- **Mémoire** → Aide les agents à se comporter de manière cohérente au fil du temps.

---

## Deux types de mémoire dans les agents

Lors de la conception de systèmes d'agents du monde réel, vous traitez généralement de **deux types de mémoire**.

<img width="571" height="372" alt="image" src="https://github.com/user-attachments/assets/7c2d9e58-a219-4cea-b9f0-6de091298d66" />


_Source de l'image : https://langchain-ai.github.io/langgraph/concepts/memory/#what-is-memory_

---

### 1. Mémoire à court terme

Limitée à une seule session ou tâche.

**Inclut :**
- La conversation jusqu'à présent
- Outils utilisés
- Réponses générées
- Documents récupérés

Pensez-y comme à un journal brut des conversations utilisateur-agent.

LangGraph, Autogen et des frameworks similaires traitent cela comme partie de l'**état** de l'agent.
Mais l'état grandit vite, et la plupart des agents performent mal lorsqu'ils sont enterrés sous un historique non pertinent.

**Stratégies pour gérer la mémoire à court terme :**
- Tailler les messages périmés
- Résumer le passé en points clés
- Filtrer basé sur ce qui est toujours pertinent

C'est un acte d'équilibre : **longueur de contexte vs clarté vs coût**.

---

### 2. Mémoire à long terme

Vit à travers les sessions, jours, semaines — même pour toujours.

**Aide les agents à se souvenir :**
- Qui est l'utilisateur
- Comment ils préfèrent interagir
- Ce qui a déjà été fait
- Contexte passé important

**Exemples :**
- "L'utilisateur préfère un ton neutre"
- "Le nom de l'utilisateur est X et il vit dans la ville Y"
- "La facture #123 a déjà été escaladée"

Plus de données ≠ mieux par défaut — il s'agit de récupérer la bonne chose au bon moment.

---

## Types de mémoire à long terme à considérer

Emprunté aux sciences cognitives :

- **Mémoire sémantique** → Faits et infos (objectif)
  _"L'utilisateur parle anglais et préfère les fichiers Excel."_

- **Mémoire épisodique** → Actions passées
  _"L'agent a déjà généré un résumé hier."_

- **Mémoire procédurale** → Préférences (subjectif)
  _"Éviter la voix passive. Prioriser les points d'action."_

---

**Exemples par cas d'usage :**

- **Chatbots face utilisateur** → Mémoire sémantique pour la personnalisation
- **Agents d'automatisation de processus** → Mémoire épisodique pour éviter les réessais ou les boucles
- **Assistants adaptatifs** → Mémoire procédurale pour ajuster les prompts basés sur les retours

---

## Questions clés de conception

Avant de dire "nous avons besoin de mémoire", demandez :
- **Quel type ?**
- **Pourquoi est-ce nécessaire ?**
- **Comment sera-t-elle stockée, récupérée et maintenue fraîche ?**

---

## Gérer la mémoire en pratique

Gérer la mémoire ressemble souvent à gérer RAG.
La partie difficile ? Décider **quoi stocker** et **quoi récupérer**.

Bourrer plus de texte dans l'entrée de l'agent aide rarement — cela **nuit souvent aux performances**.

Vous devez concevoir la mémoire intentionnellement, basé sur :
- Le travail de l'agent
- Ce dont il doit se rappeler
- Quand il doit s'en rappeler
- Comment la garder utile au fil du temps

---

## Quelques exemples d'entreprise

**Agent de support client**
- Besoins : historique de support récent, bugs connus, sentiment utilisateur
- Types de mémoire : épisodique + sémantique

**Copilote de vente**
- Besoins : pitchs précédents, objections utilisateur, statut de clôture
- Types de mémoire : sémantique + procédurale

**Agent auditeur de conformité**
- Besoins : éléments signalés, exceptions antérieures, changements de politique
- Types de mémoire : épisodique

---

Dans tous les cas, il ne s'agit pas de **combien** de données vous stockez — il s'agit de **à quel point elles sont pertinentes et structurées**.

Et oui, je l'ai dit douloureusement de nombreuses fois, mais je le redirai :
> **Problème d'abord, toujours.** La stratégie de mémoire, comme les outils ou la planification, dépend entièrement du problème que vous résolvez.

---

## À suivre

Dans la partie suivante, nous parlerons des **systèmes multi-agents** — ce qu'ils sont, comment ils se coordonnent, et si vous avez réellement besoin de plus d'un agent.

PS : Nous enseignons également un cours très apprécié sur la façon de construire réellement des systèmes IA dans cet environnement en évolution rapide, en utilisant une approche axée sur les problèmes. Il est conçu pour les PM, les leaders, les ingénieurs, les décideurs, etc. qui travaillent dans des contraintes du monde réel. Les anciens élèves viennent de Google, Meta, Apple, Netflix, AWS, Spotify, Snapchat, Deloitte et bien d'autres. Notre prochaine cohorte commence bientôt. Les tarifs early bird sont en ligne : utilisez le code "GITHUB" pour obtenir 300 $ de réduction (valide uniquement pour août 2025) pour [vous inscrire ici](https://maven.com/aishwarya-kiriti/genai-system-design) !!

