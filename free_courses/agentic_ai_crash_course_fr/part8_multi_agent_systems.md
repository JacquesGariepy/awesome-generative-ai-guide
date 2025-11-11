# Partie 8 : Systèmes multi-agents

---

Jusqu'à présent, nous avons beaucoup parlé de ce qui fait agir un **seul agent** — des **outils** et **RAG** à la **mémoire** et la **planification**.

Mais que se passe-t-il si votre pipeline agentique doit :
- Paralléliser les tâches pour accélérer les choses
- Utiliser différentes personas d'agents pour différentes parties d'une tâche
- Diviser la complexité entre des unités spécialisées, comme dans une équipe

C'est là que les **systèmes multi-agents** entrent en jeu.

---

## Pourquoi utiliser des systèmes multi-agents ?

Parfois, un seul agent ne peut tout simplement pas le faire parce que le problème exige **échelle**, **spécialisation** ou **pensée parallèle**.

**Exemples :**
- Générer une stratégie marketing qui nécessite des **insights de marché**, une **révision juridique** et des **suggestions créatives**.
- Construire un assistant de conformité qui doit **extraire des informations**, **signaler des risques** et **vérifier les politiques**.
- Automatiser un processus de vente où **un agent** parle à l'utilisateur, **un autre** enrichit les données, et **un troisième** gère les suivis.

Pourriez-vous faire cela avec un seul agent costaud ?
**Peut-être.**

Mais le diviser en **plusieurs agents spécialisés** peut permettre :
- **Parallélisation** → Les agents travaillent simultanément sur des parties d'une tâche
- **Spécialisation** → Un agent est excellent en jargon juridique, un autre en écriture d'e-mails
- **Indépendance d'outils** → Chaque agent peut avoir ses propres outils et mémoire

---

## Coordination plate vs hiérarchique des agents

Tous les systèmes multi-agents ont besoin d'un moyen de **coordonner**.
Deux modèles de communication courants :
<img width="1144" height="626" alt="image" src="https://github.com/user-attachments/assets/c6e2c1c0-bb92-48c8-ba53-dfbfdc6e3926" />


---

### 1. Modèles hiérarchiques (Plus contrôlables)

Un **agent orchestrateur** délègue les sous-tâches aux autres.
Il voit la vue d'ensemble et contrôle le flux.

**Utiliser quand :**
- Les tâches peuvent être clairement décomposées
- Vous voulez un contrôle serré
- Vous avez des rôles d'agents connus (par ex., résumeur, générateur, vérificateur)

**Pensez :** workflows d'entreprise, suites d'outils, pipelines parallèles.

---

### 2. Modèles plats (Plus dynamiques)

Les agents se parlent en tant que **pairs** — pas de patron.

**Utiliser quand :**
- Les tâches nécessitent créativité ou débat
- Vous voulez que les agents s'évaluent mutuellement
- Il n'y a pas un seul chemin de réponse "correct"

**Pensez :** brainstorming, classement d'options, raisonnement multi-vues.

---

## Ce que personne ne vous dit : Les systèmes multi-agents sont une galère

Sur le papier, cela semble génial.
Et bien sûr, vous pouvez construire des prototypes multi-agents rapides et vous amuser avec.

Mais pour les cas d'usage **clients/entreprises**… c'est douloureux.

La plupart des gens lisent un blog sur les systèmes multi-agents et s'excitent de la modularité —
> "C'est comme les microservices !" disent-ils.

Mais **les agents IA ne sont pas des microservices**.

Contrairement au code, les modèles IA sont **non déterministes**. Ils ne se comportent pas toujours de la même manière.
Ajouter plus d'agents signifie :
- Plus de **non-déterminisme** (variation entre agents, pas seulement au sein d'un)
- Plus de **complexité de mémoire et d'état** (qui sait quoi, et quand ?)
- **Latence** et **coût** plus élevés
- Plus de **bugs de coordination** et points de défaillance
- Plus de **collusion**, où les agents sont d'accord alors qu'ils ne devraient pas (arrive plus que vous ne pensez)

Honnêtement, je pourrais écrire un livre sur à quel point il est douloureux de faire fonctionner les systèmes multi-agents de manière fiable.

---

## Alors… Devriez-vous les utiliser ?

Ma règle personnelle :
> **Dans l'entreprise, ne commencez pas avec multi-agents. Commencez avec un.**

Laissez ce **seul agent** échouer — empiriquement (via des métriques d'évaluation) ou opérationnellement — avant d'évoluer.

D'après mon expérience, **70 %+ des cas d'usage d'entreprise** fonctionnent très bien avec un seul agent bien conçu — qui utilise **outils**, **mémoire**, **RAG** et **planification**.

---

### Les systèmes multi-agents brillent quand :
- La tâche est assez grande pour nécessiter une **exécution parallèle**
- Vous avez besoin d'une **spécialisation claire**
- Vous voulez un **débat créatif**, une évaluation ou une prise de décision distribuée

Même alors, vous avez besoin d'une **conception solide** — en particulier autour de la **mémoire**, de l'**état** et des **protocoles de communication**.

---

## Dernier mot : Problème d'abord, toujours

Cela a été notre mantra depuis le Jour 1 :
> Ne construisez pas un système multi-agents parce que ça sonne "agentique".
> Construisez-le si — et seulement si — votre problème en a besoin.

La seule façon de savoir ?
- Avoir les bonnes **métriques**
- Tester
- Laisser les systèmes plus simples échouer d'abord

---

## À suivre

Dans la partie suivante, nous parlerons des **agents du monde réel** et comment ils fonctionnent.

PS : Nous enseignons également un cours très apprécié sur la façon de construire réellement des systèmes IA dans cet environnement en évolution rapide, en utilisant une approche axée sur les problèmes. Il est conçu pour les PM, les leaders, les ingénieurs, les décideurs, etc. qui travaillent dans des contraintes du monde réel. Les anciens élèves viennent de Google, Meta, Apple, Netflix, AWS, Spotify, Snapchat, Deloitte et bien d'autres. Notre prochaine cohorte commence bientôt. Les tarifs early bird sont en ligne : utilisez le code "GITHUB" pour obtenir 300 $ de réduction (valide uniquement pour août 2025) pour [vous inscrire ici](https://maven.com/aishwarya-kiriti/genai-system-design) !!


