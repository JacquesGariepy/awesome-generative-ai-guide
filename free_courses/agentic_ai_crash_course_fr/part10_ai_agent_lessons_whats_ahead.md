# Partie 10 : Leçons sur les agents IA et ce qui nous attend


## Un récapitulatif rapide

Voici ce que nous avons couvert au cours des 9 dernières parties :

- **Partie 1 — Ce que sont les agents :** Pas seulement des chatbots qui génèrent du texte, mais des systèmes qui peuvent décider et agir.
- **Partie 2 — Types d'agents :** Des agents de workflow étroitement contrôlés aux entièrement autonomes, selon combien de prise de décision vous déléguez.
- **Partie 3–4 — Outils et RAG :** Le pain et le beurre de l'action des agents et du fondement des connaissances.
- **Partie 5 — MCP :** Une façon propre de structurer tout ce dont un agent a besoin (outils, mémoire, messages précédents) en un seul payload.
- **Partie 6 — Planification et modèles de raisonnement :** Pourquoi les LLM simples ne suffisent pas pour les décisions complexes, et comment les modèles plus récents sont construits pour les tâches multi-étapes.
- **Partie 7 — Mémoire :** Mémoire à court terme vs long terme, quoi stocker, comment récupérer, et pourquoi c'est important pour la continuité.
- **Partie 8 — Systèmes multi-agents :** Orchestration, collaboration pair-à-pair, et le désordre de la coordination.
- **Partie 9 — Systèmes du monde réel :** Comment Perplexity, NotebookLM et DeepResearch utilisent probablement ces modèles de différentes manières.

Nous avons couvert les **pièces mobiles** qui apparaissent dans les systèmes du monde réel.
Mais tout s'effondre si vous ne pensez pas à deux choses : **observabilité** et **évaluation**.

---

## Ce qui reste difficile

### Observabilité
L'observabilité signifie suivre ce que fait votre agent — à chaque étape. Vous voudrez :
- Journaux d'appels d'outils, décisions, réessais
- Métriques pour repérer les goulots d'étranglement en latence et coût
- Visibilité sur quand les choses déraillent
- Traçabilité étape par étape pour le débogage

Des outils comme **Comet Opik** aident avec cela.
Concevez l'observabilité **dès le premier jour**, en particulier pour les agents à haute autonomie.

---

### Évaluation
Les agents sont **non déterministes**.
Vous avez besoin d'**évaluation continue**, pas seulement de tests manuels.

Au minimum, suivez :
- Taux de complétion d'objectif ou de tâche
- Succès/échec d'appel d'outil
- Qualité RAG et métriques d'hallucination
- Sur-réflexion ou inefficacité du modèle
- Latence et utilisation de tokens à chaque étape

L'évaluation est comment vous **comprenez** et **améliorez** votre système.
Trop d'équipes font des *vérifications d'ambiance* au lieu de vraies évaluations — et se retrouvent coincées dans le **purgatoire des PoC**.

Pensez aux évaluations + observabilité comme à votre **pipeline de test** — l'équivalent agentique de la QA logicielle.
Les métriques varieront par cas d'usage, mais la discipline est la même.

---

## Où les choses se dirigent en IA agentique

Cet espace est jeune, mais voici des tendances claires :

---

### 1. Protocoles > Prompts
<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/13b78c2e-fc2b-41fd-a0e5-aa88c60187ea" />

_Source de l'image : Post LinkedIn de Reuven_

À mesure que les systèmes grandissent, nous nous éloignerons des prompts artisanaux vers des **standards** partagés.

- **MCP** (Model Context Protocol) standardise comment nous emballons le contexte structuré — outils, mémoire, RAG, instructions précédentes.
- **A2A** (Agent-to-Agent), publié par Google, se concentre sur la communication inter-agents cross-plateforme avec un schéma partagé.

Attendez-vous à des abstractions plus propres au fil du temps — bien qu'il faudra un moment avant que quoi que ce soit ne devienne aussi standard que HTTP.

---

### 2. Modèles de raisonnement hybrides
Les modèles de raisonnement évolueront vers une **planification sélective** — savoir quand planifier vs agir vite.

Nous voyons déjà cela avec **Claude 3.7** et d'autres.
L'objectif : équilibrer intelligence avec efficacité — sans trop réfléchir à chaque tâche.

---

### 3. Meilleurs systèmes de mémoire
La mémoire d'aujourd'hui est principalement **rafistolée**.
Le futur : une mémoire qui sait **quoi rappeler, quand et pourquoi**.
Attendez-vous à :
- Mémoire limitée aux tâches
- Mémoire basée sur les sessions
- Mémoire spécifique aux personas

Et **gestion plus facile**.

---

### 4. Maturité de l'écosystème d'outils
Pour l'instant, tout le monde construit des outils/enveloppes personnalisés. Au fil du temps :
- API de confiance, plug-and-play
- Meilleures couches d'abstraction
- Pratiques de sécurité partagées

Tout comme les microservices ont mûri dans le logiciel traditionnel, les outils mûriront dans la **stack agentique**.

---

## Un dernier mot

Si vous avez suivi, vous avez vu le thème :

Nous n'avons pas commencé par l'**architecture**.
Nous avons commencé par les **problèmes**.

C'est le vrai changement de mentalité :
> Ne chassez pas les agents pour le battage médiatique.
> Construisez-les quand ils rendent la résolution d'un problème plus facile, plus rapide ou plus intelligente.

**Commencez simple. Mesurez tout. Évoluez quand nécessaire.**
La pensée agent-d'abord casse. La pensée problème-d'abord évolue.

---

Merci d'avoir lu, partagé et réfléchi pendant ces 10 parties.
Si vous retenez une chose de cette série — que ce soit ceci :

> **Problème d'abord, toujours.**

Consultez le readme pour plus de conférences et de sujets avancés. Si cela a été utile, n'hésitez pas à le transmettre à quelqu'un qui cherche à apprendre dans ce domaine. Et si vous souhaitez aller plus loin, notre cours complet de 6 semaines couvre la conception de systèmes, les concepts agentiques appliqués et les vrais workflows d'évaluation, le genre qui supporte les applications de qualité production. Le cours est conçu pour tout le monde, que vous soyez Product Manager, Architecte, Directeur, leader C-suite, ou quelqu'un qui explore sérieusement l'IA agentique.

Notre prochaine cohorte commence bientôt. Les tarifs early bird sont en ligne : utilisez le code "GITHUB" pour obtenir 300 $ de réduction (valide uniquement pour août 2025) pour [vous inscrire ici](https://maven.com/aishwarya-kiriti/genai-system-design) !!

