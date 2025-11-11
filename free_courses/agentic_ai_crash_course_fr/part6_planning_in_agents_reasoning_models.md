# Partie 6 : Planification dans les agents + Modèles de raisonnement


---

## Waouh ! Nous avons dépassé la moitié de notre cours !

Au cours des dernières parties, nous avons parlé de ce que les agents peuvent faire :
- Utiliser des outils
- Récupérer des informations via RAG
- Tout passer dans un format propre en utilisant MCP

Mais tout cela suppose quelque chose de fondamental :
**Que l'agent sait réellement quoi faire ensuite.**
Et c'est là que les choses se cassent souvent.

Aujourd'hui, nous changeons de focus des outils et des entrées vers **comment les agents pensent** — plus spécifiquement, comment les modèles modernes commencent à planifier et pourquoi cela change la façon dont nous concevons les systèmes du monde réel.

---

## Pourquoi la planification est importante dans les systèmes agentiques

Voici quelques exemples pour commencer.

Si vous demandez à un agent :
> "Combien font 13 multiplié par 47 ?"
…il peut soit le résoudre directement soit appeler une calculatrice. C'est une tâche en une étape — aucune planification réelle nécessaire.

Maintenant imaginez demander :
> "Trouver tous nos clients Q1 dans le secteur de la santé, vérifier lesquels sont en retard de paiement, et rédiger des e-mails personnalisés avec de nouveaux liens de paiement."

Dans ce cas, l'agent doit :
- Comprendre l'instruction
- La décomposer en parties gérables
- Récupérer les bonnes données
- Choisir des outils
- Effectuer les étapes dans l'ordre
- Gérer les exceptions
- Savoir quand la tâche est terminée

Cette boucle d'interprétation, de séquençage et d'action est la **planification**.

On attend de l'agent (c'est-à-dire du modèle) qu'il comprenne cela seul — y compris quels outils utiliser et comment appliquer les informations qu'il a.

---

## Pourquoi les LLM traditionnels ont du mal avec la planification

La plupart des LLM à usage général n'ont jamais été entraînés pour faire cela.

Ils sont entraînés à **prédire le prochain token** basé sur le contexte précédent — rien de plus.
Ils excellent à :
- Continuer des phrases
- Générer des résumés
- Répondre à des questions directes

…mais ils se comportent plus comme des **générateurs à courte vue**.
Ils complètent ce qui est devant eux mais ne sont pas câblés pour penser à l'avance.

Lorsqu'on leur demande d'agir comme agents dans des tâches de prise de décision en plusieurs étapes, ils ont tendance à :
- Sauter des étapes
- Répéter des actions
- Compliquer excessivement des choses simples
- Perdre le fil à mi-parcours

---

## Premières tentatives pour améliorer le raisonnement

Pour combler cette lacune, les constructeurs ont expérimenté avec des techniques de prompting pour encourager le comportement de planification.

Un exemple populaire : **Prompting Chain-of-Thought** — ajouter "Pensons étape par étape" pour décomposer les tâches en étapes.

Cela a fonctionné pour les puzzles logiques et les Q&R structurées, mais a échoué pour les **vrais agents** travaillant avec :
- Outils
- Entrées imprévisibles
- État changeant

Parce que sous-jacent, ces modèles n'étaient toujours pas entraînés pour la planification — ils répondaient simplement à des **astuces de prompt**.

---

## Puis sont venus les modèles de raisonnement

Le prochain changement : entraîner des modèles à planifier **par conception**.

Cela a donné naissance aux **grands modèles de raisonnement (LRM)**.
<img width="743" height="663" alt="image" src="https://github.com/user-attachments/assets/ce4d8d91-b539-4003-adfc-1fa6dcfd3631" />

**LLMs :**
entrée → LLM → déclaration de sortie

**LRMs :**
entrée → LRM → étape de plan + déclaration de sortie



Tout reste du texte, mais les LRM sont encouragés pendant l'entraînement à **penser avant d'agir**.

---

**Exemples :**
- La **série o** d'OpenAI (o1, o3) — premiers exemples publics
- **DeepSeek-R1** de DeepSeek — ajusté pour le raisonnement et la planification augmentés par outils
- **Modèles de pensée Gemini** de Google
- **Mode de raisonnement Claude 3.7** d'Anthropic

Certains activent même le raisonnement **uniquement quand nécessaire**.

---

## Comment ils s'intègrent dans la conception agentique

La principale valeur des modèles de raisonnement est d'améliorer le **composant de planification** — la partie qui demande :
> "Que dois-je faire ensuite, et pourquoi ?"

Dans les cas d'usage d'entreprise, **la planification est là où les agents échouent souvent**.
Les modèles de raisonnement peuvent aider, mais ils ne sont pas magiques.

---

## Utilisez-les avec précaution

Les modèles de raisonnement sont encore **nouveaux** et viennent avec des compromis :
- Réfléchissent trop aux tâches simples
- Génèrent des sorties plus longues
- Augmentent la latence et le coût
- Peuvent halluciner des plans logiques mais incorrects

**Règle empirique :**
- Ne commencez pas avec un modèle de raisonnement.
- Commencez avec un modèle de base de taille moyenne.
- Ne changez que si vous voyez des échecs de planification clairs — et même alors, évaluez l'impact réel.

---

## À suivre

Dans la partie suivante, nous passerons à un autre **composant central des agents** : la **mémoire** — comment les agents peuvent se souvenir efficacement et pourquoi c'est important.

PS : Nous enseignons également un cours très apprécié sur la façon de construire réellement des systèmes IA dans cet environnement en évolution rapide, en utilisant une approche axée sur les problèmes. Il est conçu pour les PM, les leaders, les ingénieurs, les décideurs, etc. qui travaillent dans des contraintes du monde réel. Les anciens élèves viennent de Google, Meta, Apple, Netflix, AWS, Spotify, Snapchat, Deloitte et bien d'autres. Notre prochaine cohorte commence bientôt. Les tarifs early bird sont en ligne : utilisez le code "GITHUB" pour obtenir 300 $ de réduction (valide uniquement pour août 2025) pour [vous inscrire ici](https://maven.com/aishwarya-kiriti/genai-system-design) !!


