# Partie 1 : Qu'est-ce que les agents, au juste ?

Bonjour,

De nos jours, tout le monde semble se précipiter pour "construire des agents", mais arrêtons-nous un instant.
Qu'est-ce qu'un agent IA exactement ? Et pourquoi le monde entier en est-il soudainement obsédé ?

Pour être honnête, il n'existe pas de définition largement acceptée.
Mais voici une définition simple et utile pour nos besoins :

> L'IA générative est excellente pour comprendre et générer du contenu.
> **L'IA agentique va plus loin — elle comprend, génère du contenu et effectue des actions.**

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/3bf03a13-eb32-487f-9ddb-2c97919f1e80" />

---

## Un Retour Rapide en Arrière

En 2022, ChatGPT a explosé parce que, pour la première fois, l'IA semblait conversationnelle.
Vous n'aviez pas besoin d'écrire du code ou d'entraîner des modèles — vous pouviez simplement lui parler.

Comparons :
- **Programmation traditionnelle** → Nécessitait du code pour fonctionner
- **ML traditionnel** → Nécessitait de l'ingénierie de caractéristiques
- **Deep learning** → Nécessitait un entraînement spécifique à la tâche
- **ChatGPT** → Pouvait raisonner sur différentes tâches et répondre sans entraînement

C'est ce qu'on appelle l'**apprentissage zero-shot** (aucun exemple nécessaire) ou l'**apprentissage en contexte** (comprend les tâches uniquement à partir des instructions).

---

## En 2024, les gens en voulaient plus

Parler était cool — mais et si l'IA pouvait réellement faire des choses ?

Par exemple :
- Au lieu de simplement vous donner une liste de prospects, pourrait-elle leur envoyer des e-mails ?
- Au lieu de résumer un document, pourrait-elle le classer dans le bon dossier et créer une tâche dans votre workflow ?
- Au lieu de suggérer un produit à un utilisateur, pourrait-elle personnaliser automatiquement la page d'accueil ?

C'est là que les **agents** sont entrés en jeu.

---

## Comment les agents passent-ils à l'action ?

La magie réside dans les **outils**.

La plupart des agents sont associés à des API, des appels de fonction ou des plugins qui leur permettent d'interagir avec des systèmes externes.
Le LLM ne répond pas seulement avec du texte — il produit des commandes structurées comme :
- `Appeler la fonction send_email() avec les entrées suivantes…`
- `Récupérer des enregistrements du CRM en utilisant cette requête…`
- `Planifier une réunion pour mardi à 14h…`

Cela fonctionne grâce à un mécanisme appelé **utilisation d'outils** (ou **appel de fonction**).
L'agent est informé des outils disponibles, et il détermine quand et comment les utiliser — soit directement, soit par le biais d'un mécanisme de planification.

---

## Les agents plus avancés incluent :
- **Mémoire** → Pour se souvenir des étapes passées ou du contexte
- **Modules de planification** → Pour décider quoi faire ensuite, en particulier pour les tâches en plusieurs étapes
- **Gestion d'état** → Pour que l'agent puisse suivre la progression et éviter les boucles ou les échecs

Pensez au LLM comme au **cerveau**, et aux outils comme aux **mains**.
Sans outils, un agent ne fait que parler. Avec des outils, il agit.

<img width="1216" height="413" alt="image" src="https://github.com/user-attachments/assets/9cd7fa10-21a0-42a3-95fb-3bf081e10af1" />

---

## Deux façons de définir les agents

**Vue technique** → Agents = LLM + Outils + Planification + Mémoire (et les composants ci-dessus)
**Vue business** → Agents = Systèmes qui accomplissent des tâches de bout en bout

**Important :** Les agents d'aujourd'hui ne sont pas des innovations en IA.
Ce sont des **enveloppes d'ingénierie** autour de modèles IA. L'intelligence sous-jacente provient toujours des modèles IA — l'agent aide simplement à agir sur cette intelligence.

---

## Comment construire réellement des applications d'IA agentique

Voici où la plupart des gens se trompent :
Ils commencent par "Construisons un agent !" au lieu de "Quel problème du monde réel résolvons-nous ?"

Inversez le récit.
Commencez par des **points de douleur du monde réel/de l'entreprise**, comme :
- Une équipe de support noyée dans des requêtes répétitives
- Un analyste qui bascule entre les tableaux de bord pour trouver des insights
- Une équipe commerciale qui enregistre et suit manuellement l'activité des clients

Ce cours se concentre sur la construction d'agents qui fonctionnent dans le monde réel — pas seulement des démos.
Bien sûr, vous pouvez créer rapidement des agents personnels ou des prototypes sans grande structure, mais lorsque vous construisez pour l'entreprise, **les choix de conception comptent**.

---

## Un modèle mental utile : Autonomie vs. Contrôle

Une fois que vous avez identifié le problème, la décision suivante est :
**Quel degré d'autonomie votre agent devrait-il avoir ?**

Pensez-y comme à un compromis :
- Combien d'autonomie donnez-vous à l'agent
- vs.
- Combien de contrôle voulez-vous conserver du côté humain

Ce n'est pas une décision universelle — c'est contextuel.
Différents problèmes exigent différents niveaux d'implication de l'agent.

---

Dans la partie suivante, nous approfondirons ce compromis autonomie-contrôle et verrons comment concevoir des agents en fonction du niveau d'autonomie dont votre cas d'usage a réellement besoin.

PS : Nous enseignons également un cours très apprécié sur la façon de construire réellement des systèmes IA dans cet environnement en évolution rapide, en utilisant une approche axée sur les problèmes. Il est conçu pour les PM, les leaders, les ingénieurs, les décideurs, etc. qui travaillent dans des contraintes du monde réel. Les anciens élèves viennent de Google, Meta, Apple, Netflix, AWS, Spotify, Snapchat, Deloitte et bien d'autres. Notre prochaine cohorte commence bientôt. Les tarifs early bird sont en ligne : utilisez le code "GITHUB" pour obtenir 300 $ de réduction (valide uniquement pour août 2025) pour [vous inscrire ici](https://maven.com/aishwarya-kiriti/genai-system-design) !!


