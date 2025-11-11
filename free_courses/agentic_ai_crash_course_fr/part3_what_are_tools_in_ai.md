# Partie 3 : Que sont les outils en IA ?

Dans la partie précédente, nous avons parlé de différents types d'agents, des basés sur des règles aux entièrement autonomes, et comment le bon niveau d'autonomie dépend du problème que vous résolvez.

Mais voici un trait commun à tous les types d'agents, aussi simples ou complexes soient-ils :

> **Ils s'appuient sur des outils pour effectuer des actions.**

---

## Que sont les "outils" en IA ?

Dans le contexte de l'IA agentique, les outils sont des capacités externes que le LLM peut invoquer, comme :
- API
- Requêtes de base de données
- Services internes
- Systèmes tiers
- Fonctions internes écrites en code

Ils transforment le LLM de quelque chose qui **parle** seulement en quelque chose qui peut **agir**.

Rappelez-vous, les LLM seuls sont **sans état**, n'ont **aucun accès aux systèmes en temps réel**, et **ne peuvent pas agir**.

---

## Mais donnez-leur des outils, et ils peuvent :
- Récupérer des données de vos systèmes internes
- Déclencher des événements (par exemple, envoyer un e-mail, créer un ticket JIRA)
- Accéder à des données structurées comme des calendriers, des tableaux de bord ou des CRM
- Exécuter une logique pré-écrite basée sur des règles métier

C'est ainsi que **la génération se transforme en exécution**.

---

## Pourquoi les outils sont importants

1. **Ils débloquent l'exécution**
   Sans outils, votre agent n'est qu'un assistant qui fait des suggestions.
   Avec des outils, il peut accomplir des workflows de bout en bout.

2. **Ils augmentent la précision**
   Plutôt que d'halluciner, le LLM peut interroger directement le bon système —
   "Quel est le statut réel de la commande ?" au lieu d'inventer une raison de retard.

3. **Ils vous permettent de contrôler le risque**
   Vous définissez ce qui est exposé. Le LLM ne peut rien faire en dehors des outils que vous enregistrez.

4. **Ils permettent la composabilité**
   Si vous voulez combiner votre CRM, calendrier et stack d'e-mails en un seul assistant,
   vous pouvez exposer chacun d'eux comme des outils et laisser le LLM les orchestrer.

---

## Exemple étape par étape : Tâche d'agent de bout en bout utilisant des outils

**Tâche :**
> "Informer un client que sa commande est retardée et proposer une nouvelle heure de livraison."

**Voici comment le système fonctionne avec les outils :**

**Entrée** — Un humain tape :
_"Hé, peux-tu informer John que sa commande est retardée et la reprogrammer pour demain ?"_

**Planification** — Le LLM décompose :
- Vérifier le statut de la commande
- Si retardée, vérifier les créneaux de livraison
- Rédiger un e-mail
- Envoyer l'e-mail
- Enregistrer l'interaction

**Appels d'outils :**
```text
get_order_status(order_id=12345)
get_available_slots(date=today+1)
send_email(to=john@example.com, content=...)
log_event(event_type="reschedule", status="completed")
```

**Génération de texte** — Le LLM compose le message :
_"Bonjour John, je voulais vous informer que votre commande a été retardée. Nous l'avons reprogrammée pour demain. Merci de votre patience."_

**Exécution** — Le système exécute les actions, enregistre la sortie et envoie éventuellement une mise à jour de statut à un tableau de bord.

---

## Comment cela fonctionne (Visuel)
<img width="808" height="357" alt="image" src="https://github.com/user-attachments/assets/f7ce3097-873f-4519-b7ba-30b80785deae" />


Voici ce qui se passe :

1. L'utilisateur pose une question ou donne une tâche.
2. Le LLM comprend ce qui doit être fait et planifie sa prochaine étape.
3. Un analyseur convertit l'idée du LLM en un format structuré (comme `get_order_status(order_id=12345)`).
4. L'agent appelle le bon outil — API, requête de base de données ou fonction interne.
5. L'outil retourne un résultat — c'est ce qu'on appelle une **observation**.
6. Le LLM examine le résultat, décide de ce qui manque ou de ce qui vient ensuite.
7. Cette boucle continue jusqu'à ce qu'il ait assez pour générer la réponse finale ou accomplir la tâche.

Le LLM utilise le résultat de chaque outil pour guider sa prochaine décision.

---

**Rappel important :**
Le LLM lui-même ne fait toujours que générer du texte.
Ce texte est structuré en appels d'outils, exécuté en externe, et les résultats sont réinjectés dans le LLM — créant une boucle de raisonnement, d'action et de réflexion (**alias un agent**).

Cette structure est utilisée par des frameworks comme **LangChain**, **CrewAI**, **AutoGen**, et même des configurations d'orchestration personnalisées dans les équipes de production.

---

## Qu'est-ce qui rend un outil utilisable par un LLM ?

Pour enregistrer un outil avec un système d'agent, vous définissez généralement :
- **Nom** (par exemple, `create_meeting`)
- **Description** (pour que le modèle sache quand l'utiliser)
- **Paramètres d'entrée** (et types)
- **Structure de sortie** (pour que le modèle puisse utiliser le résultat)

Ces métadonnées sont ce qui permet au LLM de raisonner sur quel outil utiliser et comment.

---

## Note sur l'analyse et les sorties structurées

L'analyseur joue un rôle clé dans la conversion de la réponse du LLM en un appel d'outil structuré — quelque chose que le système peut exécuter de manière fiable (comme `get_order_status(order_id=12345)`).

Mais dans de nombreuses configurations modernes, vous n'avez pas toujours besoin d'un analyseur séparé.
La plupart des LLM populaires, en particulier ceux conçus pour l'utilisation d'outils, peuvent produire directement des sorties structurées — comme JSON ou des appels de fonction — qui peuvent être consommées par votre backend telles quelles.

De même, des outils bien conçus retournent des données structurées, ce qui facilite le raisonnement du LLM sur quoi faire ensuite.

**La structure des deux côtés** (entrée et sortie) est ce qui rend les boucles d'agents **robustes, traçables et de qualité production**.

---

## Le point à retenir

Beaucoup de ceci vous semblera familier si vous avez construit ou travaillé avec des API auparavant.
Mais si vous ne venez pas de ce monde, ne compliquez pas trop le câblage.

Souvenez-vous simplement de ceci :
> Les modèles IA seuls peuvent **comprendre** et **générer**.
> Lorsqu'ils sont connectés à des logiciels, des outils, des API et des systèmes internes — ils peuvent réellement **faire des choses**.

---

Dans la partie suivante, nous découvrirons la **génération augmentée par récupération (RAG)** — ce que c'est, quand l'utiliser, et comment elle s'intègre naturellement dans les pipelines agentiques comme une couche de mémoire ou de contexte.

PS : Nous enseignons également un cours très apprécié sur la façon de construire réellement des systèmes IA dans cet environnement en évolution rapide, en utilisant une approche axée sur les problèmes. Il est conçu pour les PM, les leaders, les ingénieurs, les décideurs, etc. qui travaillent dans des contraintes du monde réel. Les anciens élèves viennent de Google, Meta, Apple, Netflix, AWS, Spotify, Snapchat, Deloitte et bien d'autres. Notre prochaine cohorte commence bientôt. Les tarifs early bird sont en ligne : utilisez le code "GITHUB" pour obtenir 300 $ de réduction (valide uniquement pour août 2025) pour [vous inscrire ici](https://maven.com/aishwarya-kiriti/genai-system-design) !!

