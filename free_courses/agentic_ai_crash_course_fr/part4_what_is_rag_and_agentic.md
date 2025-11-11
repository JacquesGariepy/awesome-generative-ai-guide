# Partie 4 : Génération augmentée par récupération (RAG) et l'essor du RAG agentique

Dans la partie précédente, nous avons vu comment les outils aident les agents IA à interagir avec les systèmes du monde réel — envoyer des e-mails, créer des tickets, déclencher des API.

Mais que se passe-t-il si le modèle n'a pas besoin d'agir ?
Que se passe-t-il s'il a simplement besoin d'accéder aux bonnes informations ?

C'est le cas dans de nombreux contextes d'entreprise :
- Documents internes répartis entre les équipes
- PDF de politiques que personne ne se souvient avoir écrits
- Insights clients enfouis dans les notes CRM
- Tableaux de bord et e-mails avec un contexte utile

Les outils n'aideront pas ici. Le modèle doit penser avec vos données.
C'est là que le **RAG** entre en jeu.

---

## Qu'est-ce que le RAG ?

RAG signifie **Génération augmentée par récupération**.
C'est une conception de système où le modèle récupère des informations pertinentes de vos propres données — juste avant de générer une réponse.

Au lieu de s'appuyer uniquement sur ce sur quoi le modèle a été entraîné, RAG lui donne accès à des **informations contextuelles en direct** de vos systèmes d'entreprise. Cela rend les réponses plus précises, fondées et auditables.

Vous vous demandez peut-être :
> "Pourquoi ne pas simplement donner toutes les données directement au modèle ?"

Le problème est :
- Les modèles ne peuvent traiter qu'une quantité limitée de texte à la fois.
- Même dans cette limite, ils ont du mal lorsque trop d'informations non pertinentes ou bruyantes sont incluses.
- Cela rend les réponses moins ciblées et plus sujettes aux erreurs.

---

## Le processus RAG (en un coup d'œil)

<img width="1024" height="356" alt="image" src="https://github.com/user-attachments/assets/2e7c2384-a564-45c1-9a08-57cfb02ee435" />


Voici à quoi cela ressemble en pratique :

1. **Données** – Votre contenu interne (PDF, e-mails, notes, wikis)
2. **Découpage** – Divisé en parties plus petites pour une meilleure indexation
3. **Prompt + Contexte** – Au moment de la requête, le système récupère les morceaux pertinents (phase de récupération)
4. **LLM** – Le modèle utilise ce contexte pour générer une réponse
5. **Sortie** – Le résultat est basé sur vos données, pas seulement sur ce que le modèle "sait"

_Source de l'image : https://hyperight.com/7-practical-applications-of-rag-models-and-their-impact-on-society/_

---

## Pourquoi le RAG est partout dans l'IA d'entreprise

Vous entendrez souvent ce chiffre :
> D'après ce que j'ai vu chez les clients et les systèmes, **70 % des cas d'usage GenAI d'entreprise utilisent RAG**.

Pourquoi RAG est inestimable pour les entreprises :
- Les connaissances d'entreprise changent fréquemment
- Le fine-tuning des modèles est coûteux et lent
- La récupération est plus rapide, plus sûre et plus facile à contrôler
- Elle apporte structure et traçabilité dans les systèmes LLM
- Elle fonctionne sur des données à la fois non structurées (docs) et semi-structurées (tableaux de bord, notes)

Donc au lieu de demander :
> "Comment puis-je enseigner au modèle tout ce que nous savons ?"
La plupart des équipes demandent :
> "Comment puis-je laisser le modèle récupérer ce que nous avons déjà ?"

---

## RAG = LLM + Données récupérées supplémentaires

RAG est devenu le modèle dominant en 2024 pour une raison :
Il a comblé le fossé entre les LLM à usage général et les connaissances d'entreprise privées, spécifiques aux tâches.

À la base, RAG est simple :
- Vous prenez un LLM
- Vous lui alimentez des informations supplémentaires récupérées juste avant la génération

Cela rend le modèle plus précis, plus conscient du contexte et moins dépendant des faits mémorisés.
C'est particulièrement utile pour des tâches comme **Q&R, résumé et recherches de politiques** — particulièrement dans des environnements riches en données comme **juridique, finance et support**.

Pas étonnant que 2024 ait été surnommée **"l'année du RAG."**

---

## Mais maintenant nous entrons dans l'ère agentique

RAG ne disparaît pas, mais il évolue.

Les systèmes d'aujourd'hui ne récupèrent pas seulement une fois et génèrent une réponse.
Dans les **workflows agentiques**, la récupération devient partie d'une boucle de raisonnement plus large et dynamique.

Les agents planifient, récupèrent, réfléchissent et récupèrent à nouveau — pas seulement une fois, mais autant de fois que nécessaire tout au long d'une tâche.

C'est là que le **RAG agentique** entre en jeu.

---

## Qu'est-ce que le RAG agentique ?

<img width="1456" height="971" alt="image" src="https://github.com/user-attachments/assets/8a7347b0-2ead-4d56-9f33-ef57667d0f00" />


RAG traditionnel :
- Une requête
- Une récupération
- Une réponse

Il fonctionne bien pour des questions autonomes comme :
> "Quelle est notre politique sur le report de PTO ?"

Mais la plupart des workflows d'entreprise du monde réel ne sont pas à coup unique.

---

**Exemple :**
Disons que vous construisez un assistant de transaction pour votre équipe commerciale.
Dans une seule tâche, l'agent peut avoir besoin de :
- Extraire l'historique CRM du client
- Récupérer la tarification actuelle pour leur segment
- Rechercher les termes légaux régionaux
- Référencer des clauses de contrats passés
- Générer une proposition personnalisée
- Vérifier les faits
- Enregistrer l'interaction

---

Dans les **systèmes agentiques**, la récupération n'est pas seulement une étape de configuration.
C'est comment l'agent :
- Rassemble le contexte manquant
- Vérifie ses hypothèses
- S'adapte en cours de tâche

Cela signifie que RAG devient :
- Un outil pour l'apprentissage en cours de tâche
- Une méthode pour réduire les hallucinations
- Un mécanisme pour gérer les workflows dynamiques
- Un pont entre le raisonnement et les connaissances d'entreprise fondées

Le RAG agentique transforme la récupération en une **boucle de prise de décision de première classe** en utilisant la récupération comme partie du processus de pensée du modèle.

---

## RAG comme outil

Si vous y pensez, RAG est aussi une sorte d'**outil**.
Mais au lieu de déclencher une action, il aide l'agent à extraire les bonnes informations d'un grand volume de données.

En pratique, les agents combinent souvent :
- **RAG**
- **Outils**
- **Planification**

…pour accomplir des tâches complexes **de manière fiable et contextuelle**.

---

## Note sur la portée

RAG est un espace profond et en évolution rapide — honnêtement, il pourrait être son propre cours.
Si vous êtes curieux d'explorer davantage :
- J'ai organisé un **dépôt GitHub** d'articles clés sur RAG qui couvre bien le paysage
- J'ai aussi un **guide 101 sur RAG agentique**

Cela dit, toutes les optimisations RAG ne sont pas nécessaires pour chaque cas d'usage.
Dans notre cours de 6 semaines, nous nous concentrons sur vous aider à comprendre **quand et où** chaque technique a du sens, plutôt que de les appliquer aveuglément.

---


Dans la partie suivante, nous plongerons dans l'un des concepts les plus discutés dernièrement : **le protocole de contexte de modèle (MCP)**.

Pour en tirer le meilleur parti, je recommanderais de revisiter la **Partie 3 sur les outils**, car MCP se construit directement sur ce concept !

PS : Nous enseignons également un cours très apprécié sur la façon de construire réellement des systèmes IA dans cet environnement en évolution rapide, en utilisant une approche axée sur les problèmes. Il est conçu pour les PM, les leaders, les ingénieurs, les décideurs, etc. qui travaillent dans des contraintes du monde réel. Les anciens élèves viennent de Google, Meta, Apple, Netflix, AWS, Spotify, Snapchat, Deloitte et bien d'autres. Notre prochaine cohorte commence bientôt. Les tarifs early bird sont en ligne : utilisez le code "GITHUB" pour obtenir 300 $ de réduction (valide uniquement pour août 2025) pour [vous inscrire ici](https://maven.com/aishwarya-kiriti/genai-system-design) !!
