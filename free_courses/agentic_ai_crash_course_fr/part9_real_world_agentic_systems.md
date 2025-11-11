# Partie 9 : Systèmes agentiques du monde réel (Sous le capot)



---

Jusqu'à présent, nous avons couvert tous les ingrédients qui composent un agent :
**outils**, **planification**, **RAG**, **mémoire**, **structure**, et **coordination** dans les configurations multi-agents.

Mais vous pensez peut-être :
> "Où tout cela apparaît-il réellement dans le monde réel ?"

Passons en revue quelques systèmes publics qui présentent un **comportement agentique** — autant que nous puissions en juger.

⚠️ **Note :**
Ce ne sont pas open source. Nous ne connaissons pas leurs internes exacts.
Ce qui suit est une simplification informée basée sur la façon dont ils se comportent extérieurement — juste assez pour comprendre comment la stack agentique peut apparaître en pratique.

---

## **NotebookLM (Google) : Recherche agentique sur vos propres données**

NotebookLM de Google agit comme un assistant de recherche personnel. Vous téléchargez vos fichiers, et il vous aide à travailler avec eux — résumer, répondre aux questions, même générer des versions audio ou des guides d'étude.

**Focus principal :** Q&R sur votre contenu — essentiellement un système RAG personnel à grande échelle.

**Comment cela fonctionne probablement :**
1. **L'utilisateur télécharge des fichiers** (PDF, notes, slides, etc.)
2. **Prétraitement** — Les stocke pour récupération ultérieure.
3. **L'utilisateur pose une question** — par ex., _"Quels étaient les insights clés de mon deck stratégie Q2 ?"_
4. **Planification** — Interprète le type de tâche (résumé, Q&R, comparaison ?), identifie les docs/sections pertinents.
5. **RAG** — Récupère les morceaux de document les plus pertinents.
6. **Génération LLM** — Répond clairement, fondé dans votre contenu.
7. **Mémoire** —
   - Court terme : Suit la conversation.
   - Long terme : Probablement minimal ou nul.
8. **Outils** — Possiblement lecteurs de fichiers, modules de résumé.

**Ce qui le rend agentique :** Interprète les objectifs, recherche dans vos données, et compose des réponses — pas seulement des sorties statiques.

---

## **Perplexity : Recherche agentique sur le web ouvert**

Perplexity vous donne une réponse directe, semblable à une réponse avec des sources — au lieu d'une page de liens.

**Comment cela fonctionne probablement :**
1. **L'utilisateur pose une question** — par ex., _"Quelle est la dernière recherche sur les traitements d'Alzheimer ?"_
2. **Planification** — Interprète l'intention ("dernière", "crédible"), décide de l'approche de recherche.
3. **Utilisation d'outils** — Émet des requêtes via des API web.
4. **RAG** — Récupère des extraits de pages pertinents.
5. **Réponse LLM** — Synthétise une réponse avec citations.
6. **Mémoire** —
   - Court terme : Contexte de session.
   - Long terme : Peut stocker des préférences (par ex., "toujours utiliser WSJ pour les nouvelles").

**Ce qui le rend agentique :** Récupère des infos, décide quoi utiliser, et construit une réponse dans une boucle multi-étapes.

---

## **DeepResearch (OpenAI) : Workflows agentiques profonds**

DeepResearch s'attaque à des **tâches de recherche complexes et ouvertes** — par ex., analyse de marché, paysages concurrentiels, plongées techniques profondes.

**Comment cela fonctionne probablement :**
1. **L'utilisateur demande une tâche large** — par ex., _"Analyser le paysage de l'IA générative pour les startups éducatives."_
2. **Planification** — Décompose en sous-tâches (financement, tendances, entreprises, risques), forme un plan d'exécution.
3. **Outils** — Inclut probablement :
   - Recherche web
   - Lecteurs de documents (PDF)
   - Outils de données (tableurs, graphiques)
   - Modules de génération de rapports
4. **RAG agentique** — Pas de récupération à coup unique — récupère, réfléchit, re-récupère à mesure que la tâche évolue.
5. **Mémoire** —
   - Épisodique : Suit quelles parties sont terminées.
   - Sémantique : Stocke des faits/noms clés.
6. **Raisonnement multi-étapes** — Boucles : planifier → récupérer → lire → repenser → générer → raffiner → répéter.

**Ce qui le rend agentique :** Planification lourde, utilisation itérative d'outils, progression auto-dirigée.

---

## **Connexion au Jour 2 : Niveaux d'autonomie**
<img width="694" height="370" alt="image" src="https://github.com/user-attachments/assets/48812496-309d-42cd-9c86-8ef3cb345ec2" />
<img width="969" height="231" alt="image" src="https://github.com/user-attachments/assets/12489209-a75b-4a31-853e-33dda02e1aaa" />



**NotebookLM** — Entre Niveau 2 et Niveau 3.
- Agent de workflow à contrôle élevé.
- Récupération forte, prise de décision autonome limitée.

**Perplexity** — Niveau 3 (peut-être touchant le Niveau 4).
- Planifie les requêtes, organise les sources, élabore des réponses.

**DeepResearch** — Fort Niveau 4.
- Prend des objectifs de haut niveau, décompose les tâches, travaille de manière itérative avec guidance minimale.

---

## Essayez par vous-même

Ils ont tous des versions gratuites — expérimentez et observez :
- Combien de **contrôle** vous avez
- Combien le **système décide** par lui-même

C'est un excellent moyen d'aiguiser votre instinct pour la conception d'agents.

---

## À suivre

Dans la partie suivante, nous conclurons la série :
- Résumer ce que nous avons appris
- Partager les meilleures pratiques
- Jeter un coup d'œil rapide sur où **l'IA agentique** se dirige

PS : Nous enseignons également un cours très apprécié sur la façon de construire réellement des systèmes IA dans cet environnement en évolution rapide, en utilisant une approche axée sur les problèmes. Il est conçu pour les PM, les leaders, les ingénieurs, les décideurs, etc. qui travaillent dans des contraintes du monde réel. Les anciens élèves viennent de Google, Meta, Apple, Netflix, AWS, Spotify, Snapchat, Deloitte et bien d'autres. Notre prochaine cohorte commence bientôt. Les tarifs early bird sont en ligne : utilisez le code "GITHUB" pour obtenir 300 $ de réduction (valide uniquement pour août 2025) pour [vous inscrire ici](https://maven.com/aishwarya-kiriti/genai-system-design) !!

