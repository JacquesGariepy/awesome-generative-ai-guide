# Partie 5 : Qu'est-ce que le MCP et pourquoi devriez-vous vous en soucier ?

---

## D'abord, un récapitulatif rapide

- **Partie 3 :** Nous avons appris que les outils permettent aux modèles de **faire des choses**.
- **Partie 4 :** Nous avons vu que RAG aide les modèles à **trouver des informations pertinentes** avant de répondre.

Ce sont des **supports externes** — ils aident le modèle à agir plus intelligemment, mais la coordination reste à l'extérieur du modèle.

Mais que se passerait-il si vous pouviez passer **tout le contexte dont un modèle a besoin** — outils, données récupérées, mémoire, instructions — dans un format propre et structuré ?

C'est ce que le **protocole de contexte de modèle (MCP)** essaie de résoudre.

---

## Alors qu'est-ce que le MCP ?
<img width="737" height="452" alt="image" src="https://github.com/user-attachments/assets/2e17fcb6-ab35-4a4c-88b4-c74db165e4b8" />


À la base, le **protocole de contexte de modèle** est une façon standardisée de donner à un LLM tout ce dont il a besoin pour raisonner et répondre.

Pensez-y comme à l'emballage de :

- La tâche que vous voulez que le modèle fasse
- Les outils/API qu'il peut utiliser
- Les documents ou la mémoire dont il pourrait avoir besoin
- Les messages précédents dans la conversation

…et ensuite remettre tout cela en une seule fois.

Ce n'est **pas** un outil, une bibliothèque ou un produit.
C'est un **protocole** — une structure pour la communication entre le modèle et le monde extérieur.

Si vous venez du monde de la tech, les équivalents seraient : **HTTP**, **TCP/IP**, ou **SMTP**.
Si ce n'est pas le cas, souvenez-vous simplement : les gens de la tech adorent la standardisation — cela facilite la réutilisation et l'assemblage des choses.

---

## Pourquoi est-ce important ?

Disons que vous construisez un agent.
En ce moment, vous jonglez probablement avec :

- Envoyer un prompt
- Passer des documents récupérés
- Enregistrer des outils
- Gérer l'état
- Suivre ce qui s'est passé avant

MCP dit :
> "Standardisons la façon dont nous donnons tout cela au modèle, pour ne pas réinventer la roue pour chaque cas d'usage."

Et pour les **entreprises**, cela compte beaucoup.
À mesure que les agents deviennent plus complexes, coordonner **outils**, **RAG**, **mémoire** et **sorties** devient compliqué.

MCP rend cette orchestration **composable**, **modulaire** et plus facile à brancher dans d'autres systèmes.

Si vous avez déjà travaillé avec des API, pensez à MCP comme à un **schéma de requête bien défini**.
Au lieu de tout jeter dans une longue chaîne et d'espérer que le modèle comprenne, le modèle voit toujours du texte — mais il est **structuré**, avec un **contexte, des options et un fondement clairs**.

---

## Pourquoi MCP s'est-il imposé si rapidement ?

Étant donné que MCP n'est qu'un protocole, vous vous demandez peut-être :
> Qu'est-ce qui le rend meilleur, et pourquoi tout le monde s'y est-il mis ?

Voici ce qui a aidé :

1. **Natif pour l'IA** — MCP a été construit pour les agents IA. Il fait de la place pour tout ce que les agents utilisent aujourd'hui : outils, prompts, mémoire, documents, et plus.
2. **Documentation et exemples solides** — Anthropic (créateurs de MCP) a publié non seulement la spécification mais aussi des clients, des SDK, des outils de test et des démos du monde réel.
3. **Effet de réseau** — Publié discrètement en novembre 2024, la plupart des gens l'ont ignoré… jusqu'en 2025, où il a explosé. Des outils, des startups et même OpenAI ont commencé à le supporter.

---

## Malentendus courants

- **MCP n'est pas une nouvelle API ou un produit** — C'est juste un modèle, une façon propre de cadrer ce que vous envoyez au modèle.
- **Il ne rend pas les modèles plus intelligents** — Il leur donne simplement un meilleur contexte plus structuré.
- **Ce n'est pas seulement pour les agents** — Même les assistants simples bénéficient d'une meilleure gestion du contexte.

---

## Alors… Devriez-vous vous en soucier ?

Si vous construisez des prompts jouets ou des démos rapides — probablement pas (encore).

Mais si vous travaillez sur :

- Des agents de niveau entreprise
- Des workflows multi-outils
- Des LLM qui doivent accéder à **mémoire + RAG + planification**
- Des systèmes où la **gestion du contexte** est un goulot d'étranglement

…alors **oui**, vous devriez vous en soucier. MCP concerne l'amélioration du passage de contexte évolutif et structuré aux modèles.

Mais gardez à l'esprit : MCP n'est qu'un protocole.
Comme toutes les normes, il ne fonctionne que s'il est largement adopté.
Si quelque chose de meilleur arrive avant que MCP ne devienne "le HTTP des agents", l'écosystème pourrait à nouveau changer.

---

## Lectures complémentaires et ressources

- Nous avons fait une **[analyse approfondie complète](https://thenuancedperspective.substack.com/p/mcp-overhyped-misunderstood-and-actually)** sur MCP, y compris les clients, les serveurs et les cas d'usage du monde réel (écrit par Kiriti Badam, OpenAI).
- Nous avons également organisé une **session en direct gratuite** — vous pouvez regarder l'[enregistrement](https://maven.com/p/82345a) ici.

---

Dans la partie suivante, nous découvrirons le composant **planification** des systèmes agentiques et pourquoi il est important.

PS : Nous enseignons également un cours très apprécié sur la façon de construire réellement des systèmes IA dans cet environnement en évolution rapide, en utilisant une approche axée sur les problèmes. Il est conçu pour les PM, les leaders, les ingénieurs, les décideurs, etc. qui travaillent dans des contraintes du monde réel. Les anciens élèves viennent de Google, Meta, Apple, Netflix, AWS, Spotify, Snapchat, Deloitte et bien d'autres. Notre prochaine cohorte commence bientôt. Les tarifs early bird sont en ligne : utilisez le code "GITHUB" pour obtenir 300 $ de réduction (valide uniquement pour août 2025) pour [vous inscrire ici](https://maven.com/aishwarya-kiriti/genai-system-design) !!


