# Chapitre 22 - Partie 2: Tree of Thoughts et Raisonnement Avancé

## Tree of Thoughts (ToT)

**Tree of Thoughts** permet d'explorer systématiquement un arbre de possibilités, avec backtracking si nécessaire - similaire à une recherche en profondeur ou à Monte Carlo Tree Search.

```python
"""
TREE OF THOUGHTS (ToT) - Yao et al. 2023

Problème avec CoT linéaire:
  • Un seul path de raisonnement
  • Si erreur en étape 2 → toute la solution échoue
  • Pas de backtracking

Solution ToT:
  • Explorer ARBRE de pensées
  • Chaque étape = plusieurs options
  • Évaluer chaque branche
  • Backtrack si deadend

Exemple: Game of 24
  Input: [4, 9, 10, 13]
  Goal: Obtenir 24 avec +, -, ×, ÷

  Tree:
           Root: [4,9,10,13]
          /      |      \
     (4+9) =13  (4×9)=36  (10-4)=6
      [13,10,13] [36,10,13] [6,9,13]
         |
    13+10=23 ✗  13+13=26 ✗ (backtrack)
         |
    (36-13)=23
    [23, 10] ...

Algorithme:
  1. Generate: Créer N candidats pour prochaine étape
  2. Evaluate: Noter chaque candidat (prometteur?)
  3. Select: Choisir top-K
  4. Repeat ou Backtrack

Variantes de search:
  • BFS (Breadth-First): Explorer tous au même niveau
  • DFS (Depth-First): Aller profond, backtrack si échec
  • Beam Search: Garder top-K à chaque niveau

Performance vs CoT:
  Game of 24: 74% → 95% (+21%)
  Creative Writing: 64% → 85%
  Crosswords: 60% → 78%

Coût:
  ❌ Beaucoup plus de LLM calls
  ❌ Complexité exponentielle sans pruning

Paper: Tree of Thoughts: Deliberate Problem Solving with LLMs
       (Yao et al., NeurIPS 2023)
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq


class SearchStrategy(Enum):
    """Stratégie de recherche dans l'arbre"""
    BFS = "bfs"  # Breadth-First Search
    DFS = "dfs"  # Depth-First Search
    BEAM = "beam"  # Beam Search


@dataclass
class ThoughtNode:
    """Nœud dans l'arbre de pensées"""
    state: str  # État actuel (pensée)
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    score: float = 0.0  # Score d'évaluation
    depth: int = 0
    is_solution: bool = False

    def get_path(self) -> List[str]:
        """Retourne le chemin depuis la racine"""
        path = []
        current = self
        while current:
            path.insert(0, current.state)
            current = current.parent
        return path


class TreeOfThoughts:
    """
    Tree of Thoughts (ToT) avec différentes stratégies de recherche

    Exemple:
        >>> tot = TreeOfThoughts(
        ...     model=my_llm,
        ...     max_depth=5,
        ...     branching_factor=3
        ... )
        >>> solution = tot.solve(
        ...     problem="Trouver comment obtenir 24 avec [4,9,10,13]",
        ...     strategy=SearchStrategy.BFS
        ... )
    """

    def __init__(
        self,
        model: Any,
        max_depth: int = 5,
        branching_factor: int = 3,
        beam_width: int = 5
    ):
        """
        Args:
            model: Modèle LLM
            max_depth: Profondeur max de l'arbre
            branching_factor: Nombre de branches par nœud
            beam_width: Largeur pour beam search
        """
        self.model = model
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.beam_width = beam_width

        self.nodes_explored = 0
        self.llm_calls = 0

    def solve(
        self,
        problem: str,
        strategy: SearchStrategy = SearchStrategy.BFS
    ) -> Dict[str, Any]:
        """
        Résout un problème avec Tree of Thoughts

        Args:
            problem: Description du problème
            strategy: Stratégie de recherche

        Returns:
            {
                "solution": ThoughtNode ou None,
                "nodes_explored": int,
                "llm_calls": int
            }
        """
        print("\n" + "="*80)
        print("TREE OF THOUGHTS")
        print("="*80)
        print(f"Problème: {problem}")
        print(f"Stratégie: {strategy.value}")
        print(f"Max depth: {self.max_depth}")
        print(f"Branching factor: {self.branching_factor}")

        # Réinitialiser compteurs
        self.nodes_explored = 0
        self.llm_calls = 0

        # Racine de l'arbre
        root = ThoughtNode(
            state=f"Problème: {problem}",
            depth=0
        )

        # Rechercher selon stratégie
        if strategy == SearchStrategy.BFS:
            solution = self._bfs(root, problem)
        elif strategy == SearchStrategy.DFS:
            solution = self._dfs(root, problem)
        elif strategy == SearchStrategy.BEAM:
            solution = self._beam_search(root, problem)
        else:
            raise ValueError(f"Stratégie {strategy} inconnue")

        print(f"\n📊 Statistiques:")
        print(f"  Nœuds explorés: {self.nodes_explored}")
        print(f"  Appels LLM: {self.llm_calls}")

        if solution:
            print(f"\n✅ Solution trouvée à profondeur {solution.depth}:")
            path = solution.get_path()
            for i, step in enumerate(path):
                print(f"  Étape {i}: {step}")
        else:
            print(f"\n❌ Aucune solution trouvée")

        return {
            "solution": solution,
            "nodes_explored": self.nodes_explored,
            "llm_calls": self.llm_calls
        }

    def _bfs(self, root: ThoughtNode, problem: str) -> Optional[ThoughtNode]:
        """Breadth-First Search"""
        queue = [root]

        while queue:
            node = queue.pop(0)
            self.nodes_explored += 1

            # Vérifier si solution
            if self._is_solution(node, problem):
                node.is_solution = True
                return node

            # Atteint max depth?
            if node.depth >= self.max_depth:
                continue

            # Générer enfants
            children = self._generate_children(node, problem)

            # Ajouter à la queue
            queue.extend(children)

        return None

    def _dfs(self, root: ThoughtNode, problem: str) -> Optional[ThoughtNode]:
        """Depth-First Search avec backtracking"""

        def dfs_recursive(node: ThoughtNode) -> Optional[ThoughtNode]:
            self.nodes_explored += 1

            # Solution?
            if self._is_solution(node, problem):
                node.is_solution = True
                return node

            # Max depth?
            if node.depth >= self.max_depth:
                return None

            # Générer et explorer enfants
            children = self._generate_children(node, problem)

            for child in children:
                result = dfs_recursive(child)
                if result:
                    return result

            # Backtrack (aucun enfant n'a mené à solution)
            return None

        return dfs_recursive(root)

    def _beam_search(self, root: ThoughtNode, problem: str) -> Optional[ThoughtNode]:
        """Beam Search: Garder top-K candidats à chaque niveau"""
        beam = [root]

        for depth in range(self.max_depth):
            # Générer tous les candidats de ce niveau
            all_candidates = []

            for node in beam:
                self.nodes_explored += 1

                # Solution?
                if self._is_solution(node, problem):
                    node.is_solution = True
                    return node

                # Générer enfants
                children = self._generate_children(node, problem)
                all_candidates.extend(children)

            if not all_candidates:
                break

            # Garder top-K selon score
            beam = heapq.nlargest(
                self.beam_width,
                all_candidates,
                key=lambda n: n.score
            )

        # Retourner meilleur candidat du dernier beam
        if beam:
            return max(beam, key=lambda n: n.score)

        return None

    def _generate_children(
        self,
        parent: ThoughtNode,
        problem: str
    ) -> List[ThoughtNode]:
        """
        Génère enfants pour un nœud

        En production: Appel LLM pour générer N pensées candidates

        Args:
            parent: Nœud parent
            problem: Problème original

        Returns:
            Liste de nœuds enfants
        """
        # Prompt pour générer candidats
        prompt = f"""Problème: {problem}

Pensée actuelle: {parent.state}

Génère {self.branching_factor} prochaines étapes possibles de raisonnement.
Format: Une pensée par ligne."""

        # Simulation
        # En production: candidates = self.model.generate(prompt)
        self.llm_calls += 1

        # Générer candidats simulés
        candidates = [
            f"Option {i+1} depuis '{parent.state[:30]}...'"
            for i in range(self.branching_factor)
        ]

        # Créer nœuds enfants
        children = []
        for i, candidate in enumerate(candidates):
            # Évaluer chaque candidat
            score = self._evaluate_thought(candidate, problem, parent.depth + 1)
            self.llm_calls += 1

            child = ThoughtNode(
                state=candidate,
                parent=parent,
                score=score,
                depth=parent.depth + 1
            )
            children.append(child)

        # Ajouter au parent
        parent.children = children

        return children

    def _evaluate_thought(
        self,
        thought: str,
        problem: str,
        depth: int
    ) -> float:
        """
        Évalue la promesse d'une pensée

        En production: Appel LLM pour scorer

        Returns:
            Score entre 0 et 1 (1 = très prometteur)
        """
        # Prompt d'évaluation
        prompt = f"""Problème: {problem}

Pensée à évaluer: {thought}

Cette pensée est-elle prometteuse pour résoudre le problème?
Score de 0 (mauvais) à 10 (excellent):"""

        # Simulation: score diminue avec profondeur (encourage exploration large)
        import random
        base_score = random.uniform(0.3, 0.9)
        depth_penalty = depth * 0.05
        score = max(0.1, base_score - depth_penalty)

        return score

    def _is_solution(self, node: ThoughtNode, problem: str) -> bool:
        """
        Vérifie si un nœud est une solution

        En production: Appel LLM ou vérification programmatique

        Returns:
            True si solution complète
        """
        # Simulation: Solution trouvée à profondeur max avec probabilité
        if node.depth >= self.max_depth - 1:
            import random
            return random.random() < 0.3  # 30% chance

        return False


# ============================================================================
# GRAPH OF THOUGHTS (GoT)
# ============================================================================

"""
GRAPH OF THOUGHTS = Extension de ToT avec graphe général

ToT = Arbre (chaque nœud a 1 parent)
GoT = Graphe (nœuds peuvent être combinés, cycles possibles)

Opérations supplémentaires:
  1. Aggregation: Combiner plusieurs pensées
  2. Refinement: Améliorer une pensée
  3. Generation: Créer nouvelle branche

Exemple: Essay writing
  - Génère 3 introductions
  - Agrège meilleurs éléments
  - Raffine le résultat
  - Génère corps basé sur intro raffinée

Avantages vs ToT:
  ✅ Plus flexible
  ✅ Peut combiner idées de multiples branches
  ✅ Meilleur pour tâches créatives

Complexité:
  ❌ Plus complexe à implémenter
  ❌ Encore plus de LLM calls

Paper: Graph of Thoughts (Besta et al., 2023)
"""

@dataclass
class GraphNode:
    """Nœud dans un graphe de pensées"""
    id: str
    content: str
    parents: List['GraphNode'] = field(default_factory=list)
    children: List['GraphNode'] = field(default_factory=list)
    node_type: str = "thought"  # "thought", "aggregation", "refinement"
    score: float = 0.0


class GraphOfThoughts:
    """
    Graph of Thoughts (GoT)

    Plus flexible que ToT - permet agrégation et cycles
    """

    def __init__(self, model: Any):
        self.model = model
        self.nodes: Dict[str, GraphNode] = {}
        self.node_counter = 0

    def create_node(
        self,
        content: str,
        node_type: str = "thought",
        parents: Optional[List[GraphNode]] = None
    ) -> GraphNode:
        """Créer un nouveau nœud"""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1

        node = GraphNode(
            id=node_id,
            content=content,
            parents=parents or [],
            node_type=node_type
        )

        # Ajouter comme enfant aux parents
        for parent in node.parents:
            parent.children.append(node)

        self.nodes[node_id] = node
        return node

    def aggregate(
        self,
        nodes: List[GraphNode],
        task: str
    ) -> GraphNode:
        """
        Agrège plusieurs pensées en une

        Args:
            nodes: Nœuds à agréger
            task: Contexte/objectif

        Returns:
            Nouveau nœud agrégé
        """
        print(f"\n🔗 Aggregation de {len(nodes)} pensées")

        # Prompt d'agrégation
        thoughts = "\n".join([
            f"{i+1}. {node.content}"
            for i, node in enumerate(nodes)
        ])

        prompt = f"""Tâche: {task}

Pensées à agréger:
{thoughts}

Créé une pensée unifiée qui combine les meilleures idées:"""

        # En production: aggregated = self.model.generate(prompt)
        aggregated = f"Agrégation de {len(nodes)} pensées sur: {task}"

        # Créer nœud d'agrégation
        node = self.create_node(
            content=aggregated,
            node_type="aggregation",
            parents=nodes
        )

        print(f"  → {aggregated[:60]}...")

        return node

    def refine(
        self,
        node: GraphNode,
        feedback: str
    ) -> GraphNode:
        """
        Raffine une pensée

        Args:
            node: Nœud à raffiner
            feedback: Feedback pour amélioration

        Returns:
            Nœud raffiné
        """
        print(f"\n✨ Raffinement avec feedback: {feedback}")

        prompt = f"""Pensée originale: {node.content}

Feedback: {feedback}

Améliore la pensée en tenant compte du feedback:"""

        # En production: refined = self.model.generate(prompt)
        refined = f"[Raffiné] {node.content}"

        refined_node = self.create_node(
            content=refined,
            node_type="refinement",
            parents=[node]
        )

        print(f"  → {refined[:60]}...")

        return refined_node


# ============================================================================
# BEST PRACTICES ET COMPARAISON
# ============================================================================

def compare_reasoning_techniques():
    """Compare toutes les techniques de reasoning"""
    print("\n" + "="*80)
    print("COMPARAISON DES TECHNIQUES DE REASONING")
    print("="*80)

    print("""
┌────────────────────┬─────────────┬──────────────┬──────────────┬─────────────┐
│ Technique          │ Complexité  │ LLM Calls    │ Best For     │ Limitations │
├────────────────────┼─────────────┼──────────────┼──────────────┼─────────────┤
│ Direct Answer      │ O(1)        │ 1            │ Simple Q&A   │ Pas de      │
│                    │             │              │              │ reasoning   │
├────────────────────┼─────────────┼──────────────┼──────────────┼─────────────┤
│ Chain-of-Thought   │ O(1)        │ 1            │ Math, logic  │ Linear      │
│ (Zero/Few-shot)    │             │              │              │ Single path │
├────────────────────┼─────────────┼──────────────┼──────────────┼─────────────┤
│ Self-Consistency   │ O(N)        │ N            │ Critical     │ N× cost     │
│                    │             │ (N=5-40)     │ decisions    │             │
├────────────────────┼─────────────┼──────────────┼──────────────┼─────────────┤
│ Program-of-        │ O(1)        │ 1 + exec     │ Math,        │ Needs       │
│ Thoughts           │             │              │ calculations │ sandbox     │
├────────────────────┼─────────────┼──────────────┼──────────────┼─────────────┤
│ Tree-of-Thoughts   │ O(b^d)      │ b^d × 2      │ Planning,    │ Exponential │
│                    │             │ (gen+eval)   │ game-solving │ cost        │
├────────────────────┼─────────────┼──────────────┼──────────────┼─────────────┤
│ Graph-of-Thoughts  │ O(nodes²)   │ Variable     │ Creative,    │ Very        │
│                    │             │ (high)       │ composition  │ complex     │
└────────────────────┴─────────────┴──────────────┴──────────────┴─────────────┘

Où:
  b = branching factor
  d = depth
  N = nombre de samples (self-consistency)

RECOMMANDATIONS PAR USE CASE:

1. Simple Q&A:
   → Direct (pas de reasoning nécessaire)

2. Math standard:
   → Program-of-Thoughts (exact)
   → Ou Zero-Shot CoT si code impossible

3. Logique multi-étapes:
   → Few-Shot CoT avec exemples
   → Self-Consistency si critique

4. Planning complexe:
   → Tree-of-Thoughts (BFS ou Beam)

5. Écriture créative:
   → Graph-of-Thoughts (agrégation d'idées)

6. Jeux, puzzles:
   → Tree-of-Thoughts (DFS avec backtracking)

BENCHMARKS (accuracy):

GSM8K (math):
  • Direct: 17%
  • CoT: 58%
  • Self-Consistency: 74%
  • PoT: 78%

Game of 24:
  • CoT: 74%
  • ToT (DFS): 95%

Creative Writing (human eval):
  • CoT: 64%
  • ToT: 85%
  • GoT: 91%
    """)


def print_best_practices():
    """Best practices pour reasoning"""
    print("\n\n" + "="*80)
    print("BEST PRACTICES - REASONING")
    print("="*80)

    print("""
1. CHOIX DE LA TECHNIQUE
   ✅ Commencer simple (CoT) puis complexifier si nécessaire
   ✅ Tester sur petit sample avant déploiement
   ✅ Considérer coût vs bénéfice

2. PROMPT ENGINEERING
   ✅ Être explicite: "Think step by step"
   ✅ Fournir exemples de qualité (few-shot)
   ✅ Demander justifications: "Explain your reasoning"

3. ÉVALUATION
   ✅ Vérifier pas seulement réponse finale mais processus
   ✅ Chercher hallucinations dans le raisonnement
   ✅ Tester robustesse (rephrasing de la question)

4. OPTIMISATION COÛTS
   ❌ Pas de ToT pour questions simples (overkill)
   ❌ Limiter branching factor et depth
   ✅ Utiliser modèles plus petits pour évaluation
   ✅ Cacher résultats fréquents

5. DEBUGGING
   • Reasoning incorrect:
     → Améliorer exemples (few-shot)
     → Ajouter contraintes au prompt
     → Utiliser model plus puissant

   • Solutions partielles:
     → Augmenter max_depth (ToT)
     → Vérifier critères de solution

   • Trop lent:
     → Réduire branching_factor
     → Utiliser Beam search au lieu de BFS

6. PRODUCTION
   ✅ Logging complet du reasoning
   ✅ Métriques: accuracy, latency, cost
   ✅ A/B testing des techniques
   ✅ Fallback si timeout
    """)


# ============================================================================
# DÉMONSTRATIONS
# ============================================================================

def demo_tree_of_thoughts():
    """Démo Tree of Thoughts"""
    print("="*80)
    print("DÉMONSTRATION: TREE OF THOUGHTS")
    print("="*80)

    tot = TreeOfThoughts(
        model=None,
        max_depth=4,
        branching_factor=3,
        beam_width=5
    )

    problem = "Trouver comment obtenir 24 en utilisant les nombres [4, 9, 10, 13] avec +, -, ×, ÷"

    # Test BFS
    print("\n1️⃣  BFS (Breadth-First Search)")
    print("-" * 80)
    result_bfs = tot.solve(problem, strategy=SearchStrategy.BFS)

    # Test Beam Search
    print("\n\n2️⃣  BEAM SEARCH (largeur=5)")
    print("-" * 80)
    tot2 = TreeOfThoughts(model=None, max_depth=4, beam_width=5)
    result_beam = tot2.solve(problem, strategy=SearchStrategy.BEAM)


def demo_graph_of_thoughts():
    """Démo Graph of Thoughts"""
    print("\n\n" + "="*80)
    print("DÉMONSTRATION: GRAPH OF THOUGHTS")
    print("="*80)

    got = GraphOfThoughts(model=None)

    task = "Écrire une introduction pour un article sur l'IA"

    # Générer 3 introductions différentes
    print("\n📝 Génération de 3 introductions variées:")
    intro1 = got.create_node("L'intelligence artificielle transforme notre monde...", "thought")
    intro2 = got.create_node("Depuis les années 1950, l'IA a progressé...", "thought")
    intro3 = got.create_node("Les LLMs représentent une révolution...", "thought")

    print(f"  1. {intro1.content}")
    print(f"  2. {intro2.content}")
    print(f"  3. {intro3.content}")

    # Agréger
    aggregated = got.aggregate([intro1, intro2, intro3], task)

    # Raffiner
    refined = got.refine(aggregated, "Rendre plus engageant et ajouter statistique")

    print(f"\n✅ Résultat final: {refined.content}")


if __name__ == "__main__":
    # Démos
    demo_tree_of_thoughts()
    demo_graph_of_thoughts()

    # Comparaison
    compare_reasoning_techniques()

    # Best practices
    print_best_practices()

    print("\n\n" + "="*80)
    print("🎉 CHAPITRE 22 COMPLÉTÉ: REASONING AVANCÉ")
    print("="*80)
    print("""
RÉCAPITULATIF COMPLET:

PARTIE 1: Chain-of-Thought
  • Zero-Shot CoT: "Let's think step by step"
  • Few-Shot CoT: Exemples avec raisonnement
  • Self-Consistency: Vote majoritaire (N paths)
  • Program-of-Thoughts: Générer code Python

PARTIE 2: Techniques Avancées
  • Tree-of-Thoughts: Exploration d'arbre avec backtracking
  • BFS, DFS, Beam Search
  • Graph-of-Thoughts: Agrégation et raffinement
  • Comparaison complète des techniques

AMÉLIORATION D'ACCURACY:
  Direct → CoT: +40% (math)
  CoT → Self-Consistency: +25%
  CoT → ToT: +30% (planning)
  CoT → PoT: +35% (math exact)

QUAND UTILISER CHAQUE TECHNIQUE:
  Simple Q&A → Direct
  Math standard → PoT ou Zero-Shot CoT
  Logique → Few-Shot CoT
  Critique → Self-Consistency
  Planning → Tree-of-Thoughts
  Créatif → Graph-of-Thoughts

BEST PRACTICES:
  ✅ Commencer simple, complexifier si nécessaire
  ✅ Tester coût vs bénéfice
  ✅ Logger tout le reasoning
  ✅ Métriques: accuracy, latency, cost
  ✅ A/B testing en production

PROCHAINS CHAPITRES:
  • Ch.23-25: Projets pratiques complets
  • Intégration de toutes les techniques apprises
  • Déploiement production end-to-end
    """)
