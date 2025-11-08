# Chapitre 19 - Partie 3: Systèmes Multi-Agents et LangGraph

## Introduction aux Systèmes Multi-Agents

Les **systèmes multi-agents** permettent à plusieurs agents spécialisés de collaborer pour résoudre des problèmes complexes. Chaque agent a une expertise spécifique et peut communiquer avec les autres.

```python
"""
MULTI-AGENTS = Plusieurs agents qui collaborent

Pourquoi multi-agents?
  1. Spécialisation:
     • Agent researcher: Recherche d'information
     • Agent coder: Écriture de code
     • Agent reviewer: Review de code
     • Agent tester: Tests
     → Meilleure qualité qu'un agent généraliste

  2. Parallélisation:
     • Plusieurs tâches en même temps
     • Accélération du workflow

  3. Modularité:
     • Facile d'ajouter/remplacer agents
     • Chaque agent testable indépendamment

  4. Scaling:
     • Peut gérer des problèmes très complexes
     • Décomposition hiérarchique

Patterns de coordination:
  1. Sequential (chaîne):
     A → B → C → D
     Exemple: Research → Code → Review → Deploy

  2. Hierarchical (hiérarchie):
          Manager
         /    |    \
     Agent1 Agent2 Agent3
     Exemple: Chef orchestre des spécialistes

  3. Collaborative (collaboration):
     A ⟷ B ⟷ C
     Agents communiquent directement

  4. Competitive (compétition):
     A → Solution 1
     B → Solution 2  → Vote/Select best
     C → Solution 3

Frameworks:
  • LangGraph: Graph-based workflows
  • AutoGen (Microsoft): Multi-agent conversations
  • CrewAI: Role-based agents
  • MetaGPT: Software company simulation
"""

from typing import List, Dict, Any, Optional, Callable, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict


# ============================================================================
# ARCHITECTURE MULTI-AGENTS
# ============================================================================

class AgentRole(Enum):
    """Rôles possibles pour les agents"""
    MANAGER = "manager"           # Coordonne les autres
    RESEARCHER = "researcher"     # Recherche d'information
    CODER = "coder"              # Écrit du code
    REVIEWER = "reviewer"         # Review code/texte
    TESTER = "tester"            # Teste le code
    WRITER = "writer"            # Rédige du texte
    ANALYST = "analyst"          # Analyse de données


@dataclass
class Message:
    """Message entre agents"""
    sender: str                   # ID de l'agent émetteur
    receiver: str                 # ID de l'agent destinataire
    content: str                  # Contenu du message
    message_type: str = "text"    # Type: text, code, data, etc.
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "type": self.message_type,
            "metadata": self.metadata
        }


class Agent:
    """
    Agent de base pour système multi-agents

    Chaque agent a:
      • Un rôle spécifique
      • Des outils spécialisés
      • Une capacité à communiquer
      • Une mémoire locale
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        system_prompt: str,
        tools: Optional[List[Any]] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.memory: List[Message] = []

    def receive_message(self, message: Message):
        """Reçoit un message"""
        self.memory.append(message)

    def process(self, task: str) -> str:
        """
        Traite une tâche selon le rôle

        En production: appel LLM avec system_prompt + tools
        """
        # Simulation basée sur le rôle
        if self.role == AgentRole.RESEARCHER:
            return self._research(task)
        elif self.role == AgentRole.CODER:
            return self._code(task)
        elif self.role == AgentRole.REVIEWER:
            return self._review(task)
        elif self.role == AgentRole.TESTER:
            return self._test(task)
        else:
            return f"[{self.role.value}] Tâche traitée: {task}"

    def _research(self, task: str) -> str:
        """Simule recherche"""
        return f"""Résultats de recherche pour: {task}

Sources trouvées:
1. Documentation officielle: ...
2. Tutorial: ...
3. Best practices: ...

Résumé: Voici les informations pertinentes..."""

    def _code(self, task: str) -> str:
        """Simule génération de code"""
        return f'''Code généré pour: {task}

```python
def solution():
    """Implementation de la solution"""
    # Code ici
    pass
```

Explications: ...'''

    def _review(self, task: str) -> str:
        """Simule review"""
        return f"""Review de: {task}

✅ Points positifs:
  - Structure claire
  - Bonne documentation

⚠️  Suggestions:
  - Ajouter gestion d'erreurs
  - Optimiser performance

Score: 8/10"""

    def _test(self, task: str) -> str:
        """Simule tests"""
        return f"""Tests pour: {task}

Tests exécutés: 15
✅ Réussis: 14
❌ Échoués: 1

Détails: ..."""


class MultiAgentSystem:
    """
    Système multi-agents avec coordination

    Gère:
      • Création et enregistrement d'agents
      • Routage des messages
      • Coordination des tâches
      • Workflows séquentiels ou parallèles
    """

    def __init__(self, verbose: bool = True):
        self.agents: Dict[str, Agent] = {}
        self.verbose = verbose
        self.message_history: List[Message] = []

    def register_agent(self, agent: Agent):
        """Enregistre un nouvel agent"""
        self.agents[agent.agent_id] = agent
        if self.verbose:
            print(f"✅ Agent '{agent.agent_id}' ({agent.role.value}) enregistré")

    def send_message(self, message: Message):
        """Envoie un message à un agent"""
        if message.receiver not in self.agents:
            raise ValueError(f"Agent '{message.receiver}' introuvable")

        # Enregistrer dans l'historique
        self.message_history.append(message)

        # Délivrer au destinataire
        receiver_agent = self.agents[message.receiver]
        receiver_agent.receive_message(message)

        if self.verbose:
            print(f"\n📨 Message: {message.sender} → {message.receiver}")
            print(f"   Contenu: {message.content[:80]}...")

    def run_sequential_workflow(
        self,
        task: str,
        agent_sequence: List[str]
    ) -> Dict[str, Any]:
        """
        Execute un workflow séquentiel

        Exemple:
            task = "Créer une API REST"
            sequence = ["researcher", "coder", "reviewer", "tester"]

            researcher → recherche best practices
            coder → écrit le code
            reviewer → review le code
            tester → teste le code

        Args:
            task: Tâche initiale
            agent_sequence: Liste d'IDs d'agents dans l'ordre

        Returns:
            Résultats de chaque étape
        """
        if self.verbose:
            print("\n" + "="*80)
            print("WORKFLOW SÉQUENTIEL")
            print("="*80)
            print(f"Tâche: {task}")
            print(f"Pipeline: {' → '.join(agent_sequence)}")
            print("="*80)

        results = {}
        current_input = task

        for i, agent_id in enumerate(agent_sequence):
            if agent_id not in self.agents:
                raise ValueError(f"Agent '{agent_id}' introuvable")

            agent = self.agents[agent_id]

            if self.verbose:
                print(f"\n--- Étape {i+1}/{len(agent_sequence)}: {agent_id} ({agent.role.value}) ---")

            # Agent traite la tâche
            result = agent.process(current_input)
            results[agent_id] = result

            if self.verbose:
                print(f"\nRésultat:")
                print(result[:300] + "..." if len(result) > 300 else result)

            # Le résultat devient l'input de la prochaine étape
            current_input = f"Étape précédente ({agent_id}):\n{result}\n\nContinue avec cette information."

        if self.verbose:
            print("\n" + "="*80)
            print("✅ WORKFLOW TERMINÉ")
            print("="*80)

        return results

    def run_hierarchical_workflow(
        self,
        task: str,
        manager_id: str,
        worker_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Execute un workflow hiérarchique

        Manager décompose la tâche et délègue aux workers

        Exemple:
            task = "Créer une application web complète"
            manager = "project_manager"
            workers = ["backend_dev", "frontend_dev", "designer"]

            Manager → décompose en sous-tâches
            Backend dev → API
            Frontend dev → Interface
            Designer → Design
            Manager → intègre tout
        """
        if self.verbose:
            print("\n" + "="*80)
            print("WORKFLOW HIÉRARCHIQUE")
            print("="*80)
            print(f"Tâche: {task}")
            print(f"Manager: {manager_id}")
            print(f"Workers: {', '.join(worker_ids)}")
            print("="*80)

        # 1. Manager planifie
        manager = self.agents[manager_id]

        if self.verbose:
            print(f"\n📋 {manager_id} planifie...")

        plan = manager.process(f"Décompose cette tâche en sous-tâches pour les agents disponibles: {task}")

        if self.verbose:
            print(f"\nPlan:")
            print(plan[:300] + "..." if len(plan) > 300 else plan)

        # 2. Workers executent en parallèle (simulé séquentiellement ici)
        worker_results = {}

        for worker_id in worker_ids:
            if worker_id not in self.agents:
                continue

            worker = self.agents[worker_id]

            if self.verbose:
                print(f"\n👷 {worker_id} ({worker.role.value}) travaille...")

            # Message du manager au worker
            message = Message(
                sender=manager_id,
                receiver=worker_id,
                content=f"Voici ta partie: {task}"
            )
            self.send_message(message)

            # Worker execute
            result = worker.process(task)
            worker_results[worker_id] = result

            if self.verbose:
                print(f"Résultat: {result[:200]}...")

        # 3. Manager intègre les résultats
        if self.verbose:
            print(f"\n🔄 {manager_id} intègre les résultats...")

        integration = manager.process(
            f"Intègre ces résultats:\n" +
            "\n".join([f"{k}: {v[:100]}..." for k, v in worker_results.items()])
        )

        return {
            "plan": plan,
            "worker_results": worker_results,
            "integration": integration
        }


# ============================================================================
# LANGGRAPH: WORKFLOWS AVEC GRAPHES
# ============================================================================

"""
LANGGRAPH = Framework pour workflows complexes avec graphes

Concepts clés:
  1. State (État):
     • Données partagées entre nodes
     • Peut être modifié par chaque node

  2. Nodes (Nœuds):
     • Fonctions qui traitent et modifient le state
     • Peuvent être des agents, des outils, etc.

  3. Edges (Arêtes):
     • Connexions entre nodes
     • Conditional edges: routing basé sur le state

  4. Graph (Graphe):
     • Ensemble de nodes et edges
     • Définit le workflow

Exemple simple:
    Input → Node A → Node B → Node C → Output
                      ↓
                   (if condition)
                      ↓
                    Node D

Avantages vs LangChain:
  ✅ Plus de contrôle sur le flow
  ✅ État explicite et mutable
  ✅ Conditional routing clair
  ✅ Facile à visualiser et debug

Use cases:
  • Multi-agent collaboration
  • Workflows avec branchements conditionnels
  • Human-in-the-loop
  • Retry logic complexe
"""

class GraphState(TypedDict):
    """État partagé dans le graph"""
    messages: List[str]
    data: Dict[str, Any]
    current_step: str
    completed: bool


class GraphNode:
    """Nœud dans le graph"""

    def __init__(self, node_id: str, function: Callable):
        self.node_id = node_id
        self.function = function

    def execute(self, state: GraphState) -> GraphState:
        """Execute le node et retourne le state modifié"""
        return self.function(state)


class ConditionalEdge:
    """Arête conditionnelle"""

    def __init__(
        self,
        condition: Callable[[GraphState], bool],
        true_node: str,
        false_node: str
    ):
        self.condition = condition
        self.true_node = true_node
        self.false_node = false_node

    def route(self, state: GraphState) -> str:
        """Retourne le prochain node basé sur la condition"""
        if self.condition(state):
            return self.true_node
        else:
            return self.false_node


class SimpleGraph:
    """
    Implémentation simplifiée d'un graph LangGraph-like

    Exemple:
        # Définir nodes
        def research_node(state):
            state["data"]["research"] = "... résultats recherche ..."
            return state

        def code_node(state):
            state["data"]["code"] = "... code généré ..."
            return state

        # Créer graph
        graph = SimpleGraph()
        graph.add_node("research", research_node)
        graph.add_node("code", code_node)
        graph.add_edge("research", "code")
        graph.set_entry_point("research")

        # Executer
        result = graph.run({"messages": [], "data": {}})
    """

    def __init__(self, verbose: bool = True):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, str] = {}  # node_id → next_node_id
        self.conditional_edges: Dict[str, ConditionalEdge] = {}
        self.entry_point: Optional[str] = None
        self.verbose = verbose

    def add_node(self, node_id: str, function: Callable):
        """Ajoute un node"""
        self.nodes[node_id] = GraphNode(node_id, function)

        if self.verbose:
            print(f"✅ Node '{node_id}' ajouté")

    def add_edge(self, from_node: str, to_node: str):
        """Ajoute une arête simple"""
        self.edges[from_node] = to_node

    def add_conditional_edge(
        self,
        from_node: str,
        condition: Callable,
        true_node: str,
        false_node: str
    ):
        """Ajoute une arête conditionnelle"""
        self.conditional_edges[from_node] = ConditionalEdge(
            condition, true_node, false_node
        )

    def set_entry_point(self, node_id: str):
        """Définit le point d'entrée du graph"""
        self.entry_point = node_id

    def run(self, initial_state: Dict) -> GraphState:
        """
        Execute le graph

        Flow:
          1. Commence au entry_point
          2. Execute le node
          3. Détermine le prochain node (edge ou conditional)
          4. Répète jusqu'à node terminal
        """
        if not self.entry_point:
            raise ValueError("Entry point non défini")

        # Initialiser le state
        state: GraphState = {
            "messages": initial_state.get("messages", []),
            "data": initial_state.get("data", {}),
            "current_step": self.entry_point,
            "completed": False
        }

        if self.verbose:
            print("\n" + "="*80)
            print("EXÉCUTION DU GRAPH")
            print("="*80)

        current_node_id = self.entry_point
        max_steps = 20  # Sécurité contre boucles infinies
        step = 0

        while step < max_steps:
            step += 1

            if self.verbose:
                print(f"\n--- Step {step}: {current_node_id} ---")

            # Execute le node actuel
            if current_node_id not in self.nodes:
                if self.verbose:
                    print(f"⚠️  Node '{current_node_id}' introuvable, fin du graph")
                break

            node = self.nodes[current_node_id]
            state["current_step"] = current_node_id

            # Execute
            state = node.execute(state)

            if self.verbose:
                print(f"State après exécution: {list(state['data'].keys())}")

            # Vérifier si terminé
            if state.get("completed"):
                if self.verbose:
                    print("\n✅ Graph terminé (completed=True)")
                break

            # Déterminer le prochain node
            next_node = None

            # 1. Vérifier conditional edge
            if current_node_id in self.conditional_edges:
                edge = self.conditional_edges[current_node_id]
                next_node = edge.route(state)

                if self.verbose:
                    print(f"🔀 Conditional routing → {next_node}")

            # 2. Sinon, edge simple
            elif current_node_id in self.edges:
                next_node = self.edges[current_node_id]

                if self.verbose:
                    print(f"→ Next node: {next_node}")

            # 3. Sinon, fin du graph
            else:
                if self.verbose:
                    print("🏁 Pas de prochain node, fin du graph")
                break

            current_node_id = next_node

        if self.verbose:
            print("\n" + "="*80)
            print("FIN DU GRAPH")
            print("="*80)

        return state


# ============================================================================
# PROJET COMPLET: SYSTÈME DE DÉVELOPPEMENT MULTI-AGENTS
# ============================================================================

class SoftwareDevelopmentSystem:
    """
    Système multi-agents pour développement logiciel

    Agents:
      1. Product Manager: Définit les specs
      2. Researcher: Recherche solutions
      3. Architect: Conçoit l'architecture
      4. Developer: Écrit le code
      5. Reviewer: Review le code
      6. Tester: Teste
      7. DevOps: Déploie

    Workflow:
      PM specs → Research → Architecture → Code → Review → Test → Deploy
    """

    def __init__(self):
        self.system = MultiAgentSystem(verbose=True)
        self._setup_agents()

    def _setup_agents(self):
        """Créer tous les agents"""

        # Product Manager
        pm = Agent(
            agent_id="product_manager",
            role=AgentRole.MANAGER,
            system_prompt="Tu es un Product Manager qui définit les spécifications produit."
        )
        self.system.register_agent(pm)

        # Researcher
        researcher = Agent(
            agent_id="researcher",
            role=AgentRole.RESEARCHER,
            system_prompt="Tu recherches les meilleures solutions et technologies."
        )
        self.system.register_agent(researcher)

        # Developer
        developer = Agent(
            agent_id="developer",
            role=AgentRole.CODER,
            system_prompt="Tu écris du code propre et bien documenté."
        )
        self.system.register_agent(developer)

        # Reviewer
        reviewer = Agent(
            agent_id="reviewer",
            role=AgentRole.REVIEWER,
            system_prompt="Tu reviews le code pour qualité et sécurité."
        )
        self.system.register_agent(reviewer)

        # Tester
        tester = Agent(
            agent_id="tester",
            role=AgentRole.TESTER,
            system_prompt="Tu écris et executes des tests."
        )
        self.system.register_agent(tester)

    def develop_feature(self, feature_description: str) -> Dict:
        """
        Développe une nouvelle feature avec workflow complet

        Args:
            feature_description: Description de la feature

        Returns:
            Résultats de chaque étape
        """
        print("\n" + "="*80)
        print("🚀 DÉVELOPPEMENT DE FEATURE")
        print("="*80)
        print(f"Feature: {feature_description}")
        print("="*80)

        # Workflow séquentiel
        results = self.system.run_sequential_workflow(
            task=feature_description,
            agent_sequence=[
                "product_manager",  # Specs
                "researcher",       # Research
                "developer",        # Code
                "reviewer",         # Review
                "tester"           # Test
            ]
        )

        return results


def demo_langgraph_workflow():
    """Démonstration d'un workflow LangGraph simple"""

    print("\n" + "="*80)
    print("DÉMONSTRATION LANGGRAPH")
    print("="*80)

    # Définir les nodes
    def start_node(state: GraphState) -> GraphState:
        """Node de départ"""
        state["messages"].append("Démarrage du workflow")
        state["data"]["started"] = True
        return state

    def research_node(state: GraphState) -> GraphState:
        """Recherche d'information"""
        state["messages"].append("Recherche en cours...")
        state["data"]["research_done"] = True
        state["data"]["quality_score"] = 8  # Simulé
        return state

    def good_quality_node(state: GraphState) -> GraphState:
        """Path si bonne qualité"""
        state["messages"].append("✅ Bonne qualité, on continue!")
        state["completed"] = True
        return state

    def improve_node(state: GraphState) -> GraphState:
        """Path si qualité à améliorer"""
        state["messages"].append("⚠️  Qualité à améliorer, on itère...")
        state["data"]["quality_score"] = 9  # Après amélioration
        return state

    # Créer le graph
    graph = SimpleGraph(verbose=True)

    # Ajouter nodes
    graph.add_node("start", start_node)
    graph.add_node("research", research_node)
    graph.add_node("good_quality", good_quality_node)
    graph.add_node("improve", improve_node)

    # Ajouter edges
    graph.add_edge("start", "research")

    # Conditional edge après research
    def check_quality(state: GraphState) -> bool:
        """Vérifie si la qualité est suffisante"""
        return state["data"].get("quality_score", 0) >= 8

    graph.add_conditional_edge(
        from_node="research",
        condition=check_quality,
        true_node="good_quality",
        false_node="improve"
    )

    graph.add_edge("improve", "good_quality")

    # Définir entry point
    graph.set_entry_point("start")

    # Executer
    result = graph.run({
        "messages": [],
        "data": {}
    })

    print("\n" + "="*80)
    print("RÉSULTAT FINAL")
    print("="*80)
    print(f"Messages: {result['messages']}")
    print(f"Data: {result['data']}")


# ============================================================================
# DÉMONSTRATIONS
# ============================================================================

def demo_multi_agent_system():
    """Démonstration du système multi-agents"""

    print("="*80)
    print("DÉMONSTRATION: SYSTÈME DE DÉVELOPPEMENT MULTI-AGENTS")
    print("="*80)

    # Créer le système
    dev_system = SoftwareDevelopmentSystem()

    # Développer une feature
    results = dev_system.develop_feature(
        "Créer une API REST pour gestion d'utilisateurs avec authentification JWT"
    )

    print("\n\n" + "="*80)
    print("📊 RÉSUMÉ DES RÉSULTATS")
    print("="*80)

    for agent_id, result in results.items():
        print(f"\n{agent_id}:")
        print("-" * 40)
        print(result[:200] + "..." if len(result) > 200 else result)


if __name__ == "__main__":
    # Démo multi-agents
    demo_multi_agent_system()

    # Démo LangGraph
    demo_langgraph_workflow()

    print("\n\n" + "="*80)
    print("KEY TAKEAWAYS - PARTIE 3")
    print("="*80)
    print("""
1. SYSTÈMES MULTI-AGENTS
   • Agents spécialisés collaborent
   • Meilleure qualité que généraliste
   • Patterns: Sequential, Hierarchical, Collaborative

2. COORDINATION
   Sequential:  A → B → C (pipeline)
   Hierarchical: Manager délègue aux workers
   Collaborative: Agents communiquent entre eux

3. LANGGRAPH
   • Workflows basés sur graphes
   • State partagé et mutable
   • Conditional routing
   • Plus de contrôle que LangChain

4. BEST PRACTICES
   ✅ Spécialiser les agents (1 rôle = 1 agent)
   ✅ Communication claire entre agents
   ✅ État partagé bien structuré
   ✅ Gestion d'erreurs à chaque étape
   ✅ Logging détaillé du workflow

5. QUAND UTILISER MULTI-AGENTS?
   • Problèmes complexes multi-facettes
   • Besoin de spécialisation
   • Workflows avec branchements
   • Collaboration humain-IA

CHAPITRE 19 COMPLÉTÉ!
Prochains chapitres: Multimodal, Long Context, Chain-of-Thought
    """)
