# Chapitre 19: Agents AI et Multi-Agents

## Introduction

Les **agents AI** sont des systèmes capables de percevoir leur environnement, raisonner sur leurs observations, et agir de manière autonome pour atteindre des objectifs. Contrairement aux LLMs statiques qui ne font que générer du texte, les agents peuvent:

- **Utiliser des outils** (APIs, bases de données, calculateurs)
- **Planifier** des séquences d'actions complexes
- **S'auto-corriger** en observant les résultats
- **Collaborer** avec d'autres agents
- **Apprendre** de leurs expériences

```python
"""
AGENTS AI = LLMs + Tools + Reasoning + Action

Évolution des capacités:
  1. LLM simple (2020-2022):
     • Input: Question
     • Output: Réponse
     • Limite: Pas d'accès au monde réel

  2. LLM + Tools (2023):
     • Input: Question
     • Output: Réponse + Tool calls
     • Capacité: Peut chercher info, calculer, etc.

  3. Agents autonomes (2024+):
     • Input: Objectif
     • Output: Plan + Exécution + Résultat
     • Capacité: Résout problèmes complexes de manière autonome

Cas d'usage:
  ✅ Customer support: Agent qui cherche docs, crée tickets
  ✅ Data analysis: Agent qui query DB, génère rapports
  ✅ Research assistant: Agent qui lit papers, résume, compare
  ✅ Coding assistant: Agent qui code, teste, debug
  ✅ Task automation: Agent qui orchestre workflows

Architecture d'un agent:
  1. LLM (cerveau): Raisonne et décide
  2. Tools (mains): Effectue actions dans le monde
  3. Memory (mémoire): Stocke historique et contexte
  4. Planner (stratège): Décompose tâches complexes
  5. Executor (exécuteur): Execute le plan
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
import re


class AgentAction(Enum):
    """Types d'actions qu'un agent peut effectuer"""
    THINK = "think"           # Réflexion interne
    USE_TOOL = "use_tool"     # Utiliser un outil
    RESPOND = "respond"       # Répondre à l'utilisateur
    ASK_USER = "ask_user"     # Demander clarification
    FINISH = "finish"         # Terminer la tâche


@dataclass
class Tool:
    """
    Un outil que l'agent peut utiliser

    Exemple:
        calculator = Tool(
            name="calculator",
            description="Effectue des calculs mathématiques",
            function=lambda x: eval(x)
        )
    """
    name: str
    description: str
    function: Callable
    parameters: Optional[Dict[str, Any]] = None

    def __call__(self, *args, **kwargs):
        """Execute l'outil"""
        return self.function(*args, **kwargs)


@dataclass
class AgentStep:
    """Une étape dans le raisonnement de l'agent"""
    thought: str                    # Pensée de l'agent
    action: AgentAction             # Action choisie
    action_input: Optional[str]     # Input pour l'action
    observation: Optional[str]      # Résultat de l'action

    def to_dict(self) -> Dict:
        return {
            "thought": self.thought,
            "action": self.action.value,
            "action_input": self.action_input,
            "observation": self.observation
        }


class SimpleAgent:
    """
    Agent AI simple utilisant le pattern ReAct (Reasoning + Acting)

    Le pattern ReAct alterne entre:
      1. Thought: L'agent réfléchit à ce qu'il doit faire
      2. Action: L'agent choisit une action (outil ou réponse)
      3. Observation: L'agent observe le résultat
      4. [Répéter jusqu'à résolution]

    Exemple:
        >>> agent = SimpleAgent(model="gpt-4", tools=[calculator, search])
        >>> result = agent.run("Quel est le PIB de la France en 2023?")

        Étapes:
          Thought: Je dois chercher le PIB de la France
          Action: use_tool(search, "PIB France 2023")
          Observation: "Le PIB de la France en 2023 est 2.9 trillion EUR"

          Thought: J'ai trouvé l'information
          Action: respond("Le PIB de la France en 2023 est 2.9 trillion EUR")
    """

    def __init__(
        self,
        model: str = "gpt-4",
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        """
        Args:
            model: Modèle LLM à utiliser
            tools: Liste d'outils disponibles
            max_iterations: Nombre max d'itérations
            verbose: Afficher les étapes
        """
        self.model = model
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Créer un mapping nom -> outil
        self.tool_map = {tool.name: tool for tool in self.tools}

    def _create_system_prompt(self) -> str:
        """Créer le prompt système pour l'agent"""
        tools_desc = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])

        return f"""Tu es un agent AI intelligent qui résout des problèmes en utilisant des outils.

OUTILS DISPONIBLES:
{tools_desc}

FORMAT DE RÉPONSE:
Tu dois TOUJOURS répondre dans ce format exact:

Thought: [ta réflexion sur ce qu'il faut faire]
Action: [use_tool OU respond]
Action Input: [nom de l'outil et paramètre OU réponse finale]

EXEMPLE:
Question: Quel est 25 * 17 + 100?

Thought: Je dois d'abord calculer 25 * 17, puis ajouter 100
Action: use_tool
Action Input: calculator("25 * 17")

[Après observation du résultat: 425]

Thought: Maintenant j'ajoute 100 à 425
Action: use_tool
Action Input: calculator("425 + 100")

[Après observation du résultat: 525]

Thought: J'ai le résultat final
Action: respond
Action Input: Le résultat de 25 * 17 + 100 est 525.

RÈGLES:
1. Utilise les outils quand tu as besoin d'informations externes
2. Raisonne étape par étape
3. Observe les résultats avant de continuer
4. Réponds avec "Action: respond" quand tu as la réponse finale
"""

    def _parse_llm_output(self, text: str) -> AgentStep:
        """
        Parse la sortie du LLM pour extraire thought, action, action_input

        Format attendu:
            Thought: ...
            Action: ...
            Action Input: ...
        """
        # Extraire les sections
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.+?)(?=\nAction Input:|$)", text, re.DOTALL)
        input_match = re.search(r"Action Input:\s*(.+?)(?=\n\n|$)", text, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        action_str = action_match.group(1).strip() if action_match else ""
        action_input = input_match.group(1).strip() if input_match else ""

        # Déterminer le type d'action
        if "respond" in action_str.lower():
            action = AgentAction.RESPOND
        elif "use_tool" in action_str.lower():
            action = AgentAction.USE_TOOL
        else:
            action = AgentAction.THINK

        return AgentStep(
            thought=thought,
            action=action,
            action_input=action_input,
            observation=None
        )

    def _execute_action(self, step: AgentStep) -> str:
        """
        Execute l'action choisie par l'agent

        Returns:
            Observation (résultat de l'action)
        """
        if step.action == AgentAction.USE_TOOL:
            # Parser "tool_name(param)"
            match = re.match(r'(\w+)\((.*)\)', step.action_input)
            if not match:
                return f"Erreur: Format invalide. Utilise tool_name(param)"

            tool_name = match.group(1)
            param = match.group(2).strip('"').strip("'")

            # Vérifier que l'outil existe
            if tool_name not in self.tool_map:
                available = ", ".join(self.tool_map.keys())
                return f"Erreur: Outil '{tool_name}' inconnu. Disponibles: {available}"

            # Executer l'outil
            try:
                tool = self.tool_map[tool_name]
                result = tool(param)
                return f"Résultat: {result}"
            except Exception as e:
                return f"Erreur lors de l'exécution: {str(e)}"

        elif step.action == AgentAction.RESPOND:
            # Réponse finale
            return "TERMINÉ"

        else:
            return "Action non reconnue"

    def run(self, task: str) -> Dict[str, Any]:
        """
        Execute une tâche de manière autonome

        Args:
            task: La tâche à accomplir

        Returns:
            {
                "success": bool,
                "answer": str,
                "steps": List[AgentStep],
                "iterations": int
            }
        """
        if self.verbose:
            print("=" * 80)
            print(f"TÂCHE: {task}")
            print("=" * 80)

        # Historique de la conversation
        messages = [
            {"role": "system", "content": self._create_system_prompt()},
            {"role": "user", "content": task}
        ]

        steps: List[AgentStep] = []

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- Itération {iteration + 1}/{self.max_iterations} ---")

            # 1. LLM génère thought + action
            llm_response = self._call_llm(messages)

            # 2. Parser la réponse
            step = self._parse_llm_output(llm_response)

            if self.verbose:
                print(f"\n💭 Thought: {step.thought}")
                print(f"🎬 Action: {step.action.value}")
                print(f"📝 Input: {step.action_input}")

            # 3. Executer l'action
            if step.action == AgentAction.RESPOND:
                # Tâche terminée
                steps.append(step)

                if self.verbose:
                    print(f"\n✅ RÉPONSE FINALE:")
                    print(f"{step.action_input}")
                    print("=" * 80)

                return {
                    "success": True,
                    "answer": step.action_input,
                    "steps": steps,
                    "iterations": iteration + 1
                }

            else:
                # Executer l'action et observer
                observation = self._execute_action(step)
                step.observation = observation
                steps.append(step)

                if self.verbose:
                    print(f"👁️  Observation: {observation}")

                # Ajouter à l'historique
                messages.append({
                    "role": "assistant",
                    "content": llm_response
                })
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })

        # Max iterations atteint
        if self.verbose:
            print("\n❌ Nombre max d'itérations atteint")

        return {
            "success": False,
            "answer": "Impossible de résoudre (max iterations)",
            "steps": steps,
            "iterations": self.max_iterations
        }

    def _call_llm(self, messages: List[Dict]) -> str:
        """
        Appel au LLM (simulé pour cet exemple)

        En production, utiliser OpenAI API, Anthropic, etc.
        """
        # SIMULATION - En production, remplacer par:
        # import openai
        # response = openai.ChatCompletion.create(
        #     model=self.model,
        #     messages=messages
        # )
        # return response.choices[0].message.content

        # Pour l'exemple, on simule des réponses
        return """Thought: Je dois utiliser l'outil approprié pour résoudre ce problème.
Action: use_tool
Action Input: calculator("2 + 2")"""


# ============================================================================
# OUTILS COMMUNS POUR AGENTS
# ============================================================================

def create_calculator_tool() -> Tool:
    """Créer un outil calculateur"""
    def calculate(expression: str) -> str:
        """Évalue une expression mathématique"""
        try:
            # Sécurité: whitelist des opérations autorisées
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "Erreur: Caractères non autorisés"

            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Erreur de calcul: {str(e)}"

    return Tool(
        name="calculator",
        description="Calcule des expressions mathématiques (ex: '25 * 17 + 100')",
        function=calculate
    )


def create_search_tool() -> Tool:
    """Créer un outil de recherche (simulé)"""
    # Base de connaissances simulée
    knowledge_base = {
        "pib france 2023": "Le PIB de la France en 2023 est environ 2.9 trillions EUR",
        "population france": "La population de la France est environ 67 millions d'habitants",
        "capitale france": "La capitale de la France est Paris",
        "tour eiffel hauteur": "La Tour Eiffel mesure 330 mètres de hauteur",
    }

    def search(query: str) -> str:
        """Recherche dans la base de connaissances"""
        query_lower = query.lower()

        for key, value in knowledge_base.items():
            if key in query_lower:
                return value

        return f"Aucun résultat trouvé pour: {query}"

    return Tool(
        name="search",
        description="Recherche des informations (ex: 'PIB France 2023')",
        function=search
    )


def create_python_tool() -> Tool:
    """Créer un outil d'exécution Python"""
    def execute_python(code: str) -> str:
        """Execute du code Python de manière sécurisée"""
        try:
            # ATTENTION: En production, utiliser un sandbox sécurisé!
            # Ici c'est une simulation simplifiée

            # Whitelist de modules autorisés
            safe_globals = {
                "__builtins__": {
                    "len": len,
                    "range": range,
                    "sum": sum,
                    "max": max,
                    "min": min,
                    "sorted": sorted,
                }
            }

            # Capturer stdout
            from io import StringIO
            import sys

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            exec(code, safe_globals)

            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            return output or "Code exécuté avec succès (pas de sortie)"

        except Exception as e:
            return f"Erreur: {str(e)}"

    return Tool(
        name="python",
        description="Execute du code Python simple",
        function=execute_python
    )


# ============================================================================
# EXEMPLE COMPLET
# ============================================================================

def demo_simple_agent():
    """Démonstration d'un agent simple"""
    print("=" * 80)
    print("DÉMONSTRATION: AGENT AI SIMPLE")
    print("=" * 80)

    # Créer les outils
    calculator = create_calculator_tool()
    search = create_search_tool()

    # Créer l'agent
    agent = SimpleAgent(
        model="gpt-4",
        tools=[calculator, search],
        max_iterations=10,
        verbose=True
    )

    # Tâche 1: Calcul mathématique
    print("\n\n" + "=" * 80)
    print("TÂCHE 1: CALCUL MATHÉMATIQUE")
    print("=" * 80)

    result1 = agent.run("Calcule (15 * 8) + (120 / 4) - 7")

    # Tâche 2: Recherche d'information
    print("\n\n" + "=" * 80)
    print("TÂCHE 2: RECHERCHE D'INFORMATION")
    print("=" * 80)

    result2 = agent.run("Quelle est la population de la France?")

    # Tâche 3: Combinaison recherche + calcul
    print("\n\n" + "=" * 80)
    print("TÂCHE 3: RECHERCHE + CALCUL")
    print("=" * 80)

    result3 = agent.run(
        "Quelle est la hauteur de la Tour Eiffel en mètres? "
        "Puis calcule combien ça fait en pieds (1 mètre = 3.28084 pieds)"
    )

    # Résumé
    print("\n\n" + "=" * 80)
    print("RÉSUMÉ DES RÉSULTATS")
    print("=" * 80)

    for i, result in enumerate([result1, result2, result3], 1):
        status = "✅ Succès" if result["success"] else "❌ Échec"
        print(f"\nTâche {i}: {status}")
        print(f"  Réponse: {result['answer']}")
        print(f"  Itérations: {result['iterations']}")
        print(f"  Étapes: {len(result['steps'])}")


# ============================================================================
# REACT: REASONING + ACTING
# ============================================================================

class ReActAgent:
    """
    Agent utilisant le pattern ReAct (Yao et al., 2022)

    ReAct = Reasoning (raisonnement) + Acting (action)

    Le pattern alterne:
      1. Thought: Raisonner sur la situation actuelle
      2. Action: Choisir et executer une action
      3. Observation: Observer le résultat
      4. [Retour à 1]

    Avantages vs LLM simple:
      ✅ Plus interprétable (on voit le raisonnement)
      ✅ Peut s'auto-corriger
      ✅ Meilleure décomposition de tâches complexes
      ✅ Utilise outils externes

    Paper: https://arxiv.org/abs/2210.03629
    """

    def __init__(
        self,
        llm_client: Any,
        tools: List[Tool],
        max_steps: int = 10
    ):
        self.llm = llm_client
        self.tools = tools
        self.max_steps = max_steps
        self.tool_map = {t.name: t for t in tools}

    def solve(self, task: str) -> Dict:
        """
        Résout une tâche avec ReAct

        Exemple de trace:
            Task: Quelle est la capitale du pays avec 67M habitants?

            Thought 1: Je dois d'abord trouver quel pays a 67M habitants
            Action 1: search("pays 67 millions habitants")
            Obs 1: La France a environ 67 millions d'habitants

            Thought 2: Maintenant je dois trouver la capitale de la France
            Action 2: search("capitale France")
            Obs 2: La capitale de la France est Paris

            Thought 3: J'ai la réponse
            Action 3: respond("La capitale est Paris")
        """

        prompt = self._build_react_prompt(task)
        history = []

        for step in range(self.max_steps):
            # Générer thought + action
            response = self.llm.generate(prompt)

            # Parser
            thought, action, action_input = self._parse_response(response)

            # Observer
            if action == "respond":
                return {
                    "answer": action_input,
                    "steps": history,
                    "success": True
                }

            observation = self._execute_tool(action, action_input)

            # Mémoriser
            history.append({
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation
            })

            # Mettre à jour le prompt
            prompt += f"\n\nObservation {step+1}: {observation}"

        return {
            "answer": "Max steps reached",
            "steps": history,
            "success": False
        }

    def _build_react_prompt(self, task: str) -> str:
        """Construit le prompt ReAct"""
        tools_desc = "\n".join([
            f"{t.name}: {t.description}" for t in self.tools
        ])

        return f"""Résous cette tâche en utilisant le format ReAct:

Thought: [ton raisonnement]
Action: [outil à utiliser]
Action Input: [paramètre de l'outil]

Outils disponibles:
{tools_desc}

Tâche: {task}

Commence:
Thought 1:"""

    def _parse_response(self, text: str) -> tuple:
        """Parse thought, action, action_input"""
        # Implémentation simplifiée
        thought = re.search(r"Thought.*?:(.*?)(?=Action|$)", text, re.DOTALL)
        action = re.search(r"Action.*?:(.*?)(?=Action Input|$)", text, re.DOTALL)
        action_input = re.search(r"Action Input.*?:(.*?)$", text, re.DOTALL)

        return (
            thought.group(1).strip() if thought else "",
            action.group(1).strip() if action else "",
            action_input.group(1).strip() if action_input else ""
        )

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute un outil"""
        if tool_name not in self.tool_map:
            return f"Outil '{tool_name}' introuvable"

        try:
            return self.tool_map[tool_name](tool_input)
        except Exception as e:
            return f"Erreur: {str(e)}"


# ============================================================================
# COMPARAISON: LLM vs AGENT
# ============================================================================

def demo_llm_vs_agent():
    """Compare LLM simple vs Agent avec outils"""
    print("=" * 80)
    print("COMPARAISON: LLM SIMPLE vs AGENT AVEC OUTILS")
    print("=" * 80)

    question = "Quelle est la hauteur de la Tour Eiffel en pieds?"

    print(f"\nQuestion: {question}\n")

    # Scénario 1: LLM simple
    print("-" * 80)
    print("SCÉNARIO 1: LLM SIMPLE (sans outils)")
    print("-" * 80)
    print("""
Limitations:
  ❌ Peut halluciner les chiffres
  ❌ Pas de conversion précise mètres -> pieds
  ❌ Pas de source vérifiable
  ❌ Réponse figée (training cutoff)

Réponse typique:
  "La Tour Eiffel mesure environ 1,083 pieds de hauteur."

  Problem: Est-ce exact? On ne sait pas!
    """)

    # Scénario 2: Agent avec outils
    print("-" * 80)
    print("SCÉNARIO 2: AGENT AVEC OUTILS")
    print("-" * 80)
    print("""
Capacités:
  ✅ Recherche la hauteur exacte (330m)
  ✅ Utilise calculateur pour conversion précise
  ✅ Source vérifiable
  ✅ Peut se corriger si erreur

Trace d'exécution:

  Thought 1: Je dois d'abord trouver la hauteur en mètres
  Action 1: search("hauteur Tour Eiffel")
  Obs 1: "La Tour Eiffel mesure 330 mètres de hauteur"

  Thought 2: Maintenant je dois convertir 330m en pieds (1m = 3.28084 pieds)
  Action 2: calculator("330 * 3.28084")
  Obs 2: "1082.6772"

  Thought 3: J'ai la réponse précise
  Action 3: respond("La Tour Eiffel mesure 1,082.68 pieds")

Résultat: 1,082.68 pieds (PRÉCIS et VÉRIFIABLE)
    """)

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
Les agents avec outils sont supérieurs pour:
  • Questions factuelles nécessitant des données récentes
  • Calculs précis
  • Tâches multi-étapes
  • Situations nécessitant vérification

LLM simple reste meilleur pour:
  • Tâches purement créatives
  • Questions générales
  • Rapidité (pas d'appels d'outils)
    """)


if __name__ == "__main__":
    # Démo de l'agent simple
    demo_simple_agent()

    # Comparaison LLM vs Agent
    demo_llm_vs_agent()

    print("\n\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. AGENTS = LLM + TOOLS + REASONING
   • LLM = cerveau qui raisonne
   • Tools = mains qui agissent
   • Loop = observe, raisonne, agit, répète

2. PATTERN REACT (Reasoning + Acting)
   • Thought: "Que dois-je faire?"
   • Action: Execute un outil
   • Observation: "Quel est le résultat?"
   • [Répète jusqu'à résolution]

3. AVANTAGES
   ✅ Accès au monde réel (APIs, DBs, calculateurs)
   ✅ Auto-correction possible
   ✅ Décomposition de tâches complexes
   ✅ Traçabilité du raisonnement

4. DÉFIS
   ❌ Plus lent (multiple appels LLM + outils)
   ❌ Plus coûteux
   ❌ Peut boucler infiniment
   ❌ Debugging plus complexe

PROCHAINE PARTIE: Function calling et Tool use avancés
    """)
