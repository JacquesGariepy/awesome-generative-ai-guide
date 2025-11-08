# Chapitre 19 - Partie 2: Function Calling et LangChain

## Function Calling avec OpenAI et Anthropic

Le **function calling** permet aux LLMs de générer des appels de fonctions structurés au lieu de texte libre. Les modèles modernes (GPT-4, Claude 3+) peuvent:

- Détecter quand appeler une fonction
- Choisir la bonne fonction parmi plusieurs
- Générer les paramètres au bon format (JSON)
- Gérer les erreurs et réessayer

```python
"""
FUNCTION CALLING = LLM qui génère des appels de fonction structurés

Évolution:
  2022: Prompt engineering
    "Tu as accès à calculator(). Pour calculer 2+2, dis: CALL calculator(2+2)"
    ❌ Fragile, parsing difficile

  2023+: Function calling natif
    Model comprend les fonctions et génère JSON structuré
    ✅ Robuste, type-safe, facile à parser

Providers:
  • OpenAI: GPT-4, GPT-3.5 (parameter: functions, tools)
  • Anthropic: Claude 3+ (parameter: tools)
  • Google: Gemini (parameter: function_declarations)
  • Mistral: Mistral Large (supporte function calling)

Cas d'usage:
  ✅ Chatbots qui réservent des vols
  ✅ Assistants qui créent des tickets
  ✅ Agents qui query des bases de données
  ✅ Code assistants qui executent du code

Workflow:
  1. Définir les fonctions disponibles (schéma JSON)
  2. Envoyer requête utilisateur + schémas au LLM
  3. LLM décide s'il faut appeler une fonction
  4. Si oui: LLM génère function call (JSON)
  5. Developer execute la fonction
  6. Renvoyer résultat au LLM
  7. LLM génère réponse finale pour l'utilisateur
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import inspect


# ============================================================================
# OPENAI FUNCTION CALLING
# ============================================================================

@dataclass
class FunctionParameter:
    """Paramètre d'une fonction"""
    name: str
    type: str              # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None


@dataclass
class FunctionDefinition:
    """
    Définition d'une fonction pour function calling

    Exemple:
        get_weather = FunctionDefinition(
            name="get_weather",
            description="Récupère la météo pour une ville",
            parameters=[
                FunctionParameter("location", "string", "Nom de la ville", required=True),
                FunctionParameter("unit", "string", "Unité (celsius/fahrenheit)",
                                required=False, enum=["celsius", "fahrenheit"])
            ],
            function=lambda location, unit="celsius": f"Il fait 22°C à {location}"
        )
    """
    name: str
    description: str
    parameters: List[FunctionParameter]
    function: Callable

    def to_openai_schema(self) -> Dict:
        """Convertit en schéma OpenAI"""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def to_anthropic_schema(self) -> Dict:
        """Convertit en schéma Anthropic (Claude)"""
        input_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param in self.parameters:
            input_schema["properties"][param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                input_schema["properties"][param.name]["enum"] = param.enum

            if param.required:
                input_schema["required"].append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_schema
        }

    def execute(self, **kwargs) -> Any:
        """Execute la fonction avec les paramètres"""
        return self.function(**kwargs)


class OpenAIFunctionAgent:
    """
    Agent utilisant function calling d'OpenAI

    Exemple:
        >>> agent = OpenAIFunctionAgent(api_key="sk-...")
        >>> agent.add_function(get_weather_function)
        >>> response = agent.chat("Quelle est la météo à Paris?")
    """

    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        self.api_key = api_key
        self.model = model
        self.functions: List[FunctionDefinition] = []
        self.conversation_history: List[Dict] = []

    def add_function(self, func: FunctionDefinition):
        """Ajoute une fonction disponible"""
        self.functions.append(func)

    def chat(self, user_message: str) -> str:
        """
        Envoie un message et gère les function calls

        Flow:
          1. User: "Quelle est la météo à Paris?"
          2. LLM: function_call = get_weather(location="Paris")
          3. Execute: result = get_weather("Paris")
          4. LLM: "Il fait 22°C à Paris"
          5. Return réponse à l'utilisateur
        """
        # Ajouter message utilisateur
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Préparer schémas de fonctions
        tools = [
            {
                "type": "function",
                "function": func.to_openai_schema()
            }
            for func in self.functions
        ]

        # Appel à OpenAI (simulé pour l'exemple)
        print("\n" + "="*80)
        print("APPEL OPENAI API")
        print("="*80)
        print(f"Messages: {len(self.conversation_history)}")
        print(f"Tools disponibles: {len(tools)}")

        # SIMULATION - En production:
        # import openai
        # response = openai.chat.completions.create(
        #     model=self.model,
        #     messages=self.conversation_history,
        #     tools=tools,
        #     tool_choice="auto"
        # )

        # Simuler une réponse avec function call
        simulated_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({
                                "location": "Paris",
                                "unit": "celsius"
                            })
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }

        response_message = simulated_response["choices"][0]["message"]
        finish_reason = simulated_response["choices"][0]["finish_reason"]

        # Vérifier si function call
        if finish_reason == "tool_calls" and response_message.get("tool_calls"):
            print("\n🔧 LLM a décidé d'appeler une fonction")

            # Ajouter réponse LLM à l'historique
            self.conversation_history.append(response_message)

            # Executer chaque function call
            for tool_call in response_message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])

                print(f"\n📞 Appel: {function_name}({function_args})")

                # Trouver la fonction
                func = next((f for f in self.functions if f.name == function_name), None)

                if func:
                    # Executer
                    result = func.execute(**function_args)
                    print(f"📊 Résultat: {result}")

                    # Ajouter résultat à l'historique
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": str(result)
                    })

            # Rappeler LLM avec les résultats
            print("\n🔄 Rappel LLM avec résultats...")

            # Simuler réponse finale
            final_response = "Il fait actuellement 22°C à Paris avec un ciel dégagé."

            self.conversation_history.append({
                "role": "assistant",
                "content": final_response
            })

            return final_response

        else:
            # Pas de function call, réponse directe
            content = response_message.get("content", "")
            self.conversation_history.append(response_message)
            return content


# ============================================================================
# ANTHROPIC (CLAUDE) FUNCTION CALLING
# ============================================================================

class ClaudeFunctionAgent:
    """
    Agent utilisant Claude 3+ avec tool use

    Claude utilise un format légèrement différent d'OpenAI:
      - "tools" au lieu de "functions"
      - "tool_use" au lieu de "function_call"
      - Format de réponse différent

    Exemple:
        >>> agent = ClaudeFunctionAgent(api_key="sk-ant-...")
        >>> agent.add_tool(calculator_tool)
        >>> response = agent.chat("Combien fait 25 * 17?")
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self.tools: List[FunctionDefinition] = []
        self.messages: List[Dict] = []

    def add_tool(self, tool: FunctionDefinition):
        """Ajoute un outil disponible"""
        self.tools.append(tool)

    def chat(self, user_message: str, max_iterations: int = 5) -> str:
        """
        Conversation avec tool use

        Claude peut faire plusieurs tool calls en séquence
        """
        # Ajouter message utilisateur
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        # Préparer schémas d'outils
        tools_schema = [tool.to_anthropic_schema() for tool in self.tools]

        print("\n" + "="*80)
        print("APPEL CLAUDE API")
        print("="*80)
        print(f"Messages: {len(self.messages)}")
        print(f"Tools: {len(tools_schema)}")

        for iteration in range(max_iterations):
            print(f"\n--- Itération {iteration + 1} ---")

            # SIMULATION - En production:
            # import anthropic
            # client = anthropic.Anthropic(api_key=self.api_key)
            # response = client.messages.create(
            #     model=self.model,
            #     max_tokens=4096,
            #     tools=tools_schema,
            #     messages=self.messages
            # )

            # Simuler une réponse avec tool use
            if iteration == 0:
                # Premier appel: Claude demande à utiliser calculator
                simulated_response = {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Je vais calculer 25 * 17 pour vous."
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "calculator",
                            "input": {
                                "expression": "25 * 17"
                            }
                        }
                    ],
                    "stop_reason": "tool_use"
                }
            else:
                # Deuxième appel: Claude répond avec le résultat
                simulated_response = {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Le résultat de 25 × 17 est 425."
                        }
                    ],
                    "stop_reason": "end_turn"
                }

            response = simulated_response
            stop_reason = response["stop_reason"]

            # Ajouter réponse à l'historique
            self.messages.append({
                "role": "assistant",
                "content": response["content"]
            })

            # Vérifier si tool use
            if stop_reason == "tool_use":
                print("🔧 Claude veut utiliser un outil")

                # Extraire et executer les tool uses
                tool_results = []

                for content_block in response["content"]:
                    if content_block.get("type") == "tool_use":
                        tool_name = content_block["name"]
                        tool_input = content_block["input"]
                        tool_use_id = content_block["id"]

                        print(f"\n📞 Tool: {tool_name}")
                        print(f"📝 Input: {tool_input}")

                        # Trouver et executer l'outil
                        tool = next((t for t in self.tools if t.name == tool_name), None)

                        if tool:
                            result = tool.execute(**tool_input)
                            print(f"📊 Résultat: {result}")

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": str(result)
                            })

                # Ajouter résultats à l'historique
                self.messages.append({
                    "role": "user",
                    "content": tool_results
                })

                # Continuer la boucle pour obtenir réponse finale

            else:
                # Fin de la conversation
                # Extraire le texte de la réponse
                text_content = ""
                for block in response["content"]:
                    if block.get("type") == "text":
                        text_content += block["text"]

                return text_content

        return "Max iterations atteint"


# ============================================================================
# OUTILS COMMUNS
# ============================================================================

def create_weather_function() -> FunctionDefinition:
    """Créer fonction météo"""

    # Base de données simulée
    weather_db = {
        "paris": {"temp": 22, "condition": "Ensoleillé", "humidity": 65},
        "londres": {"temp": 15, "condition": "Nuageux", "humidity": 80},
        "new york": {"temp": 28, "condition": "Partiellement nuageux", "humidity": 70},
        "tokyo": {"temp": 25, "condition": "Pluie légère", "humidity": 85},
    }

    def get_weather(location: str, unit: str = "celsius") -> str:
        """Récupère la météo"""
        location_lower = location.lower()

        if location_lower not in weather_db:
            return f"Désolé, pas de données météo pour {location}"

        data = weather_db[location_lower]
        temp = data["temp"]

        # Convertir si fahrenheit
        if unit == "fahrenheit":
            temp = (temp * 9/5) + 32

        return json.dumps({
            "location": location,
            "temperature": temp,
            "unit": unit,
            "condition": data["condition"],
            "humidity": data["humidity"]
        })

    return FunctionDefinition(
        name="get_weather",
        description="Récupère les informations météo pour une ville donnée",
        parameters=[
            FunctionParameter(
                name="location",
                type="string",
                description="Nom de la ville (ex: Paris, Londres)",
                required=True
            ),
            FunctionParameter(
                name="unit",
                type="string",
                description="Unité de température",
                required=False,
                enum=["celsius", "fahrenheit"]
            )
        ],
        function=get_weather
    )


def create_database_query_function() -> FunctionDefinition:
    """Créer fonction de requête base de données"""

    # Base de données simulée
    users_db = [
        {"id": 1, "name": "Alice Martin", "email": "alice@example.com", "role": "admin"},
        {"id": 2, "name": "Bob Durand", "email": "bob@example.com", "role": "user"},
        {"id": 3, "name": "Charlie Petit", "email": "charlie@example.com", "role": "user"},
    ]

    def query_users(filter_field: Optional[str] = None, filter_value: Optional[str] = None) -> str:
        """Query la base de données utilisateurs"""
        results = users_db

        # Appliquer filtre si spécifié
        if filter_field and filter_value:
            results = [
                user for user in results
                if str(user.get(filter_field, "")).lower() == filter_value.lower()
            ]

        return json.dumps({
            "count": len(results),
            "users": results
        }, indent=2)

    return FunctionDefinition(
        name="query_users",
        description="Recherche des utilisateurs dans la base de données",
        parameters=[
            FunctionParameter(
                name="filter_field",
                type="string",
                description="Champ sur lequel filtrer (name, email, role)",
                required=False,
                enum=["name", "email", "role"]
            ),
            FunctionParameter(
                name="filter_value",
                type="string",
                description="Valeur du filtre",
                required=False
            )
        ],
        function=query_users
    )


def create_calculator_function() -> FunctionDefinition:
    """Créer fonction calculateur"""

    def calculate(expression: str) -> str:
        """Calcule une expression mathématique"""
        try:
            # Validation de sécurité
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "Erreur: Expression contient des caractères non autorisés"

            result = eval(expression)
            return json.dumps({
                "expression": expression,
                "result": result
            })
        except Exception as e:
            return json.dumps({
                "error": str(e)
            })

    return FunctionDefinition(
        name="calculator",
        description="Calcule des expressions mathématiques",
        parameters=[
            FunctionParameter(
                name="expression",
                type="string",
                description="Expression mathématique à calculer (ex: '25 * 17 + 100')",
                required=True
            )
        ],
        function=calculate
    )


# ============================================================================
# LANGCHAIN: FRAMEWORK POUR AGENTS
# ============================================================================

"""
LANGCHAIN = Framework populaire pour construire des applications LLM

Composants principaux:
  1. LLMs: Wrappers pour OpenAI, Anthropic, etc.
  2. Prompts: Templates réutilisables
  3. Chains: Enchaînement d'opérations
  4. Agents: Agents avec tools
  5. Memory: Gestion de la mémoire conversationnelle
  6. Tools: Outils pré-construits

Avantages:
  ✅ Abstractions de haut niveau
  ✅ Beaucoup d'outils pré-faits
  ✅ Intégrations multiples (OpenAI, Anthropic, HF, etc.)
  ✅ Community active

Inconvénients:
  ❌ Abstraction parfois trop complexe
  ❌ Overhead de performance
  ❌ Breaking changes fréquents

Alternative: LangGraph (plus bas niveau, plus de contrôle)
"""

class SimpleLangChainAgent:
    """
    Implémentation simplifiée d'un agent LangChain-like

    Démontre les concepts sans la complexité de LangChain complet

    Composants:
      • LLM: Modèle de langage
      • Tools: Outils disponibles
      • Prompt: Template de prompt
      • Memory: Historique conversationnel
      • Agent Executor: Loop principal
    """

    def __init__(
        self,
        llm: Any,
        tools: List[FunctionDefinition],
        verbose: bool = True
    ):
        self.llm = llm
        self.tools = tools
        self.verbose = verbose
        self.memory: List[Dict] = []

    def run(self, task: str, max_iterations: int = 10) -> str:
        """
        Execute une tâche

        Flow:
          1. Créer prompt avec task + tools + memory
          2. LLM génère action
          3. Execute action
          4. Observe résultat
          5. Répéter jusqu'à réponse finale
        """
        if self.verbose:
            print("\n" + "="*80)
            print(f"TÂCHE: {task}")
            print("="*80)

        # Ajouter tâche à la mémoire
        self.memory.append({
            "role": "user",
            "content": task
        })

        for iteration in range(max_iterations):
            if self.verbose:
                print(f"\n--- Itération {iteration + 1} ---")

            # Construire prompt
            prompt = self._build_prompt()

            # LLM génère action
            response = self._call_llm(prompt)

            # Parser action
            action_type, action_input = self._parse_action(response)

            if self.verbose:
                print(f"Action: {action_type}")
                print(f"Input: {action_input}")

            # Execute action
            if action_type == "respond":
                # Tâche terminée
                self.memory.append({
                    "role": "assistant",
                    "content": action_input
                })
                return action_input

            elif action_type == "use_tool":
                # Utiliser un outil
                observation = self._execute_tool(action_input)

                if self.verbose:
                    print(f"Observation: {observation}")

                # Ajouter à la mémoire
                self.memory.append({
                    "role": "assistant",
                    "content": f"Action: {action_input}"
                })
                self.memory.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })

        return "Max iterations atteint"

    def _build_prompt(self) -> str:
        """Construit le prompt avec memory et tools"""
        tools_desc = "\n".join([
            f"- {t.name}: {t.description}"
            for t in self.tools
        ])

        memory_str = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in self.memory
        ])

        return f"""Tu es un agent intelligent avec accès à des outils.

OUTILS DISPONIBLES:
{tools_desc}

HISTORIQUE:
{memory_str}

Choisis une action:
1. use_tool: <nom_outil>(<paramètres>)
2. respond: <réponse finale>

Action:"""

    def _call_llm(self, prompt: str) -> str:
        """Appel LLM (simulé)"""
        # En production: appel réel au LLM
        return "respond: Voici la réponse"

    def _parse_action(self, response: str) -> tuple:
        """Parse l'action du LLM"""
        if response.startswith("respond:"):
            return "respond", response[8:].strip()
        elif response.startswith("use_tool:"):
            return "use_tool", response[9:].strip()
        else:
            return "unknown", response

    def _execute_tool(self, tool_call: str) -> str:
        """Execute un appel d'outil"""
        # Parser "tool_name(param)"
        match = re.match(r'(\w+)\((.*)\)', tool_call)
        if not match:
            return "Erreur: Format invalide"

        tool_name = match.group(1)
        param = match.group(2).strip('"').strip("'")

        # Trouver l'outil
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            return f"Erreur: Outil '{tool_name}' introuvable"

        # Executer
        try:
            result = tool.execute(param)
            return str(result)
        except Exception as e:
            return f"Erreur: {str(e)}"


# ============================================================================
# DÉMONSTRATIONS
# ============================================================================

def demo_function_calling():
    """Démo function calling"""
    print("="*80)
    print("DÉMONSTRATION: FUNCTION CALLING")
    print("="*80)

    # Créer fonctions
    weather_func = create_weather_function()
    calc_func = create_calculator_function()
    db_func = create_database_query_function()

    print("\n📋 FONCTIONS DISPONIBLES:")
    print("-" * 80)

    for func in [weather_func, calc_func, db_func]:
        print(f"\n{func.name}:")
        print(f"  Description: {func.description}")
        print(f"  Paramètres:")
        for param in func.parameters:
            req = "obligatoire" if param.required else "optionnel"
            print(f"    - {param.name} ({param.type}, {req}): {param.description}")

    # Schémas OpenAI
    print("\n\n🔧 SCHÉMA OPENAI:")
    print("-" * 80)
    print(json.dumps(weather_func.to_openai_schema(), indent=2))

    # Schéma Anthropic
    print("\n\n🔧 SCHÉMA ANTHROPIC (CLAUDE):")
    print("-" * 80)
    print(json.dumps(weather_func.to_anthropic_schema(), indent=2))

    # Test exécution
    print("\n\n✅ TEST EXÉCUTION:")
    print("-" * 80)

    result = weather_func.execute(location="Paris", unit="celsius")
    print(f"\nget_weather(location='Paris', unit='celsius'):")
    print(json.dumps(json.loads(result), indent=2))

    result2 = calc_func.execute(expression="25 * 17")
    print(f"\ncalculator(expression='25 * 17'):")
    print(json.dumps(json.loads(result2), indent=2))

    result3 = db_func.execute(filter_field="role", filter_value="admin")
    print(f"\nquery_users(filter_field='role', filter_value='admin'):")
    print(result3)


def demo_comparison():
    """Compare les différentes approches"""
    print("\n\n" + "="*80)
    print("COMPARAISON DES APPROCHES")
    print("="*80)

    print("""
┌─────────────────────┬──────────────────┬────────────────────┬─────────────────┐
│ Approche            │ Complexité       │ Flexibilité        │ Recommandé pour │
├─────────────────────┼──────────────────┼────────────────────┼─────────────────┤
│ Agent simple        │ ⭐ Faible        │ ⭐⭐ Moyenne       │ Prototypes      │
│ (ReAct DIY)         │                  │                    │ Learning        │
├─────────────────────┼──────────────────┼────────────────────┼─────────────────┤
│ Function calling    │ ⭐⭐ Moyenne     │ ⭐⭐⭐ Haute      │ Production      │
│ (OpenAI/Anthropic)  │                  │                    │ Reliability     │
├─────────────────────┼──────────────────┼────────────────────┼─────────────────┤
│ LangChain           │ ⭐⭐⭐ Haute    │ ⭐⭐⭐⭐ Très haute│ Prototypage     │
│                     │                  │                    │ Applications    │
├─────────────────────┼──────────────────┼────────────────────┼─────────────────┤
│ LangGraph           │ ⭐⭐⭐⭐ Très  │ ⭐⭐⭐⭐⭐ Max   │ Production      │
│                     │ haute            │                    │ Multi-agents    │
└─────────────────────┴──────────────────┴────────────────────┴─────────────────┘

RECOMMANDATIONS:

1. Débutant / Prototype:
   → Agent simple (ReAct DIY)
   → Comprendre les concepts fondamentaux

2. Production simple:
   → Function calling natif (OpenAI/Anthropic)
   → Robuste, type-safe, bien documenté

3. Application complexe:
   → LangChain pour prototypage rapide
   → LangGraph pour contrôle fin en production

4. Multi-agents / Workflows complexes:
   → LangGraph
   → Contrôle total sur les transitions
    """)


if __name__ == "__main__":
    # Démonstration function calling
    demo_function_calling()

    # Comparaison
    demo_comparison()

    print("\n\n" + "="*80)
    print("KEY TAKEAWAYS - PARTIE 2")
    print("="*80)
    print("""
1. FUNCTION CALLING = LLM qui génère JSON structuré
   • OpenAI: parameter "tools" avec schéma JSON
   • Anthropic: parameter "tools" avec "input_schema"
   • Robuste, type-safe, facile à intégrer

2. WORKFLOW STANDARD
   User → LLM (decide) → Tool call → Execute → LLM (respond) → User
   • LLM décide quand et quelle fonction appeler
   • Developer execute la fonction
   • LLM intègre résultat dans réponse

3. LANGCHAIN
   ✅ Framework complet pour applications LLM
   ✅ Beaucoup d'abstractions et d'outils
   ❌ Complexité élevée
   ❌ Overhead

4. BEST PRACTICES
   • Descriptions claires des fonctions
   • Validation des paramètres
   • Gestion d'erreurs robuste
   • Logging de tous les appels

PROCHAINE PARTIE: Multi-agents et LangGraph
    """)
